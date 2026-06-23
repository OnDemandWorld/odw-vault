"""Phase 3: Format-specific triage.

Cheap per-file inspection to estimate extraction cost.
Runs in a thread pool with per-thread DB connections.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz  # pymupdf
from rich.progress import Progress, SpinnerColumn, TextColumn

from pipeline.config import Config
from pipeline.helpers import record_failure
from pipeline.logging import PhaseLogger

# Language detection via lingua
try:
    from lingua import Language, LanguageDetectorBuilder

    HAS_LINGUA = True
    _detector = None

    def _get_detector():
        global _detector
        if _detector is None:
            _detector = (
                LanguageDetectorBuilder.from_languages(
                    Language.ENGLISH,
                    Language.CHINESE,
                )
                .with_minimum_relative_distance(0.0)
                .build()
            )
        return _detector
except ImportError:
    HAS_LINGUA = False


def _triage_pdf(filepath: str, config: Config) -> dict:
    out = {
        "is_encrypted": 0,
        "is_corrupt": 0,
        "page_count": None,
        "has_text_layer": None,
        "category_override": None,
        "triage_json": None,
        "detected_language": None,
    }
    try:
        doc = fitz.open(filepath)
    except Exception as e:
        out["is_corrupt"] = 1
        out["triage_json"] = json.dumps({"error": str(e)}, ensure_ascii=False)
        return out

    if doc.is_encrypted and not doc.authenticate(""):
        out["is_encrypted"] = 1
        doc.close()
        return out

    out["page_count"] = doc.page_count
    n = doc.page_count
    if n == 0:
        doc.close()
        out["triage_json"] = json.dumps({"sampled_pages": [], "avg_chars": 0}, ensure_ascii=False)
        out["has_text_layer"] = 0
        out["category_override"] = "pdf-scanned"
        return out

    sample_pages = sorted({0, n // 2, max(0, n - 1)})
    chars_total = 0
    text_sample = []
    for i in sample_pages:
        try:
            text = doc.load_page(i).get_text("text")
            chars_total += len(text)
            text_sample.append(text)
        except Exception:
            pass

    avg = chars_total / max(1, len(sample_pages))
    threshold = config.triage.pdf_text_threshold_chars_per_page
    out["has_text_layer"] = 1 if avg >= threshold else 0
    out["category_override"] = "pdf-text" if avg >= threshold else "pdf-scanned"

    if HAS_LINGUA and text_sample:
        combined = " ".join(text_sample)
        if len(combined) > 50:
            lang = _get_detector().detect_language_of(combined)
            if lang:
                out["detected_language"] = lang.name

    out["triage_json"] = json.dumps(
        {
            "sampled_pages": sample_pages,
            "avg_chars_per_sampled_page": round(avg, 1),
            "detected_language": out.get("detected_language"),
        },
        ensure_ascii=False,
    )
    doc.close()
    return out


def _triage_media(filepath: str) -> dict:
    out = {"duration_seconds": None, "triage_json": None, "is_corrupt": 0}
    if not shutil.which("ffprobe"):
        out["triage_json"] = json.dumps({"error": "ffprobe not found"}, ensure_ascii=False)
        return out
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", filepath],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            fmt = data.get("format", {})
            out["duration_seconds"] = float(fmt.get("duration", 0))
            out["triage_json"] = json.dumps(fmt, ensure_ascii=False)
        else:
            out["is_corrupt"] = 1
            out["triage_json"] = json.dumps({"error": result.stderr}, ensure_ascii=False)
    except subprocess.TimeoutExpired:
        out["triage_json"] = json.dumps({"error": "timeout"}, ensure_ascii=False)
    except Exception as e:
        out["is_corrupt"] = 1
        out["triage_json"] = json.dumps({"error": str(e)}, ensure_ascii=False)
    return out


def _triage_image(filepath: str) -> dict:
    out = {"width": None, "height": None, "triage_json": None, "is_corrupt": 0}
    try:
        pix = fitz.Pixmap(filepath)
        out["width"] = pix.width
        out["height"] = pix.height
        out["triage_json"] = json.dumps(
            {"width": pix.width, "height": pix.height}, ensure_ascii=False
        )
        pix = None
    except Exception as e:
        out["is_corrupt"] = 1
        out["triage_json"] = json.dumps({"error": str(e)}, ensure_ascii=False)
    return out


def _detect_language(filepath: str, category: str) -> str | None:
    if not HAS_LINGUA:
        return None
    if category not in ("document", "spreadsheet", "pdf-text", "data", "email", "ebook"):
        return None
    try:
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            text = f.read(4096)
        if len(text) < 50:
            return None
        lang = _get_detector().detect_language_of(text)
        return lang.name if lang else None
    except Exception:
        return None


def run_phase3(
    db,
    config: Config,
    plog: PhaseLogger,
    workers: int | None = None,
    categories: list[str] | None = None,
    db_path: str | None = None,
) -> dict:
    """Run triage on pending files."""
    where_clause = "triage_status='pending' AND excluded=0 AND is_dup_primary=1"
    category_params: list[str] = []
    if categories:
        placeholders = ",".join("?" for _ in categories)
        where_clause += f" AND category IN ({placeholders})"
        category_params = list(categories)

    files_to_triage = list(
        db.query(f"SELECT id, path, category FROM file WHERE {where_clause}", category_params)
    )

    if not files_to_triage:
        plog.info("No files pending triage.")
        return {"files_processed": 0, "files_failed": 0}

    n_workers = workers or min(os.cpu_count() or 4, 8)
    if db_path is None:
        db_path = config.cache_root_path / "corpus.db"

    # Accumulate results in main thread, then batch-update
    results: list[tuple[int, dict | None, bool]] = []

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True
    ) as progress:
        task = progress.add_task(
            f"Triage {len(files_to_triage)} files...", total=len(files_to_triage)
        )

        def _triage_one(file_row):
            # Open a new DB connection in this thread
            import sqlite_utils

            tdb = sqlite_utils.Database(db_path)
            tdb.conn.execute("PRAGMA foreign_keys = ON")

            file_id = file_row["id"]
            filepath = file_row["path"]
            category = file_row["category"]

            if not Path(filepath).exists():
                return file_id, {"error": "file not found"}, True

            try:
                if category in ("pdf-text", "pdf-scanned") or (category and "pdf" in category):
                    result = _triage_pdf(filepath, config)
                    if result.get("category_override"):
                        tdb.execute(
                            "UPDATE file SET category=? WHERE id=?",
                            [result["category_override"], file_id],
                        )
                    tdb.execute(
                        "UPDATE file SET is_encrypted=?, is_corrupt=?, page_count=?, "
                        "has_text_layer=?, triage_json=?, triage_status='done' WHERE id=?",
                        [
                            result["is_encrypted"],
                            result["is_corrupt"],
                            result["page_count"],
                            result["has_text_layer"],
                            result["triage_json"],
                            file_id,
                        ],
                    )
                    tdb.conn.commit()  # Ensure triage results are persisted
                elif category in ("audio", "video"):
                    result = _triage_media(filepath)
                    tdb.execute(
                        "UPDATE file SET duration_seconds=?, triage_json=?, triage_status='done' WHERE id=?",
                        [result["duration_seconds"], result["triage_json"], file_id],
                    )
                    tdb.conn.commit()
                elif category == "image":
                    result = _triage_image(filepath)
                    tdb.execute(
                        "UPDATE file SET triage_json=?, triage_status='done' WHERE id=?",
                        [result["triage_json"], file_id],
                    )
                    tdb.conn.commit()
                else:
                    # Generic text-bearing files — detect language
                    lang = _detect_language(filepath, category)
                    tdb.execute(
                        "UPDATE file SET triage_json=?, triage_status='done' WHERE id=?",
                        [
                            json.dumps({"detected_language": lang}, ensure_ascii=False)
                            if lang
                            else None,
                            file_id,
                        ],
                    )
                    tdb.conn.commit()
                    return file_id, None, False

                if result.get("is_corrupt") or result.get("error"):
                    return file_id, result, True
                return file_id, result, False
            finally:
                tdb.conn.close()

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_triage_one, fr): fr for fr in files_to_triage}
            for future in as_completed(futures):
                file_id, result, error = future.result()
                if error and result:
                    record_failure(
                        db,
                        file_id=file_id,
                        phase="triage",
                        error_message=str(result.get("error", "triage error")),
                    )
                    results.append((file_id, result, True))
                else:
                    results.append((file_id, result, False))
                progress.update(task, advance=1)

    triaged = sum(1 for _, _, e in results if not e)
    failed = sum(1 for _, _, e in results if e)

    plog.info(
        f"Triage complete: {triaged} triaged, {failed} failed", triaged=triaged, failed=failed
    )
    return {"files_processed": triaged, "files_failed": failed}
