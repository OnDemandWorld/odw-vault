"""Phase 8: Text extraction pipeline.

Dispatches files to the appropriate extractor based on extract_strategy,
writes extraction artifacts to .rag-cache/extractions/, and records
results in the extraction and failure tables.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.progress import Progress, SpinnerColumn, TextColumn

from pipeline.config import AppConfig, ExtractConfig
from pipeline.db import finish_model_run, heartbeat_model_run, start_model_run
from pipeline.helpers import record_failure
from pipeline.logging import PhaseLogger

# Import extractors
from rag.extractors.docling_extractor import extract_docling
from rag.extractors.filename_only_extractor import extract_filename_only
from rag.extractors.metadata_only_extractor import extract_metadata_only
from rag.extractors.ocr_extractor import extract_ocr
from rag.extractors.textutil_extractor import extract_textutil
from rag.extractors.tika_extractor import extract_tika

# Strategy -> extractor mapping
EXTRACTOR_MAP = {
    "docling": extract_docling,
    "tika": extract_tika,
    "ocr": extract_ocr,
    "textutil": extract_textutil,
    "filename-only": extract_filename_only,
    "metadata-only": extract_metadata_only,
}

MAX_EXTRACT_TEXT_DB = 1_000_000  # Store only first 1MB in DB


def _build_config_hash(cfg: AppConfig) -> str:
    """Build a deterministic hash of the extraction config."""
    block = cfg.extract.model_dump()
    return hashlib.sha256(json.dumps(block, sort_keys=True).encode("utf-8")).hexdigest()


def _ensure_extraction_dir(cache_root: Path, sha256: str) -> Path:
    """Ensure the extraction output directory exists."""
    out_dir = cache_root / "extractions" / sha256[:2]
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _write_extraction_artifact(
    cache_root: Path,
    sha256: str,
    text: str | None,
    metadata: dict,
    strategy: str,
) -> str | None:
    """Write extraction markdown to .rag-cache/extractions/<sha[:2]>/<sha>.md."""
    if not text:
        return None
    try:
        out_dir = _ensure_extraction_dir(cache_root, sha256)
        out_path = out_dir / f"{sha256}.md"
        lines = [
            f"# Extracted: {metadata.get('filename', sha256)}",
            f"strategy: {strategy}",
            "---",
            "",
            text,
        ]
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return str(out_path)
    except Exception:
        return None


def _extract_one(
    file_row: dict,
    corpus_root: Path,
    cache_root: Path,
    extract_cfg: ExtractConfig,
) -> tuple[int, bool, str | None, str | None, int, int | None]:
    """Extract text for a single file.

    Returns (file_id, succeeded, error_message, artifact_path, char_count, page_count).
    """
    file_id = file_row["id"]
    filepath = file_row["path"]
    strategy = file_row["extract_strategy"]
    sha256 = file_row.get("sha256", "") or ""

    if not Path(filepath).exists():
        return file_id, False, "file not found on disk", None, 0, None

    extractor = EXTRACTOR_MAP.get(strategy)
    if extractor is None:
        return file_id, False, f"no extractor for strategy: {strategy}", None, 0, None

    try:
        # Tika extractor needs extra kwargs
        if strategy == "tika":
            text, meta, succeeded, err = extract_tika(
                filepath,
                tika_url=extract_cfg.tika_url,
                brute_force=extract_cfg.tika_brute_force_fallback,
            )
        else:
            text, meta, succeeded, err = extractor(filepath)
    except Exception as e:
        return file_id, False, str(e), None, 0, None

    if not succeeded:
        return file_id, False, err, None, 0, meta.get("page_count")

    char_count = len(text) if text else 0
    page_count = meta.get("page_count")

    # Write artifact
    artifact_path = _write_extraction_artifact(cache_root, sha256, text, meta, strategy)

    return file_id, True, None, artifact_path, char_count, page_count


def run_extract(
    db,
    cfg: AppConfig,
    plog: PhaseLogger | None = None,
    workers: int | None = None,
    strategy: str | None = None,
    limit: int | None = None,
    reextract: bool = False,
) -> dict:
    """Run extraction on eligible files.

    Args:
        db: sqlite_utils Database (must have a .conn for commit).
        cfg: AppConfig with extract sub-config.
        plog: Optional PhaseLogger.
        workers: Number of parallel workers (default: docling_workers from config).
        strategy: Filter to a single extract_strategy.
        limit: Process at most N files.
        reextract: Re-process files that already have an extraction row.

    Returns:
        (processed, failed) counts.
    """
    extract_cfg = cfg.extract
    corpus_root = cfg.corpus_root_path
    cache_root = cfg.cache_root_path
    n_workers = workers or extract_cfg.docling_workers

    # Select files to process
    where_parts = [
        "is_dup_primary=1",
        "excluded=0",
        "extract_strategy IS NOT NULL",
        "extract_strategy NOT IN ('skip', 'manual', 'unsupported')",
    ]

    if strategy:
        where_parts.append(f"extract_strategy='{strategy}'")

    if not reextract:
        # Exclude files that already have an extraction row
        where_parts.append("id NOT IN (SELECT file_id FROM extraction WHERE succeeded=1)")

    where_clause = " AND ".join(where_parts)

    query = f"SELECT * FROM file WHERE {where_clause}"
    if limit:
        query += f" LIMIT {limit}"

    files = list(db.query(query))

    if not files:
        msg = "No files eligible for extraction."
        if plog:
            plog.info(msg)
        return {"files_processed": 0, "files_failed": 0}

    # Start model run tracking
    config_hash = _build_config_hash(cfg)
    run_id = start_model_run(
        db,
        role="extraction",
        model_name="multi-extractor",
        config_hash=config_hash,
        phase="extract",
        tool="docling/tika/ocr/textutil",
    )

    processed = 0
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task(f"Extracting {len(files)} files...", total=len(files))

        def _process_row(row):
            return _extract_one(row, corpus_root, cache_root, extract_cfg)

        if n_workers <= 1:
            # Sequential
            results = [_process_row(row) for row in files]
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_process_row, row): row for row in files}
                results = []
                for future in as_completed(futures):
                    results.append(future.result())

        for file_id, succeeded, error_msg, artifact_path, char_count, page_count in results:
            # Get sha256 for this file
            file_row = next(
                iter(db.query("SELECT sha256, extract_strategy FROM file WHERE id=?", [file_id])),
                {},
            )
            strat = file_row.get("extract_strategy", "unknown")

            if succeeded:
                # Read full text from artifact if we need it for the DB
                text_for_db = None
                if artifact_path and Path(artifact_path).exists():
                    full_text = Path(artifact_path).read_text(encoding="utf-8")
                    # Strip the metadata header (first 4 lines)
                    lines = full_text.split("\n", 3)
                    text_for_db = lines[3] if len(lines) > 3 else full_text
                    # Truncate to 1MB for DB storage
                    if len(text_for_db) > MAX_EXTRACT_TEXT_DB:
                        text_for_db = text_for_db[:MAX_EXTRACT_TEXT_DB]

                db["extraction"].insert(
                    {
                        "file_id": file_id,
                        "tool": strat,
                        "text_extracted": text_for_db,
                        "char_count": char_count,
                        "page_count": page_count,
                        "succeeded": 1,
                        "error_message": None,
                    }
                )
                db.conn.commit()

                # Mark file as done (no extract_status column; track via extraction table)
                processed += 1

            else:
                db["extraction"].insert(
                    {
                        "file_id": file_id,
                        "tool": strat,
                        "text_extracted": None,
                        "char_count": 0,
                        "page_count": page_count,
                        "succeeded": 0,
                        "error_message": error_msg,
                    }
                )
                db.conn.commit()

                record_failure(
                    db,
                    file_id=file_id,
                    phase="extract",
                    tool=strat,
                    error_message=error_msg,
                )
                failed += 1

            # Heartbeat every 10 files
            if (processed + failed) % 10 == 0:
                heartbeat_model_run(db, run_id, processed, failed)

            progress.update(task, advance=1)

    status = "done" if failed == 0 and processed > 0 else ("done" if processed > 0 else "failed")
    notes = None
    if failed > 0:
        notes = f"{failed} file(s) failed extraction"

    finish_model_run(
        db,
        run_id,
        status=status,
        items_processed=processed,
        items_failed=failed,
        notes=notes,
    )

    msg = f"Extraction complete: {processed} extracted, {failed} failed"
    if plog:
        plog.info(msg, processed=processed, failed=failed)
    return {"files_processed": processed, "files_failed": failed}
