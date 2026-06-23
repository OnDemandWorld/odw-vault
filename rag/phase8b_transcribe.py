"""Phase 8b: Audio/video transcription pipeline.

Uses pywhispercpp with the large-v3 model for transcription.
Stores word-timestamp JSON under .rag-cache/transcripts/, and
a markdown rendering under .rag-cache/extractions/.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
from pathlib import Path

from rich.progress import Progress, SpinnerColumn, TextColumn

from pipeline.config import AppConfig
from pipeline.db import finish_model_run, heartbeat_model_run, start_model_run
from pipeline.helpers import record_failure
from pipeline.logging import PhaseLogger


def _build_config_hash(cfg: AppConfig) -> str:
    """Deterministic hash of transcription config."""
    block = cfg.models.transcription.model_dump()
    return hashlib.sha256(json.dumps(block, sort_keys=True).encode("utf-8")).hexdigest()


def _ensure_transcript_dir(cache_root: Path, sha256: str) -> Path:
    out_dir = cache_root / "transcripts" / sha256[:2]
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _transcribe_file(
    filepath: str,
    sha256: str,
    cache_root: Path,
    model_name: str = "large-v3",
    language: str = "auto",
    threads: int = 8,
    word_timestamps: bool = True,
) -> tuple[bool, str | None, dict | None]:
    """Transcribe a single audio/video file.

    Returns (succeeded, transcript_json_path, error_info).
    """
    try:
        from pywhispercpp.model import Model

        model = Model(model=model_name, fname_audio=filepath)
        model.print_system_info()

        # Run transcription with parameters
        params = model.new_full_params(
            language=None if language == "auto" else language,
            n_threads=threads,
            no_timestamps=False,
            word_timestamps=word_timestamps,
        )

        model.process(
            fname_audio=filepath,
            params=params,
            new_segment_callback=lambda seg: None,  # placeholder
        )

        # Extract results
        segments = []
        for seg in model.segments:
            seg_data = {
                "text": seg.text,
                "start": seg.start,
                "end": seg.end,
            }
            if word_timestamps and hasattr(seg, "words"):
                seg_data["words"] = [
                    {"text": w.text, "start": w.start, "end": w.end} for w in seg.words
                ]
            segments.append(seg_data)

        transcript = {
            "sha256": sha256,
            "model": model_name,
            "language": language,
            "segments": segments,
        }

        # Write transcript JSON
        out_dir = _ensure_transcript_dir(cache_root, sha256)
        transcript_path = out_dir / f"{sha256}.json"
        transcript_path.write_text(
            json.dumps(transcript, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Write markdown rendering
        md_lines = [f"# Transcript: {Path(filepath).name}", ""]
        for seg in segments:
            md_lines.append(seg["text"].strip())
        md_path = cache_root / "extractions" / sha256[:2] / f"{sha256}.md"
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text("\n".join(md_lines), encoding="utf-8")

        return True, str(transcript_path), None

    except ImportError:
        return False, None, {"error": "pywhispercpp not installed"}
    except Exception as e:
        return False, None, {"error": str(e)}


def run_transcribe(
    db,
    cfg: AppConfig,
    plog: PhaseLogger | None = None,
    workers: int = 1,
    folder: str | None = None,
    limit: int | None = None,
    model: str | None = None,
) -> dict:
    """Run transcription on eligible audio/video files.

    Args:
        db: sqlite_utils Database.
        cfg: AppConfig with models.transcription sub-config.
        plog: Optional PhaseLogger.
        workers: Number of parallel transcription workers.
        folder: Filter by rel_path prefix.
        limit: Transcribe at most N files.
        model: Override transcription model name.

    Returns:
        (processed, failed) counts.
    """
    tcfg = cfg.models.transcription
    cache_root = cfg.cache_root_path
    model_name = model or tcfg.model

    # Select audio/video files
    where_parts = [
        "is_dup_primary=1",
        "excluded=0",
        "category IN ('audio', 'video')",
    ]

    if folder:
        where_parts.append(f"rel_path LIKE '{folder}/%'")

    where_clause = " AND ".join(where_parts)

    query = f"SELECT * FROM file WHERE {where_clause}"
    if limit:
        query += f" LIMIT {limit}"

    files = list(db.query(query))

    if not files:
        msg = "No audio/video files eligible for transcription."
        if plog:
            plog.info(msg)
        return {"files_processed": 0, "files_failed": 0}

    # Apply opt-in globs if configured
    opt_in = tcfg.opt_in_globs
    if opt_in:
        filtered = []
        for f in files:
            rel = f.get("rel_path", "")
            for pattern in opt_in:
                if fnmatch.fnmatch(rel, pattern):
                    filtered.append(f)
                    break
        files = filtered

    if not files:
        msg = "No files match opt-in glob patterns."
        if plog:
            plog.info(msg)
        return {"files_processed": 0, "files_failed": 0}

    # Start model run tracking
    config_hash = _build_config_hash(cfg)
    run_id = start_model_run(
        db,
        role="transcription",
        model_name=model_name,
        config_hash=config_hash,
        phase="transcribe",
        tool="whisper.cpp",
    )

    processed = 0
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task(f"Transcribing {len(files)} files...", total=len(files))

        for file_row in files:
            file_id = file_row["id"]
            filepath = file_row["path"]
            sha256 = (file_row.get("sha256") or "").strip()

            if not Path(filepath).exists():
                record_failure(
                    db,
                    file_id=file_id,
                    phase="transcribe",
                    tool=model_name,
                    error_message="file not found on disk",
                )
                db["extraction"].insert(
                    {
                        "file_id": file_id,
                        "tool": "whisper.cpp",
                        "text_extracted": None,
                        "char_count": 0,
                        "page_count": None,
                        "succeeded": 0,
                        "error_message": "file not found on disk",
                    }
                )
                db.conn.commit()
                failed += 1
                # Heartbeat every 5 files
                if (processed + failed) % 5 == 0:
                    heartbeat_model_run(db, run_id, processed, failed)
                progress.update(task, advance=1)
                continue

            succeeded, transcript_path, error_info = _transcribe_file(
                filepath,
                sha256,
                cache_root,
                model_name=model_name,
                language=tcfg.language,
                threads=tcfg.threads,
                word_timestamps=tcfg.word_timestamps,
            )

            if succeeded:
                # Read transcript to get text for the DB
                if transcript_path:
                    with open(transcript_path, encoding="utf-8") as f:
                        data = json.load(f)
                    full_text = "\n".join(s["text"].strip() for s in data.get("segments", []))
                else:
                    full_text = ""

                db["extraction"].insert(
                    {
                        "file_id": file_id,
                        "tool": "whisper.cpp",
                        "text_extracted": full_text[:1_000_000] if full_text else None,
                        "char_count": len(full_text),
                        "page_count": None,
                        "succeeded": 1,
                        "error_message": None,
                    }
                )
                db.conn.commit()
                processed += 1

            else:
                err_msg = error_info.get("error", "unknown") if error_info else "unknown"
                db["extraction"].insert(
                    {
                        "file_id": file_id,
                        "tool": "whisper.cpp",
                        "text_extracted": None,
                        "char_count": 0,
                        "page_count": None,
                        "succeeded": 0,
                        "error_message": err_msg,
                    }
                )
                db.conn.commit()

                record_failure(
                    db,
                    file_id=file_id,
                    phase="transcribe",
                    tool=model_name,
                    error_message=err_msg,
                )
                failed += 1

            progress.update(task, advance=1)

            # Heartbeat every 5 files
            if (processed + failed) % 5 == 0:
                heartbeat_model_run(db, run_id, processed, failed)

    status = "done" if failed == 0 and processed > 0 else ("done" if processed > 0 else "failed")
    notes = f"{failed} file(s) failed transcription" if failed > 0 else None

    finish_model_run(
        db,
        run_id,
        status=status,
        items_processed=processed,
        items_failed=failed,
        notes=notes,
    )

    msg = f"Transcription complete: {processed} transcribed, {failed} failed"
    if plog:
        plog.info(msg, processed=processed, failed=failed)
    return {"files_processed": processed, "files_failed": failed}
