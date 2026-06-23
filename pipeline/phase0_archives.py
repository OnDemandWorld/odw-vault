"""Phase 0: Recursive archive expansion.

Expands archives in place, creating sibling <archive>.extracted/ folders.
Idempotent: skips already-expanded archives.
"""

from __future__ import annotations

import os
from pathlib import Path

import patoolib
from rich.progress import Progress, SpinnerColumn, TextColumn

from pipeline.config import Config
from pipeline.helpers import (
    DOC_ARCHIVES,
    classify_archive_error,
    is_archive_extension,
    is_hidden_or_system,
    is_system_dir,
    now_iso,
    record_expansion,
    record_failure,
)
from pipeline.logging import PhaseLogger


def _find_archives(root: Path, config: Config) -> list[Path]:
    archives = []
    cache_resolved = config.cache_root_path.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d
            for d in dirnames
            if not is_hidden_or_system(d)
            and not is_system_dir(d)
            and not (Path(dirpath) / d).resolve().is_relative_to(cache_resolved)
        ]
        for fn in filenames:
            if is_hidden_or_system(fn):
                continue
            fp = Path(dirpath) / fn
            ext = fp.suffix.lower()
            if ext in DOC_ARCHIVES:
                continue
            if is_archive_extension(ext):
                archives.append(fp)
    return archives


def _ensure_folder(db, folder_path: str, folder_rel: str, config: Config) -> int | None:
    """Ensure a folder row exists. Returns folder_id."""
    existing = next(db["folder"].rows_where("path = ?", [folder_path]), None)
    if existing:
        return existing["id"]

    parent_abs = str(Path(folder_path).parent.resolve())
    parent_rel = str(Path(folder_rel).parent) if folder_rel else ""
    parent_id = None
    if parent_rel:
        parent_row = next(db["folder"].rows_where("path = ?", [parent_abs]), None)
        if parent_row:
            parent_id = parent_row["id"]

    db["folder"].insert(
        {
            "path": folder_path,
            "rel_path": folder_rel,
            "parent_id": parent_id,
            "name": Path(folder_path).name,
            "depth": len(Path(folder_rel).parts) if folder_rel else 0,
        }
    )
    folder = next(db["folder"].rows_where("path = ?", [folder_path]), None)
    return folder["id"] if folder else None


def _ensure_file_in_db(db, file_path: Path, config: Config) -> int | None:
    """Ensure file and folder rows exist in DB. Returns file_id."""
    abs_path = str(file_path.resolve())
    rel_path = str(file_path.relative_to(config.corpus_root_path))
    folder_path = str(file_path.parent.resolve())
    folder_rel = str(file_path.parent.relative_to(config.corpus_root_path))

    # Ensure folder
    folder_id = _ensure_folder(db, folder_path, folder_rel, config)

    existing = next(db["file"].rows_where("path = ?", [abs_path]), None)
    if existing:
        return existing["id"]

    stat = file_path.stat()
    db["file"].insert(
        {
            "folder_id": folder_id,
            "path": abs_path,
            "rel_path": rel_path,
            "name": file_path.name,
            "extension": file_path.suffix.lower(),
            "size_bytes": stat.st_size,
            "mtime": now_iso(),
            "hash_status": "skipped",
            "identify_status": "skipped",
            "triage_status": "skipped",
        },
        ignore=True,
    )
    row = next(db["file"].rows_where("path = ?", [abs_path]), None)
    return row["id"] if row else None


def run_phase0(
    db, config: Config, plog: PhaseLogger, max_depth: int | None = None, dry_run: bool = False
) -> dict:
    depth_limit = max_depth or config.archives.max_depth
    root = config.corpus_root_path
    expanded_total = 0
    failed_total = 0

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True
    ) as progress:
        task = progress.add_task("Expanding archives...", total=None)

        for depth in range(depth_limit):
            archives = _find_archives(root, config)

            already = set()
            for r in db.query(
                "SELECT f.path FROM archive_expansion ae JOIN file f ON f.id = ae.archive_file_id WHERE ae.succeeded = 1"
            ):
                already.add(r["path"])
            archives = [a for a in archives if str(a.resolve()) not in already]

            if not archives:
                plog.info(
                    f"No archives to expand at depth {depth}." if depth else "No archives found."
                )
                break

            progress.update(
                task, description=f"Expanding {len(archives)} archives (depth {depth})..."
            )

            for arc in archives:
                if dry_run:
                    expanded_total += 1
                    continue

                target = arc.parent / f"{arc.name}.extracted"
                target.mkdir(parents=True, exist_ok=True)

                try:
                    patoolib.extract_archive(str(arc), outdir=str(target), verbosity=-1)
                    file_count = sum(1 for _ in target.rglob("*") if _.is_file())

                    arc_id = _ensure_file_in_db(db, arc, config)

                    folder_path = str(target.resolve())
                    folder_rel = str(target.relative_to(config.corpus_root_path))
                    parent_abs = str(target.parent.resolve())
                    parent_row = next(db["folder"].rows_where("path = ?", [parent_abs]), None)
                    parent_id = parent_row["id"] if parent_row else None
                    db["folder"].insert(
                        {
                            "path": folder_path,
                            "rel_path": folder_rel,
                            "parent_id": parent_id,
                            "name": target.name,
                            "depth": len(Path(folder_rel).parts),
                            "is_extracted_archive": 1,
                        }
                    )
                    folder_row = next(db["folder"].rows_where("path = ?", [folder_path]), None)
                    folder_id = folder_row["id"] if folder_row else None

                    record_expansion(
                        db,
                        archive_file_id=arc_id or 0,
                        extracted_to_path=str(target.resolve()),
                        extracted_to_folder_id=folder_id,
                        tool="patool",
                        succeeded=True,
                        file_count=file_count,
                        error_message=None,
                    )
                    expanded_total += 1
                except Exception as e:
                    arc_id = _ensure_file_in_db(db, arc, config)
                    record_expansion(
                        db,
                        archive_file_id=arc_id or 0,
                        extracted_to_path=str(target.resolve()),
                        extracted_to_folder_id=None,
                        tool="patool",
                        succeeded=False,
                        file_count=0,
                        error_message=str(e),
                    )
                    record_failure(
                        db,
                        file_id=arc_id,
                        phase="archives",
                        tool="patool",
                        error_class=classify_archive_error(e),
                        error_message=str(e),
                    )
                    failed_total += 1
                    plog.warning(f"Failed to expand {arc}: {e}")

    plog.info(f"Archive expansion: {expanded_total} expanded, {failed_total} failed")
    return {"files_processed": expanded_total, "files_failed": failed_total}
