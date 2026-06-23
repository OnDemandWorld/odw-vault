"""Phase 1: Walk the corpus tree and compute SHA-256 hashes.

Inserts folder and file rows into the database. Skips hidden/system files,
.rag-cache contents, and oversized files. Hashes files in parallel using
ProcessPoolExecutor.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from rich.progress import Progress, SpinnerColumn, TextColumn

from pipeline.config import Config
from pipeline.helpers import (
    is_hidden_or_system,
    is_system_dir,
    now_iso,
    record_failure,
    sha256_file,
)
from pipeline.logging import PhaseLogger


def _should_skip_file(name: str, path: Path, config: Config) -> bool:
    """Determine if a file should be skipped during walk."""
    if is_hidden_or_system(name):
        return True
    if path.resolve().is_relative_to(config.cache_root_path.resolve()):
        return True
    return False


def _should_skip_dir(name: str) -> bool:
    return is_hidden_or_system(name) or is_system_dir(name)


def _row_by_path(db, table: str, abs_path: str):
    """Get a single row from table by path, or None."""
    return next(db[table].rows_where("path = ?", [abs_path]), None)


def _ensure_folder(db, folder_path: Path, corpus_root: Path) -> int:
    """Ensure a folder row exists. Returns folder_id."""
    abs_path = str(folder_path.resolve())
    rel_path = str(folder_path.relative_to(corpus_root))
    depth = len(Path(rel_path).parts)

    existing = _row_by_path(db, "folder", abs_path)
    if existing:
        return existing["id"]

    parent_id = None
    if depth > 0:
        parent_abs = str(folder_path.parent.resolve())
        if parent_abs != abs_path:
            parent_row = _row_by_path(db, "folder", parent_abs)
            if parent_row:
                parent_id = parent_row["id"]

    db["folder"].insert(
        {
            "path": abs_path,
            "rel_path": rel_path,
            "parent_id": parent_id,
            "name": folder_path.name if folder_path.name else corpus_root.name,
            "depth": depth,
        }
    )
    row = _row_by_path(db, "folder", abs_path)
    return row["id"] if row else 0


def run_phase1(
    db,
    config: Config,
    plog: PhaseLogger,
    workers: int | None = None,
    rehash: bool = False,
) -> dict:
    """Walk corpus and hash all files."""
    root = config.corpus_root_path
    cache_root = config.cache_root_path
    max_size = config.walk.max_file_size_bytes
    n_workers = workers or config.walk.hash_workers or os.cpu_count() or 4

    files_to_hash: list[str] = []
    total_files = 0
    total_folders = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Walking corpus tree...", total=None)

        for dirpath, dirnames, filenames in os.walk(root):
            dir_path = Path(dirpath)
            dirnames[:] = [d for d in dirnames if not _should_skip_dir(d)]
            cache_resolved = config.cache_root_path.resolve()
            dirnames[:] = [
                d for d in dirnames
                if not (dir_path / d).resolve().is_relative_to(cache_resolved)
            ]

            _ensure_folder(db, dir_path, root)
            total_folders += 1

            for fn in filenames:
                fp = Path(dirpath) / fn
                if _should_skip_file(fn, fp, config):
                    continue

                abs_path = str(fp.resolve())
                rel_path = str(fp.relative_to(root))

                if not rehash:
                    existing = _row_by_path(db, "file", abs_path)
                    if existing and existing.get("hash_status") in ("done", "skipped"):
                        # File already in DB (from phase0 or previous walk), just add to hash list
                        files_to_hash.append(abs_path)
                        total_files += 1
                        continue
                else:
                    # Check if file exists even in rehash mode
                    existing = _row_by_path(db, "file", abs_path)

                try:
                    stat = fp.stat()
                    size = stat.st_size
                    mtime = now_iso()
                except OSError as e:
                    record_failure(
                        db,
                        phase="walk",
                        error_class="permission",
                        error_message=f"Cannot stat {abs_path}: {e}",
                    )
                    continue

                if size > max_size:
                    record_failure(
                        db,
                        phase="walk",
                        error_class="oversized",
                        error_message=f"File too large: {size} bytes ({abs_path})",
                    )
                    continue

                folder_id = _ensure_folder(db, dir_path, root)

                if existing:
                    # Update existing file row instead of replacing
                    db.execute(
                        "UPDATE file SET hash_status='pending', identify_status='pending', "
                        "triage_status='pending', size_bytes=?, mtime=? WHERE id=?",
                        [size, mtime, existing["id"]],
                    )
                else:
                    db["file"].insert(
                        {
                            "folder_id": folder_id,
                            "path": abs_path,
                            "rel_path": rel_path,
                            "name": fn,
                            "extension": fp.suffix.lower() if fp.suffix else None,
                            "size_bytes": size,
                            "mtime": mtime,
                            "hash_status": "pending",
                            "identify_status": "pending",
                            "triage_status": "pending",
                        }
                    )
                files_to_hash.append(abs_path)
                total_files += 1

        progress.update(
            task, description=f"Hashing {len(files_to_hash)} files with {n_workers} workers..."
        )

    hashed = 0
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Computing SHA-256...", total=len(files_to_hash))

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(sha256_file, p): p for p in files_to_hash}
            for future in as_completed(futures):
                path, digest, error = future.result()
                if error:
                    db.execute(
                        "UPDATE file SET hash_status='failed', error_message=? WHERE path=?",
                        [error, path],
                    )
                    record_failure(
                        db,
                        phase="walk",
                        error_class="permission",
                        error_message=f"Hash failed for {path}: {error}",
                    )
                    failed += 1
                else:
                    db.execute(
                        "UPDATE file SET sha256=?, hash_status='done' WHERE path=?",
                        [digest, path],
                    )
                    hashed += 1
                progress.update(task, advance=1)

    plog.info(
        f"Walk complete: {total_files} files, {total_folders} folders, "
        f"{hashed} hashed, {failed} failed",
        files=total_files,
        folders=total_folders,
        hashed=hashed,
        failed=failed,
    )
    return {"files_processed": hashed, "files_failed": failed}
