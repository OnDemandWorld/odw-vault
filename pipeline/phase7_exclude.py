"""Phase 7: Mark exclusions and approval sign-off.

Allows marking individual files/folders as excluded, batch CSV import,
and final approval of the pre-flight results.
"""

from __future__ import annotations

import csv
from pathlib import Path

from sqlite_utils.db import NotFoundError

from pipeline.config import Config


def _exclude_folder_tree(db, folder_id: int, reason: str) -> None:
    """Exclude a folder, all descendant folders, and all their files.

    OCR audit #2: marking only the folder row left descendant files at
    excluded=0, so they were still extracted, embedded, and served.
    """
    rows = db.query(
        """
        WITH RECURSIVE subtree(id) AS (
            SELECT ?
            UNION ALL
            SELECT f.id FROM folder f JOIN subtree s ON f.parent_id = s.id
        )
        SELECT id FROM subtree
        """,
        [folder_id],
    )
    ids = [r["id"] for r in rows]
    if not ids:
        return
    placeholders = ",".join("?" for _ in ids)
    db.execute(
        f"UPDATE folder SET excluded = 1, exclusion_reason = ? WHERE id IN ({placeholders})",
        [reason, *ids],
    )
    db.execute(
        f"UPDATE file SET excluded = 1, exclusion_reason = ? WHERE folder_id IN ({placeholders})",
        [reason, *ids],
    )
    db.conn.commit()


def run_phase7_exclude(
    db,
    config: Config,
    target: str,
    item_id: int,
    reason: str,
) -> None:
    """Mark a single file or folder as excluded."""
    table = "folder" if target == "folder" else "file"
    try:
        db[table].get(item_id)
    except NotFoundError:
        raise ValueError(f"{target} with id {item_id} not found") from None

    if target == "folder":
        _exclude_folder_tree(db, item_id, reason)
        return

    db[table].update(
        item_id,
        {
            "excluded": 1,
            "exclusion_reason": reason,
        },
    )


def run_phase7_exclude_batch(
    db,
    config: Config,
    csv_path: Path,
) -> tuple[int, int]:
    """Batch exclude from CSV. Schema: target,id,reason.

    Returns (applied_count, skipped_count).
    """
    count = 0
    skipped = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            target = row["target"].strip()
            item_id = int(row["id"].strip())
            reason = row["reason"].strip()
            table = "folder" if target == "folder" else "file"
            try:
                existing = db[table].get(item_id)
            except NotFoundError:
                skipped += 1
                continue  # Skip nonexistent IDs with a count
            if existing:
                if target == "folder":
                    _exclude_folder_tree(db, item_id, reason)
                else:
                    db[table].update(
                        item_id,
                        {
                            "excluded": 1,
                            "exclusion_reason": reason,
                        },
                    )
                count += 1
    return count, skipped
