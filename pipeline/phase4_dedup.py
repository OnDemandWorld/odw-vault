"""Phase 4: Exact deduplication via SHA-256 hash grouping.

Pure SQL operation. Assigns dup_group_id to all files sharing a SHA-256,
marking all but one member as is_dup_primary=0. Canonical copy: shortest
rel_path, tiebreaker oldest mtime.
"""

from __future__ import annotations

from pipeline.config import Config
from pipeline.logging import PhaseLogger


def run_phase4(
    db,
    config: Config,
    plog: PhaseLogger,
) -> dict:
    """Run exact deduplication."""
    # Find duplicate groups (files sharing SHA-256)
    dup_groups = list(
        db.query("""
        SELECT sha256, COUNT(*) AS copies, MIN(id) AS min_id
        FROM file
        WHERE sha256 IS NOT NULL AND hash_status='done' AND excluded=0
        GROUP BY sha256
        HAVING copies > 1
    """)
    )

    if not dup_groups:
        plog.info("No duplicate files found.")
        return {"files_processed": 0, "files_failed": 0}

    total_dups = 0
    groups = len(dup_groups)

    plog.info(f"Found {groups} duplicate groups, deduplicating...")

    with db.conn:
        for group in dup_groups:
            sha256 = group["sha256"]
            min_id = group["min_id"]

            # Assign dup_group_id = smallest file.id in the group
            db.execute(
                "UPDATE file SET dup_group_id=? WHERE sha256=?",
                [min_id, sha256],
            )

            # Mark all but the canonical as non-primary
            db.execute(
                """
                UPDATE file SET is_dup_primary=0
                WHERE sha256=?
                AND id NOT IN (
                    SELECT id FROM file
                    WHERE sha256=?
                    ORDER BY LENGTH(rel_path) ASC, mtime ASC, id ASC
                    LIMIT 1
                )
            """,
                [sha256, sha256],
            )

            total_dups += group["copies"] - 1

    plog.info(
        f"Dedup complete: {groups} groups, {total_dups} duplicates marked",
        groups=groups,
        dups=total_dups,
    )
    return {"files_processed": groups, "files_failed": 0}
