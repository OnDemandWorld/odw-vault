"""Folder filter resolution for retrieval scoping."""

from __future__ import annotations

import sqlite_utils


def resolve_folder_filter(db: sqlite_utils.Database, folder_filter: dict) -> set[int] | None:
    """Resolve folder_filter to a set of allowed file_ids.

    Supports:
    - {"path_prefix": "Project/Gleneagles"} — files under folders matching rel_path LIKE prefix%
    - {"folder_id": 42} — files directly in the given folder
    - {"inferred_category": "deployment"} — files in folders with the given inferred_category
    - {"workspace": "team-a"} — files tagged with the given workspace (V1.1 M4);
      composes via intersection with any of the filters above. When omitted the
      whole corpus is eligible (V1.0 behavior).

    Returns None if no filter is provided (no scoping needed).
    """
    if not folder_filter:
        return None

    # Workspace constraint (V1.1 M4, additive). Resolved first so it can be
    # intersected with the path/folder/category filters below. Absent key =>
    # no constraint => behavior identical to V1.0.
    workspace_ids: set[int] | None = None
    if folder_filter.get("workspace") is not None:
        rows = db.query(
            "SELECT id FROM file WHERE workspace = ? AND excluded = 0",
            [folder_filter["workspace"]],
        )
        workspace_ids = {r["id"] for r in rows}
        if not workspace_ids:
            return None

    allowed_ids: set[int] = set()

    if "path_prefix" in folder_filter:
        prefix = folder_filter["path_prefix"]
        rows = db.query(
            "SELECT f.id FROM file f JOIN folder fo ON f.folder_id = fo.id "
            "WHERE fo.rel_path LIKE ? AND f.excluded = 0",
            [f"{prefix}%"],
        )
        allowed_ids.update(r["id"] for r in rows)

    if "folder_id" in folder_filter:
        fid = folder_filter["folder_id"]
        rows = db.query("SELECT id FROM file WHERE folder_id = ? AND excluded = 0", [fid])
        allowed_ids.update(r["id"] for r in rows)

    if "inferred_category" in folder_filter:
        cat = folder_filter["inferred_category"]
        rows = db.query(
            "SELECT f.id FROM file f JOIN folder fo ON f.folder_id = fo.id "
            "WHERE fo.inferred_category = ? AND f.excluded = 0",
            [cat],
        )
        allowed_ids.update(r["id"] for r in rows)

    # Compose the workspace constraint (intersection) with the union above.
    if workspace_ids is not None:
        allowed_ids = (allowed_ids & workspace_ids) if allowed_ids else workspace_ids

    return allowed_ids if allowed_ids else None
