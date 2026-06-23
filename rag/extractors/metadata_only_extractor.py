"""Metadata-only extractor for files where only file-level metadata is useful."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path


def extract_metadata_only(filepath: str) -> tuple[str | None, dict, bool, str | None]:
    """Return file metadata as structured text. No document text content.

    Returns (text, metadata, succeeded, error_message).
    """
    try:
        p = Path(filepath)
        stat = p.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
        meta = {
            "size_bytes": stat.st_size,
            "mtime": mtime,
            "parent_folder": p.parent.name,
            "full_path": str(p),
        }
        text = (
            f"File: {p.name}\n"
            f"Folder: {p.parent.name}\n"
            f"Size: {stat.st_size} bytes\n"
            f"Modified: {mtime}"
        )
        return text, meta, True, None
    except Exception as e:
        return None, {}, False, str(e)
