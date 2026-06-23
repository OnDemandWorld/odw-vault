"""Filename-only extractor for non-extractable files (CAD, executables, etc.)."""

from __future__ import annotations

from pathlib import Path


def extract_filename_only(filepath: str) -> tuple[str | None, dict, bool, str | None]:
    """Return a one-line string with the filename and parent folder name.

    Returns (text, metadata, succeeded, error_message).
    """
    try:
        p = Path(filepath)
        text = f"{p.name} (folder: {p.parent.name})"
        return text, {"filename": p.name, "parent_folder": p.parent.name}, True, None
    except Exception as e:
        return None, {}, False, str(e)
