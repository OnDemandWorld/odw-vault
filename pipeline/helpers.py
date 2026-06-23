"""Shared utility helpers used across all pipeline phases."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# macOS-specific system files to skip
SYSTEM_FILES = {".DS_Store", ".localized", ".fseventsd", ".Spotlight-V100", ".Trashes"}
ARCHIVE_OS_PATTERNS = {"__MACOSX", ".AppleDouble", ".AppleDesktop"}

# Document extensions that look like archives but should NOT be expanded
DOC_ARCHIVES = {".pages", ".numbers", ".key", ".docx", ".xlsx", ".pptx", ".epub", ".jar"}

ARCHIVE_EXTENSIONS = {
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz2",
    ".tar.xz",
    ".txz",
    ".gz",
    ".bz2",
    ".xz",
    ".lz",
    ".lzma",
    ".lzh",
    ".zst",
    ".cab",
    ".arj",
    ".cpio",
    ".rpm",
    ".deb",
    ".iso",
}

ARCHIVE_MAGIC = {
    b"PK\x03\x04": "zip",
    b"PK\x05\x06": "zip",  # empty zip
    b"PK\x07\x08": "zip",  # spanned zip
    b"Rar!\x1a\x07": "rar",
    b"7z\xbc\xaf\x27\x1c": "7z",
    b"\x1f\x8b": "gzip",
    b"BZh": "bzip2",
    b"\xfd7zXZ": "xz",
    b"\x1d": "lzh",
}


def now_iso() -> str:
    """Return current UTC time in ISO-8601 format."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str | Path) -> tuple[str, str | None, str | None]:
    """Compute SHA-256 of a file in 1 MB chunks. Returns (path, digest, error)."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return str(path), h.hexdigest(), None
    except OSError as e:
        return str(path), None, str(e)


def is_hidden_or_system(name: str) -> bool:
    """Check if a file/folder name is hidden (dot-prefixed) or a macOS system file."""
    return name.startswith(".") or name in SYSTEM_FILES


def is_system_dir(name: str) -> bool:
    """Check if a directory name is a macOS system directory."""
    return name in ARCHIVE_OS_PATTERNS


def is_archive_extension(ext: str) -> bool:
    """Check if a file extension is an archive type (excluding doc archives)."""
    ext = ext.lower()
    # DOC_ARCHIVES (docx, xlsx, etc.) do not overlap with ARCHIVE_EXTENSIONS,
    # but the check is kept defensively in case either set is extended.
    return ext in ARCHIVE_EXTENSIONS and ext not in DOC_ARCHIVES


def classify_archive_error(exc: Exception) -> str:
    """Map an archive extraction exception to an error class."""
    msg = str(exc).lower()
    if "encrypted" in msg or "password" in msg:
        return "encrypted"
    if "corrupt" in msg or "invalid" in msg or "damaged" in msg:
        return "corrupt"
    if "permission" in msg or "denied" in msg:
        return "permission"
    if "timeout" in msg:
        return "timeout"
    return "unknown"


def sanitize_json_for_sqlite(obj: Any) -> str:
    """Serialize an object to JSON string for SQLite storage."""
    return json.dumps(obj, ensure_ascii=False, default=str)


def record_failure(
    db: Any,
    *,
    file_id: int | None = None,
    folder_id: int | None = None,
    phase: str,
    tool: str | None = None,
    error_class: str | None = None,
    error_message: str | None = None,
    traceback_str: str | None = None,
) -> None:
    """Insert a row into the failure table."""
    db["failure"].insert(
        {
            "file_id": file_id,
            "folder_id": folder_id,
            "phase": phase,
            "tool": tool,
            "error_class": error_class,
            "error_message": error_message,
            "traceback": traceback_str,
        }
    )


def record_expansion(
    db: Any,
    archive_file_id: int,
    extracted_to_path: str,
    extracted_to_folder_id: int | None,
    tool: str,
    succeeded: bool,
    file_count: int | None,
    error_message: str | None,
) -> None:
    """Insert a row into the archive_expansion table."""
    db["archive_expansion"].insert(
        {
            "archive_file_id": archive_file_id,
            "extracted_to_path": extracted_to_path,
            "extracted_to_folder_id": extracted_to_folder_id,
            "tool": tool,
            "succeeded": 1 if succeeded else 0,
            "file_count": file_count,
            "error_message": error_message,
        }
    )
