"""textutil-based extractor for legacy macOS text files (.rtf, .doc, .txt, etc.)."""

from __future__ import annotations

import shutil
import subprocess


def extract_textutil(filepath: str) -> tuple[str | None, dict, bool, str | None]:
    """Use macOS textutil to convert the file to plain text.

    Returns (text, metadata, succeeded, error_message).
    """
    if not shutil.which("textutil"):
        return None, {}, False, "textutil not found (macOS only)"

    try:
        result = subprocess.run(
            ["textutil", "-convert", "txt", "-stdout", filepath],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            text = result.stdout
            return text, {"char_count": len(text)}, True, None
        else:
            return None, {}, False, result.stderr.strip() or "textutil returned non-zero exit code"
    except subprocess.TimeoutExpired:
        return None, {}, False, "textutil timed out after 60s"
    except Exception as e:
        return None, {}, False, str(e)
