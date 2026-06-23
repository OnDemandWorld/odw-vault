"""Base extractor interface.

All extractors return (text, metadata, succeeded, error_message).
text: extracted string (None on failure)
metadata: dict of extractor-specific info (page counts, tool version, etc.)
succeeded: bool
error_message: str or None
"""

from __future__ import annotations

from typing import NamedTuple


class ExtractResult(NamedTuple):
    """Result from an extraction run."""

    text: str | None
    metadata: dict
    succeeded: bool
    error_message: str | None
