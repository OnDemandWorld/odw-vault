"""Text extraction backends for Phase 8."""

from __future__ import annotations

from pipeline.config import ExtractConfig

from .base import ExtractResult as ExtractResult

# Re-export individual extraction functions
from .docling_extractor import extract_docling
from .filename_only_extractor import extract_filename_only
from .metadata_only_extractor import extract_metadata_only
from .ocr_extractor import extract_ocr
from .textutil_extractor import extract_textutil
from .tika_extractor import extract_tika
from .whisper_extractor import extract_whisper

_STRATEGY_MAP = {
    "docling": extract_docling,
    "tika": extract_tika,
    "ocr": extract_ocr,
    "whisper": extract_whisper,
    "filename-only": extract_filename_only,
    "metadata-only": extract_metadata_only,
    "textutil": extract_textutil,
}


def get_extractor(strategy: str, config: ExtractConfig | None = None):
    """Return a callable for the given strategy.

    The returned function accepts (filepath: str) and returns
    (text: str | None, metadata: dict, succeeded: bool, error_message: str | None).
    """
    fn = _STRATEGY_MAP.get(strategy)
    if fn is None:
        raise ValueError(f"Unknown extract strategy: {strategy}")
    if strategy == "tika" and config:
        return lambda fp: extract_tika(
            fp, tika_url=config.tika_url, brute_force=config.tika_brute_force_fallback
        )
    if strategy == "whisper" and config:
        from pipeline.config import TranscriptionConfig

        tc = TranscriptionConfig()
        return lambda fp: extract_whisper(
            fp, model_name=tc.model, language=tc.language, threads=tc.threads
        )
    return fn
