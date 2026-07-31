"""Language-aware tokenization for BM25/FTS retrieval (V1.2 M2 — F-Vault-2).

Provides a small ``tokenize(text, lang)`` abstraction used to improve Chinese
retrieval at the BM25 layer:

  * ``lang == "en"`` (default) → whitespace tokenization, i.e. the existing
    behaviour. The English BM25 index/query path is intentionally left
    untouched; this abstraction only *adds* a Chinese path.
  * ``lang == "zh"`` → Chinese word segmentation. Uses ``jieba`` when it is
    importable (declared as an *optional* dependency); otherwise falls back to
    a dependency-free character-bigram tokenizer so retrieval always works and
    is testable offline.

Language detection reuses the existing fasttext ``lid`` model when available,
falling back to ``lingua`` and then a CJK heuristic, defaulting to ``en`` on
failure. Tokenizers are lazily loaded and cached at module level to avoid
repeated initialization cost.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re

logger = logging.getLogger(__name__)

# CJK Unified Ideographs range (covers common simplified + traditional chars).
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")
_LATIN_RE = re.compile(r"[A-Za-z0-9]+")


# ---------------------------------------------------------------------------
# Lazy, cached tokenizer / detector loading
# ---------------------------------------------------------------------------

_jieba = None
_jieba_checked = False
_lingua_detector = None
_lingua_checked = False
_fasttext_cache: dict[str, object] = {}


def _get_jieba():
    """Lazily import jieba. Returns the module or ``None`` if unavailable."""
    global _jieba, _jieba_checked
    if not _jieba_checked:
        _jieba_checked = True
        try:
            import jieba

            with contextlib.suppress(Exception):
                jieba.setLogLevel(logging.ERROR)  # silence first-run banner
            _jieba = jieba
        except Exception:
            _jieba = None
    return _jieba


def jieba_available() -> bool:
    """Return True if jieba is importable in the current environment."""
    return _get_jieba() is not None


def _get_lingua():
    """Lazily build a lingua detector for {English, Chinese}. Cached."""
    global _lingua_detector, _lingua_checked
    if not _lingua_checked:
        _lingua_checked = True
        try:
            from lingua import Language, LanguageDetectorBuilder

            _lingua_detector = LanguageDetectorBuilder.from_languages(
                Language.ENGLISH, Language.CHINESE
            ).build()
        except Exception:
            _lingua_detector = None
    return _lingua_detector


def _get_fasttext(model_path: str | None):
    """Lazily load a fasttext lid model (cached per path). ``None`` if unavailable."""
    if not model_path:
        return None
    if model_path in _fasttext_cache:
        return _fasttext_cache[model_path]
    model = None
    try:
        if os.path.exists(model_path):
            import fasttext

            model = fasttext.load_model(model_path)
    except Exception:
        model = None
    _fasttext_cache[model_path] = model
    return model


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


def has_cjk(text: str) -> bool:
    """Return True if ``text`` contains any CJK ideograph."""
    return bool(_CJK_RE.search(text or ""))


def detect_language(text: str, model_path: str | None = None) -> str:
    """Detect the language of ``text``.

    Order: fasttext lid (if a model path is provided and loadable) → lingua
    (English/Chinese) → CJK heuristic. Defaults to ``"en"`` on failure.
    """
    if not text or not text.strip():
        return "en"

    ft = _get_fasttext(model_path)
    if ft is not None:
        try:
            label = ft.predict(text.replace("\n", " "), k=1)[0][0].replace("__label__", "")
            if label:
                return label
        except Exception:
            logger.debug("fasttext detection failed; trying fallbacks")

    detector = _get_lingua()
    if detector is not None:
        try:
            lang = detector.detect_language_of(text)
            if lang is not None:
                return lang.iso_code_639_1.name.lower()
        except Exception:
            logger.debug("lingua detection failed; using heuristic")

    return "zh" if has_cjk(text) else "en"


# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------


def tokenize_en(text: str) -> list[str]:
    """English tokenization — whitespace split (the existing behaviour)."""
    if not text:
        return []
    return text.split()


def bigram_tokenize(text: str) -> list[str]:
    """Dependency-free Chinese tokenization via character bigrams.

    Each contiguous CJK run of length N yields N-1 overlapping bigrams (a
    single character yields itself). Latin/digit words are kept as lowercase
    tokens so mixed-language chunks remain searchable.
    """
    if not text:
        return []
    tokens: list[str] = []
    for run in _CJK_RUN_RE.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            for i in range(len(run) - 1):
                tokens.append(run[i : i + 2])
    for word in _LATIN_RE.findall(text):
        tokens.append(word.lower())
    return tokens


def tokenize_zh(text: str) -> list[str]:
    """Chinese tokenization — jieba when available, else character bigrams."""
    if not text:
        return []
    jieba = _get_jieba()
    if jieba is not None:
        try:
            raw = [t.strip() for t in jieba.cut(text) if t and t.strip()]
            return [t for t in raw if has_cjk(t) or t.isalnum()]
        except Exception:
            logger.debug("jieba segmentation failed; falling back to bigrams")
    return bigram_tokenize(text)


def tokenize(text: str, lang: str = "en") -> list[str]:
    """Tokenize ``text`` for the given language.

    ``en`` (and any non-Chinese lang) uses the unchanged whitespace path;
    ``zh`` uses jieba-or-bigram segmentation.
    """
    if not text:
        return []
    if lang and str(lang).lower().startswith("zh"):
        return tokenize_zh(text)
    return tokenize_en(text)


def zh_index_text(text: str) -> str:
    """Space-joined Chinese tokens, suitable for storing in an FTS5 text column."""
    return " ".join(tokenize_zh(text))


# ---------------------------------------------------------------------------
# FTS index helpers (BM25 layer wiring)
# ---------------------------------------------------------------------------


def index_zh_chunks(db, rows, model_path: str | None = None) -> int:
    """Populate the ``chunk_fts_zh`` index for Chinese chunks.

    ``rows`` is an iterable of ``(chunk_id, text)``. Only chunks that contain
    CJK text and are detected as Chinese are indexed (English chunks are
    skipped via a cheap CJK pre-check, so the default path adds no meaningful
    overhead). Never raises — any failure is logged and ignored so ingestion
    is never broken by the additive Chinese index.

    Returns the number of chunks indexed.
    """
    indexed = 0
    try:
        for chunk_id, text in rows:
            if not text or not has_cjk(text):
                continue
            lang = detect_language(text, model_path)
            if not str(lang).lower().startswith("zh"):
                continue
            tokens = zh_index_text(text)
            if not tokens.strip():
                continue
            db.execute(
                "INSERT INTO chunk_fts_zh(rowid, text) VALUES (?, ?)",
                [chunk_id, tokens],
            )
            indexed += 1
        if indexed:
            db.conn.commit()
    except Exception as exc:  # pragma: no cover - defensive, never break ingestion
        logger.debug("chunk_fts_zh population skipped: %s", exc)
    return indexed


def remove_zh_index_for_file(db, file_id: int) -> None:
    """Delete ``chunk_fts_zh`` entries for a file's chunks (kept consistent on
    re-chunk / removal). No-op if the index table is absent."""
    try:
        db.execute(
            "DELETE FROM chunk_fts_zh WHERE rowid IN "
            "(SELECT id FROM chunk WHERE file_id = ?)",
            [file_id],
        )
    except Exception:  # pragma: no cover - defensive
        logger.debug("chunk_fts_zh cleanup skipped for file_id=%s", file_id)
