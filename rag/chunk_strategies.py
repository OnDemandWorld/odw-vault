"""Pluggable chunking strategies (V1.2 M1 — F-Vault-1).

This module introduces a small strategy abstraction over text chunking so the
ingestion pipeline can select between multiple chunkers without changing the
default behaviour.

Design constraints (see roadmap/V1.2_VAULT_DESIGN.md):
  * Strictly additive — the default strategy (``sentence_window``) reproduces
    the pre-existing sentence-window chunker *exactly*, so existing ingestion
    output and tests are unchanged.
  * A strategy turns raw text into a list of :class:`ChunkSpan` (text + char
    offsets + ordinal). The ingestion path is responsible for turning spans
    into full ``chunk`` table rows (token estimate, page mapping, metadata).

Three strategies are registered by default:
  * ``sentence_window`` — focal sentence +/- N surrounding sentences (default).
  * ``recursive`` — split by a separator hierarchy (``\\n\\n`` -> ``\\n`` ->
    sentence terminators -> space) respecting a target ``chunk_size``/overlap.
  * ``paragraph`` — split on blank lines; over-long paragraphs are further
    split/merged by sentence.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Sentence boundary regex — mirrors rag.phase10_chunk so the default
# sentence-window strategy reproduces the legacy splitting exactly. Kept here
# (rather than imported) to avoid a circular import with phase10_chunk, which
# imports the strategy registry from this module.
_SENTENCE_RE = re.compile(r"(?<=[.!?。！？\n])\s+")  # noqa: RUF001


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences (identical to rag.phase10_chunk._split_sentences)."""
    raw = _SENTENCE_RE.split(text)
    return [s for s in raw if s.strip()]


@dataclass
class ChunkSpan:
    """A single chunk produced by a strategy.

    Mirrors the information the ingestion path needs from the legacy
    sentence-window loop: the chunk ``text``, its character ``start``/``end``
    offsets into the source extraction text, and its ordinal ``index``.
    """

    text: str
    start: int
    end: int
    index: int


class ChunkStrategy(ABC):
    """Interface for a chunking strategy."""

    #: Registry name. Subclasses must override.
    name: str = "base"

    @abstractmethod
    def chunk(self, text: str, **opts) -> list[ChunkSpan]:
        """Split ``text`` into an ordered list of :class:`ChunkSpan`."""


# ---------------------------------------------------------------------------
# Sentence-window strategy (wraps the existing chunker; default)
# ---------------------------------------------------------------------------


class SentenceWindowStrategy(ChunkStrategy):
    """Focal sentence +/- ``window_size`` surrounding sentences.

    This reproduces the original ``rag.phase10_chunk`` behaviour exactly:
    identical sentence splitting, identical ``" ".join(window)`` chunk text,
    and identical character offsets.
    """

    name = "sentence_window"

    def chunk(self, text: str, **opts) -> list[ChunkSpan]:
        window_size = int(opts.get("window_size", 5))

        sentences = _split_sentences(text)
        if not sentences:
            return []

        # Pre-compute character offsets for each sentence (same algorithm as
        # the legacy chunker so offsets match byte-for-byte).
        offsets: list[tuple[int, int]] = []
        pos = 0
        for s in sentences:
            start = text.find(s, pos)
            if start < 0:
                start = pos  # fallback
            end = start + len(s)
            offsets.append((start, end))
            pos = end

        spans: list[ChunkSpan] = []
        last = len(sentences) - 1
        for i in range(len(sentences)):
            lo = max(0, i - window_size)
            hi = min(last, i + window_size)
            chunk_text = " ".join(sentences[lo : hi + 1])
            spans.append(
                ChunkSpan(
                    text=chunk_text,
                    start=offsets[lo][0],
                    end=offsets[hi][1],
                    index=i,
                )
            )
        return spans


# ---------------------------------------------------------------------------
# Recursive strategy
# ---------------------------------------------------------------------------

# Separator hierarchy (highest priority first). Each entry is a regex that
# matches a split boundary. Sentence terminators cover English + Chinese.
_RECURSIVE_SEPARATORS: tuple[str, ...] = (
    r"\n\n",
    r"\n",
    r"(?<=[。！？.!?])\s*",  # noqa: RUF001
    r" ",
)


def _split_keep(text: str, pattern: str) -> list[tuple[str, int]]:
    """Split ``text`` on ``pattern``, keeping the delimiter with the preceding
    piece. Returns ``(piece, start_offset)`` tuples covering the whole string.
    """
    pieces: list[tuple[str, int]] = []
    last = 0
    for m in re.finditer(pattern, text):
        if m.end() == last:
            # Zero-width match (e.g. lookbehind sentence boundary) — skip to
            # avoid empty pieces / infinite loops.
            continue
        pieces.append((text[last : m.end()], last))
        last = m.end()
    if last < len(text):
        pieces.append((text[last:], last))
    return pieces


def _atomic_split(
    text: str,
    start: int,
    separators: tuple[str, ...],
    chunk_size: int,
    out: list[tuple[str, int]],
) -> None:
    """Recursively split ``text`` into segments each ``<= chunk_size``.

    Tries separators in priority order; falls back to a hard character split
    once separators are exhausted. Appends ``(segment, start_offset)`` to
    ``out``.
    """
    if len(text) <= chunk_size:
        if text.strip():
            out.append((text, start))
        return

    if not separators:
        # Hard character split (last resort).
        for i in range(0, len(text), chunk_size):
            piece = text[i : i + chunk_size]
            if piece.strip():
                out.append((piece, start + i))
        return

    sep, *rest = separators
    parts = _split_keep(text, sep)
    if len(parts) <= 1:
        # Separator not present — try the next, finer separator.
        _atomic_split(text, start, tuple(rest), chunk_size, out)
        return

    for piece, piece_start in parts:
        _atomic_split(piece, start + piece_start, tuple(rest), chunk_size, out)


def _merge_atomic(
    atomic: list[tuple[str, int]],
    chunk_size: int,
    overlap: int,
) -> list[ChunkSpan]:
    """Greedily pack atomic segments into chunks ``<= chunk_size``.

    Consecutive chunks share up to ``overlap`` trailing characters worth of
    segments. Atomic segments already carry their separators, so joining is a
    plain concatenation that preserves the original text.
    """
    spans: list[ChunkSpan] = []
    n = len(atomic)
    i = 0
    idx = 0
    while i < n:
        buf: list[tuple[str, int]] = []
        buf_len = 0
        j = i
        while j < n and buf_len + len(atomic[j][0]) <= chunk_size:
            buf.append(atomic[j])
            buf_len += len(atomic[j][0])
            j += 1
        if not buf:
            # A single atomic segment exceeds chunk_size — emit it whole.
            buf = [atomic[j]]
            j += 1

        text = "".join(t for t, _ in buf)
        if text.strip():
            spans.append(
                ChunkSpan(
                    text=text,
                    start=buf[0][1],
                    end=buf[-1][1] + len(buf[-1][0]),
                    index=idx,
                )
            )
            idx += 1

        if j >= n:
            break

        # Pull the next start back so the following chunk re-includes up to
        # ``overlap`` characters of trailing segments. ``max(..., i + 1)``
        # guarantees forward progress even with a very large overlap.
        k = j
        ov_len = 0
        while k > i and ov_len + len(atomic[k - 1][0]) <= overlap:
            ov_len += len(atomic[k - 1][0])
            k -= 1
        i = max(k, i + 1)

    return spans


class RecursiveStrategy(ChunkStrategy):
    """Recursively split by a separator hierarchy to a target size."""

    name = "recursive"

    def chunk(self, text: str, **opts) -> list[ChunkSpan]:
        chunk_size = int(opts.get("chunk_size", 2000))
        overlap = int(opts.get("chunk_overlap", 200))
        chunk_size = max(1, chunk_size)
        overlap = max(0, min(overlap, chunk_size - 1))

        if not text or not text.strip():
            return []

        atomic: list[tuple[str, int]] = []
        _atomic_split(text, 0, _RECURSIVE_SEPARATORS, chunk_size, atomic)
        if not atomic:
            return []
        return _merge_atomic(atomic, chunk_size, overlap)


# ---------------------------------------------------------------------------
# Paragraph strategy
# ---------------------------------------------------------------------------

_PARAGRAPH_RE = r"\n[ \t]*\n"


class ParagraphStrategy(ChunkStrategy):
    """Split on blank lines; over-long paragraphs are split/merged by sentence."""

    name = "paragraph"

    def chunk(self, text: str, **opts) -> list[ChunkSpan]:
        chunk_size = int(opts.get("chunk_size", 2000))
        chunk_size = max(1, chunk_size)

        if not text or not text.strip():
            return []

        spans: list[ChunkSpan] = []
        idx = 0
        for para, para_start in _split_keep(text, _PARAGRAPH_RE):
            if not para.strip():
                continue
            if len(para) <= chunk_size:
                spans.append(
                    ChunkSpan(
                        text=para,
                        start=para_start,
                        end=para_start + len(para),
                        index=idx,
                    )
                )
                idx += 1
                continue

            # Over-long paragraph: split by sentence and greedily merge up to
            # chunk_size (reusing the shared sentence splitter).
            sentences = _split_sentences(para)
            if not sentences:
                spans.append(
                    ChunkSpan(
                        text=para,
                        start=para_start,
                        end=para_start + len(para),
                        index=idx,
                    )
                )
                idx += 1
                continue

            # Offsets of each sentence relative to the source text.
            rel: list[tuple[str, int]] = []
            search_from = para_start
            for s in sentences:
                found = text.find(s, search_from)
                if found < 0:
                    found = search_from
                rel.append((s, found))
                search_from = found + len(s)

            buf: list[tuple[str, int]] = []
            buf_len = 0
            for s, s_start in rel:
                if buf and buf_len + len(s) > chunk_size:
                    chunk_text = " ".join(t for t, _ in buf)
                    spans.append(
                        ChunkSpan(
                            text=chunk_text,
                            start=buf[0][1],
                            end=buf[-1][1] + len(buf[-1][0]),
                            index=idx,
                        )
                    )
                    idx += 1
                    buf = []
                    buf_len = 0
                buf.append((s, s_start))
                buf_len += len(s)
            if buf:
                chunk_text = " ".join(t for t, _ in buf)
                spans.append(
                    ChunkSpan(
                        text=chunk_text,
                        start=buf[0][1],
                        end=buf[-1][1] + len(buf[-1][0]),
                        index=idx,
                    )
                )
                idx += 1

        return spans


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ChunkStrategyRegistry:
    """Register and retrieve chunking strategies by name."""

    DEFAULT = "sentence_window"

    def __init__(self) -> None:
        self._strategies: dict[str, ChunkStrategy] = {}

    def register(self, strategy: ChunkStrategy) -> None:
        """Register a strategy instance under its ``name``."""
        if not getattr(strategy, "name", None):
            raise ValueError("Strategy must define a non-empty 'name'")
        self._strategies[strategy.name] = strategy

    def get(self, name: str | None) -> ChunkStrategy:
        """Return the strategy for ``name``.

        Falls back to the default strategy when ``name`` is ``None``, empty,
        or unknown (logging a warning for unknown names). This guarantees the
        ingestion path always has a working chunker and that the default
        behaviour is preserved for unrecognised configuration.
        """
        if name and name in self._strategies:
            return self._strategies[name]
        if name:
            logger.warning(
                "Unknown chunk strategy '%s', falling back to '%s'",
                name,
                self.DEFAULT,
            )
        return self._strategies[self.DEFAULT]

    def names(self) -> list[str]:
        """Return the sorted list of registered strategy names."""
        return sorted(self._strategies)

    def __contains__(self, name: str) -> bool:
        return name in self._strategies


def build_default_registry() -> ChunkStrategyRegistry:
    """Build a registry populated with the built-in strategies."""
    registry = ChunkStrategyRegistry()
    registry.register(SentenceWindowStrategy())
    registry.register(RecursiveStrategy())
    registry.register(ParagraphStrategy())
    return registry


#: Module-level default registry used by the ingestion path.
_REGISTRY = build_default_registry()


def get_chunker(name: str | None = None) -> ChunkStrategy:
    """Return a strategy by name from the default registry (default: sentence_window)."""
    return _REGISTRY.get(name)


def get_registry() -> ChunkStrategyRegistry:
    """Return the module-level default registry."""
    return _REGISTRY


# Legacy CLI/config alias -> registry name.
_NAME_ALIASES: dict[str, str] = {
    "sentence-window": "sentence_window",
    "sentencewindow": "sentence_window",
}


def normalize_strategy_name(name: str | None) -> str | None:
    """Map legacy/alias chunker names (e.g. ``sentence-window``) to registry names."""
    if not name:
        return name
    return _NAME_ALIASES.get(name, name)


def resolve_strategy_name(cfg, category: str | None = None, override: str | None = None) -> str:
    """Resolve the effective strategy name for a file.

    Priority: explicit ``override`` (e.g. CLI ``--chunker``) > per-category
    override (``cfg.chunk.category_strategies``) > ``cfg.chunk.strategy``.
    Always returns a non-empty name; the registry falls back to the default
    for unknown names.
    """
    if override:
        return normalize_strategy_name(override) or ChunkStrategyRegistry.DEFAULT

    chunk_cfg = getattr(cfg, "chunk", None)
    cat_map = getattr(chunk_cfg, "category_strategies", None) or {}
    if category and category in cat_map:
        return normalize_strategy_name(cat_map[category]) or ChunkStrategyRegistry.DEFAULT

    configured = getattr(chunk_cfg, "strategy", None)
    return normalize_strategy_name(configured) or ChunkStrategyRegistry.DEFAULT
