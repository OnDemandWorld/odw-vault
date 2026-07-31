"""Tests for rag/chunk_strategies.py (V1.2 M1 — multi-chunking strategies)."""

from __future__ import annotations

from itertools import pairwise

import pytest

from rag.chunk_strategies import (
    ChunkSpan,
    ChunkStrategy,
    ChunkStrategyRegistry,
    ParagraphStrategy,
    RecursiveStrategy,
    SentenceWindowStrategy,
    build_default_registry,
    get_chunker,
    get_registry,
    normalize_strategy_name,
    resolve_strategy_name,
)
from rag.phase10_chunk import _split_sentences

# ---------------------------------------------------------------------------
# Registry (C1)
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_default_registry_has_three_strategies(self):
        reg = build_default_registry()
        assert set(reg.names()) == {"sentence_window", "recursive", "paragraph"}

    def test_default_is_sentence_window(self):
        assert ChunkStrategyRegistry.DEFAULT == "sentence_window"
        assert get_chunker().name == "sentence_window"
        assert get_chunker(None).name == "sentence_window"

    def test_get_by_name(self):
        assert get_chunker("recursive").name == "recursive"
        assert get_chunker("paragraph").name == "paragraph"

    def test_unknown_falls_back_to_default(self):
        assert get_chunker("does-not-exist").name == "sentence_window"

    def test_contains(self):
        reg = get_registry()
        assert "sentence_window" in reg
        assert "nope" not in reg

    def test_register_custom_strategy(self):
        class Dummy(ChunkStrategy):
            name = "dummy"

            def chunk(self, text, **opts):
                return [ChunkSpan(text=text, start=0, end=len(text), index=0)]

        reg = build_default_registry()
        reg.register(Dummy())
        assert "dummy" in reg
        assert reg.get("dummy").name == "dummy"

    def test_register_requires_name(self):
        class Bad(ChunkStrategy):
            name = ""

            def chunk(self, text, **opts):
                return []

        reg = ChunkStrategyRegistry()
        with pytest.raises(ValueError):
            reg.register(Bad())


class TestNameResolution:
    @staticmethod
    def _cfg(strategy="sentence_window", category_strategies=None):
        from types import SimpleNamespace

        return SimpleNamespace(
            chunk=SimpleNamespace(
                strategy=strategy,
                category_strategies=category_strategies or {},
            )
        )

    def test_normalize_alias(self):
        assert normalize_strategy_name("sentence-window") == "sentence_window"
        assert normalize_strategy_name("recursive") == "recursive"
        assert normalize_strategy_name(None) is None

    def test_resolve_default(self):
        assert resolve_strategy_name(self._cfg()) == "sentence_window"

    def test_resolve_override_wins(self):
        assert resolve_strategy_name(self._cfg(), override="paragraph") == "paragraph"
        # legacy alias override
        assert resolve_strategy_name(self._cfg(), override="sentence-window") == "sentence_window"

    def test_resolve_category_override(self):
        cfg = self._cfg(category_strategies={"document": "paragraph"})
        assert resolve_strategy_name(cfg, category="document") == "paragraph"
        # Unmapped category falls back to configured default
        assert resolve_strategy_name(cfg, category="data") == "sentence_window"


# ---------------------------------------------------------------------------
# Sentence-window strategy — must match legacy behaviour exactly (C1)
# ---------------------------------------------------------------------------


def _legacy_windows(text: str, window_size: int) -> list[tuple[str, int, int]]:
    """Reconstruct the pre-V1.2 sentence-window algorithm for equivalence checks."""
    sentences = _split_sentences(text)
    offsets: list[tuple[int, int]] = []
    pos = 0
    for s in sentences:
        start = text.find(s, pos)
        if start < 0:
            start = pos
        end = start + len(s)
        offsets.append((start, end))
        pos = end
    out = []
    last = len(sentences) - 1
    for i in range(len(sentences)):
        lo = max(0, i - window_size)
        hi = min(last, i + window_size)
        out.append((" ".join(sentences[lo : hi + 1]), offsets[lo][0], offsets[hi][1]))
    return out


class TestSentenceWindowStrategy:
    def test_matches_legacy_exactly(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        for window_size in (0, 1, 2, 5):
            spans = SentenceWindowStrategy().chunk(text, window_size=window_size)
            legacy = _legacy_windows(text, window_size)
            assert len(spans) == len(legacy)
            for span, (lt, ls, le) in zip(spans, legacy, strict=True):
                assert span.text == lt
                assert span.start == ls
                assert span.end == le

    def test_indices_sequential(self):
        spans = SentenceWindowStrategy().chunk("A. B. C. D.", window_size=1)
        assert [s.index for s in spans] == list(range(len(spans)))

    def test_empty_text(self):
        assert SentenceWindowStrategy().chunk("") == []
        assert SentenceWindowStrategy().chunk("   \n  ") == []

    def test_default_window_size(self):
        # No opts -> window_size defaults to 5
        spans = SentenceWindowStrategy().chunk("A. B. C.")
        assert len(spans) == 3


# ---------------------------------------------------------------------------
# Recursive strategy (C2)
# ---------------------------------------------------------------------------


class TestRecursiveStrategy:
    def test_splits_on_paragraph_boundary_first(self):
        text = "Alpha beta.\n\nGamma delta.\n\nEpsilon zeta."
        spans = RecursiveStrategy().chunk(text, chunk_size=25, chunk_overlap=0)
        # Each paragraph (<=25 chars) becomes its own chunk
        assert len(spans) == 3
        assert "Alpha beta." in spans[0].text
        assert "Gamma delta." in spans[1].text
        assert "Epsilon zeta." in spans[2].text

    def test_respects_chunk_size(self):
        text = " ".join(f"word{i}" for i in range(200))
        spans = RecursiveStrategy().chunk(text, chunk_size=60, chunk_overlap=0)
        assert len(spans) > 1
        for s in spans:
            assert len(s.text) <= 60

    def test_falls_back_to_finer_separators(self):
        # No newlines -> must still split (by sentence terminator / space)
        text = " ".join(["hello world."] * 50)
        spans = RecursiveStrategy().chunk(text, chunk_size=40, chunk_overlap=0)
        assert len(spans) > 1
        for s in spans:
            assert len(s.text) <= 40

    def test_overlap_shares_content(self):
        text = " ".join(f"w{i}" for i in range(100))
        spans = RecursiveStrategy().chunk(text, chunk_size=30, chunk_overlap=15)
        assert len(spans) > 2
        # The tail of one chunk should reappear at the start of the next.
        shared = 0
        for a, b in pairwise(spans):
            tail = a.text[-10:]
            if tail and tail in b.text:
                shared += 1
        assert shared >= 1

    def test_offsets_are_consistent(self):
        text = "One two three.\n\nFour five six."
        spans = RecursiveStrategy().chunk(text, chunk_size=20, chunk_overlap=0)
        for s in spans:
            assert 0 <= s.start < s.end <= len(text)
            # The span text should be a slice of the source (after strip of
            # trailing separators it still appears in the source).
            assert s.text.strip()[:5] in text

    def test_empty_text(self):
        assert RecursiveStrategy().chunk("") == []
        assert RecursiveStrategy().chunk("   ") == []

    def test_indices_sequential(self):
        text = "a b c d e f g h " * 20
        spans = RecursiveStrategy().chunk(text, chunk_size=30, chunk_overlap=0)
        assert [s.index for s in spans] == list(range(len(spans)))


# ---------------------------------------------------------------------------
# Paragraph strategy (C3)
# ---------------------------------------------------------------------------


class TestParagraphStrategy:
    def test_splits_on_blank_lines(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        spans = ParagraphStrategy().chunk(text, chunk_size=200)
        assert len(spans) == 3
        assert "Para one." in spans[0].text
        assert "Para two." in spans[1].text
        assert "Para three." in spans[2].text

    def test_overlong_paragraph_split_by_sentence(self):
        long_para = "This is a sentence. " * 30  # ~600 chars
        text = f"Intro.\n\n{long_para}"
        spans = ParagraphStrategy().chunk(text, chunk_size=100)
        # Intro is one chunk; the long paragraph is split into several.
        assert len(spans) > 2
        for s in spans:
            # Allow a single-sentence overflow but not the whole 600-char para.
            assert len(s.text) < len(long_para)

    def test_short_paragraphs_kept_whole(self):
        text = "A.\n\nB.\n\nC."
        spans = ParagraphStrategy().chunk(text, chunk_size=100)
        assert len(spans) == 3

    def test_empty_text(self):
        assert ParagraphStrategy().chunk("") == []
        assert ParagraphStrategy().chunk("\n\n\n") == []

    def test_offsets_within_bounds(self):
        text = "First para.\n\nSecond para is a bit longer than the first one."
        spans = ParagraphStrategy().chunk(text, chunk_size=200)
        for s in spans:
            assert 0 <= s.start < s.end <= len(text)


# ---------------------------------------------------------------------------
# Ingestion wiring (C4) — strategy selection through run_chunk
# ---------------------------------------------------------------------------


def _make_app_config(tmp_path, **chunk_kwargs):
    from pipeline.config import (
        AppConfig,
        ChunkConfig,
        ContextualRetrievalConfig,
        EmbeddingConfig,
        GenerationConfig,
        ModelsConfig,
        PathsConfig,
        SummarizationConfig,
    )

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    cache = corpus / ".rag-cache"
    cache.mkdir()
    return AppConfig(
        paths=PathsConfig(corpus_root=str(corpus), cache_root=str(cache)),
        chunk=ChunkConfig(**chunk_kwargs),
        models=ModelsConfig(
            embedding=EmbeddingConfig(name="test-embed", collection_suffix="_test"),
            summarization=SummarizationConfig(name="test-summarize"),
            contextual_retrieval=ContextualRetrievalConfig(name="test-context"),
            generation=GenerationConfig(
                name="test-gen", fallback_name="test-fb", alternate_name="test-alt"
            ),
        ),
    )


class TestRunChunkStrategySelection:
    def test_default_strategy_is_sentence_window(self, tmp_path, test_db):
        import json

        from rag.phase10_chunk import run_chunk
        from tests.conftest import seed_test_extractions, seed_test_files

        cfg = _make_app_config(tmp_path)  # default strategy
        file_ids = seed_test_files(test_db)
        seed_test_extractions(test_db, file_ids, text="A. B. C. D.")
        total, _ = run_chunk(test_db, cfg)
        assert total > 0
        for ch in test_db["chunk"].rows:
            meta = json.loads(ch["metadata_json"])
            assert meta["chunk_strategy"] == "sentence_window"

    def test_paragraph_strategy_via_config(self, tmp_path, test_db):
        import json

        from rag.phase10_chunk import run_chunk
        from tests.conftest import seed_test_extractions, seed_test_files

        cfg = _make_app_config(tmp_path, strategy="paragraph", chunk_size=50)
        file_ids = seed_test_files(test_db)
        seed_test_extractions(
            test_db, file_ids, text="Para one here.\n\nPara two here.\n\nPara three here."
        )
        total, _ = run_chunk(test_db, cfg)
        assert total >= 3
        strategies = {json.loads(ch["metadata_json"])["chunk_strategy"] for ch in test_db["chunk"].rows}
        assert strategies == {"paragraph"}

    def test_cli_override_alias(self, tmp_path, test_db):
        import json

        from rag.phase10_chunk import run_chunk
        from tests.conftest import seed_test_extractions, seed_test_files

        cfg = _make_app_config(tmp_path)  # default sentence_window
        file_ids = seed_test_files(test_db)
        seed_test_extractions(test_db, file_ids, text="A. B. C.")
        # Legacy CLI alias "sentence-window" must map to the default strategy.
        total, _ = run_chunk(test_db, cfg, chunker="sentence-window")
        assert total > 0
        for ch in test_db["chunk"].rows:
            assert json.loads(ch["metadata_json"])["chunk_strategy"] == "sentence_window"
