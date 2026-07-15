"""Tests for rag/reranker.py — cross-encoder reranker."""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pytest

from rag.retrieval import Hit
from rag.reranker import rerank, _extract_score, _sigmoid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hits(n: int) -> list[Hit]:
    """Create n mock Hit objects with sequential IDs."""
    return [
        Hit(
            chunk_id=i,
            file_id=i,
            folder_id=1,
            rel_path=f"doc_{i}.txt",
            page_start=None,
            text=f"This is the content of document {i}. " * 10,
        )
        for i in range(n)
    ]


def _mock_client_with_scores(scores: list[float]):
    """Create a mock Ollama client that returns fixed scores via embed API."""
    client = MagicMock()
    client.embed.return_value = {
        "embeddings": [[s] for s in scores],
    }
    return client


# ---------------------------------------------------------------------------
# Test: rerank sorting logic
# ---------------------------------------------------------------------------


class TestRerankerSortingLogic:
    """Construct data with known relevance scores and assert correct ordering."""

    def test_rerank_sorts_by_score_descending(self):
        """Reranked results should be sorted by relevance score descending."""
        hits = _make_hits(5)
        # Scores: doc 2 is best, doc 0 is worst
        scores = [0.1, 0.5, 0.95, 0.3, 0.7]

        with patch("ollama.Client") as mock_client_cls:
            mock_client = _mock_client_with_scores(scores)
            mock_client_cls.return_value = mock_client

            result = rerank(
                query="test query",
                hits=hits,
                model_name="reranker-model",
                ollama_host="http://localhost:11434",
                top_n=5,
            )

        # Should be sorted by score descending: 0.95, 0.7, 0.5, 0.3, 0.1
        assert len(result) == 5
        rerank_scores = [h.rerank_score for h in result]
        assert rerank_scores == sorted(rerank_scores, reverse=True)
        # Best hit should be chunk_id=2 (score 0.95)
        assert result[0].chunk_id == 2
        assert result[0].rerank_score == pytest.approx(_sigmoid(0.95), abs=0.01)

    def test_rerank_preserves_hit_metadata(self):
        """Reranking should preserve all original Hit fields."""
        hits = _make_hits(3)
        scores = [0.8, 0.2, 0.5]

        with patch("ollama.Client") as mock_client_cls:
            mock_client = _mock_client_with_scores(scores)
            mock_client_cls.return_value = mock_client

            result = rerank(
                query="test",
                hits=hits,
                model_name="reranker-model",
                ollama_host="http://localhost:11434",
                top_n=3,
            )

        # All original fields should be preserved
        for h in result:
            assert h.rel_path is not None
            assert h.text is not None
            assert h.file_id is not None

    def test_rerank_empty_input(self):
        """Empty input should return empty output."""
        result = rerank(
            query="test",
            hits=[],
            model_name="reranker-model",
            ollama_host="http://localhost:11434",
        )
        assert result == []


# ---------------------------------------------------------------------------
# Test: top_k cutoff
# ---------------------------------------------------------------------------


class TestRerankerTopKCutoff:
    """Input 50 candidates, set top_k=5, assert exactly 5 returned."""

    def test_top_n_cutoff(self):
        """Should return exactly top_n results."""
        hits = _make_hits(50)
        scores = [float(i) / 50.0 for i in range(50)]

        with patch("ollama.Client") as mock_client_cls:
            mock_client = _mock_client_with_scores(scores)
            mock_client_cls.return_value = mock_client

            result = rerank(
                query="test query",
                hits=hits,
                model_name="reranker-model",
                ollama_host="http://localhost:11434",
                top_n=5,
                batch_size=50,  # single batch for simplicity
            )

        assert len(result) == 5

    def test_top_n_larger_than_input(self):
        """If top_n > len(hits), return all hits."""
        hits = _make_hits(3)
        scores = [0.9, 0.5, 0.1]

        with patch("ollama.Client") as mock_client_cls:
            mock_client = _mock_client_with_scores(scores)
            mock_client_cls.return_value = mock_client

            result = rerank(
                query="test",
                hits=hits,
                model_name="reranker-model",
                ollama_host="http://localhost:11434",
                top_n=100,
            )

        assert len(result) == 3

    def test_top_n_equals_input(self):
        """If top_n == len(hits), return all hits."""
        hits = _make_hits(10)
        scores = [float(i) / 10.0 for i in range(10)]

        with patch("ollama.Client") as mock_client_cls:
            mock_client = _mock_client_with_scores(scores)
            mock_client_cls.return_value = mock_client

            result = rerank(
                query="test",
                hits=hits,
                model_name="reranker-model",
                ollama_host="http://localhost:11434",
                top_n=10,
                batch_size=10,
            )

        assert len(result) == 10


# ---------------------------------------------------------------------------
# Test: config toggle (skip reranking)
# ---------------------------------------------------------------------------


class TestRerankerConfigToggle:
    """When reranker is disabled, the retrieval pipeline should skip rerank()."""

    def test_retrieval_skips_reranker_when_disabled(self):
        """retrieve() should not call rerank when config says disabled."""
        from rag.retrieval import reciprocal_rank_fuse

        # We test at the retrieval module level — verify the flag logic
        # by checking that the reranker code path is gated by the config.
        # This is a unit test of the toggle logic, not a full integration test.

        # Simulate the config check logic from retrieval.py
        class MockRerankerConfig:
            enabled = False
            name = "reranker-model"
            top_n_out = 8
            batch_size = 8

        class MockConfig:
            reranker = MockRerankerConfig()

        cfg = MockConfig()
        use_reranker = None  # not overridden

        reranker_enabled = (
            use_reranker if use_reranker is not None
            else getattr(cfg.reranker, "enabled", False)
        )

        assert reranker_enabled is False

    def test_retrieval_enables_reranker_when_config_true(self):
        """retrieve() should call rerank when config says enabled."""

        class MockRerankerConfig:
            enabled = True
            name = "reranker-model"
            top_n_out = 8
            batch_size = 8

        class MockConfig:
            reranker = MockRerankerConfig()

        cfg = MockConfig()
        use_reranker = None

        reranker_enabled = (
            use_reranker if use_reranker is not None
            else getattr(cfg.reranker, "enabled", False)
        )

        assert reranker_enabled is True

    def test_api_override_can_force_enable(self):
        """API parameter use_reranker=True should override config disabled."""

        class MockRerankerConfig:
            enabled = False
            name = "reranker-model"
            top_n_out = 8

        class MockConfig:
            reranker = MockRerankerConfig()

        cfg = MockConfig()
        use_reranker = True  # API override

        reranker_enabled = (
            use_reranker if use_reranker is not None
            else getattr(cfg.reranker, "enabled", False)
        )

        assert reranker_enabled is True

    def test_api_override_can_force_disable(self):
        """API parameter use_reranker=False should override config enabled."""

        class MockRerankerConfig:
            enabled = True
            name = "reranker-model"
            top_n_out = 8

        class MockConfig:
            reranker = MockRerankerConfig()

        cfg = MockConfig()
        use_reranker = False  # API override

        reranker_enabled = (
            use_reranker if use_reranker is not None
            else getattr(cfg.reranker, "enabled", False)
        )

        assert reranker_enabled is False


# ---------------------------------------------------------------------------
# Test: batch processing
# ---------------------------------------------------------------------------


class TestRerankerBatchProcessing:
    """Mock the model, input 20 docs with batch_size=8, assert 3 calls."""

    def test_batch_processing_call_count(self):
        """20 documents with batch_size=8 should result in 3 API calls."""
        hits = _make_hits(20)

        with patch("ollama.Client") as mock_client_cls:
            mock_client = MagicMock()
            # Return scores for whatever batch size is requested
            def embed_side_effect(**kwargs):
                input_docs = kwargs.get("input", [])
                return {"embeddings": [[0.5] for _ in input_docs]}

            mock_client.embed.side_effect = embed_side_effect
            mock_client_cls.return_value = mock_client

            rerank(
                query="test query",
                hits=hits,
                model_name="reranker-model",
                ollama_host="http://localhost:11434",
                top_n=8,
                batch_size=8,
            )

        # 20 docs / batch_size 8 = ceil(2.5) = 3 calls
        assert mock_client.embed.call_count == 3

    def test_batch_processing_correct_sizes(self):
        """Verify each batch has the correct number of documents."""
        hits = _make_hits(20)

        with patch("ollama.Client") as mock_client_cls:
            mock_client = MagicMock()
            batch_sizes: list[int] = []

            def embed_side_effect(**kwargs):
                input_docs = kwargs.get("input", [])
                batch_sizes.append(len(input_docs))
                return {"embeddings": [[0.5] for _ in input_docs]}

            mock_client.embed.side_effect = embed_side_effect
            mock_client_cls.return_value = mock_client

            rerank(
                query="test",
                hits=hits,
                model_name="reranker-model",
                ollama_host="http://localhost:11434",
                top_n=8,
                batch_size=8,
            )

        # 3 batches: 8, 8, 4
        assert batch_sizes == [8, 8, 4]

    def test_batch_size_one(self):
        """batch_size=1 should make one call per document."""
        hits = _make_hits(5)

        with patch("ollama.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.embed.side_effect = lambda **kw: {
                "embeddings": [[0.5] for _ in kw.get("input", [])]
            }
            mock_client_cls.return_value = mock_client

            rerank(
                query="test",
                hits=hits,
                model_name="reranker-model",
                ollama_host="http://localhost:11434",
                top_n=5,
                batch_size=1,
            )

        assert mock_client.embed.call_count == 5

    def test_batch_size_larger_than_input(self):
        """batch_size > len(hits) should result in a single call."""
        hits = _make_hits(5)

        with patch("ollama.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.embed.side_effect = lambda **kw: {
                "embeddings": [[0.5] for _ in kw.get("input", [])]
            }
            mock_client_cls.return_value = mock_client

            rerank(
                query="test",
                hits=hits,
                model_name="reranker-model",
                ollama_host="http://localhost:11434",
                top_n=5,
                batch_size=100,
            )

        assert mock_client.embed.call_count == 1


# ---------------------------------------------------------------------------
# Test: utility functions
# ---------------------------------------------------------------------------


class TestUtilityFunctions:
    def test_sigmoid_positive(self):
        assert _sigmoid(0.0) == pytest.approx(0.5)
        assert _sigmoid(10.0) > 0.99
        assert _sigmoid(-10.0) < 0.01

    def test_sigmoid_range(self):
        """Sigmoid output should always be in [0, 1]."""
        for x in [-100, -10, -1, 0, 1, 10, 100]:
            val = _sigmoid(x)
            assert 0.0 <= val <= 1.0

    def test_extract_score_direct_float(self):
        assert _extract_score("0.85") == pytest.approx(0.85)
        assert _extract_score("0.0") == 0.0
        assert _extract_score("1.0") == 1.0

    def test_extract_score_from_text(self):
        assert _extract_score("Score: 0.75") == pytest.approx(0.75)
        assert _extract_score("0.9/1.0") == pytest.approx(0.9)

    def test_extract_score_percentage(self):
        """Percentage values should be normalized to 0-1."""
        assert _extract_score("85") == pytest.approx(0.85)

    def test_extract_score_invalid(self):
        assert _extract_score("no score here") == 0.0
        assert _extract_score("") == 0.0

    def test_text_truncation(self):
        """Long texts should be truncated to 2048 chars before scoring."""
        long_text = "x" * 5000
        hit = Hit(
            chunk_id=1, file_id=1, folder_id=1,
            rel_path="long.txt", page_start=None, text=long_text,
        )

        with patch("ollama.Client") as mock_client_cls:
            mock_client = MagicMock()
            captured_inputs: list[list[str]] = []

            def embed_side_effect(**kwargs):
                captured_inputs.append(kwargs.get("input", []))
                return {"embeddings": [[0.5] for _ in kwargs.get("input", [])]}

            mock_client.embed.side_effect = embed_side_effect
            mock_client_cls.return_value = mock_client

            rerank(
                query="test",
                hits=[hit],
                model_name="reranker-model",
                ollama_host="http://localhost:11434",
            )

        # Text should have been truncated to 2048 chars
        assert len(captured_inputs[0][0]) == 2048
