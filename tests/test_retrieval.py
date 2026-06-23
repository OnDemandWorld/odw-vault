"""Tests for rag/retrieval.py — pure functions."""

from rag.retrieval import Hit, reciprocal_rank_fuse


class TestHitDataclass:
    def test_hit_defaults(self):
        h = Hit(chunk_id=1, file_id=1, folder_id=1, rel_path="x", page_start=None, text="y")
        assert h.dense_score is None
        assert h.bm25_score is None
        assert h.rerank_score is None
        assert h.fused_score is None

    def test_hit_all_fields(self):
        h = Hit(
            chunk_id=1,
            file_id=2,
            folder_id=3,
            rel_path="a/b.txt",
            page_start=5,
            text="hello",
            dense_score=0.9,
            bm25_score=0.8,
            rerank_score=0.85,
            fused_score=0.87,
        )
        assert h.chunk_id == 1
        assert h.page_start == 5
        assert h.dense_score == 0.9


class TestReciprocalRankFuse:
    def _make_dense(self, n: int) -> list[Hit]:
        return [
            Hit(chunk_id=i, file_id=i, folder_id=1, rel_path=f"f{i}.txt", page_start=None, text=f"t{i}", dense_score=1.0 - i * 0.1)
            for i in range(1, n + 1)
        ]

    def _make_bm25(self, n: int) -> list[Hit]:
        return [
            Hit(chunk_id=i, file_id=i, folder_id=1, rel_path=f"f{i}.txt", page_start=None, text=f"t{i}", bm25_score=1.0 - i * 0.1)
            for i in range(1, n + 1)
        ]

    def test_empty_both(self):
        assert reciprocal_rank_fuse([], [], k=60) == []

    def test_only_dense_hits(self):
        hits = self._make_dense(3)
        result = reciprocal_rank_fuse(hits, [], k=60)
        assert len(result) == 3
        assert result[0].fused_score is not None

    def test_only_bm25_hits(self):
        hits = self._make_bm25(3)
        result = reciprocal_rank_fuse([], hits, k=60)
        assert len(result) == 3

    def test_overlapping_hits(self):
        dense = self._make_dense(2)
        bm25 = self._make_bm25(2)
        result = reciprocal_rank_fuse(dense, bm25, k=60)
        # Both lists have same chunk_ids, so they should be merged
        assert len(result) == 2
        # Fused score should be positive (sum of reciprocal ranks)
        assert result[0].fused_score > 0

    def test_ordering_descending(self):
        dense = self._make_dense(5)
        bm25 = self._make_bm25(5)
        result = reciprocal_rank_fuse(dense, bm25, k=60)
        scores = [h.fused_score for h in result]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_k_parameter(self):
        # Must create separate Hit objects for each call since RRF mutates in place
        dense1 = self._make_dense(2)
        bm25_1 = self._make_bm25(2)
        r1 = reciprocal_rank_fuse(dense1, bm25_1, k=60)

        dense2 = self._make_dense(2)
        bm25_2 = self._make_bm25(2)
        r2 = reciprocal_rank_fuse(dense2, bm25_2, k=1)

        # Different k should produce different fused scores
        assert r1[0].fused_score > 0
        assert r2[0].fused_score > 0
        assert r1[0].fused_score != r2[0].fused_score

    def test_fused_score_formula(self):
        """Verify 1/(k+rank+1) formula for a known case."""
        dense = [Hit(chunk_id=1, file_id=1, folder_id=1, rel_path="a", page_start=None, text="t")]
        bm25 = []
        result = reciprocal_rank_fuse(dense, bm25, k=60)
        # rank=0, so score = 1/(60+0+1) = 1/61
        assert abs(result[0].fused_score - 1 / 61) < 1e-10

    def test_preserves_hit_data(self):
        dense = [
            Hit(
                chunk_id=42,
                file_id=7,
                folder_id=3,
                rel_path="path/to/doc.txt",
                page_start=5,
                text="important content",
            )
        ]
        result = reciprocal_rank_fuse(dense, [], k=60)
        assert result[0].chunk_id == 42
        assert result[0].file_id == 7
        assert result[0].rel_path == "path/to/doc.txt"
        assert result[0].page_start == 5
        assert result[0].text == "important content"
