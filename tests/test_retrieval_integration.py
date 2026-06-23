"""Tests for rag/retrieval.py — retrieval helpers with mocked deps."""

from __future__ import annotations

from unittest.mock import MagicMock

from rag.retrieval import (
    Hit,
    _assemble_context,
    _dense_retrieve_with_embedding,
    reciprocal_rank_fuse,
)


class TestDenseRetrieveWithEmbedding:
    def test_returns_empty_on_collection_error(self):
        coll = MagicMock()
        coll.query.side_effect = Exception("Chroma error")

        result = _dense_retrieve_with_embedding(
            coll, [0.1, 0.2, 0.3], n_results=5, candidate_file_ids=None
        )
        assert result == []

    def test_returns_hits_from_results(self):
        coll = MagicMock()
        coll.query.return_value = {
            "ids": [["c_1", "c_2"]],
            "metadatas": [[
                {"chunk_id": 1, "file_id": 10, "folder_id": 1, "rel_path": "doc.txt", "start_page": 3},
                {"chunk_id": 2, "file_id": 10, "folder_id": 1, "rel_path": "doc.txt", "start_page": None},
            ]],
            "documents": [["text one", "text two"]],
            "distances": [[0.1, 0.3]],
        }

        result = _dense_retrieve_with_embedding(
            coll, [0.1, 0.2, 0.3], n_results=5, candidate_file_ids=None
        )
        assert len(result) == 2
        assert result[0].chunk_id == 1
        assert result[0].file_id == 10
        assert result[0].rel_path == "doc.txt"
        assert result[0].page_start == 3
        assert result[0].text == "text one"
        assert result[0].dense_score is not None  # 1.0 - 0.1 = 0.9

    def test_score_formula(self):
        coll = MagicMock()
        coll.query.return_value = {
            "ids": [["c_1"]],
            "metadatas": [[{"chunk_id": 1, "file_id": 1, "folder_id": 1, "rel_path": "x"}]],
            "documents": [["text"]],
            "distances": [[0.2]],
        }
        result = _dense_retrieve_with_embedding(
            coll, [0.1], n_results=1, candidate_file_ids=None
        )
        assert abs(result[0].dense_score - 0.8) < 1e-10  # 1.0 - 0.2

    def test_filters_by_candidate_file_ids(self):
        coll = MagicMock()
        coll.query.return_value = {
            "ids": [["c_1"]],
            "metadatas": [[{"chunk_id": 1, "file_id": 1, "folder_id": 1, "rel_path": "x"}]],
            "documents": [["text"]],
            "distances": [[0.1]],
        }

        _dense_retrieve_with_embedding(
            coll, [0.1], n_results=5, candidate_file_ids={1, 2, 3}
        )
        # Verify where clause was passed
        call_kwargs = coll.query.call_args.kwargs
        assert call_kwargs["where"] == {"file_id": {"$in": [1, 2, 3]}}

    def test_no_candidate_ids_passes_no_where(self):
        coll = MagicMock()
        coll.query.return_value = {
            "ids": [["c_1"]],
            "metadatas": [[{"chunk_id": 1, "file_id": 1, "folder_id": 1, "rel_path": "x"}]],
            "documents": [["text"]],
            "distances": [[0.1]],
        }

        _dense_retrieve_with_embedding(
            coll, [0.1], n_results=5, candidate_file_ids=None
        )
        call_kwargs = coll.query.call_args.kwargs
        assert call_kwargs["where"] is None


class TestAssembleContext:
    def test_sorts_by_file_id_then_chunk_id(self):
        hits = [
            Hit(chunk_id=5, file_id=2, folder_id=1, rel_path="b.txt", page_start=None, text="b"),
            Hit(chunk_id=1, file_id=1, folder_id=1, rel_path="a.txt", page_start=None, text="a"),
            Hit(chunk_id=2, file_id=1, folder_id=1, rel_path="a.txt", page_start=None, text="a2"),
        ]
        _assemble_context(hits, MagicMock())
        assert hits[0].file_id == 1
        assert hits[0].chunk_id == 1
        assert hits[1].file_id == 1
        assert hits[1].chunk_id == 2
        assert hits[2].file_id == 2

    def test_numbers_consecutive(self):
        hits = [
            Hit(chunk_id=1, file_id=1, folder_id=1, rel_path="doc.txt", page_start=None, text="content"),
            Hit(chunk_id=2, file_id=1, folder_id=1, rel_path="doc.txt", page_start=3, text="more content"),
        ]
        _assemble_context(hits, MagicMock())
        assert hits[0].text.startswith("[1]")
        assert hits[1].text.startswith("[2]")

    def test_page_info_included(self):
        hits = [
            Hit(chunk_id=1, file_id=1, folder_id=1, rel_path="doc.pdf", page_start=7, text="content"),
        ]
        _assemble_context(hits, MagicMock())
        assert "page 7" in hits[0].text

    def test_no_page_info(self):
        hits = [
            Hit(chunk_id=1, file_id=1, folder_id=1, rel_path="doc.txt", page_start=None, text="content"),
        ]
        _assemble_context(hits, MagicMock())
        assert "page" not in hits[0].text


class TestRRFWithAssembleContext:
    """Integration test: RRF followed by _assemble_context."""

    def test_rrf_then_assemble(self):
        dense = [
            Hit(chunk_id=1, file_id=1, folder_id=1, rel_path="a.txt", page_start=None, text="dense hit"),
        ]
        bm25 = [
            Hit(chunk_id=2, file_id=2, folder_id=1, rel_path="b.txt", page_start=None, text="bm25 hit"),
        ]
        fused = reciprocal_rank_fuse(dense, bm25, k=60)
        _assemble_context(fused, MagicMock())

        assert len(fused) == 2
        assert fused[0].text.startswith("[1]")
        assert fused[1].text.startswith("[2]")
