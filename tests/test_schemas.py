"""Tests for api/schemas.py — Pydantic model validation."""

import pytest
from pydantic import ValidationError

from api.schemas import (
    Citation,
    FeedbackRequest,
    FileResponse,
    FolderFilter,
    FolderNode,
    HealthResponse,
    Metrics,
    ModelInfo,
    QueryRequest,
    QueryResponse,
)


class TestQueryRequestValidation:
    def test_valid_request(self):
        req = QueryRequest(query="hello")
        assert req.query == "hello"
        assert req.stream is False

    def test_empty_query_rejected(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="")

    def test_oversized_query_rejected(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="x" * 2001)

    def test_optional_fields_absent(self):
        req = QueryRequest(query="test")
        assert req.folder_filter is None
        assert req.top_k_chunks is None
        assert req.use_reranker is None

    def test_folder_filter_nested(self):
        req = QueryRequest(query="test", folder_filter={"path_prefix": "Project/"})
        assert req.folder_filter is not None

    def test_top_k_out_of_range_low(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="test", top_k_chunks=0)

    def test_top_k_out_of_range_high(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="test", top_k_chunks=21)

    def test_thinking_flag(self):
        req = QueryRequest(query="test", thinking=True)
        assert req.thinking is True


class TestCitationModel:
    def test_citation_valid(self):
        c = Citation(
            marker="[1]",
            file_id=1,
            rel_path="doc.txt",
            page=None,
            chunk_id=1,
            snippet="hello",
        )
        assert c.marker == "[1]"

    def test_citation_page_present(self):
        c = Citation(
            marker="[1]",
            file_id=1,
            rel_path="doc.txt",
            page=5,
            chunk_id=1,
            snippet="hello",
        )
        assert c.page == 5


class TestQueryResponse:
    def test_response_valid(self):
        r = QueryResponse(
            answer="test answer",
            citations=[],
            retrieved_chunks=[],
            metrics=Metrics(retrieval_ms=50, generation_ms=50, total_ms=100),
            models=ModelInfo(embedding="e", generation="g", reranker=None, contextual_augmentation=None),
            query_log_id=1,
        )
        assert r.answer == "test answer"

    def test_response_missing_answer(self):
        with pytest.raises(ValidationError):
            QueryResponse(
                citations=[],
                retrieved_chunks=[],
                metrics=Metrics(retrieval_ms=50, generation_ms=50, total_ms=100),
                models=ModelInfo(embedding="e", generation="g"),
                query_log_id=1,
            )


class TestFolderFilterModel:
    def test_all_fields_optional(self):
        ff = FolderFilter()
        assert ff.path_prefix is None
        assert ff.folder_id is None
        assert ff.inferred_category is None

    def test_single_field_set(self):
        ff = FolderFilter(path_prefix="Project/")
        assert ff.path_prefix == "Project/"


class TestFeedbackRequest:
    def test_valid_feedback(self):
        f = FeedbackRequest(query_log_id=1, feedback="up")
        assert f.feedback == "up"

    def test_feedback_down(self):
        f = FeedbackRequest(query_log_id=1, feedback="down")
        assert f.feedback == "down"

    def test_note_optional(self):
        f = FeedbackRequest(query_log_id=1, feedback="up")
        assert f.note is None


class TestFileResponse:
    def test_valid_file_response(self):
        r = FileResponse(
            id=1,
            rel_path="a.txt",
            name="a.txt",
            category="document",
            format_name="Text",
            page_count=None,
            folder_id=1,
            parent_folder=None,
            summary=None,
            extraction_path=None,
        )
        assert r.id == 1
        assert r.category == "document"


class TestHealthResponse:
    def test_all_booleans(self):
        h = HealthResponse(ollama=True, chroma=True, database=True, fasttext=False)
        assert h.ollama is True
        assert h.fasttext is False


class TestFolderNode:
    def test_recursive_children(self):
        child = FolderNode(id=2, rel_path="sub", name="sub", inferred_category=None, inferred_label=None, children=[])
        parent = FolderNode(id=1, rel_path="", name="root", inferred_category=None, inferred_label=None, children=[child])
        assert len(parent.children) == 1
        assert parent.children[0].name == "sub"
