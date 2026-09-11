"""Tests for POST /query and POST /query/stream in api/main.py.

These exercise the two query endpoints over HTTP with fastapi.testclient,
mocking the retrieval + generation seam so no Ollama / Chroma / network is
required.  The conversation helpers (rag.conversation.*) are pure SQLite and
run against the real migrated test DB.
"""

import sqlite3
from unittest.mock import MagicMock, patch

import sqlite_utils
from fastapi.testclient import TestClient

from api.main import app
from pipeline.db import migrate
from rag.retrieval import Hit


def _make_test_db(tmp_path):
    """Create a test DB with check_same_thread=False for TestClient."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    db = sqlite_utils.Database(conn)
    migrate(db)
    return db


def _make_cfg(tmp_path):
    """Build a MagicMock config exposing every attribute /query touches."""
    cfg = MagicMock()
    cfg.ollama.host = "http://localhost:11434"
    cfg.chroma_root_path = str(tmp_path / "chroma")
    cfg.models.embedding.collection_suffix = "test"
    cfg.models.embedding.name = "test-embed"
    cfg.models.generation.name = "test-gen"
    cfg.models.generation.thinking = False
    cfg.models.reranker.enabled = False
    cfg.models.contextual_retrieval.enabled = False
    return cfg


def _make_hit():
    return Hit(
        chunk_id=1,
        file_id=1,
        folder_id=1,
        rel_path="test/doc.txt",
        page_start=None,
        text="The Gleneagles deployment uses robot platform X [1].",
        dense_score=0.9,
        bm25_score=0.8,
        fused_score=0.85,
    )


def _gen_result():
    return {
        "answer": "The deployment uses robot platform X [1].",
        "citations": [
            {
                "citation_number": 1,
                "file_id": 1,
                "rel_path": "test/doc.txt",
                "page_start": None,
                "chunk_id": 1,
                "snippet": "robot platform X",
            }
        ],
        "generation_ms": 12.0,
        "model": "test-gen",
        "refused": False,
    }


class TestQueryEndpoint:
    @patch("api.main.generate_answer")
    @patch("api.main.retrieve")
    @patch("api.main._load_config")
    @patch("api.main.ollama.Client")
    @patch("api.main.chromadb.PersistentClient")
    def test_happy_path(
        self, mock_chroma, mock_ollama, mock_load_cfg, mock_retrieve, mock_generate, tmp_path
    ):
        db = _make_test_db(tmp_path)
        mock_load_cfg.return_value = _make_cfg(tmp_path)
        mock_ollama.return_value.list.return_value = {"models": []}
        mock_chroma.return_value.get_collection.return_value = MagicMock()
        mock_retrieve.return_value = (
            [_make_hit()],
            {"retrieval_ms": 5.0, "query_lang": "en"},
        )
        mock_generate.return_value = _gen_result()

        patcher = patch("api.main._get_db", return_value=db)
        patcher.start()
        try:
            client = TestClient(app)
            response = client.post("/query", json={"query": "What robot platform?"})

            assert response.status_code == 200
            data = response.json()
            assert data["answer"] == "The deployment uses robot platform X [1]."
            assert isinstance(data["citations"], list)
            assert data["citations"][0]["marker"] == "[1]"
            assert data["citations"][0]["rel_path"] == "test/doc.txt"
            assert isinstance(data["retrieved_chunks"], list)
            assert data["retrieved_chunks"][0]["rank"] == 1
            assert data["retrieved_chunks"][0]["rel_path"] == "test/doc.txt"
            assert data["query_log_id"] is not None
            assert data["models"]["embedding"] == "test-embed"
            assert data["models"]["generation"] == "test-gen"

            # query_log row was persisted
            rows = list(db.query("SELECT id, answer_text FROM query_log"))
            assert len(rows) == 1
            assert rows[0]["id"] == data["query_log_id"]
        finally:
            patcher.stop()

    @patch("api.main._load_config")
    @patch("api.main.ollama.Client")
    @patch("api.main.chromadb.PersistentClient")
    def test_folder_filter_no_match_returns_422(
        self, mock_chroma, mock_ollama, mock_load_cfg, tmp_path
    ):
        db = _make_test_db(tmp_path)
        mock_load_cfg.return_value = _make_cfg(tmp_path)
        mock_ollama.return_value.list.return_value = {"models": []}
        mock_chroma.return_value.get_collection.return_value = MagicMock()

        patcher = patch("api.main._get_db", return_value=db)
        patcher.start()
        try:
            client = TestClient(app)
            response = client.post(
                "/query",
                json={
                    "query": "anything",
                    "folder_filter": {"path_prefix": "does/not/exist"},
                },
            )
            assert response.status_code == 422
            assert "folder_filter" in response.json()["detail"]
        finally:
            patcher.stop()

    @patch("api.main.retrieve")
    @patch("api.main._load_config")
    @patch("api.main.ollama.Client")
    @patch("api.main.chromadb.PersistentClient")
    def test_retrieval_runtime_error_returns_503(
        self, mock_chroma, mock_ollama, mock_load_cfg, mock_retrieve, tmp_path
    ):
        db = _make_test_db(tmp_path)
        mock_load_cfg.return_value = _make_cfg(tmp_path)
        mock_ollama.return_value.list.return_value = {"models": []}
        mock_chroma.return_value.get_collection.return_value = MagicMock()
        mock_retrieve.side_effect = RuntimeError("Chroma collection missing")

        patcher = patch("api.main._get_db", return_value=db)
        patcher.start()
        try:
            client = TestClient(app)
            response = client.post("/query", json={"query": "boom"})
            assert response.status_code == 503
            assert "Chroma collection missing" in response.json()["detail"]
        finally:
            patcher.stop()


class TestQueryStreamEndpoint:
    @patch("api.main.retrieve")
    @patch("api.main._load_config")
    @patch("api.main.ollama.Client")
    @patch("api.main.ollama.AsyncClient")
    @patch("api.main.chromadb.PersistentClient")
    def test_stream_emits_expected_events(
        self, mock_chroma, mock_async_ollama, mock_ollama, mock_load_cfg, mock_retrieve, tmp_path
    ):
        db = _make_test_db(tmp_path)
        mock_load_cfg.return_value = _make_cfg(tmp_path)

        # Reachability check uses the sync client; streaming uses the async
        # client whose .chat awaits to an async iterator of chunks.
        mock_ollama.return_value.list.return_value = {"models": []}

        chunks = [
            {"message": {"content": "The answer "}},
            {"message": {"content": "is platform X [1]."}},
        ]

        async def _fake_chat(**kwargs):
            async def _gen():
                for chunk in chunks:
                    yield chunk

            return _gen()

        mock_async_ollama.return_value.chat.side_effect = _fake_chat
        mock_chroma.return_value.get_collection.return_value = MagicMock()
        mock_retrieve.return_value = (
            [_make_hit()],
            {"retrieval_ms": 5.0, "query_lang": "en"},
        )

        patcher = patch("api.main._get_db", return_value=db)
        patcher.start()
        try:
            client = TestClient(app)
            response = client.post("/query/stream", json={"query": "What platform?"})

            assert response.status_code == 200
            text = response.text
            for event_name in ("retrieval", "token", "citations", "done"):
                assert f"event: {event_name}" in text
        finally:
            patcher.stop()
