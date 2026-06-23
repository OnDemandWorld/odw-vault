"""Tests for api/main.py — FastAPI TestClient with mocked dependencies."""

import sqlite3
from unittest.mock import MagicMock, patch

import sqlite_utils
from fastapi.testclient import TestClient

from api.main import app
from pipeline.db import migrate


def _make_test_db(tmp_path):
    """Create a test DB with check_same_thread=False for TestClient."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    db = sqlite_utils.Database(conn)
    migrate(db)
    return db


def _client_with_db(db):
    """Create a TestClient with _get_db patched to return the test DB."""
    patcher = patch("api.main._get_db", return_value=db)
    patcher.start()
    return TestClient(app), patcher


class TestHealthEndpoint:
    @patch("api.main._load_config")
    @patch("api.main.ollama.Client")
    @patch("api.main.chromadb.PersistentClient")
    def test_all_services_healthy(self, mock_chroma, mock_ollama, mock_load_cfg, tmp_path):
        db = _make_test_db(tmp_path)
        client, patcher = _client_with_db(db)
        try:
            mock_cfg = MagicMock()
            mock_cfg.ollama.host = "http://localhost:11434"
            mock_cfg.chroma_root_path = str(tmp_path / "chroma")
            mock_cfg.models.embedding.collection_suffix = "test"
            mock_cfg.models.language_id.model_path = str(tmp_path / "model")
            mock_load_cfg.return_value = mock_cfg

            mock_ollama.return_value.list.return_value = {"models": []}
            mock_chroma.return_value.get_collection.return_value = MagicMock()

            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["ollama"] is True
            assert data["chroma"] is True
            assert data["database"] is True
        finally:
            patcher.stop()

    @patch("api.main._load_config")
    @patch("api.main.ollama.Client")
    @patch("api.main.chromadb.PersistentClient")
    def test_ollama_unreachable(self, mock_chroma, mock_ollama, mock_load_cfg, tmp_path):
        db = _make_test_db(tmp_path)
        client, patcher = _client_with_db(db)
        try:
            mock_cfg = MagicMock()
            mock_cfg.ollama.host = "http://localhost:11434"
            mock_cfg.chroma_root_path = str(tmp_path / "chroma")
            mock_cfg.models.embedding.collection_suffix = "test"
            mock_cfg.models.language_id.model_path = str(tmp_path / "model")
            mock_load_cfg.return_value = mock_cfg

            mock_ollama.side_effect = Exception("Connection refused")
            mock_chroma.return_value.get_collection.return_value = MagicMock()

            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["ollama"] is False
        finally:
            patcher.stop()


class TestFoldersEndpoint:
    @patch("api.main._load_config")
    def test_empty_folders(self, mock_load_cfg, tmp_path):
        db = _make_test_db(tmp_path)
        client, patcher = _client_with_db(db)
        try:
            mock_load_cfg.return_value = MagicMock()
            response = client.get("/folders")
            assert response.status_code == 200
            assert response.json() == []
        finally:
            patcher.stop()

    @patch("api.main._load_config")
    def test_returns_folders(self, mock_load_cfg, tmp_path):
        db = _make_test_db(tmp_path)
        db["folder"].insert({
            "path": "Project/Alpha",
            "rel_path": "Project/Alpha",
            "name": "Alpha",
            "depth": 1,
            "excluded": 0,
            "inferred_category": "project",
            "inferred_label": "Alpha",
        })
        db.conn.commit()
        client, patcher = _client_with_db(db)
        try:
            mock_load_cfg.return_value = MagicMock()
            response = client.get("/folders")
            assert response.status_code == 200
            data = response.json()
            assert len(data) >= 1
            assert data[0]["name"] == "Alpha"
        finally:
            patcher.stop()


class TestFilesEndpoint:
    @patch("api.main._load_config")
    def test_file_not_found(self, mock_load_cfg, tmp_path):
        db = _make_test_db(tmp_path)
        client, patcher = _client_with_db(db)
        try:
            mock_load_cfg.return_value = MagicMock()
            response = client.get("/files/999")
            assert response.status_code == 404
        finally:
            patcher.stop()

    @patch("api.main._load_config")
    def test_get_file(self, mock_load_cfg, tmp_path):
        db = _make_test_db(tmp_path)
        db["folder"].insert({"path": "test", "rel_path": "test", "name": "test", "depth": 0})
        db.conn.commit()
        db["file"].insert({
            "name": "report.pdf",
            "path": "/tmp/test/report.pdf",
            "rel_path": "report.pdf",
            "folder_id": 1,
            "sha256": "abc123",
            "size_bytes": 1000,
            "mtime": "2026-01-01",
            "hash_status": "done",
            "identify_status": "done",
            "triage_status": "pending",
            "is_dup_primary": 1,
            "excluded": 0,
            "category": "document",
            "format_name": "PDF",
            "page_count": 5,
        })
        db.conn.commit()
        client, patcher = _client_with_db(db)
        try:
            mock_load_cfg.return_value = MagicMock()
            response = client.get("/files/1")
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "report.pdf"
            assert data["category"] == "document"
            assert data["page_count"] == 5
        finally:
            patcher.stop()


class TestFeedbackEndpoint:
    @patch("api.main._load_config")
    def test_feedback_not_found(self, mock_load_cfg, tmp_path):
        db = _make_test_db(tmp_path)
        client, patcher = _client_with_db(db)
        try:
            mock_load_cfg.return_value = MagicMock()
            response = client.post("/feedback", json={"query_log_id": 1, "feedback": "up"})
            assert response.status_code == 404
        finally:
            patcher.stop()

    @patch("api.main._load_config")
    def test_feedback_success(self, mock_load_cfg, tmp_path):
        db = _make_test_db(tmp_path)
        db.conn.execute(
            "INSERT INTO query_log (user, query_text, answer_text, retrieved_chunks_json, answer_model, embedding_model, latency_ms) VALUES ('test', 'hello', 'answer', '[]', 'test-model', 'test-embed', 100)",
        )
        db.conn.commit()
        client, patcher = _client_with_db(db)
        try:
            mock_load_cfg.return_value = MagicMock()
            response = client.post("/feedback", json={"query_log_id": 1, "feedback": "up"})
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
        finally:
            patcher.stop()


class TestModelsEndpoint:
    @patch("api.main._load_config")
    @patch("api.main.chromadb.PersistentClient")
    def test_returns_models(self, mock_chroma, mock_load_cfg, tmp_path):
        mock_cfg = MagicMock()
        mock_cfg.chroma_root_path = str(tmp_path / "chroma")
        mock_cfg.models.embedding.name = "test-embed"
        mock_cfg.models.generation.name = "test-gen"
        mock_cfg.models.generation.fallback_name = "test-fb"
        mock_cfg.models.generation.alternate_name = "test-alt"
        mock_cfg.models.summarization.name = "test-sum"
        mock_cfg.models.contextual_retrieval.enabled = False
        mock_cfg.models.reranker.enabled = False
        mock_cfg.models.language_id.backend = "lingua"
        mock_load_cfg.return_value = mock_cfg

        mock_chroma.return_value.list_collections.return_value = []

        client = TestClient(app)
        response = client.get("/models")

        assert response.status_code == 200
        data = response.json()
        assert data["embedding"] == "test-embed"
        assert data["generation"] == "test-gen"
        assert "chroma_collections" in data


class TestFileTextEndpoint:
    @patch("api.main._load_config")
    def test_no_text_returns_404(self, mock_load_cfg, tmp_path):
        db = _make_test_db(tmp_path)
        client, patcher = _client_with_db(db)
        try:
            mock_load_cfg.return_value = MagicMock()
            response = client.get("/files/1/text")
            assert response.status_code == 404
        finally:
            patcher.stop()

    @patch("api.main._load_config")
    def test_returns_text(self, mock_load_cfg, tmp_path):
        db = _make_test_db(tmp_path)
        # Insert a folder and file first for FK
        db["folder"].insert({"path": "test", "rel_path": "test", "name": "test", "depth": 0})
        db.conn.commit()
        db["file"].insert({
            "name": "test.txt",
            "path": "/tmp/test/test.txt",
            "rel_path": "test.txt",
            "folder_id": 1,
            "sha256": "abc",
            "size_bytes": 100,
            "mtime": "2026-01-01",
            "hash_status": "done",
            "identify_status": "done",
            "triage_status": "pending",
            "is_dup_primary": 1,
            "excluded": 0,
        })
        db.conn.commit()
        db["extraction"].insert({
            "file_id": 1,
            "text_extracted": "Hello, world!",
            "tool": "test",
            "succeeded": 1,
        })
        db.conn.commit()
        client, patcher = _client_with_db(db)
        try:
            mock_load_cfg.return_value = MagicMock()
            response = client.get("/files/1/text")
            assert response.status_code == 200
            assert "Hello, world!" in response.text
        finally:
            patcher.stop()
