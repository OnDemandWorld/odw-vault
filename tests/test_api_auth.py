"""Tests for the optional inbound API-key auth and the GET /metrics endpoint.

The auth layer in api/main.py is additive and backward-compatible: when
VAULT_API_KEY is unset or empty every endpoint stays open (preserving
INTEGRATION_CONTRACT.md §1 and the existing unauthenticated suite); when it is
set, all endpoints except /health and the API docs require a matching
``Authorization: Bearer <VAULT_API_KEY>`` header.

The /query mocking mirrors tests/test_api_query.py so no Ollama / Chroma /
network access is required.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from tests.test_api_query import _gen_result, _make_cfg, _make_hit, _make_test_db

API_KEY = "test-secret-key"


@pytest.fixture
def query_client(tmp_path):
    """TestClient with the /query retrieval+generation seams mocked out."""
    db = _make_test_db(tmp_path)
    with (
        patch("api.main.generate_answer", return_value=_gen_result()),
        patch(
            "api.main.retrieve",
            return_value=([_make_hit()], {"retrieval_ms": 5.0, "query_lang": "en"}),
        ),
        patch("api.main._load_config", return_value=_make_cfg(tmp_path)),
        patch("api.main.ollama.Client") as mock_ollama,
        patch("api.main.chromadb.PersistentClient") as mock_chroma,
        patch("api.main._get_db", return_value=db),
    ):
        mock_ollama.return_value.list.return_value = {"models": []}
        mock_chroma.return_value.get_collection.return_value = MagicMock()
        yield TestClient(app)


class TestApiKeyAuth:
    def test_query_open_when_key_unset(self, query_client, monkeypatch):
        monkeypatch.delenv("VAULT_API_KEY", raising=False)
        resp = query_client.post("/query", json={"query": "What robot platform?"})
        assert resp.status_code == 200
        assert resp.json()["answer"]

    def test_query_open_when_key_empty(self, query_client, monkeypatch):
        monkeypatch.setenv("VAULT_API_KEY", "   ")
        resp = query_client.post("/query", json={"query": "What robot platform?"})
        assert resp.status_code == 200

    def test_missing_header_returns_401(self, query_client, monkeypatch):
        monkeypatch.setenv("VAULT_API_KEY", API_KEY)
        resp = query_client.post("/query", json={"query": "What robot platform?"})
        assert resp.status_code == 401

    def test_wrong_key_returns_401(self, query_client, monkeypatch):
        monkeypatch.setenv("VAULT_API_KEY", API_KEY)
        resp = query_client.post(
            "/query",
            json={"query": "What robot platform?"},
            headers={"Authorization": "Bearer not-the-right-key"},
        )
        assert resp.status_code == 401

    def test_wrong_scheme_returns_401(self, query_client, monkeypatch):
        monkeypatch.setenv("VAULT_API_KEY", API_KEY)
        resp = query_client.post(
            "/query",
            json={"query": "What robot platform?"},
            headers={"Authorization": f"Basic {API_KEY}"},
        )
        assert resp.status_code == 401

    def test_correct_key_returns_200(self, query_client, monkeypatch):
        monkeypatch.setenv("VAULT_API_KEY", API_KEY)
        resp = query_client.post(
            "/query",
            json={"query": "What robot platform?"},
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
        assert resp.status_code == 200
        assert resp.json()["answer"]

    def test_health_exempt_when_key_set(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VAULT_API_KEY", API_KEY)
        db = _make_test_db(tmp_path)
        with (
            patch("api.main._load_config", return_value=_make_cfg(tmp_path)),
            patch("api.main.ollama.Client") as mock_ollama,
            patch("api.main.chromadb.PersistentClient") as mock_chroma,
            patch("api.main._get_db", return_value=db),
        ):
            mock_ollama.return_value.list.return_value = {"models": []}
            mock_chroma.return_value.get_collection.return_value = MagicMock()
            resp = TestClient(app).get("/health")
            assert resp.status_code == 200

    def test_docs_exempt_when_key_set(self, monkeypatch):
        monkeypatch.setenv("VAULT_API_KEY", API_KEY)
        client = TestClient(app)
        assert client.get("/openapi.json").status_code == 200


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VAULT_API_KEY", raising=False)
        db = _make_test_db(tmp_path)
        with patch("api.main._get_db", return_value=db):
            resp = TestClient(app).get("/metrics")
            assert resp.status_code == 200
            assert "vault_queries_total" in resp.text
            assert "vault_files_total" in resp.text
            assert "vault_chunks_total" in resp.text

    def test_metrics_protected_when_key_set(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VAULT_API_KEY", API_KEY)
        db = _make_test_db(tmp_path)
        with patch("api.main._get_db", return_value=db):
            assert TestClient(app).get("/metrics").status_code == 401
