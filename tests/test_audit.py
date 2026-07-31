"""Tests for V1.3 F-Vault-1 audit logging (api/audit.py + GET /audit).

Covers: migration idempotency, ``record_audit`` write/read + best-effort
safety, audit records produced by real endpoints (mocked at the same seams as
tests/test_api_query.py and tests/test_file_management.py so no
Ollama/Chroma/network is needed), GET /audit filtering, auth protection, and
best-effort degradation (a failing audit write never changes the main response).
"""

from __future__ import annotations

import io
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.audit import record_audit
from api.main import _resolve_actor, app
from pipeline.db import migrate, open_db
from tests.test_api_query import _gen_result, _make_cfg, _make_hit, _make_test_db

API_KEY = "test-secret-key"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _audit_rows(db):
    return list(db.query("SELECT * FROM audit_log ORDER BY id"))


def _make_file_cfg(tmp_path: Path):
    """Mock AppConfig for the file upload/delete endpoints."""
    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    chroma = tmp_path / "chroma"
    chroma.mkdir(exist_ok=True)
    cfg = MagicMock()
    cfg.corpus_root_path = corpus
    cfg.chroma_root_path = chroma
    cfg.models.embedding.collection_suffix = "test"
    return cfg


def _seed_query_log(db) -> int:
    """Insert a minimal query_log row and return its id."""
    db.conn.execute(
        "INSERT INTO query_log "
        "(query_text, retrieved_chunks_json, answer_text, answer_model, embedding_model, latency_ms) "
        "VALUES ('q', '[]', 'a', 'm', 'e', 1)"
    )
    db.conn.commit()
    return next(iter(db.query("SELECT id FROM query_log")))["id"]


# ---------------------------------------------------------------------------
# AU1 — migration + record_audit
# ---------------------------------------------------------------------------


class TestAuditMigration:
    def test_audit_log_table_created(self, tmp_path):
        db = open_db(tmp_path / "test.db")
        migrate(db)
        tables = {
            r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert "audit_log" in tables

    def test_idempotent(self, tmp_path):
        db = open_db(tmp_path / "test.db")
        migrate(db)
        migrate(db)  # must not raise
        cols = [row[1] for row in db.execute("PRAGMA table_info(audit_log)").fetchall()]
        assert cols == [
            "id",
            "ts",
            "actor",
            "action",
            "resource_type",
            "resource_id",
            "detail",
            "status",
        ]

    def test_schema_version_7_recorded(self, tmp_path):
        db = open_db(tmp_path / "test.db")
        migrate(db)
        versions = {r[0] for r in db.execute("SELECT version FROM schema_version").fetchall()}
        assert 7 in versions

    def test_indexes_created(self, tmp_path):
        db = open_db(tmp_path / "test.db")
        migrate(db)
        indexes = {
            r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
        }
        assert "idx_audit_log_action" in indexes
        assert "idx_audit_log_ts" in indexes


class TestRecordAudit:
    def test_write_and_read(self, tmp_path):
        db = _make_test_db(tmp_path)
        record_audit(db, "alice", "query", "query_log", resource_id=7, detail="what?")
        rows = _audit_rows(db)
        assert len(rows) == 1
        r = rows[0]
        assert r["actor"] == "alice"
        assert r["action"] == "query"
        assert r["resource_type"] == "query_log"
        assert r["resource_id"] == "7"  # stored as text
        assert r["detail"] == "what?"
        assert r["status"] == "ok"
        assert r["ts"]  # timestamp populated by default

    def test_defaults(self, tmp_path):
        db = _make_test_db(tmp_path)
        record_audit(db, "bob", "feedback", "query_log")
        r = _audit_rows(db)[0]
        assert r["resource_id"] is None
        assert r["detail"] is None
        assert r["status"] == "ok"

    def test_best_effort_never_raises(self):
        bad_db = MagicMock()
        bad_db.conn.execute.side_effect = sqlite3.OperationalError(
            "no such table: audit_log"
        )
        # Must not raise even though the write fails.
        record_audit(bad_db, "x", "query", "query_log")


# ---------------------------------------------------------------------------
# AU2 — operations produce audit records
# ---------------------------------------------------------------------------


class TestAuditOnOperation:
    @patch("api.main.generate_answer")
    @patch("api.main.retrieve")
    @patch("api.main._load_config")
    @patch("api.main.ollama.Client")
    @patch("api.main.chromadb.PersistentClient")
    def test_query_produces_audit_record(
        self, mock_chroma, mock_ollama, mock_load_cfg, mock_retrieve, mock_generate,
        tmp_path, monkeypatch,
    ):
        monkeypatch.delenv("VAULT_API_KEY", raising=False)
        monkeypatch.delenv("VAULT_AUDIT_ACTOR", raising=False)
        db = _make_test_db(tmp_path)
        mock_load_cfg.return_value = _make_cfg(tmp_path)
        mock_ollama.return_value.list.return_value = {"models": []}
        mock_chroma.return_value.get_collection.return_value = MagicMock()
        mock_retrieve.return_value = ([_make_hit()], {"retrieval_ms": 5.0, "query_lang": "en"})
        mock_generate.return_value = _gen_result()

        with patch("api.main._get_db", return_value=db):
            resp = TestClient(app).post("/query", json={"query": "What robot platform?"})

        assert resp.status_code == 200
        rows = _audit_rows(db)
        assert len(rows) == 1
        assert rows[0]["action"] == "query"
        assert rows[0]["resource_type"] == "query_log"
        assert rows[0]["actor"] == "anonymous"
        assert rows[0]["detail"] == "What robot platform?"
        assert rows[0]["resource_id"] == str(resp.json()["query_log_id"])

    def test_feedback_produces_audit_record(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VAULT_API_KEY", raising=False)
        db = _make_test_db(tmp_path)
        qid = _seed_query_log(db)

        with patch("api.main._get_db", return_value=db):
            resp = TestClient(app).post("/feedback", json={"query_log_id": qid, "feedback": "up"})

        assert resp.status_code == 200
        rows = _audit_rows(db)
        assert len(rows) == 1
        assert rows[0]["action"] == "feedback"
        assert rows[0]["resource_type"] == "query_log"
        assert rows[0]["resource_id"] == str(qid)
        assert rows[0]["detail"] == "up"

    def test_upload_produces_audit_record(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VAULT_API_KEY", raising=False)
        db = _make_test_db(tmp_path)
        cfg = _make_file_cfg(tmp_path)

        with patch("api.main._load_config", return_value=cfg), \
             patch("api.main._get_db", return_value=db):
            resp = TestClient(app).post(
                "/files/upload",
                files=[("files", ("test.txt", io.BytesIO(b"hello"), "text/plain"))],
            )

        assert resp.status_code == 200
        assert resp.json()["uploaded"] == 1
        rows = _audit_rows(db)
        assert len(rows) == 1
        assert rows[0]["action"] == "file.upload"
        assert rows[0]["resource_type"] == "file"
        assert rows[0]["detail"] == "test.txt"

    def test_delete_produces_audit_record(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VAULT_API_KEY", raising=False)
        db = _make_test_db(tmp_path)
        cfg = _make_file_cfg(tmp_path)

        # Seed a folder + file row.
        db["folder"].insert({"path": ".", "rel_path": ".", "name": ".", "depth": 0, "excluded": 0})
        db.conn.commit()
        folder_id = next(iter(db.query("SELECT id FROM folder LIMIT 1")))["id"]
        test_file = cfg.corpus_root_path / "gone.txt"
        test_file.write_bytes(b"bye")
        db["file"].insert({
            "folder_id": folder_id,
            "path": str(test_file),
            "rel_path": "gone.txt",
            "name": "gone.txt",
            "size_bytes": 3,
            "mtime": "2026-01-01T00:00:00",
            "is_dup_primary": 1,
            "excluded": 0,
        })
        db.conn.commit()
        file_id = next(iter(db.query("SELECT MAX(id) AS id FROM file")))["id"]

        with patch("api.main._load_config", return_value=cfg), \
             patch("api.main._get_db", return_value=db), \
             patch("api.main.chromadb.PersistentClient", return_value=MagicMock()):
            resp = TestClient(app).delete(f"/files/{file_id}")

        assert resp.status_code == 200
        rows = _audit_rows(db)
        assert len(rows) == 1
        assert rows[0]["action"] == "file.delete"
        assert rows[0]["resource_type"] == "file"
        assert rows[0]["resource_id"] == str(file_id)
        assert rows[0]["detail"] == "gone.txt"


# ---------------------------------------------------------------------------
# AU3 — GET /audit
# ---------------------------------------------------------------------------


class TestGetAudit:
    def test_returns_list_most_recent_first(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VAULT_API_KEY", raising=False)
        db = _make_test_db(tmp_path)
        record_audit(db, "a", "query", "query_log", resource_id=1)
        record_audit(db, "a", "file.upload", "file", resource_id=2)
        record_audit(db, "a", "query", "query_log", resource_id=3)

        with patch("api.main._get_db", return_value=db):
            resp = TestClient(app).get("/audit")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3
        assert len(data["items"]) == 3
        assert data["items"][0]["resource_id"] == "3"  # newest first

    def test_filter_by_action(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VAULT_API_KEY", raising=False)
        db = _make_test_db(tmp_path)
        record_audit(db, "a", "query", "query_log")
        record_audit(db, "a", "file.upload", "file")

        with patch("api.main._get_db", return_value=db):
            resp = TestClient(app).get("/audit", params={"action": "query"})

        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["action"] == "query"

    def test_limit(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VAULT_API_KEY", raising=False)
        db = _make_test_db(tmp_path)
        for i in range(5):
            record_audit(db, "a", "query", "query_log", resource_id=i)

        with patch("api.main._get_db", return_value=db):
            resp = TestClient(app).get("/audit", params={"limit": 2})

        assert resp.json()["total"] == 2

    def test_open_when_key_unset(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VAULT_API_KEY", raising=False)
        db = _make_test_db(tmp_path)
        with patch("api.main._get_db", return_value=db):
            assert TestClient(app).get("/audit").status_code == 200

    def test_protected_when_key_set(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VAULT_API_KEY", API_KEY)
        db = _make_test_db(tmp_path)
        with patch("api.main._get_db", return_value=db):
            assert TestClient(app).get("/audit").status_code == 401
            ok = TestClient(app).get("/audit", headers={"Authorization": f"Bearer {API_KEY}"})
            assert ok.status_code == 200


# ---------------------------------------------------------------------------
# AU4 — best-effort degradation
# ---------------------------------------------------------------------------


class TestBestEffortDegradation:
    def test_upload_still_200_when_audit_table_missing(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VAULT_API_KEY", raising=False)
        db = _make_test_db(tmp_path)
        cfg = _make_file_cfg(tmp_path)
        # Force audit writes to fail.
        db.conn.execute("DROP TABLE audit_log")
        db.conn.commit()

        with patch("api.main._load_config", return_value=cfg), \
             patch("api.main._get_db", return_value=db):
            resp = TestClient(app).post(
                "/files/upload",
                files=[("files", ("t.txt", io.BytesIO(b"x"), "text/plain"))],
            )

        assert resp.status_code == 200
        assert resp.json()["uploaded"] == 1

    def test_feedback_still_200_when_audit_write_fails(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VAULT_API_KEY", raising=False)
        db = _make_test_db(tmp_path)
        qid = _seed_query_log(db)
        db.conn.execute("DROP TABLE audit_log")
        db.conn.commit()

        with patch("api.main._get_db", return_value=db):
            resp = TestClient(app).post("/feedback", json={"query_log_id": qid, "feedback": "down"})

        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Actor resolution
# ---------------------------------------------------------------------------


class TestActorResolution:
    def test_anonymous_when_no_auth(self, monkeypatch):
        monkeypatch.delenv("VAULT_API_KEY", raising=False)
        monkeypatch.delenv("VAULT_AUDIT_ACTOR", raising=False)
        assert _resolve_actor() == "anonymous"

    def test_authenticated_when_key_set(self, monkeypatch):
        monkeypatch.setenv("VAULT_API_KEY", API_KEY)
        monkeypatch.delenv("VAULT_AUDIT_ACTOR", raising=False)
        assert _resolve_actor() == "authenticated"

    def test_configured_actor_wins(self, monkeypatch):
        monkeypatch.setenv("VAULT_AUDIT_ACTOR", "vault-service")
        assert _resolve_actor() == "vault-service"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
