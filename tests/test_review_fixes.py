"""Regression tests for bugs found in the September 2026 code review.

Each test pins a specific bug that was fixed:
- upload path traversal (api/main.py)
- feedback CHECK-constraint 500 (api/schemas.py)
- IncrementalIndexer._run_extraction extractor contract (rag/indexer.py)
- folder/workspace scope leaking in retrieval (rag/retrieval.py)
"""

from __future__ import annotations

import io
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import sqlite_utils
from fastapi.testclient import TestClient

from api.main import app
from pipeline.db import migrate
from rag.indexer import IncrementalIndexer
from rag.retrieval import _hierarchical_narrowing
from tests.test_indexer import _make_app_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db(tmp_path: Path):
    """Create a test DB with check_same_thread=False for TestClient."""
    db_path = tmp_path / "test_review.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    db = sqlite_utils.Database(conn)
    migrate(db)
    return db


# ---------------------------------------------------------------------------
# Upload path traversal
# ---------------------------------------------------------------------------


class TestUploadPathTraversal:
    def _upload(self, db, cfg, filename: str):
        with patch("api.main._load_config", return_value=cfg), \
             patch("api.main._get_db", return_value=db):
            client = TestClient(app)
            return client.post(
                "/files/upload",
                files=[("files", (filename, io.BytesIO(b"evil"), "text/plain"))],
            )

    def test_parent_segments_flattened_into_corpus(self, tmp_path):
        db = _make_db(tmp_path)
        cfg = _make_app_config(tmp_path / "corpus", tmp_path / "cache", tmp_path / "chroma")
        response = self._upload(db, cfg, "../../pwned.txt")
        assert response.status_code == 200
        assert response.json()["uploaded"] == 1
        # The file must land inside the corpus under its basename — never
        # outside the corpus root.
        assert (cfg.corpus_root_path / "pwned.txt").exists()
        assert not (tmp_path / "pwned.txt").exists()
        assert not (tmp_path.parent / "pwned.txt").exists()

    def test_absolute_path_flattened_into_corpus(self, tmp_path):
        db = _make_db(tmp_path)
        cfg = _make_app_config(tmp_path / "corpus", tmp_path / "cache", tmp_path / "chroma")
        # Raw multipart body: httpx strips path components from filenames,
        # so craft the Content-Disposition by hand.
        body = (
            "--B\r\n"
            'Content-Disposition: form-data; name="files"; filename="/etc/pwned.txt"\r\n'
            "Content-Type: text/plain\r\n\r\n"
            "evil\r\n"
            "--B--\r\n"
        )
        with patch("api.main._load_config", return_value=cfg), \
             patch("api.main._get_db", return_value=db):
            client = TestClient(app)
            response = client.post(
                "/files/upload",
                content=body.encode(),
                headers={"Content-Type": "multipart/form-data; boundary=B"},
            )
        assert response.status_code == 200
        assert response.json()["uploaded"] == 1
        assert (cfg.corpus_root_path / "pwned.txt").exists()
        assert not Path("/etc/pwned.txt").exists()

    def test_bare_name_still_works(self, tmp_path):
        db = _make_db(tmp_path)
        cfg = _make_app_config(tmp_path / "corpus", tmp_path / "cache", tmp_path / "chroma")
        response = self._upload(db, cfg, "ok.txt")
        assert response.status_code == 200
        assert response.json()["uploaded"] == 1
        assert (cfg.corpus_root_path / "ok.txt").exists()


# ---------------------------------------------------------------------------
# Feedback validation
# ---------------------------------------------------------------------------


class TestFeedbackValidation:
    def test_invalid_feedback_value_is_422(self, tmp_path):
        db = _make_db(tmp_path)
        with patch("api.main._get_db", return_value=db):
            client = TestClient(app)
            response = client.post("/feedback", json={"query_log_id": 1, "feedback": "upvote"})
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Indexer extraction contract (regression: tuple vs dict)
# ---------------------------------------------------------------------------


class TestRunExtractionContract:
    def test_real_extractor_writes_succeeded_row(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        db = _make_db(tmp_path)
        cfg = _make_app_config(corpus, tmp_path / "cache", tmp_path / "chroma")

        txt = corpus / "doc.txt"
        txt.write_text("hello world", encoding="utf-8")
        db["folder"].insert({"path": str(corpus), "rel_path": ".", "name": "corpus", "depth": 0})
        folder_id = next(iter(db.query("SELECT id FROM folder WHERE rel_path = '.'")))["id"]
        db["file"].insert({
            "folder_id": folder_id, "path": str(txt), "rel_path": "doc.txt",
            "name": "doc.txt", "extension": "txt", "size_bytes": 11,
            "mtime": "2026-01-01T00:00:00", "sha256": "", "mime_type": "text/plain",
            "hash_status": "pending", "identify_status": "pending",
            "triage_status": "pending", "is_dup_primary": 1, "excluded": 0,
        })
        file_id = next(iter(db.query("SELECT id FROM file WHERE rel_path = 'doc.txt'")))["id"]

        indexer = IncrementalIndexer(db, cfg, chroma_client=MagicMock())
        # Real extractor ("filename-only") — exercises the actual contract.
        indexer._run_extraction(file_id, txt, "filename-only")

        rows = list(db.query("SELECT * FROM extraction WHERE file_id = ?", [file_id]))
        assert len(rows) == 1
        assert rows[0]["succeeded"] == 1
        assert rows[0]["char_count"] > 0
        assert not list(db.query("SELECT * FROM failure WHERE file_id = ?", [file_id]))

    def test_extractor_failure_records_failure_row(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        db = _make_db(tmp_path)
        cfg = _make_app_config(corpus, tmp_path / "cache", tmp_path / "chroma")

        txt = corpus / "doc.txt"
        txt.write_text("hello world", encoding="utf-8")
        db["folder"].insert({"path": str(corpus), "rel_path": ".", "name": "corpus", "depth": 0})
        folder_id = next(iter(db.query("SELECT id FROM folder WHERE rel_path = '.'")))["id"]
        db["file"].insert({
            "folder_id": folder_id, "path": str(txt), "rel_path": "doc.txt",
            "name": "doc.txt", "extension": "txt", "size_bytes": 11,
            "mtime": "2026-01-01T00:00:00", "sha256": "", "mime_type": "text/plain",
            "hash_status": "pending", "identify_status": "pending",
            "triage_status": "pending", "is_dup_primary": 1, "excluded": 0,
        })
        file_id = next(iter(db.query("SELECT id FROM file WHERE rel_path = 'doc.txt'")))["id"]

        indexer = IncrementalIndexer(db, cfg, chroma_client=MagicMock())
        fake_map = {"failing": lambda p: (None, {}, False, "boom")}
        with patch("rag.phase8_extract.EXTRACTOR_MAP", fake_map):
            indexer._run_extraction(file_id, txt, "failing")

        rows = list(db.query("SELECT * FROM extraction WHERE file_id = ?", [file_id]))
        assert rows and rows[0]["succeeded"] == 0
        failures = list(db.query("SELECT * FROM failure WHERE file_id = ?", [file_id]))
        assert failures and failures[0]["phase"] == "extract"


# ---------------------------------------------------------------------------
# Retrieval scoping (regression: empty intersection returned None)
# ---------------------------------------------------------------------------


class TestHierarchicalNarrowingScope:
    def _call(self, db, allowed):
        cfg = MagicMock()
        cfg.retrieval.top_k_folders = 3
        cfg.retrieval.top_k_documents = 5
        suffix = "test"
        client = MagicMock()
        # get_collection raises → both folder and summary legs are skipped
        client.get_collection.side_effect = RuntimeError("missing")
        return _hierarchical_narrowing(client, suffix, [0.0], "query", cfg, db, allowed)

    def test_filter_present_narrowing_empty_returns_allowed_scope(self, tmp_path):
        db = _make_db(tmp_path)
        allowed = {1, 2, 3}
        result = self._call(db, allowed)
        assert result == allowed, "empty narrowing must fall back to the allowed scope, never None"

    def test_no_filter_narrowing_empty_returns_none(self, tmp_path):
        db = _make_db(tmp_path)
        result = self._call(db, None)
        assert result is None, "no filter and no narrowing → unrestricted search is intended"

    def test_narrowing_hits_outside_scope_fall_back_to_scope(self, tmp_path):
        db = _make_db(tmp_path)
        allowed = {7}
        client = MagicMock()
        folder_coll = MagicMock()
        folder_coll.query.return_value = {
            "ids": [["f1"]],
            "metadatas": [[{"folder_id": 99}]],
        }
        client.get_collection.return_value = folder_coll
        # folder 99 contains file 42 — outside the allowed scope
        db["folder"].insert({"id": 99, "path": "/x", "rel_path": "x", "name": "x", "depth": 0, "excluded": 0})
        db["file"].insert({
            "folder_id": 99, "path": "/x/f.txt", "rel_path": "f.txt",
            "name": "f.txt", "extension": "txt", "size_bytes": 1,
            "mtime": "2026-01-01T00:00:00", "sha256": "", "mime_type": "text/plain",
            "hash_status": "done", "identify_status": "done",
            "triage_status": "done", "is_dup_primary": 1, "excluded": 0,
        })
        cfg = MagicMock()
        cfg.retrieval.top_k_folders = 3
        cfg.retrieval.top_k_documents = 5
        result = _hierarchical_narrowing(client, "test", [0.0], "query", cfg, db, allowed)
        assert result == allowed
