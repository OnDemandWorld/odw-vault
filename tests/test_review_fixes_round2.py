"""Regression tests for round 2 of the September 2026 review (deferred items now fixed).

Covers:
- upload byte cap + streamed write (api/main.py)
- CSV formula-injection guard on /audit/export (api/main.py)
- LIKE wildcard escaping in folder scoping + query search (rag/filters.py, api/main.py)
- X-Trace-Id header validation (api/main.py)
- OTLP traceId 32-hex format (api/spans.py)
- phase1 incremental hashing / real mtime / size-cap on known files (pipeline/phase1_walk.py)
"""

from __future__ import annotations

import io
import json
import re
import sqlite3
from datetime import UTC
from pathlib import Path
from unittest.mock import patch

import sqlite_utils
from fastapi.testclient import TestClient

from api.main import app
from api.spans import OtlpHttpSpanExporter, Span
from pipeline.db import migrate
from rag.filters import resolve_folder_filter


def _make_db(tmp_path: Path):
    db_path = tmp_path / "test_r2.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    db = sqlite_utils.Database(conn)
    migrate(db)
    return db


def _make_cfg(tmp_path: Path):
    from tests.test_file_management import _make_cfg as base

    return base(tmp_path)


# ---------------------------------------------------------------------------
# Upload size cap
# ---------------------------------------------------------------------------


class TestUploadSizeCap:
    def test_oversized_upload_rejected_and_no_leftover(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VAULT_MAX_UPLOAD_BYTES", "100")
        db = _make_db(tmp_path)
        cfg = _make_cfg(tmp_path)
        payload = b"x" * 500
        with patch("api.main._load_config", return_value=cfg), \
             patch("api.main._get_db", return_value=db):
            client = TestClient(app)
            response = client.post(
                "/files/upload",
                files=[("files", ("big.bin", io.BytesIO(payload), "application/octet-stream"))],
            )
        assert response.status_code == 200
        data = response.json()
        assert data["uploaded"] == 0
        assert data["failed"] == ["big.bin"]
        # Partial write must be cleaned up
        assert not list(cfg.corpus_root_path.rglob("big.bin*"))

    def test_under_cap_still_succeeds(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VAULT_MAX_UPLOAD_BYTES", "10000")
        db = _make_db(tmp_path)
        cfg = _make_cfg(tmp_path)
        with patch("api.main._load_config", return_value=cfg), \
             patch("api.main._get_db", return_value=db):
            client = TestClient(app)
            response = client.post(
                "/files/upload",
                files=[("files", ("ok.txt", io.BytesIO(b"small"), "text/plain"))],
            )
        assert response.json()["uploaded"] == 1
        assert (cfg.corpus_root_path / "ok.txt").read_bytes() == b"small"


# ---------------------------------------------------------------------------
# CSV formula-injection guard
# ---------------------------------------------------------------------------


class TestAuditCsvFormulaGuard:
    def test_formula_cells_are_prefixed(self, tmp_path):
        db = _make_db(tmp_path)
        db["audit_log"].insert(
            {
                "actor": "tester",
                "action": "query",
                "resource_type": "query_log",
                "resource_id": "1",
                "detail": "=cmd|'/C calc'!A0",
                "status": "ok",
            }
        )
        db.conn.commit()
        with patch("api.main._get_db", return_value=db):
            client = TestClient(app)
            response = client.get("/audit/export", params={"format": "csv"})
        assert response.status_code == 200
        assert "'=cmd" in response.text, "formula cell must be neutralized"

    def test_normal_cells_untouched(self, tmp_path):
        db = _make_db(tmp_path)
        db["audit_log"].insert(
            {
                "actor": "tester",
                "action": "query",
                "resource_type": "query_log",
                "resource_id": "2",
                "detail": "what is the platform",
                "status": "ok",
            }
        )
        db.conn.commit()
        with patch("api.main._get_db", return_value=db):
            client = TestClient(app)
            response = client.get("/audit/export", params={"format": "csv"})
        assert "what is the platform" in response.text
        assert "'what is" not in response.text


# ---------------------------------------------------------------------------
# LIKE escaping
# ---------------------------------------------------------------------------


class TestLikeEscaping:
    def _seed(self, db):
        db["folder"].insert(
            {"path": "/corpus/docs", "rel_path": "docs", "name": "docs", "depth": 1}
        )
        fid = next(iter(db.query("SELECT id FROM folder WHERE rel_path='docs'")))["id"]
        db["file"].insert(
            {
                "folder_id": fid,
                "path": "/corpus/docs/a.txt",
                "rel_path": "docs/a.txt",
                "name": "a.txt",
                "extension": "txt",
                "size_bytes": 1,
                "mtime": "2026-01-01T00:00:00",
                "sha256": "",
                "hash_status": "done",
                "identify_status": "done",
                "triage_status": "done",
                "is_dup_primary": 1,
                "excluded": 0,
            }
        )
        db.conn.commit()

    def test_wildcard_prefix_does_not_broaden_scope(self, tmp_path):
        db = _make_db(tmp_path)
        self._seed(db)
        # "%" as a literal path prefix must match nothing, not every folder
        assert resolve_folder_filter(db, {"path_prefix": "%"}) is None
        # "docs" still matches as before
        assert resolve_folder_filter(db, {"path_prefix": "docs"}) is not None

    def test_keyword_search_escapes_wildcards(self, tmp_path):
        db = _make_db(tmp_path)
        common = {
            "answer_text": "a",
            "answer_model": "m",
            "embedding_model": "e",
            "latency_ms": 1,
            "retrieved_chunks_json": "[]",
        }
        db["query_log"].insert({"user": None, "query_text": "100% cotton", **common})
        db["query_log"].insert({"user": None, "query_text": "100 dollars", **common})
        db.conn.commit()
        with patch("api.main._get_db", return_value=db):
            client = TestClient(app)
            response = client.get("/queries", params={"keyword": "100%"})
        items = response.json()["items"]
        assert len(items) == 1, "% must match a literal percent, not any suffix"
        assert items[0]["query_text"] == "100% cotton"


# ---------------------------------------------------------------------------
# Trace-id header validation
# ---------------------------------------------------------------------------


class TestTraceIdValidation:
    def test_valid_header_is_echoed(self, tmp_path):
        db = _make_db(tmp_path)
        with patch("api.main._get_db", return_value=db):
            client = TestClient(app)
            response = client.get("/health", headers={"X-Trace-Id": "abc-123_XY"})
        assert response.headers.get("X-Trace-Id") == "abc-123_XY"

    def test_log_injection_header_is_replaced(self, tmp_path):
        db = _make_db(tmp_path)
        evil = "x" + "\r\nFAKE: 1" + "y" * 200
        with patch("api.main._get_db", return_value=db):
            client = TestClient(app)
            response = client.get("/health", headers={"X-Trace-Id": evil})
        echoed = response.headers.get("X-Trace-Id")
        assert echoed != evil
        assert re.fullmatch(r"[A-Za-z0-9-]{36}", echoed), "invalid header → fresh UUID"


# ---------------------------------------------------------------------------
# OTLP traceId format
# ---------------------------------------------------------------------------


class TestOtlpTraceId:
    def test_payload_trace_id_is_32_hex(self):
        span = Span(
            name="vault.test",
            trace_id="11111111-2222-3333-4444-555555555555",
            span_id="abcdef0123456789",
            parent_span_id=None,
            start_ms=1.0,
            duration_ms=2.0,
        )
        payload = json.loads(
            OtlpHttpSpanExporter(endpoint="http://example/v1/traces")._payload(span)
        )
        trace_id = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["traceId"]
        assert re.fullmatch(r"[0-9a-f]{32}", trace_id)


# ---------------------------------------------------------------------------
# Phase 1 walk: incremental hashing, real mtime, size cap on known files
# ---------------------------------------------------------------------------


class TestPhase1Incremental:
    def _run(self, db, cfg, plog, rehash=False):
        from pipeline.phase1_walk import run_phase1

        return run_phase1(db, cfg, plog, workers=1, rehash=rehash)

    def test_second_walk_does_not_rehash(self, test_db, test_corpus, mock_plog):
        from tests.test_phase1_walk import TestPhase1Walk

        root, _ = test_corpus
        cfg = TestPhase1Walk()._make_config(root)
        r1 = self._run(test_db, cfg, mock_plog)
        assert r1["files_processed"] > 0
        # No mtime change, no rehash flag → nothing should be re-hashed
        r2 = self._run(test_db, cfg, mock_plog)
        assert r2["files_processed"] == 0, "incremental walk must skip known files"

    def test_rehash_flag_requeues(self, test_db, test_corpus, mock_plog):
        from tests.test_phase1_walk import TestPhase1Walk

        root, _ = test_corpus
        cfg = TestPhase1Walk()._make_config(root)
        self._run(test_db, cfg, mock_plog)
        r = self._run(test_db, cfg, mock_plog, rehash=True)
        assert r["files_processed"] > 0

    def test_mtime_is_real_not_now(self, test_db, test_corpus, mock_plog):
        import datetime as dt
        import os

        from tests.test_phase1_walk import TestPhase1Walk

        root, _ = test_corpus
        cfg = TestPhase1Walk()._make_config(root)
        # Backdate one file's mtime to a known UTC instant
        some_file = root / "readme.txt"
        old_ts = dt.datetime(2020, 1, 1, tzinfo=dt.UTC).timestamp()
        os.utime(some_file, (old_ts, old_ts))
        self._run(test_db, cfg, mock_plog)
        row = next(
            iter(
                test_db.query(
                    "SELECT mtime FROM file WHERE path = ?", [str(some_file.resolve())]
                )
            )
        )
        assert row["mtime"].startswith("2020-01-01"), "mtime column must hold real stat mtime"

    def test_grown_file_oversized_is_flagged(self, test_db, test_corpus, mock_plog):
        from tests.test_phase1_walk import TestPhase1Walk

        root, _ = test_corpus
        cfg = TestPhase1Walk()._make_config(root, max_size=10_485_760)
        self._run(test_db, cfg, mock_plog)
        # Grow a known file past the cap and re-walk: the size check must
        # still run for already-known files (previously bypassed by the
        # hash-skip continue, so grown files were silently accepted).
        target = root / "readme.txt"
        target.write_bytes(b"x" * 20_971_520)
        self._run(test_db, cfg, mock_plog)
        fails = list(
            test_db.query("SELECT * FROM failure WHERE phase='walk' AND error_class='oversized'")
        )
        assert fails, "grown file must be recorded as oversized"


# ---------------------------------------------------------------------------
# Sync repair pass: previously-failed files get retried
# ---------------------------------------------------------------------------


class TestSyncRepairPass:
    def test_chunk_without_embedding_is_requeued(self, tmp_path):
        from unittest.mock import MagicMock

        from pipeline.db import open_db
        from rag.indexer import IncrementalIndexer
        from tests.test_indexer import _make_app_config

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        cache = tmp_path / "cache"
        cache.mkdir()
        chroma = tmp_path / "chroma"
        chroma.mkdir()
        db = open_db(tmp_path / "r.db")
        migrate(db)
        cfg = _make_app_config(corpus, cache, chroma)

        txt = corpus / "healed.txt"
        content = "This is a test sentence. " * 30
        txt.write_text(content, encoding="utf-8")
        stat = txt.stat()

        db["folder"].insert({"path": str(corpus), "rel_path": ".", "name": "corpus", "depth": 0})
        folder_id = next(iter(db.query("SELECT id FROM folder LIMIT 1")))["id"]
        from datetime import datetime

        mtime_iso = datetime.fromtimestamp(stat.st_mtime, tz=UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        # File already in DB, hashed + chunked, but the chunk has no
        # embedding (simulates the embed phase failing while Ollama was down).
        import hashlib

        digest = hashlib.sha256(content.encode()).hexdigest()
        db["file"].insert(
            {
                "folder_id": folder_id,
                "path": str(txt),
                "rel_path": "healed.txt",
                "name": "healed.txt",
                "extension": "txt",
                "size_bytes": stat.st_size,
                "mtime": mtime_iso,
                "sha256": digest,
                "hash_status": "done",
                "identify_status": "done",
                "triage_status": "done",
                "extract_strategy": "filename-only",
                "is_dup_primary": 1,
                "excluded": 0,
            }
        )
        file_id = next(iter(db.query("SELECT id FROM file LIMIT 1")))["id"]
        db["chunk"].insert(
            {
                "file_id": file_id,
                "chunk_index": 0,
                "text": content[:200],
                "token_count": 40,
            }
        )
        db.conn.commit()

        indexer = IncrementalIndexer(db, cfg, chroma_client=MagicMock())
        embed_calls = []
        with patch(
            "rag.indexer.IncrementalIndexer._run_embedding",
            side_effect=lambda *a, **k: embed_calls.append(a),
        ), patch(
            "rag.indexer.IncrementalIndexer._run_summarization",
            side_effect=lambda *a, **k: None,
        ):
            summary = indexer.sync_all()

        assert summary["retried"] == 1, "file with un-embedded chunk must be requeued"
        assert summary["new"] == 0 and summary["modified"] == 0
        assert embed_calls, "the repair pass must re-run embedding for it"


class TestFtsCleanupNoCorruption:
    def test_cleanup_then_reindex_keeps_db_healthy(self, tmp_path):
        """Regression: _cleanup_file_derivatives must not double-delete FTS
        postings (manual delete + AFTER DELETE trigger) — that corrupted the
        database on every re-index of an existing file."""
        from rag.indexer import _cleanup_file_derivatives

        db = _make_db(tmp_path)
        db["folder"].insert(
            {"path": "/c", "rel_path": ".", "name": "c", "depth": 0}
        )
        fid = next(iter(db.query("SELECT id FROM folder")))["id"]
        db["file"].insert(
            {
                "folder_id": fid, "path": "/c/x.txt", "rel_path": "x.txt",
                "name": "x.txt", "extension": "txt", "size_bytes": 5,
                "mtime": "2026-01-01", "sha256": "", "hash_status": "done",
                "identify_status": "done", "triage_status": "done",
                "is_dup_primary": 1, "excluded": 0,
            }
        )
        file_id = next(iter(db.query("SELECT id FROM file")))["id"]
        db["chunk"].insert(
            {"file_id": file_id, "chunk_index": 0, "text": "hello world test", "token_count": 3}
        )
        db["chunk"].insert(
            {"file_id": file_id, "chunk_index": 1, "text": "second chunk here", "token_count": 3}
        )
        db.conn.commit()

        _cleanup_file_derivatives(db, file_id)
        db.conn.commit()
        assert next(iter(db.query("PRAGMA integrity_check")))["integrity_check"] == "ok"
        assert not list(db.query("SELECT id FROM chunk WHERE file_id = ?", [file_id]))
