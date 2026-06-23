"""Tests for pipeline/db.py."""


import sqlite3
from unittest.mock import patch

from pipeline.db import (
    cleanup_stale_runs,
    heartbeat_model_run,
    is_db_initialized,
    migrate,
    open_db,
    start_model_run,
)

EXPECTED_TABLES = [
    "schema_version",
    "config",
    "pipeline_run",
    "folder",
    "file",
    "archive_expansion",
    "format_policy",
    "failure",
    "extraction",
    "summary",
    "chunk",
    "embedding_ref",
]
EXPECTED_VIEWS = [
    "v_format_histogram",
    "v_category_summary",
    "v_ocr_workload",
    "v_transcription_workload",
    "v_duplicate_summary",
    "v_problem_files",
    "v_unknown_formats",
]


class TestOpenDb:
    def test_creates_file(self, tmp_path):
        db_path = tmp_path / "test.db"
        db = open_db(db_path)
        assert db_path.exists()

    def test_pragmas(self, tmp_path):
        db_path = tmp_path / "test.db"
        db = open_db(db_path)
        fk = db.execute("PRAGMA foreign_keys").fetchone()
        assert fk[0] == 1
        jm = db.execute("PRAGMA journal_mode").fetchone()
        assert jm[0] == "wal"

    def test_idempotent(self, tmp_path):
        db_path = tmp_path / "test.db"
        db = open_db(db_path)
        db2 = open_db(db_path)
        # Both should work
        assert db.execute("SELECT 1").fetchone() == (1,)
        assert db2.execute("SELECT 1").fetchone() == (1,)


class TestMigrate:
    def test_creates_tables(self, tmp_path):
        db_path = tmp_path / "test.db"
        db = open_db(db_path)
        migrate(db)
        tables = {
            r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        for t in EXPECTED_TABLES:
            assert t in tables

    def test_creates_views(self, tmp_path):
        db_path = tmp_path / "test.db"
        db = open_db(db_path)
        migrate(db)
        views = {
            r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='view'").fetchall()
        }
        for v in EXPECTED_VIEWS:
            assert v in views

    def test_idempotent(self, tmp_path):
        db_path = tmp_path / "test.db"
        db = open_db(db_path)
        migrate(db)
        migrate(db)  # Should not raise
        assert True

    def test_schema_version_recorded(self, tmp_path):
        db_path = tmp_path / "test.db"
        db = open_db(db_path)
        migrate(db)
        rows = list(db.execute("SELECT * FROM schema_version"))
        assert len(rows) >= 1
        assert rows[0][0] == 1  # version

    def test_creates_indexes(self, tmp_path):
        db_path = tmp_path / "test.db"
        db = open_db(db_path)
        migrate(db)
        indexes = {
            r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
        }
        assert "idx_file_sha256" in indexes
        assert "idx_file_category" in indexes
        assert "idx_folder_parent" in indexes


class TestIsDbInitialized:
    def test_false_before_migration(self, tmp_path):
        db_path = tmp_path / "test.db"
        assert is_db_initialized(db_path) is False

    def test_true_after_migration(self, tmp_path):
        db_path = tmp_path / "test.db"
        db = open_db(db_path)
        migrate(db)
        assert is_db_initialized(db_path) is True

    def test_false_nonexistent(self, tmp_path):
        assert is_db_initialized(tmp_path / "nope.db") is False


class TestViewsQueryable:
    def test_v_format_histogram_empty(self, test_db):
        rows = list(test_db.query("SELECT * FROM v_format_histogram"))
        assert rows == []

    def test_v_category_summary_empty(self, test_db):
        rows = list(test_db.query("SELECT * FROM v_category_summary"))
        assert rows == []

    def test_v_ocr_workload_empty(self, test_db):
        rows = list(test_db.query("SELECT * FROM v_ocr_workload"))
        assert len(rows) == 1
        assert rows[0]["scanned_pdfs"] == 0


class TestDatabaseQueryReturnsList:
    """db.query() returns list by default, not a lazy generator."""

    def test_query_returns_list(self, tmp_path):
        db = open_db(tmp_path / "test.db")
        migrate(db)
        db["folder"].insert({"path": "a", "rel_path": "a", "name": "a", "depth": 0})
        db.conn.commit()
        rows = db.query("SELECT * FROM folder")
        assert isinstance(rows, list)
        assert len(rows) == 1

    def test_query_can_be_iterated_twice(self, tmp_path):
        db = open_db(tmp_path / "test.db")
        migrate(db)
        db["folder"].insert({"path": "a", "rel_path": "a", "name": "a", "depth": 0})
        db.conn.commit()
        rows = db.query("SELECT * FROM folder")
        first = list(rows)
        second = list(rows)
        assert len(first) == 1
        assert len(second) == 1

    def test_query_lazy_returns_generator(self, tmp_path):
        db = open_db(tmp_path / "test.db")
        migrate(db)
        rows = db.query("SELECT 1", lazy=True)
        assert not isinstance(rows, list)


class TestDatabaseTransaction:
    """db.transaction() provides atomic writes."""

    def test_transaction_commit(self, tmp_path):
        db = open_db(tmp_path / "test.db")
        migrate(db)
        with db.transaction():
            db.conn.execute(
                "INSERT INTO folder (path, rel_path, name, depth) VALUES ('a', 'a', 'a', 0)"
            )
        rows = list(db.query("SELECT * FROM folder"))
        assert len(rows) == 1

    def test_transaction_rollback(self, tmp_path):
        db = open_db(tmp_path / "test.db")
        migrate(db)
        try:
            with db.transaction():
                db.conn.execute(
                    "INSERT INTO folder (path, rel_path, name, depth) VALUES ('a', 'a', 'a', 0)"
                )
                raise ValueError("abort")
        except ValueError:
            pass
        rows = list(db.query("SELECT * FROM folder"))
        assert len(rows) == 0


class TestRetryOnLocked:
    """_retry_on_locked retries on 'database is locked' errors."""

    def test_retries_on_locked(self, tmp_path):
        from pipeline.db import _retry_on_locked

        call_count = 0

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise sqlite3.OperationalError("database is locked")
            return "ok"

        result = _retry_on_locked(flaky, max_retries=5, backoff=0.01)
        assert result == "ok"
        assert call_count == 3

    def test_raises_after_max_retries(self, tmp_path):
        from pipeline.db import _retry_on_locked

        def always_locked():
            raise sqlite3.OperationalError("database is locked")

        with patch("pipeline.db.time.sleep"):
            try:
                _retry_on_locked(always_locked, max_retries=3, backoff=0.01)
            except sqlite3.OperationalError as e:
                assert "locked" in str(e).lower()
            else:
                raise AssertionError("should have raised")


class TestHeartbeatModelRun:
    """heartbeat_model_run updates progress counters."""

    def test_heartbeat_updates_counters(self, tmp_path):
        db = open_db(tmp_path / "test.db")
        migrate(db)
        run_id = start_model_run(db, role="extraction", model_name="test", phase="extract", config_hash="abc")
        heartbeat_model_run(db, run_id, items_processed=10, items_failed=2)
        row = list(db.query("SELECT items_processed, items_failed FROM model_run WHERE id=?", [run_id]))[0]
        assert row["items_processed"] == 10
        assert row["items_failed"] == 2

    def test_heartbeat_with_zero_values(self, tmp_path):
        db = open_db(tmp_path / "test.db")
        migrate(db)
        run_id = start_model_run(db, role="extraction", model_name="test", phase="extract", config_hash="abc")
        heartbeat_model_run(db, run_id)
        row = list(db.query("SELECT items_processed, items_failed FROM model_run WHERE id=?", [run_id]))[0]
        assert row["items_processed"] == 0
        assert row["items_failed"] == 0


class TestCleanupStaleRuns:
    """cleanup_stale_runs marks stuck runs as aborted."""

    def test_cleans_up_stale_runs(self, tmp_path):
        db = open_db(tmp_path / "test.db")
        migrate(db)
        # Insert a fake "running" run with an old started_at
        db.conn.execute(
            "INSERT INTO model_run (role, model_name, phase, status, started_at, config_hash) "
            "VALUES ('extraction', 'test', 'extract', 'running', datetime('now', '-25 hours'), 'abc')"
        )
        db.conn.commit()
        count = cleanup_stale_runs(db, max_age_hours=24)
        assert count == 1
        rows = list(db.query("SELECT status FROM model_run"))
        assert rows[0]["status"] == "aborted"

    def test_skips_recent_runs(self, tmp_path):
        db = open_db(tmp_path / "test.db")
        migrate(db)
        db.conn.execute(
            "INSERT INTO model_run (role, model_name, phase, status, started_at, config_hash) "
            "VALUES ('extraction', 'test', 'extract', 'running', datetime('now', '-1 hours'), 'abc')"
        )
        db.conn.commit()
        count = cleanup_stale_runs(db, max_age_hours=24)
        assert count == 0
        rows = list(db.query("SELECT status FROM model_run"))
        assert rows[0]["status"] == "running"

    def test_no_stale_runs(self, tmp_path):
        db = open_db(tmp_path / "test.db")
        migrate(db)
        count = cleanup_stale_runs(db, max_age_hours=24)
        assert count == 0
