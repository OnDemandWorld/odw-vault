"""Tests for V1.4 F-3' compliance audit report export (GET /audit/export).

Covers the extended audit read seam (time-range / actor / action filtering via
``api.audit.query_audit_records``), the ``GET /audit/export`` JSON + CSV output
shapes, auth protection (shared ``_require_api_key`` middleware), and a guard
that the existing ``GET /audit`` behaviour is unchanged. Mocked at the same
seams as tests/test_audit.py / tests/test_api_query.py (patch ``api.main._get_db``
with a real migrated SQLite DB) so no Ollama / Chroma / network is required.
"""

from __future__ import annotations

import csv
import io
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.audit import query_audit_records
from api.main import app
from tests.test_api_query import _make_test_db

API_KEY = "test-secret-key"

COLUMNS = ["id", "ts", "actor", "action", "resource_type", "resource_id", "detail", "status"]


def _insert(
    db,
    ts,
    actor,
    action,
    resource_type="query_log",
    resource_id=None,
    detail=None,
    status="ok",
):
    """Insert an audit row with an explicit ``ts`` (record_audit uses now())."""
    db.conn.execute(
        "INSERT INTO audit_log "
        "(ts, actor, action, resource_type, resource_id, detail, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            ts,
            actor,
            action,
            resource_type,
            None if resource_id is None else str(resource_id),
            detail,
            status,
        ),
    )
    db.conn.commit()


def _seed_time_range(db):
    """Three rows spanning Jan-Feb 2026 with distinct actors/actions."""
    _insert(db, "2026-01-01 10:00:00", "alice", "query", resource_id=1)
    _insert(db, "2026-01-15 10:00:00", "bob", "file.upload", "file", resource_id=2)
    _insert(db, "2026-02-01 10:00:00", "alice", "query", resource_id=3)


# ---------------------------------------------------------------------------
# X1 — query_audit_records filtering (the shared audit read seam)
# ---------------------------------------------------------------------------


class TestQueryAuditRecords:
    def test_no_filters_returns_all_most_recent_first(self, tmp_path):
        db = _make_test_db(tmp_path)
        _seed_time_range(db)
        rows = query_audit_records(db)
        assert len(rows) == 3
        # Most recent ts first.
        assert rows[0]["ts"] == "2026-02-01 10:00:00"
        assert rows[-1]["ts"] == "2026-01-01 10:00:00"

    def test_filter_by_action(self, tmp_path):
        db = _make_test_db(tmp_path)
        _seed_time_range(db)
        rows = query_audit_records(db, action="query")
        assert len(rows) == 2
        assert {r["action"] for r in rows} == {"query"}

    def test_filter_by_actor(self, tmp_path):
        db = _make_test_db(tmp_path)
        _seed_time_range(db)
        rows = query_audit_records(db, actor="bob")
        assert len(rows) == 1
        assert rows[0]["actor"] == "bob"
        assert rows[0]["action"] == "file.upload"

    def test_filter_by_start(self, tmp_path):
        db = _make_test_db(tmp_path)
        _seed_time_range(db)
        rows = query_audit_records(db, start="2026-01-10 00:00:00")
        assert {r["resource_id"] for r in rows} == {"2", "3"}

    def test_filter_by_end(self, tmp_path):
        db = _make_test_db(tmp_path)
        _seed_time_range(db)
        rows = query_audit_records(db, end="2026-01-20 00:00:00")
        assert {r["resource_id"] for r in rows} == {"1", "2"}

    def test_filter_by_start_and_end(self, tmp_path):
        db = _make_test_db(tmp_path)
        _seed_time_range(db)
        rows = query_audit_records(
            db, start="2026-01-10 00:00:00", end="2026-01-20 00:00:00"
        )
        assert {r["resource_id"] for r in rows} == {"2"}

    def test_bounds_are_inclusive(self, tmp_path):
        db = _make_test_db(tmp_path)
        _seed_time_range(db)
        rows = query_audit_records(
            db, start="2026-01-15 10:00:00", end="2026-01-15 10:00:00"
        )
        assert {r["resource_id"] for r in rows} == {"2"}

    def test_iso_T_separator_accepted(self, tmp_path):
        db = _make_test_db(tmp_path)
        _seed_time_range(db)
        # ISO-8601 with a 'T' separator normalises via SQLite datetime().
        rows = query_audit_records(db, start="2026-01-10T00:00:00")
        assert {r["resource_id"] for r in rows} == {"2", "3"}

    def test_combined_actor_action_time(self, tmp_path):
        db = _make_test_db(tmp_path)
        _seed_time_range(db)
        rows = query_audit_records(
            db, actor="alice", action="query", start="2026-01-10 00:00:00"
        )
        assert {r["resource_id"] for r in rows} == {"3"}

    def test_limit(self, tmp_path):
        db = _make_test_db(tmp_path)
        _seed_time_range(db)
        rows = query_audit_records(db, limit=2)
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# X2 — GET /audit/export (JSON)
# ---------------------------------------------------------------------------


class TestExportJson:
    def test_default_json_shape(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VAULT_API_KEY", raising=False)
        db = _make_test_db(tmp_path)
        _seed_time_range(db)

        with patch("api.main._get_db", return_value=db):
            resp = TestClient(app).get("/audit/export")

        assert resp.status_code == 200
        data = resp.json()
        assert set(data.keys()) == {"items", "total", "filters"}
        assert data["total"] == 3
        assert len(data["items"]) == 3
        # Every item carries the full audit column set.
        assert set(data["items"][0].keys()) == set(COLUMNS)

    def test_json_filters_echoed(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VAULT_API_KEY", raising=False)
        db = _make_test_db(tmp_path)
        _seed_time_range(db)

        with patch("api.main._get_db", return_value=db):
            resp = TestClient(app).get(
                "/audit/export",
                params={
                    "format": "json",
                    "start": "2026-01-10T00:00:00",
                    "end": "2026-01-20T00:00:00",
                    "actor": "bob",
                    "action": "file.upload",
                    "limit": 50,
                },
            )

        data = resp.json()
        assert data["filters"] == {
            "start": "2026-01-10T00:00:00",
            "end": "2026-01-20T00:00:00",
            "actor": "bob",
            "action": "file.upload",
            "limit": 50,
        }
        assert data["total"] == 1
        assert data["items"][0]["actor"] == "bob"

    def test_json_time_filter_applied(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VAULT_API_KEY", raising=False)
        db = _make_test_db(tmp_path)
        _seed_time_range(db)

        with patch("api.main._get_db", return_value=db):
            resp = TestClient(app).get(
                "/audit/export", params={"start": "2026-01-10 00:00:00"}
            )

        data = resp.json()
        assert data["total"] == 2
        assert {i["resource_id"] for i in data["items"]} == {"2", "3"}

    def test_invalid_format_rejected(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VAULT_API_KEY", raising=False)
        db = _make_test_db(tmp_path)
        with patch("api.main._get_db", return_value=db):
            resp = TestClient(app).get("/audit/export", params={"format": "xml"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# X2 — GET /audit/export (CSV)
# ---------------------------------------------------------------------------


class TestExportCsv:
    def test_csv_headers(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VAULT_API_KEY", raising=False)
        db = _make_test_db(tmp_path)
        _seed_time_range(db)

        with patch("api.main._get_db", return_value=db):
            resp = TestClient(app).get("/audit/export", params={"format": "csv"})

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/csv")
        assert "attachment" in resp.headers["content-disposition"]
        assert "audit_export.csv" in resp.headers["content-disposition"]

    def test_csv_columns_and_rows(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VAULT_API_KEY", raising=False)
        db = _make_test_db(tmp_path)
        _seed_time_range(db)

        with patch("api.main._get_db", return_value=db):
            resp = TestClient(app).get("/audit/export", params={"format": "csv"})

        parsed = list(csv.reader(io.StringIO(resp.text)))
        header, *rows = parsed
        assert header == COLUMNS
        assert len(rows) == 3
        # Most recent first: first data row is the 2026-02-01 alice/query row.
        first = dict(zip(header, rows[0], strict=True))
        assert first["ts"] == "2026-02-01 10:00:00"
        assert first["actor"] == "alice"
        assert first["action"] == "query"
        assert first["status"] == "ok"

    def test_csv_respects_filters(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VAULT_API_KEY", raising=False)
        db = _make_test_db(tmp_path)
        _seed_time_range(db)

        with patch("api.main._get_db", return_value=db):
            resp = TestClient(app).get(
                "/audit/export", params={"format": "csv", "actor": "bob"}
            )

        parsed = list(csv.reader(io.StringIO(resp.text)))
        header, *rows = parsed
        assert header == COLUMNS
        assert len(rows) == 1
        assert dict(zip(header, rows[0], strict=True))["actor"] == "bob"

    def test_csv_empty_has_header_only(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VAULT_API_KEY", raising=False)
        db = _make_test_db(tmp_path)
        _seed_time_range(db)

        with patch("api.main._get_db", return_value=db):
            resp = TestClient(app).get(
                "/audit/export", params={"format": "csv", "actor": "nobody"}
            )

        parsed = list(csv.reader(io.StringIO(resp.text)))
        assert parsed == [COLUMNS]


# ---------------------------------------------------------------------------
# X3 — auth protection (shared _require_api_key middleware)
# ---------------------------------------------------------------------------


class TestExportAuth:
    def test_open_when_key_unset(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VAULT_API_KEY", raising=False)
        db = _make_test_db(tmp_path)
        with patch("api.main._get_db", return_value=db):
            assert TestClient(app).get("/audit/export").status_code == 200

    def test_protected_when_key_set(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VAULT_API_KEY", API_KEY)
        db = _make_test_db(tmp_path)
        with patch("api.main._get_db", return_value=db):
            assert TestClient(app).get("/audit/export").status_code == 401
            ok = TestClient(app).get(
                "/audit/export", headers={"Authorization": f"Bearer {API_KEY}"}
            )
            assert ok.status_code == 200

    def test_csv_protected_when_key_set(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VAULT_API_KEY", API_KEY)
        db = _make_test_db(tmp_path)
        with patch("api.main._get_db", return_value=db):
            assert (
                TestClient(app).get("/audit/export", params={"format": "csv"}).status_code
                == 401
            )


# ---------------------------------------------------------------------------
# Guard — GET /audit behaviour is unchanged by the additive export endpoint
# ---------------------------------------------------------------------------


class TestGetAuditUnchanged:
    def test_get_audit_shape_unchanged(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VAULT_API_KEY", raising=False)
        db = _make_test_db(tmp_path)
        _seed_time_range(db)

        with patch("api.main._get_db", return_value=db):
            resp = TestClient(app).get("/audit")

        assert resp.status_code == 200
        data = resp.json()
        # Original shape: only items + total (no filters key), newest first.
        assert set(data.keys()) == {"items", "total"}
        assert data["total"] == 3
        assert data["items"][0]["resource_id"] == "3"

    def test_get_audit_action_filter_still_works(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VAULT_API_KEY", raising=False)
        db = _make_test_db(tmp_path)
        _seed_time_range(db)

        with patch("api.main._get_db", return_value=db):
            resp = TestClient(app).get("/audit", params={"action": "file.upload"})

        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["action"] == "file.upload"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
