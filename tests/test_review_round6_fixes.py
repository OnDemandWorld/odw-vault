"""Regression tests for the 2026-10 deep-review round (V-1/V-9/V-13)."""

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
import sqlite_utils
from fastapi.testclient import TestClient

import api.main as api_main
from api.main import app, _normalize_date_bound
from pipeline.db import migrate


@pytest.fixture
def tmp_db(tmp_path: Path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    db = sqlite_utils.Database(conn)
    migrate(db)
    yield db
    conn.close()


@pytest.fixture
def patched_db(tmp_db):
    patcher = patch.object(api_main, "_get_db", return_value=tmp_db)
    patcher.start()
    yield tmp_db
    patcher.stop()


def _seed_one_file(db, name="broken.pdf"):
    db["folder"].insert({"path": ".", "rel_path": ".", "name": ".", "depth": 0, "excluded": 0})
    folder_id = next(iter(db.query("SELECT id FROM folder LIMIT 1")))["id"]
    db["file"].insert(
        {
            "folder_id": folder_id,
            "path": f"/tmp/{name}",
            "rel_path": name,
            "name": name,
            "size_bytes": 10,
            "category": "document",
            "mime_type": "application/pdf",
            "created_at": "2026-01-03T00:00:00",
            "mtime": "2026-01-03T00:00:00Z",
            "hash_status": "done",
            "identify_status": "done",
            "triage_status": "done",
            "is_dup_primary": 1,
            "excluded": 0,
        }
    )
    db.conn.commit()
    return next(iter(db.query("SELECT MAX(id) AS id FROM file")))["id"]


def test_files_failed_status_matches_real_phase_labels(patched_db):
    """V-1: pipeline writers record failures as 'extract' (not 'extraction')."""
    db = patched_db
    failed_id = _seed_one_file(db)
    db["failure"].insert(
        {
            "file_id": failed_id,
            "phase": "extract",  # the label rag/phase8_extract.py actually writes
            "tool": "test",
            "error_class": "Error",
            "error_message": "fail",
        }
    )
    db.conn.commit()

    client = TestClient(app)
    response = client.get("/files", params={"status": "failed"})
    assert response.status_code == 200
    items = response.json()["items"]
    assert any(i["id"] == failed_id for i in items), "extract-phase failure must be listed as failed"

    pending = client.get("/files", params={"status": "pending"}).json()
    assert not any(i["id"] == failed_id for i in pending["items"])


def test_normalize_date_bound_is_inclusive():
    """V-9: a bare end date covers the whole final day."""
    assert _normalize_date_bound("2026-01-31", end=True) == "2026-01-31 23:59:59"
    assert _normalize_date_bound("2026-01-31", end=False) == "2026-01-31 00:00:00"
    assert _normalize_date_bound("2026-01-31T10:00:00", end=True) == "2026-01-31 10:00:00"
    assert _normalize_date_bound("2026-01-31 10:00:00", end=True) == "2026-01-31 10:00:00"


def test_eval_run_is_async(patched_db):
    """V-13: /eval/run returns immediately with a run id instead of blocking."""
    client = TestClient(app)
    response = client.post("/eval/run")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "running"
    assert body["run_id"]
    status = client.get(f"/eval/run/{body['run_id']}")
    assert status.status_code == 200
