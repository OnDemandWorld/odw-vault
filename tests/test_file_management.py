"""Tests for file upload, listing, and deletion endpoints."""

from __future__ import annotations

import io
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import sqlite_utils
from fastapi.testclient import TestClient

from api.main import app
from pipeline.db import migrate, open_db


def _make_db(tmp_path: Path):
    """Create a test DB with check_same_thread=False for TestClient."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    db = sqlite_utils.Database(conn)
    migrate(db)
    return db


def _make_cfg(tmp_path: Path):
    """Build a mock AppConfig."""
    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    chroma = tmp_path / "chroma"
    chroma.mkdir(exist_ok=True)

    cfg = MagicMock()
    cfg.corpus_root_path = corpus
    cfg.chroma_root_path = chroma
    cfg.models.embedding.collection_suffix = "test"
    return cfg


def _seed_folder(db, rel_path=".", name="."):
    """Insert a folder row and return its ID."""
    db["folder"].insert({
        "path": rel_path,
        "rel_path": rel_path,
        "name": name,
        "depth": 0,
        "excluded": 0,
    })
    db.conn.commit()
    return next(iter(db.query("SELECT id FROM folder LIMIT 1")))["id"]


def _get_last_file_id(db):
    """Get the ID of the last inserted file."""
    return next(iter(db.query("SELECT MAX(id) as id FROM file")))["id"]


def _get_last_chunk_id(db):
    """Get the ID of the last inserted chunk."""
    return next(iter(db.query("SELECT MAX(id) as id FROM chunk")))["id"]


class TestFileUpload:
    def test_upload_and_save(self, tmp_path):
        db = _make_db(tmp_path)
        cfg = _make_cfg(tmp_path)

        with patch("api.main._load_config", return_value=cfg), \
             patch("api.main._get_db", return_value=db):
            client = TestClient(app)
            response = client.post(
                "/files/upload",
                files=[("files", ("test.txt", io.BytesIO(b"hello world"), "text/plain"))],
            )

        assert response.status_code == 200
        data = response.json()
        assert data["uploaded"] == 1
        assert data["failed"] == []

        # Verify file saved to disk
        saved = cfg.corpus_root_path / "test.txt"
        assert saved.exists()
        assert saved.read_bytes() == b"hello world"

        # Verify DB record
        rows = list(db.query("SELECT * FROM file WHERE name = 'test.txt'"))
        assert len(rows) == 1
        assert rows[0]["size_bytes"] == 11
        assert rows[0]["mime_type"] == "text/plain"
        assert rows[0]["rel_path"] == "test.txt"

    def test_upload_conflict(self, tmp_path):
        db = _make_db(tmp_path)
        cfg = _make_cfg(tmp_path)

        # Pre-create file to trigger conflict
        (cfg.corpus_root_path / "test.txt").write_bytes(b"original")

        with patch("api.main._load_config", return_value=cfg), \
             patch("api.main._get_db", return_value=db):
            client = TestClient(app)
            response = client.post(
                "/files/upload",
                files=[("files", ("test.txt", io.BytesIO(b"new content"), "text/plain"))],
            )

        assert response.status_code == 200
        data = response.json()
        assert data["uploaded"] == 1

        # Original file unchanged
        assert (cfg.corpus_root_path / "test.txt").read_bytes() == b"original"
        # New file with counter suffix
        assert (cfg.corpus_root_path / "test (1).txt").exists()
        assert (cfg.corpus_root_path / "test (1).txt").read_bytes() == b"new content"

    def test_upload_multiple(self, tmp_path):
        db = _make_db(tmp_path)
        cfg = _make_cfg(tmp_path)

        with patch("api.main._load_config", return_value=cfg), \
             patch("api.main._get_db", return_value=db):
            client = TestClient(app)
            response = client.post(
                "/files/upload",
                files=[
                    ("files", ("a.txt", io.BytesIO(b"aaa"), "text/plain")),
                    ("files", ("b.txt", io.BytesIO(b"bbb"), "text/plain")),
                ],
            )

        assert response.status_code == 200
        data = response.json()
        assert data["uploaded"] == 2
        assert data["failed"] == []


class TestFileList:
    def test_list_files(self, tmp_path):
        db = _make_db(tmp_path)
        cfg = _make_cfg(tmp_path)
        folder_id = _seed_folder(db)

        # Insert files with different statuses
        db["file"].insert({
            "folder_id": folder_id,
            "path": str(cfg.corpus_root_path / "indexed.txt"),
            "rel_path": "indexed.txt",
            "name": "indexed.txt",
            "size_bytes": 100,
            "mtime": "2026-01-01T00:00:00",
            "mime_type": "text/plain",
            "category": "document",
            "created_at": "2026-01-01T00:00:00",
            "hash_status": "done",
            "identify_status": "done",
            "triage_status": "done",
            "is_dup_primary": 1,
            "excluded": 0,
        })
        file_id = _get_last_file_id(db)

        # Extraction + chunk + embedding_ref -> indexed
        db["extraction"].insert({
            "file_id": file_id,
            "tool": "test",
            "text_extracted": "content",
            "succeeded": 1,
        })
        db["chunk"].insert({
            "file_id": file_id,
            "chunk_index": 0,
            "text": "content",
            "token_count": 1,
        })
        chunk_id = _get_last_chunk_id(db)
        db["embedding_ref"].insert({
            "chunk_id": chunk_id,
            "model": "test",
            "is_current": 1,
        })

        # Pending file
        db["file"].insert({
            "folder_id": folder_id,
            "path": str(cfg.corpus_root_path / "pending.txt"),
            "rel_path": "pending.txt",
            "name": "pending.txt",
            "size_bytes": 200,
            "mtime": "2026-01-02T00:00:00",
            "created_at": "2026-01-02T00:00:00",
            "hash_status": "pending",
            "identify_status": "pending",
            "triage_status": "pending",
            "is_dup_primary": 1,
            "excluded": 0,
        })

        # Failed file
        db["file"].insert({
            "folder_id": folder_id,
            "path": str(cfg.corpus_root_path / "failed.txt"),
            "rel_path": "failed.txt",
            "name": "failed.txt",
            "size_bytes": 300,
            "mtime": "2026-01-03T00:00:00",
            "created_at": "2026-01-03T00:00:00",
            "hash_status": "done",
            "identify_status": "done",
            "triage_status": "done",
            "is_dup_primary": 1,
            "excluded": 0,
        })
        failed_id = _get_last_file_id(db)
        db["failure"].insert({
            "file_id": failed_id,
            "phase": "extraction",
            "tool": "test",
            "error_class": "Error",
            "error_message": "fail",
        })

        db.conn.commit()

        with patch("api.main._get_db", return_value=db):
            client = TestClient(app)
            response = client.get("/files")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["items"]) == 3
        assert data["page"] == 1
        assert data["size"] == 20

        statuses = {item["name"]: item["status"] for item in data["items"]}
        assert statuses["indexed.txt"] == "indexed"
        assert statuses["pending.txt"] == "pending"
        assert statuses["failed.txt"] == "failed"

        indexed_item = next(i for i in data["items"] if i["name"] == "indexed.txt")
        assert indexed_item["is_indexed"] is True

    def test_list_empty(self, tmp_path):
        db = _make_db(tmp_path)

        with patch("api.main._get_db", return_value=db):
            client = TestClient(app)
            response = client.get("/files")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["items"] == []

    def test_list_filter_by_status(self, tmp_path):
        db = _make_db(tmp_path)
        cfg = _make_cfg(tmp_path)
        folder_id = _seed_folder(db)

        # Pending file
        db["file"].insert({
            "folder_id": folder_id,
            "path": str(cfg.corpus_root_path / "pending.txt"),
            "rel_path": "pending.txt",
            "name": "pending.txt",
            "size_bytes": 100,
            "mtime": "2026-01-01T00:00:00",
            "hash_status": "pending",
            "identify_status": "pending",
            "triage_status": "pending",
            "is_dup_primary": 1,
            "excluded": 0,
        })

        # Indexed file
        db["file"].insert({
            "folder_id": folder_id,
            "path": str(cfg.corpus_root_path / "indexed.txt"),
            "rel_path": "indexed.txt",
            "name": "indexed.txt",
            "size_bytes": 200,
            "mtime": "2026-01-02T00:00:00",
            "hash_status": "done",
            "identify_status": "done",
            "triage_status": "done",
            "is_dup_primary": 1,
            "excluded": 0,
        })
        file_id = _get_last_file_id(db)
        db["extraction"].insert({
            "file_id": file_id,
            "tool": "test",
            "text_extracted": "content",
            "succeeded": 1,
        })
        db["chunk"].insert({
            "file_id": file_id,
            "chunk_index": 0,
            "text": "content",
            "token_count": 1,
        })
        chunk_id = _get_last_chunk_id(db)
        db["embedding_ref"].insert({
            "chunk_id": chunk_id,
            "model": "test",
            "is_current": 1,
        })

        db.conn.commit()

        with patch("api.main._get_db", return_value=db):
            client = TestClient(app)
            response = client.get("/files?status=pending")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["items"][0]["name"] == "pending.txt"


class TestFileDeletion:
    def test_delete_file(self, tmp_path):
        db = _make_db(tmp_path)
        cfg = _make_cfg(tmp_path)
        folder_id = _seed_folder(db)

        # Create file on disk
        test_file = cfg.corpus_root_path / "test.txt"
        test_file.write_bytes(b"test content")

        # Insert DB records
        db["file"].insert({
            "folder_id": folder_id,
            "path": str(test_file),
            "rel_path": "test.txt",
            "name": "test.txt",
            "size_bytes": 12,
            "mtime": "2026-01-01T00:00:00",
            "hash_status": "done",
            "identify_status": "done",
            "triage_status": "done",
            "is_dup_primary": 1,
            "excluded": 0,
        })
        file_id = _get_last_file_id(db)

        db["extraction"].insert({
            "file_id": file_id,
            "tool": "test",
            "text_extracted": "test content",
            "succeeded": 1,
        })

        db["chunk"].insert({
            "file_id": file_id,
            "chunk_index": 0,
            "text": "test content",
            "token_count": 2,
        })
        chunk_id = _get_last_chunk_id(db)

        db["embedding_ref"].insert({
            "chunk_id": chunk_id,
            "model": "test",
            "is_current": 1,
            "collection": "chunks__test",
            "external_id": "emb_1",
        })

        db.conn.commit()

        # Mock Chroma
        mock_collection = MagicMock()
        mock_chroma_client = MagicMock()
        mock_chroma_client.get_collection.return_value = mock_collection

        with patch("api.main._load_config", return_value=cfg), \
             patch("api.main._get_db", return_value=db), \
             patch("api.main.chromadb.PersistentClient", return_value=mock_chroma_client):
            client = TestClient(app)

            # Verify file exists before deletion
            assert test_file.exists()
            rows = list(db.query("SELECT id FROM file WHERE id = ?", [file_id]))
            assert len(rows) == 1

            response = client.delete(f"/files/{file_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"
        assert data["file_id"] == file_id

        # Verify file removed from disk
        assert not test_file.exists()

        # Verify DB records cleaned up
        assert len(list(db.query("SELECT id FROM file WHERE id = ?", [file_id]))) == 0
        assert len(list(db.query("SELECT id FROM extraction WHERE file_id = ?", [file_id]))) == 0
        assert len(list(db.query("SELECT id FROM chunk WHERE file_id = ?", [file_id]))) == 0
        assert len(list(db.query("SELECT id FROM embedding_ref WHERE chunk_id = ?", [chunk_id]))) == 0

        # Verify Chroma deletion called
        mock_chroma_client.get_collection.assert_called_with("chunks__test")
        mock_collection.delete.assert_called_with(ids=["emb_1"])

    def test_delete_nonexistent(self, tmp_path):
        db = _make_db(tmp_path)
        cfg = _make_cfg(tmp_path)

        with patch("api.main._load_config", return_value=cfg), \
             patch("api.main._get_db", return_value=db):
            client = TestClient(app)
            response = client.delete("/files/999")

        assert response.status_code == 404
