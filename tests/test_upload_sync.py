"""Regression: files uploaded via POST /files/upload must be indexed by sync.

Upload inserts a file row with the correct mtime/size but an empty sha256
(hash_status='pending'). sync_all's mtime+size fast-path must NOT treat an
empty hash as "unchanged" — otherwise the file is never hashed, never routed
to sync_file, and never becomes searchable.
"""

import io
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import sqlite_utils
from fastapi.testclient import TestClient

from pipeline.db import migrate
from rag.indexer import IncrementalIndexer


def _make_config(tmp_path):
    from pipeline.config import (
        AppConfig,
        ContextualRetrievalConfig,
        EmbeddingConfig,
        GenerationConfig,
        ModelsConfig,
        PathsConfig,
        SummarizationConfig,
    )

    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    cache = corpus / ".rag-cache"
    cache.mkdir(exist_ok=True)
    return AppConfig(
        paths=PathsConfig(corpus_root=str(corpus), cache_root=str(cache)),
        models=ModelsConfig(
            embedding=EmbeddingConfig(name="test-embed", collection_suffix="_test"),
            summarization=SummarizationConfig(name="test-summarize"),
            contextual_retrieval=ContextualRetrievalConfig(name="test-context"),
            generation=GenerationConfig(
                name="test-gen", fallback_name="test-fb", alternate_name="test-alt"
            ),
        ),
    )


def _insert_uploaded_row(db, corpus, fp):
    stat = fp.stat()
    mtime_iso = datetime.fromtimestamp(stat.st_mtime, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    db["folder"].insert(
        {"path": str(corpus.resolve()), "rel_path": ".", "name": ".", "depth": 0}
    )
    folder_id = next(iter(db.query("SELECT id FROM folder WHERE rel_path='.'")))["id"]
    fid = db["file"].insert(
        {
            "folder_id": folder_id,
            "path": str(fp.resolve()),
            "rel_path": fp.name,
            "name": fp.name,
            "extension": fp.suffix.lstrip("."),
            "size_bytes": stat.st_size,
            "mtime": mtime_iso,
            "sha256": "",
            "mime_type": "text/plain",
            "hash_status": "pending",
            "identify_status": "pending",
            "triage_status": "pending",
            "is_dup_primary": 1,
            "excluded": 0,
            "workspace": "default",
        }
    ).last_rowid
    db.conn.commit()
    return fid


class TestUploadedFileIndexed:
    def test_sync_routes_uploaded_file_to_pipeline(self, tmp_path, test_db):
        cfg = _make_config(tmp_path)
        corpus = Path(cfg.corpus_root_path)
        fp = corpus / "uploaded.txt"
        fp.write_text("Gamma project uses the Falcon X2 platform with eight units.")
        fid = _insert_uploaded_row(test_db, corpus, fp)

        indexer = IncrementalIndexer(test_db, cfg, chroma_client=MagicMock())
        calls = []

        def fake_sync_file(path, force=False):
            calls.append(str(Path(path).resolve()))
            return {"status": "success", "file_id": fid}

        indexer.sync_file = fake_sync_file
        summary = indexer.sync_all()

        assert str(fp.resolve()) in calls, "uploaded file was skipped by sync"
        assert summary["processed"] >= 1

    def test_unchanged_hashed_file_still_skipped(self, tmp_path, test_db):
        """The fast-path must still skip genuinely unchanged, hashed files."""
        import hashlib

        cfg = _make_config(tmp_path)
        corpus = Path(cfg.corpus_root_path)
        fp = corpus / "stable.txt"
        content = b"Stable content that has already been indexed."
        fp.write_bytes(content)
        fid = _insert_uploaded_row(test_db, corpus, fp)
        real_hash = hashlib.sha256(content).hexdigest()
        test_db.execute("UPDATE file SET sha256=?, hash_status='done' WHERE id=?", [real_hash, fid])
        test_db.conn.commit()

        indexer = IncrementalIndexer(test_db, cfg, chroma_client=MagicMock())
        calls = []
        indexer.sync_file = lambda path, force=False: calls.append(str(path)) or {
            "status": "success",
            "file_id": fid,
        }
        summary = indexer.sync_all()

        assert str(fp.resolve()) not in calls
        assert summary["processed"] == 0


class TestUploadHandlerRowId:
    def test_upload_sets_created_at_and_audit_resource(self, tmp_path):
        from api.main import app

        cfg = _make_config(tmp_path)
        conn = sqlite3.connect(str(tmp_path / "api.db"), check_same_thread=False)
        db = sqlite_utils.Database(conn)
        migrate(db)

        with (
            patch("api.main._load_config", return_value=cfg),
            patch("api.main._get_db", return_value=db),
        ):
            client = TestClient(app)
            resp = client.post(
                "/files/upload",
                files=[("files", ("note.txt", io.BytesIO(b"hello upload"), "text/plain"))],
            )
        assert resp.status_code == 200
        assert resp.json()["uploaded"] == 1

        row = next(iter(db.query("SELECT id, created_at FROM file WHERE name='note.txt'")))
        assert row["created_at"] is not None, "created_at not set (last_rowid was None)"
        audit = next(
            iter(db.query("SELECT resource_id FROM audit_log WHERE action='file.upload'"))
        )
        assert int(audit["resource_id"]) == row["id"]
