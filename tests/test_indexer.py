"""Tests for rag.indexer.IncrementalIndexer.

External dependencies (Ollama, Chroma, extractor binaries) are mocked so
the tests run fast and require no services.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.config import (
    AppConfig,
    ChunkConfig,
    EmbeddingConfig,
    ExtractConfig,
    GenerationConfig,
    GenerationRuntimeConfig,
    ModelsConfig,
    OllamaConfig,
    PathsConfig,
    RetrievalConfig,
    RerankerConfig,
    SummarizationConfig,
    ContextualRetrievalConfig,
)
from pipeline.db import migrate, open_db
from rag.indexer import IncrementalIndexer, _last_rowid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app_config(corpus: Path, cache: Path, chroma: Path) -> AppConfig:
    """Build a minimal AppConfig for indexer tests."""
    return AppConfig(
        paths=PathsConfig(
            corpus_root=str(corpus),
            cache_root=str(cache),
            chroma_root=str(chroma),
        ),
        ollama=OllamaConfig(),
        models=ModelsConfig(
            embedding=EmbeddingConfig(name="test-embed", collection_suffix="test"),
            summarization=SummarizationConfig(name="test-summary"),
            contextual_retrieval=ContextualRetrievalConfig(enabled=False, name="test-summary"),
            generation=GenerationConfig(
                name="test-gen", fallback_name="test-gen", alternate_name="test-gen"
            ),
            reranker=RerankerConfig(enabled=False),
        ),
        generation_runtime=GenerationRuntimeConfig(),
        chunk=ChunkConfig(window_size=2),
        retrieval=RetrievalConfig(),
        extract=ExtractConfig(size_threshold_for_summary=100),
    )


def _make_db(tmp_path: Path):
    """Create a migrated test database."""
    db_path = tmp_path / "test_indexer.db"
    db = open_db(db_path)
    migrate(db)
    return db


def _make_chroma_mock():
    """Return (mock_client, mock_collection)."""
    coll = MagicMock()
    coll.name = "chunks__test"
    coll.metadata = {"embedding_model": "test-embed", "dim": 64, "config_hash": "x"}

    client = MagicMock()
    client.get_collection.return_value = coll
    client.list_collections.return_value = [coll]
    client.create_collection.return_value = coll

    return client, coll


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env(tmp_path: Path):
    """Set up corpus, cache, chroma dirs + DB + config."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    cache = corpus / ".rag-cache"
    cache.mkdir()
    chroma = tmp_path / "chroma"
    chroma.mkdir()

    db = _make_db(tmp_path)
    cfg = _make_app_config(corpus, cache, chroma)
    return corpus, cfg, db


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIncrementalIndexNewFile:
    """sync_file on a brand-new file creates the full set of records."""

    def test_incremental_index_new_file(self, env):
        corpus, cfg, db = env

        # Create a text file with enough content for chunking
        txt = corpus / "hello.txt"
        content = "This is a test sentence. " * 30  # ~750 chars
        txt.write_text(content, encoding="utf-8")

        chroma_client, chroma_coll = _make_chroma_mock()

        with (
            patch("rag.indexer.IncrementalIndexer._run_extraction") as mock_ext,
            patch("rag.indexer.IncrementalIndexer._run_summarization") as mock_sum,
            patch("rag.indexer.IncrementalIndexer._run_chunking") as mock_chunk,
            patch("rag.indexer.IncrementalIndexer._run_embedding") as mock_emb,
        ):
            # Make extraction insert a real extraction row so chunking has data
            def fake_extraction(file_id, file_path, strategy):
                db["extraction"].insert({
                    "file_id": file_id,
                    "tool": "textutil",
                    "text_extracted": content,
                    "char_count": len(content),
                    "succeeded": 1,
                })
                db.conn.commit()

            mock_ext.side_effect = fake_extraction

            indexer = IncrementalIndexer(db, cfg, chroma_client=chroma_client)
            result = indexer.sync_file(txt)

        assert result["status"] == "success"
        assert result["action"] == "created"
        assert result["file_id"] is not None

        # Verify file record
        file_row = next(iter(db.query("SELECT * FROM file WHERE id = ?", [result["file_id"]])))
        assert file_row["sha256"] != ""
        assert file_row["hash_status"] == "done"
        assert file_row["extract_strategy"] == "textutil"

        # Verify extraction record (inserted by our fake)
        ext_rows = list(db.query("SELECT * FROM extraction WHERE file_id = ?", [result["file_id"]]))
        assert len(ext_rows) == 1
        assert ext_rows[0]["succeeded"] == 1

        # Verify pipeline stages were called
        mock_ext.assert_called_once()
        mock_sum.assert_called_once()
        mock_chunk.assert_called_once()
        mock_emb.assert_called_once()


class TestIncrementalSkipUnchanged:
    """sync_file called twice on the same content should skip the second time."""

    def test_incremental_skip_unchanged(self, env):
        corpus, cfg, db = env

        txt = corpus / "stable.txt"
        txt.write_text("Content that does not change. " * 20, encoding="utf-8")

        chroma_client, _ = _make_chroma_mock()

        with patch("rag.indexer.IncrementalIndexer._run_extraction"), \
             patch("rag.indexer.IncrementalIndexer._run_summarization"), \
             patch("rag.indexer.IncrementalIndexer._run_chunking"), \
             patch("rag.indexer.IncrementalIndexer._run_embedding"):

            indexer = IncrementalIndexer(db, cfg, chroma_client=chroma_client)

            # First call — processes the file
            r1 = indexer.sync_file(txt)
            assert r1["status"] == "success"
            assert r1["action"] == "created"
            file_id_1 = r1["file_id"]

            # Second call — should skip (same hash)
            r2 = indexer.sync_file(txt)
            assert r2["status"] == "skipped_unchanged"
            assert r2["file_id"] == file_id_1

        # Exactly 1 file record
        count = next(iter(db.query("SELECT COUNT(*) as c FROM file WHERE path = ?", [str(txt)])))["c"]
        assert count == 1


class TestIncrementalUpdateModified:
    """Modifying a file and re-syncing replaces old derived data."""

    def test_incremental_update_modified(self, env):
        corpus, cfg, db = env

        txt = corpus / "mutable.txt"
        txt.write_text("Original content. " * 30, encoding="utf-8")

        chroma_client, chroma_coll = _make_chroma_mock()

        with patch("rag.indexer.IncrementalIndexer._run_extraction") as mock_ext, \
             patch("rag.indexer.IncrementalIndexer._run_summarization"), \
             patch("rag.indexer.IncrementalIndexer._run_chunking"), \
             patch("rag.indexer.IncrementalIndexer._run_embedding"):

            def fake_extraction(file_id, file_path, strategy):
                text = Path(file_path).read_text(encoding="utf-8")
                db["extraction"].insert({
                    "file_id": file_id,
                    "tool": "textutil",
                    "text_extracted": text,
                    "char_count": len(text),
                    "succeeded": 1,
                })
                db.conn.commit()

            mock_ext.side_effect = fake_extraction

            indexer = IncrementalIndexer(db, cfg, chroma_client=chroma_client)

            # First sync
            r1 = indexer.sync_file(txt)
            assert r1["status"] == "success"
            file_id = r1["file_id"]

            # Modify the file
            txt.write_text("Completely new content that is different. " * 30, encoding="utf-8")

            # Second sync — should detect change and reprocess
            r2 = indexer.sync_file(txt)
            assert r2["status"] == "success"
            assert r2["action"] == "updated"
            assert r2["file_id"] == file_id  # same file_id, updated in place

        # Verify hash was updated
        file_row = next(iter(db.query("SELECT sha256 FROM file WHERE id = ?", [file_id])))
        import hashlib
        expected_hash = hashlib.sha256(txt.read_bytes()).hexdigest()
        assert file_row["sha256"] == expected_hash

        # Old extraction should have been cleaned up; only 1 extraction row
        ext_count = next(iter(
            db.query("SELECT COUNT(*) as c FROM extraction WHERE file_id = ?", [file_id])
        ))["c"]
        assert ext_count == 1


class TestIncrementalRemoveFile:
    """remove_file deletes all derived data."""

    def test_incremental_remove_file(self, env):
        corpus, cfg, db = env

        txt = corpus / "doomed.txt"
        txt.write_text("This file will be removed. " * 20, encoding="utf-8")

        chroma_client, chroma_coll = _make_chroma_mock()

        with patch("rag.indexer.IncrementalIndexer._run_extraction") as mock_ext, \
             patch("rag.indexer.IncrementalIndexer._run_summarization"), \
             patch("rag.indexer.IncrementalIndexer._run_chunking"), \
             patch("rag.indexer.IncrementalIndexer._run_embedding"):

            def fake_extraction(file_id, file_path, strategy):
                db["extraction"].insert({
                    "file_id": file_id,
                    "tool": "textutil",
                    "text_extracted": "Some text. " * 50,
                    "char_count": 550,
                    "succeeded": 1,
                })
                db.conn.commit()

            mock_ext.side_effect = fake_extraction

            indexer = IncrementalIndexer(db, cfg, chroma_client=chroma_client)
            r = indexer.sync_file(txt)
            file_id = r["file_id"]

            # Seed some chunks to verify deletion
            for i in range(3):
                db["chunk"].insert({
                    "file_id": file_id,
                    "chunk_index": i,
                    "text": f"chunk {i}",
                    "token_count": 10,
                })
                chunk_id = _last_rowid(db)
                db.execute("INSERT INTO chunk_fts(rowid, text) VALUES (?, ?)", [chunk_id, f"chunk {i}"])
                db["embedding_ref"].insert({
                    "chunk_id": chunk_id,
                    "vector_store": "chroma",
                    "collection": "chunks__test",
                    "external_id": f"c_{chunk_id}",
                    "embedding_model": "test-embed",
                    "dim": 64,
                    "config_hash": "x",
                    "is_current": 1,
                })
            db.conn.commit()

        # Now remove
        result = indexer.remove_file(file_id)
        assert result["status"] == "removed"

        # Verify all records deleted
        assert next(iter(db.query("SELECT COUNT(*) as c FROM file WHERE id = ?", [file_id])))["c"] == 0
        assert next(iter(db.query("SELECT COUNT(*) as c FROM extraction WHERE file_id = ?", [file_id])))["c"] == 0
        assert next(iter(db.query("SELECT COUNT(*) as c FROM chunk WHERE file_id = ?", [file_id])))["c"] == 0
        assert next(iter(db.query("SELECT COUNT(*) as c FROM embedding_ref WHERE chunk_id IN (SELECT id FROM chunk WHERE file_id = ?)", [file_id])))["c"] == 0

        # Verify Chroma delete was called
        chroma_coll.delete.assert_called()


class TestIdempotency:
    """Calling sync_file twice on the same file produces exactly 1 set of records."""

    def test_idempotency(self, env):
        corpus, cfg, db = env

        txt = corpus / "idempotent.txt"
        txt.write_text("Idempotent content for testing. " * 20, encoding="utf-8")

        chroma_client, _ = _make_chroma_mock()

        with patch("rag.indexer.IncrementalIndexer._run_extraction") as mock_ext, \
             patch("rag.indexer.IncrementalIndexer._run_summarization"), \
             patch("rag.indexer.IncrementalIndexer._run_chunking"), \
             patch("rag.indexer.IncrementalIndexer._run_embedding"):

            call_count = 0

            def fake_extraction(file_id, file_path, strategy):
                nonlocal call_count
                call_count += 1
                db["extraction"].insert({
                    "file_id": file_id,
                    "tool": "textutil",
                    "text_extracted": "Some text. " * 50,
                    "char_count": 550,
                    "succeeded": 1,
                })
                db.conn.commit()

            mock_ext.side_effect = fake_extraction

            indexer = IncrementalIndexer(db, cfg, chroma_client=chroma_client)

            # First call
            r1 = indexer.sync_file(txt)
            assert r1["status"] == "success"

            # Second call — should skip entirely
            r2 = indexer.sync_file(txt)
            assert r2["status"] == "skipped_unchanged"

        # Extraction was only called once
        assert call_count == 1

        # Exactly 1 file record
        file_count = next(iter(
            db.query("SELECT COUNT(*) as c FROM file WHERE path = ?", [str(txt)])
        ))["c"]
        assert file_count == 1

        # Exactly 1 extraction record
        ext_count = next(iter(
            db.query("SELECT COUNT(*) as c FROM extraction WHERE file_id = ?", [r1["file_id"]])
        ))["c"]
        assert ext_count == 1


class TestSyncAll:
    """sync_all detects new, modified, and deleted files."""

    def test_sync_all_detects_changes(self, env):
        corpus, cfg, db = env

        # Create initial files
        (corpus / "a.txt").write_text("File A content. " * 20, encoding="utf-8")
        (corpus / "b.txt").write_text("File B content. " * 20, encoding="utf-8")

        chroma_client, _ = _make_chroma_mock()

        with patch("rag.indexer.IncrementalIndexer._run_extraction") as mock_ext, \
             patch("rag.indexer.IncrementalIndexer._run_summarization"), \
             patch("rag.indexer.IncrementalIndexer._run_chunking"), \
             patch("rag.indexer.IncrementalIndexer._run_embedding"):

            def fake_extraction(file_id, file_path, strategy):
                db["extraction"].insert({
                    "file_id": file_id,
                    "tool": "textutil",
                    "text_extracted": "text " * 100,
                    "char_count": 500,
                    "succeeded": 1,
                })
                db.conn.commit()

            mock_ext.side_effect = fake_extraction

            indexer = IncrementalIndexer(db, cfg, chroma_client=chroma_client)

            # First sync — 2 new files
            result = indexer.sync_all()
            assert result["new"] == 2
            assert result["processed"] == 2
            assert result["deleted"] == 0

            # Modify one file, delete another, add a new one
            (corpus / "a.txt").write_text("Modified A. " * 30, encoding="utf-8")
            (corpus / "b.txt").unlink()
            (corpus / "c.txt").write_text("New file C. " * 20, encoding="utf-8")

            result2 = indexer.sync_all()
            assert result2["modified"] == 1
            assert result2["deleted"] == 1
            assert result2["new"] == 1
