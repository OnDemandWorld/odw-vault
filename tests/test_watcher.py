"""Tests for rag.watcher.CorpusWatcher.

External dependencies (watchdog Observer, Ollama, Chroma) are mocked so
the tests run fast and require no services or filesystem watching.
"""

from __future__ import annotations

import time
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
    WatcherConfig,
)
from pipeline.db import migrate, open_db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app_config(corpus: Path, cache: Path, chroma: Path, **watcher_kwargs) -> AppConfig:
    """Build a minimal AppConfig for watcher tests."""
    watcher_cfg = WatcherConfig(**watcher_kwargs)
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
        watcher=watcher_cfg,
    )


def _make_db(tmp_path: Path):
    """Create a migrated test database."""
    db_path = tmp_path / "test_watcher.db"
    db = open_db(db_path)
    migrate(db)
    return db


def _make_chroma_mock():
    """Return a mock Chroma client."""
    coll = MagicMock()
    coll.name = "chunks__test"
    coll.metadata = {"embedding_model": "test-embed", "dim": 64, "config_hash": "x"}
    client = MagicMock()
    client.get_collection.return_value = coll
    client.create_collection.return_value = coll
    return client


def _make_event(src_path: str, is_directory: bool = False, dest_path: str | None = None):
    """Create a mock watchdog event."""
    event = MagicMock()
    event.src_path = src_path
    event.is_directory = is_directory
    if dest_path:
        event.dest_path = dest_path
    return event


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
    cfg = _make_app_config(corpus, cache, chroma, enabled=True, debounce_seconds=0.1)
    chroma_client = _make_chroma_mock()
    return {"corpus": corpus, "db": db, "cfg": cfg, "chroma": chroma_client}


# ---------------------------------------------------------------------------
# Tests: event filtering
# ---------------------------------------------------------------------------


class TestEventFiltering:
    """Test that the watcher correctly filters events."""

    def test_ignore_ds_store(self, env):
        from rag.watcher import CorpusWatcher

        watcher = CorpusWatcher(env["cfg"], env["db"], env["chroma"])
        assert watcher._should_skip("/corpus/.DS_Store") is True

    def test_ignore_rag_cache(self, env):
        from rag.watcher import CorpusWatcher

        watcher = CorpusWatcher(env["cfg"], env["db"], env["chroma"])
        assert watcher._should_skip("/corpus/.rag-cache/models/lid.bin") is True

    def test_ignore_thumbs_db(self, env):
        from rag.watcher import CorpusWatcher

        watcher = CorpusWatcher(env["cfg"], env["db"], env["chroma"])
        assert watcher._should_skip("/corpus/Thumbs.db") is True

    def test_ignore_tmp_files(self, env):
        from rag.watcher import CorpusWatcher

        watcher = CorpusWatcher(env["cfg"], env["db"], env["chroma"])
        assert watcher._should_skip("/corpus/document.tmp") is True

    def test_ignore_macosx_dir(self, env):
        from rag.watcher import CorpusWatcher

        watcher = CorpusWatcher(env["cfg"], env["db"], env["chroma"])
        assert watcher._should_skip("/corpus/__MACOSX/file.txt") is True

    def test_allow_normal_file(self, env):
        from rag.watcher import CorpusWatcher

        watcher = CorpusWatcher(env["cfg"], env["db"], env["chroma"])
        assert watcher._should_skip("/corpus/project/report.pdf") is False

    def test_allow_nested_file(self, env):
        from rag.watcher import CorpusWatcher

        watcher = CorpusWatcher(env["cfg"], env["db"], env["chroma"])
        assert watcher._should_skip("/corpus/Project/Gleneagles/spec.docx") is False


# ---------------------------------------------------------------------------
# Tests: debounce coalescing
# ---------------------------------------------------------------------------


class TestDebounce:
    """Test that rapid events are coalesced."""

    @patch("rag.watcher.IncrementalIndexer")
    def test_coalesces_rapid_events(self, MockIndexer, env):
        from rag.watcher import CorpusWatcher

        watcher = CorpusWatcher(env["cfg"], env["db"], env["chroma"])
        watcher._indexer = MockIndexer.return_value

        # Simulate 5 rapid modify events for the same path
        path = str(env["corpus"] / "test.txt")
        for _ in range(5):
            watcher._enqueue(path, "sync")

        # Only one entry should be pending
        assert len(watcher._pending) == 1

        # Flush with force=True
        watcher._flush_ready(force=True)

        # sync_file should be called exactly once
        watcher._indexer.sync_file.assert_called_once_with(Path(path))

    @patch("rag.watcher.IncrementalIndexer")
    def test_different_paths_not_coalesced(self, MockIndexer, env):
        from rag.watcher import CorpusWatcher

        watcher = CorpusWatcher(env["cfg"], env["db"], env["chroma"])
        watcher._indexer = MockIndexer.return_value

        path1 = str(env["corpus"] / "file1.txt")
        path2 = str(env["corpus"] / "file2.txt")
        watcher._enqueue(path1, "sync")
        watcher._enqueue(path2, "sync")

        assert len(watcher._pending) == 2

        watcher._flush_ready(force=True)
        assert watcher._indexer.sync_file.call_count == 2


# ---------------------------------------------------------------------------
# Tests: sync and delete triggers
# ---------------------------------------------------------------------------


class TestSyncDelete:
    """Test that events trigger correct indexer calls."""

    @patch("rag.watcher.IncrementalIndexer")
    def test_created_file_triggers_sync(self, MockIndexer, env):
        from rag.watcher import CorpusWatcher, _SyncHandler

        watcher = CorpusWatcher(env["cfg"], env["db"], env["chroma"])
        watcher._indexer = MockIndexer.return_value
        handler = _SyncHandler(watcher)

        path = str(env["corpus"] / "new_file.pdf")
        event = _make_event(path)
        handler.on_created(event)

        watcher._flush_ready(force=True)
        watcher._indexer.sync_file.assert_called_once_with(Path(path))

    @patch("rag.watcher.IncrementalIndexer")
    def test_deleted_file_triggers_remove(self, MockIndexer, env):
        from rag.watcher import CorpusWatcher, _SyncHandler

        # Seed a folder + file row in DB
        env["db"]["folder"].insert({
            "path": str(env["corpus"]),
            "rel_path": ".",
            "name": "corpus",
            "depth": 0,
            "excluded": 0,
        })
        env["db"].conn.commit()
        folder_id = env["db"].conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        env["db"]["file"].insert({
            "path": str(env["corpus"] / "old.txt"),
            "rel_path": "old.txt",
            "name": "old.txt",
            "folder_id": folder_id,
            "sha256": "abc123",
            "size_bytes": 100,
            "mtime": "2025-01-01T00:00:00Z",
            "excluded": 0,
        })
        env["db"].conn.commit()
        file_id = env["db"].conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        watcher = CorpusWatcher(env["cfg"], env["db"], env["chroma"])
        watcher._indexer = MockIndexer.return_value
        handler = _SyncHandler(watcher)

        path = str(env["corpus"] / "old.txt")
        event = _make_event(path)
        handler.on_deleted(event)

        watcher._flush_ready(force=True)
        watcher._indexer.remove_file.assert_called_once_with(file_id)

    @patch("rag.watcher.IncrementalIndexer")
    def test_moved_file_triggers_delete_and_sync(self, MockIndexer, env):
        from rag.watcher import CorpusWatcher, _SyncHandler

        watcher = CorpusWatcher(env["cfg"], env["db"], env["chroma"])
        watcher._indexer = MockIndexer.return_value
        handler = _SyncHandler(watcher)

        src = str(env["corpus"] / "old_name.txt")
        dest = str(env["corpus"] / "new_name.txt")
        event = _make_event(src, dest_path=dest)
        handler.on_moved(event)

        # Both paths should be pending
        assert src in watcher._pending
        assert dest in watcher._pending
        assert watcher._pending[src][0] == "delete"
        assert watcher._pending[dest][0] == "sync"


# ---------------------------------------------------------------------------
# Tests: directory events ignored
# ---------------------------------------------------------------------------


class TestDirectoryEvents:
    """Test that directory events are ignored."""

    @patch("rag.watcher.IncrementalIndexer")
    def test_directory_created_ignored(self, MockIndexer, env):
        from rag.watcher import CorpusWatcher, _SyncHandler

        watcher = CorpusWatcher(env["cfg"], env["db"], env["chroma"])
        watcher._indexer = MockIndexer.return_value
        handler = _SyncHandler(watcher)

        event = _make_event(str(env["corpus"] / "new_dir"), is_directory=True)
        handler.on_created(event)

        assert len(watcher._pending) == 0


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Test that errors in sync don't crash the watcher."""

    @patch("rag.watcher.IncrementalIndexer")
    def test_error_in_sync_does_not_crash(self, MockIndexer, env):
        from rag.watcher import CorpusWatcher

        watcher = CorpusWatcher(env["cfg"], env["db"], env["chroma"])
        watcher._indexer = MockIndexer.return_value
        watcher._indexer.sync_file.side_effect = RuntimeError("Ollama down")

        path = str(env["corpus"] / "bad.txt")
        watcher._enqueue(path, "sync")

        # Should not raise
        watcher._flush_ready(force=True)

        assert watcher._stats["failed"] == 1
        assert watcher._stats["processed"] == 0


# ---------------------------------------------------------------------------
# Tests: status reporting
# ---------------------------------------------------------------------------


class TestStatus:
    """Test watcher status reporting."""

    def test_status_structure(self, env):
        from rag.watcher import CorpusWatcher

        watcher = CorpusWatcher(env["cfg"], env["db"], env["chroma"])
        status = watcher.status()

        assert "watching" in status
        assert "pending" in status
        assert "processed" in status
        assert "failed" in status
        assert "last_event_at" in status
        assert status["watching"] is False
        assert status["pending"] == 0

    def test_status_pending_count(self, env):
        from rag.watcher import CorpusWatcher

        watcher = CorpusWatcher(env["cfg"], env["db"], env["chroma"])
        watcher._enqueue("/corpus/a.txt", "sync")
        watcher._enqueue("/corpus/b.txt", "sync")

        status = watcher.status()
        assert status["pending"] == 2


# ---------------------------------------------------------------------------
# Tests: watcher disabled
# ---------------------------------------------------------------------------


class TestWatcherDisabled:
    """Test behavior when watcher is disabled."""

    def test_disabled_config(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        cache = corpus / ".rag-cache"
        cache.mkdir()
        chroma = tmp_path / "chroma"
        chroma.mkdir()

        cfg = _make_app_config(corpus, cache, chroma, enabled=False)
        assert cfg.watcher.enabled is False
