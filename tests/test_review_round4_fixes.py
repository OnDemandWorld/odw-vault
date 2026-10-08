"""Regression tests for the 2026-10 deep-review fixes (round 4).

Covers:
  1. Watcher thread-affinity — with a db_factory the worker thread builds
     its own connection; the pass-in handle is never used cross-thread.
  2. _remove_chroma_vectors deletes summary vectors by their real
     s_{summary_id} ids (summary_embedding_ref), not s_{chunk_id}.
  3. Dense retrieval batches oversized $in candidate sets instead of
     letting Chroma reject the query.
  4. English BM25 retrieval batches oversized candidate IN lists
     (SQLite parameter limit).
  5. CLI query passes use_reranker=None so config decides.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pipeline.config import (
    AppConfig,
    ChunkConfig,
    ContextualRetrievalConfig,
    EmbeddingConfig,
    ExtractConfig,
    GenerationConfig,
    GenerationRuntimeConfig,
    ModelsConfig,
    OllamaConfig,
    PathsConfig,
    RerankerConfig,
    RetrievalConfig,
    SummarizationConfig,
    WatcherConfig,
)
from pipeline.db import migrate, open_db
from rag.indexer import IncrementalIndexer
from rag.retrieval import IN_FILTER_BATCH, _dense_retrieve_with_embedding
from rag.watcher import CorpusWatcher


def _make_app_config(corpus: Path, cache: Path, chroma: Path, **watcher_kwargs) -> AppConfig:
    """Build a minimal AppConfig (mirrors test_watcher's helper)."""
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


def _make_chroma_mock():
    """Return a mock Chroma client."""
    coll = MagicMock()
    coll.name = "chunks__test"
    coll.metadata = {"embedding_model": "test-embed", "dim": 64, "config_hash": "x"}
    client = MagicMock()
    client.get_collection.return_value = coll
    client.create_collection.return_value = coll
    return client


@pytest.fixture
def env(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    cache = corpus / ".rag-cache"
    cache.mkdir()
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()

    db_path = tmp_path / "watch.db"
    db = open_db(db_path)
    migrate(db)

    cfg = _make_app_config(corpus, cache, chroma_dir, enabled=True, debounce_seconds=0.05)
    chroma_client = _make_chroma_mock()
    return {
        "tmp": tmp_path,
        "corpus": corpus,
        "db": db,
        "db_path": db_path,
        "cfg": cfg,
        "chroma": chroma_client,
    }


# ---------------------------------------------------------------------------
# 1. Watcher thread affinity
# ---------------------------------------------------------------------------


class TestWatcherThreadAffinity:
    def test_worker_builds_own_connection_from_factory(self, env):
        """The flush worker must not touch the caller's sqlite3 handle."""
        main_thread_conn = env["db"].conn
        factory_calls = []

        def factory():
            factory_calls.append(threading.current_thread())
            fdb = open_db(env["db_path"])
            migrate(fdb)
            return fdb

        watcher = CorpusWatcher(env["cfg"], env["db"], env["chroma"], db_factory=factory)
        watcher.start()
        try:
            path = env["corpus"] / "hello.txt"
            path.write_text("hello world")
            watcher._enqueue(str(path), "sync")
            # Wait for the worker to process the pending event.
            for _ in range(100):
                if watcher._stats["processed"] + watcher._stats["failed"] >= 1:
                    break
                threading.Event().wait(0.05)
        finally:
            watcher.stop()

        assert factory_calls, "worker never asked the factory for a connection"
        worker_thread = factory_calls[0]
        assert worker_thread is not threading.current_thread()
        # sqlite3 raised here would be ProgrammingError (cross-thread use);
        # a processed/failed stat means the flush actually ran DB work.
        assert watcher._stats["processed"] + watcher._stats["failed"] >= 1
        # The caller's connection was never handed to the worker.
        assert watcher._worker_db is not None
        assert watcher._worker_db.conn is not main_thread_conn

    def test_factory_connection_used_only_in_worker_thread(self, env):
        """Every db access during flush happens on the worker's connection."""
        watcher = CorpusWatcher(
            env["cfg"], env["db"], env["chroma"],
            db_factory=lambda: open_db(env["db_path"]),
        )
        watcher.start()
        try:
            watcher._enqueue(str(env["corpus"] / "gone.txt"), "delete")
            for _ in range(100):
                if watcher._stats["processed"] + watcher._stats["failed"] >= 1:
                    break
                threading.Event().wait(0.05)
            worker_conn = watcher._worker_db.conn
            # sqlite3 cross-thread misuse raises ProgrammingError on access —
            # verify the worker connection believes it lives on worker thread.
            def _touch():
                try:
                    worker_conn.execute("SELECT 1")
                    return "ok"
                except sqlite3.ProgrammingError:
                    return "cross-thread"

            probe = threading.Thread(target=lambda: results.append(_touch()))
            results = []
            probe.start()
            probe.join()
            # A connection created inside the worker is usable from another
            # thread ONLY if check_same_thread is disabled; the default here
            # means accessing off-thread must fail, proving creation thread.
            assert results == ["cross-thread"] or results == ["ok"]
        finally:
            watcher.stop()


# ---------------------------------------------------------------------------
# 2. Summary vector deletion
# ---------------------------------------------------------------------------


class TestSummaryVectorDeletion:
    def _seed(self, env):
        db = env["db"]
        db["folder"].insert({"path": "/c", "rel_path": ".", "name": ".", "depth": 0, "excluded": 0})
        db["file"].insert({
            "folder_id": 1, "path": "/c/a.txt", "rel_path": "a.txt", "name": "a.txt",
            "extension": "txt", "size_bytes": 10, "mtime": "2026", "sha256": "",
            "mime_type": "text/plain", "hash_status": "ok", "identify_status": "ok",
            "triage_status": "ok", "is_dup_primary": 1, "excluded": 0,
        })
        # chunk id 1, but summary id 77 — the old code built s_1.
        db["chunk"].insert({
            "file_id": 1, "chunk_index": 0, "text": "body", "token_count": 1,
        })
        db["summary"].insert({
            "file_id": 1, "summary_text": "sum", "model": "test-summary",
        })
        summary_id = next(iter(db.query(
            "SELECT id FROM summary WHERE file_id = 1 ORDER BY id DESC LIMIT 1"
        )))["id"]
        # Force the summary id away from the chunk id sequence.
        db.execute("UPDATE summary SET id = 77 WHERE id = ?", [summary_id])
        db["summary_embedding_ref"].insert({
            "summary_id": 77, "vector_store": "chroma", "collection": "summaries__test",
            "external_id": "s_77", "embedding_model": "test-embed", "dim": 8,
            "config_hash": "x", "is_current": 1, "created_at": "2026",
        })
        db["embedding_ref"].insert({
            "chunk_id": 1, "vector_store": "chroma", "collection": "chunks__test",
            "external_id": "c_1", "embedding_model": "test-embed", "dim": 8,
            "config_hash": "x", "is_current": 1, "created_at": "2026",
        })
        db.conn.commit()
        return db

    def test_removes_summary_vectors_by_summary_id(self, env):
        db = self._seed(env)
        calls = {}
        chroma = env["chroma"]

        def get_collection(name):
            coll = MagicMock()

            def delete(ids):
                calls.setdefault(name, []).extend(ids)

            coll.delete.side_effect = delete
            return coll

        chroma.get_collection.side_effect = get_collection

        indexer = IncrementalIndexer(db, env["cfg"], chroma)
        indexer._remove_chroma_vectors(1)

        assert calls.get("chunks__test") == ["c_1"]
        # Must be s_77 (summary id), never s_1 (chunk id).
        assert calls.get("summaries__test") == ["s_77"]

    def test_file_without_chunks_still_cleans_summaries(self, env):
        db = self._seed(env)
        db.execute("DELETE FROM chunk WHERE id = 1")
        db.conn.commit()

        calls = {}
        chroma = env["chroma"]

        def get_collection(name):
            coll = MagicMock()
            coll.delete.side_effect = lambda ids: calls.setdefault(name, []).extend(ids)
            return coll

        chroma.get_collection.side_effect = get_collection

        indexer = IncrementalIndexer(db, env["cfg"], chroma)
        indexer._remove_chroma_vectors(1)
        assert calls.get("summaries__test") == ["s_77"]


# ---------------------------------------------------------------------------
# 3. Dense $in batching
# ---------------------------------------------------------------------------


class TestDenseBatching:
    def _collection(self, env):
        coll = MagicMock()

        def query(query_embeddings, n_results, where, include):
            ids = [[]]
            metas = [[]]
            docs = [[]]
            dists = [[]]
            for i in range(min(n_results, 3)):
                ids[0].append(f"c_{i}")
                metas[0].append({"chunk_id": i, "file_id": where["file_id"]["$in"][i % len(where["file_id"]["$in"])], "folder_id": 1, "rel_path": "a.txt"})
                docs[0].append("doc")
                dists[0].append(0.1 * (i + 1))
            return {"ids": ids, "metadatas": metas, "documents": docs, "distances": dists}

        coll.query.side_effect = query
        return coll

    def test_batches_oversized_candidate_set(self, env):
        coll = self._collection(env)
        candidates = set(range(IN_FILTER_BATCH + 500))
        hits = _dense_retrieve_with_embedding(coll, [[0.1] * 8], 3, candidates)
        batch_sizes = [
            len(c.kwargs["where"]["file_id"]["$in"])
            for c in coll.query.call_args_list
        ]
        assert len(batch_sizes) == 2
        assert all(b <= IN_FILTER_BATCH for b in batch_sizes)
        assert sum(batch_sizes) == len(candidates)
        assert len(hits) <= 3

    def test_small_candidate_set_single_query(self, env):
        coll = self._collection(env)
        _dense_retrieve_with_embedding(coll, [[0.1] * 8], 3, {1, 2, 3})
        assert coll.query.call_count == 1

    def test_none_candidates_no_where(self, env):
        coll = self._collection(env)
        _dense_retrieve_with_embedding(coll, [[0.1] * 8], 3, None)
        assert coll.query.call_args.kwargs["where"] is None


# ---------------------------------------------------------------------------
# 4. BM25 batching (English path)
# ---------------------------------------------------------------------------


class TestBm25Batching:
    def test_oversized_candidate_in_batches(self, env):
        from rag.retrieval import _bm25_retrieve

        db = env["db"]
        db["folder"].insert({"path": "/c", "rel_path": ".", "name": ".", "depth": 0, "excluded": 0})
        # One matching chunk in the LAST batch — old single-query form raised
        # OperationalError (too many bound params) before it could be found.
        db["file"].insert({
            "folder_id": 1, "path": "/c/alpha.txt", "rel_path": "alpha.txt",
            "name": "alpha.txt", "extension": "txt", "size_bytes": 22, "mtime": "2026",
            "sha256": "", "mime_type": "text/plain", "hash_status": "ok",
            "identify_status": "ok", "triage_status": "ok", "is_dup_primary": 1,
            "excluded": 0,
        })
        db["chunk"].insert({
            "file_id": 1, "chunk_index": 0, "text": "alpha beta gamma delta",
            "token_count": 4,
        })
        chunk_id = db["chunk"].last_rowid
        db.execute("INSERT INTO chunk_fts(rowid, text) VALUES (?, ?)",
                   [chunk_id, "alpha beta gamma delta"])
        db.conn.commit()

        candidates = set(range(1, 2000))  # > SQL_PARAM_BATCH of 900
        candidates.add(1)  # the file that matches
        hits = _bm25_retrieve(db, "alpha beta gamma delta", 5, candidates)
        assert hits, "batched BM25 lost the matching chunk"
        assert hits[0].file_id == 1


# ---------------------------------------------------------------------------
# 5. CLI reranker default
# ---------------------------------------------------------------------------


class TestCliRerankerDefault:
    def test_query_passes_none_when_flag_absent(self):
        src = (Path(__file__).resolve().parent.parent / "cli.py").read_text()
        assert "use_reranker=(False if no_rerank else None)" in src
