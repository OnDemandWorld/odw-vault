"""Round-7 deep-review regression tests.

Pins the fixes found in the 2026-11 launch-readiness review:
- pipeline/env.py NO_PROXY sanitizer (import-time httpx crash guard)
- ollama/httpx client lifecycle: every per-call client is closed on both
  success and failure (generation, retrieval, reranker, indexer summarize)
- Chinese BM25 + path-match SQL parameter batching (>999 candidates)
- empty candidate-set short-circuit in retrieve() (no ``IN ()`` crash)
- UI _on_chat no longer replays the current question as history
- eval run_eval actually applies --no-augment and model overrides
"""

from __future__ import annotations

import os
import sqlite3
from unittest.mock import MagicMock, patch

import httpx
import pytest

from pipeline.env import sanitize_proxy_env
from rag.retrieval import _bm25_retrieve_zh, _path_match_retrieve, retrieve
from tests.conftest import seed_test_extractions, seed_test_files

ZH_TEXT = "知识库检索系统支持中文查询与分词"


def _insert_chunk(db, file_id: int, text: str, index: int = 0) -> int:
    db["chunk"].insert(
        {
            "file_id": file_id,
            "chunk_index": index,
            "text": text,
            "token_count": max(1, len(text) // 4),
        }
    )
    db.conn.commit()
    return next(iter(db.query("SELECT id FROM chunk ORDER BY id DESC LIMIT 1")))["id"]


def _cap_sql_params(db, limit: int = 999):
    """Simulate older SQLite builds: raise when a statement binds too many
    parameters. Wraps db.query in place; returns the restore callable."""
    real_query = db.query

    def guarded(sql, params=None):
        if params is not None and len(params) > limit:
            raise sqlite3.OperationalError("too many SQL variables")
        return real_query(sql, params)

    db.query = guarded
    return lambda: setattr(db, "query", real_query)


# ---------------------------------------------------------------------------
# 1. Proxy env sanitizer
# ---------------------------------------------------------------------------


class TestProxyEnvSanitizer:
    @pytest.fixture(autouse=True)
    def _isolate_proxy_env(self, monkeypatch):
        for key in (
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
            "http_proxy", "https_proxy", "all_proxy",
        ):
            monkeypatch.delenv(key, raising=False)

    def test_bracketed_ipv6_no_longer_crashes_httpx(self, monkeypatch):
        hostile = "localhost,.localhost,127.0.0.1,127.0.0.0/8,[::1],::1"
        monkeypatch.setenv("NO_PROXY", hostile)
        monkeypatch.setenv("no_proxy", hostile)
        sanitize_proxy_env()
        for key in ("NO_PROXY", "no_proxy"):
            value = os.environ[key]
            assert "[::1]" not in value
            assert "::1" in value.split(",")
        # The exact crash scenario: constructing any trust_env client used
        # to raise httpx.InvalidURL ("Invalid port: ':1]'").
        with httpx.Client(trust_env=True) as client:
            assert client is not None

    def test_invalid_bracketed_entries_dropped(self, monkeypatch):
        monkeypatch.setenv("NO_PROXY", "[not-an-ip],[::1")
        sanitize_proxy_env()
        parts = os.environ["NO_PROXY"].split(",")
        assert "[not-an-ip]" not in parts
        assert "[::1" not in parts

    def test_loopback_bypass_added(self, monkeypatch):
        monkeypatch.setenv("NO_PROXY", "example.com")
        sanitize_proxy_env()
        parts = os.environ["NO_PROXY"].split(",")
        assert "localhost" in parts
        assert "127.0.0.1" in parts
        assert "example.com" in parts

    def test_idempotent(self, monkeypatch):
        monkeypatch.setenv("NO_PROXY", "example.com,[::1]")
        sanitize_proxy_env()
        once = os.environ["NO_PROXY"]
        sanitize_proxy_env()
        assert os.environ["NO_PROXY"] == once


# ---------------------------------------------------------------------------
# 2. Client lifecycle — generation
# ---------------------------------------------------------------------------


class TestGenerationClientLifecycle:
    def test_client_closed_on_success(self, app_config, hit_factory):
        with patch("rag.generation._make_client") as mock_make:
            client = MagicMock()
            client.chat.return_value = {"message": {"content": "Answer [1]."}}
            mock_make.return_value = client
            from rag.generation import generate_answer

            result = generate_answer("q", [hit_factory()], app_config)
        assert result["answer"] == "Answer [1]."
        client.close.assert_called_once()

    def test_client_closed_on_failure(self, app_config, hit_factory):
        with patch("rag.generation._make_client") as mock_make, \
             patch("rag.generation._ollama_chat", side_effect=RuntimeError("boom")):
            client = MagicMock()
            mock_make.return_value = client
            from rag.generation import generate_answer

            with pytest.raises(RuntimeError, match="boom"):
                generate_answer("q", [hit_factory()], app_config)
        client.close.assert_called_once()

    def test_refusal_creates_no_client(self, app_config):
        with patch("rag.generation._make_client") as mock_make:
            from rag.generation import generate_answer

            result = generate_answer("q", [], app_config)
        assert result["refused"] is True
        mock_make.assert_not_called()


# ---------------------------------------------------------------------------
# 3. Client lifecycle — retrieval query embedding
# ---------------------------------------------------------------------------


def _mock_chroma_for_retrieval():
    coll = MagicMock()
    coll.metadata = {"embedding_model": "test-embed"}
    coll.query.return_value = {
        "ids": [[]], "metadatas": [[]], "documents": [[]], "distances": [[]],
    }
    client = MagicMock()
    client.get_collection.return_value = coll
    return client


class TestRetrievalClientLifecycle:
    def test_query_embed_client_closed(self, app_config, test_db):
        with patch("chromadb.PersistentClient", return_value=_mock_chroma_for_retrieval()), \
             patch("ollama.Client") as mock_client, \
             patch("rag.retrieval._detect_language", return_value="en"):
            instance = mock_client.return_value
            instance.embed.return_value = {"embeddings": [[0.1] * 8]}
            hits, _metrics = retrieve("test query", test_db, None, "/chroma", app_config)
        assert hits == []
        instance.close.assert_called_once()

    def test_query_embed_client_closed_on_failure(self, app_config, test_db):
        with patch("chromadb.PersistentClient", return_value=_mock_chroma_for_retrieval()), \
             patch("ollama.Client") as mock_client, \
             patch("rag.retrieval._detect_language", return_value="en"):
            instance = mock_client.return_value
            instance.embed.side_effect = RuntimeError("ollama down")
            hits, _metrics = retrieve("test query", test_db, None, "/chroma", app_config)
        assert hits == []
        instance.close.assert_called_once()


# ---------------------------------------------------------------------------
# 4. SQL parameter batching — Chinese BM25 and path-match
# ---------------------------------------------------------------------------


class TestSqlParamBatching:
    def test_zh_bm25_batches_oversized_candidate_set(self, test_db):
        from rag.tokenization import index_zh_chunks

        file_ids = seed_test_files(
            test_db,
            files=[
                {"name": "zh1.txt", "rel_path": "zh1.txt"},
                {"name": "zh2.txt", "rel_path": "zh2.txt"},
            ],
        )
        c1 = _insert_chunk(test_db, file_ids[0], ZH_TEXT)
        c2 = _insert_chunk(test_db, file_ids[1], ZH_TEXT)
        assert index_zh_chunks(test_db, [(c1, ZH_TEXT), (c2, ZH_TEXT)]) == 2

        # 1500+ candidates: a single unbatched IN list exceeds the 999
        # parameter cap and the except-swallow turned that into zero recall.
        candidates = set(file_ids) | set(range(10000, 11500))
        restore = _cap_sql_params(test_db)
        try:
            hits = _bm25_retrieve_zh(test_db, "中文查询", 10, candidates, None)
        finally:
            restore()
        assert {h.chunk_id for h in hits} == {c1, c2}

    def test_path_match_batches_oversized_candidate_set(self, test_db):
        file_ids = seed_test_files(
            test_db,
            files=[{"name": "kwh_summary.txt", "rel_path": "reports/kwh_summary.txt"}],
        )
        chunk_id = _insert_chunk(test_db, file_ids[0], "energy readings")

        candidates = set(file_ids) | set(range(10000, 11500))
        restore = _cap_sql_params(test_db)
        try:
            hits = _path_match_retrieve(test_db, "kwh", candidates, None, n_results=10)
        finally:
            restore()
        assert any(h.chunk_id == chunk_id for h in hits)


# ---------------------------------------------------------------------------
# 5. Empty candidate set short-circuit
# ---------------------------------------------------------------------------


class TestEmptyCandidateShortCircuit:
    def test_all_candidates_excluded_returns_empty_not_crash(self, app_config, test_db):
        # Only file in scope is the excluded meta file: candidate set becomes
        # empty after exclusion. Downstream IN () must never be built.
        file_ids = seed_test_files(
            test_db,
            files=[{"name": "preflight_report.md", "rel_path": "preflight_report.md"}],
        )
        folder_id = next(
            iter(test_db.query("SELECT folder_id FROM file WHERE id = ?", [file_ids[0]]))
        )["folder_id"]
        app_config.retrieval.hierarchical = False

        with patch("chromadb.PersistentClient", return_value=_mock_chroma_for_retrieval()), \
             patch("ollama.Client") as mock_client, \
             patch("rag.retrieval._detect_language", return_value="en"):
            mock_client.return_value.embed.return_value = {"embeddings": [[0.1] * 8]}
            hits, metrics = retrieve(
                "preflight report",
                test_db,
                None,
                "/chroma",
                app_config,
                folder_filter={"folder_id": folder_id},
            )
        assert hits == []
        assert "retrieval_ms" in metrics


# ---------------------------------------------------------------------------
# 6. UI history duplication
# ---------------------------------------------------------------------------


class TestUiHistoryDuplication:
    def test_current_question_not_replayed_as_history(self, hit_factory):
        import ui.gradio_app as ga

        captured: list = []

        def fake_stream(query, hits, history=None):
            captured.append(list(history or []))
            yield "Answer [1]."
            yield "Answer [1].", []

        prior = [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
        ]
        with patch.object(ga, "retrieve", return_value=([hit_factory()], {})), \
             patch.object(ga, "_stream_tokens", side_effect=fake_stream), \
             patch.object(ga, "_ensure_conv_tables"), \
             patch.object(ga, "_save_conversation_message"), \
             patch.object(ga, "_record_query_log"), \
             patch.object(ga, "_get_db", return_value=MagicMock()):
            list(ga._on_chat("new question", prior, "All folders", None))

        assert len(captured) == 1
        # Exactly the prior turns — no bare copy of "new question" (it already
        # lives inside the generation prompt).
        assert captured[0] == prior


# ---------------------------------------------------------------------------
# 7. eval overrides actually applied
# ---------------------------------------------------------------------------


class TestEvalOverrides:
    def test_no_augment_and_generation_override_forwarded(self, test_db, app_config):
        from eval.runner import add_question, run_eval

        add_question(test_db, "q?", [1])
        with patch("eval.runner.retrieve", return_value=([], {"retrieval_ms": 1.0})) as mock_ret, \
             patch("eval.runner.generate_answer", return_value={"answer": "ok"}), \
             patch("eval.runner.chromadb.PersistentClient"):
            summary = run_eval(test_db, app_config, generation_model="gen-x", no_augment=True)

        assert summary["total_questions"] == 1
        kwargs = mock_ret.call_args.kwargs
        assert kwargs["use_augmentation"] is False
        assert kwargs["cfg"].models.generation.name == "gen-x"
        # Caller's config must not be mutated.
        assert app_config.models.generation.name == "gemma4:latest"
        row = next(iter(test_db.query("SELECT * FROM eval_run ORDER BY id DESC LIMIT 1")))
        assert row["generation_model"] == "gen-x"
        assert row["contextual_augmentation"] == 0

    def test_reranker_override_enables_reranker(self, test_db, app_config):
        from eval.runner import add_question, run_eval

        add_question(test_db, "q?", [1])
        with patch("eval.runner.retrieve", return_value=([], {"retrieval_ms": 1.0})) as mock_ret, \
             patch("eval.runner.generate_answer", return_value={"answer": "ok"}), \
             patch("eval.runner.chromadb.PersistentClient"):
            run_eval(test_db, app_config, reranker_model="rr-x")

        used_cfg = mock_ret.call_args.kwargs["cfg"]
        assert used_cfg.models.reranker.enabled is True
        assert used_cfg.models.reranker.name == "rr-x"
        assert app_config.models.reranker.enabled is False
        row = next(iter(test_db.query("SELECT * FROM eval_run ORDER BY id DESC LIMIT 1")))
        assert row["reranker_model"] == "rr-x"


# ---------------------------------------------------------------------------
# 8. Client lifecycle — reranker and indexer summarization
# ---------------------------------------------------------------------------


class TestRerankerClientLifecycle:
    def test_client_closed(self, hit_factory):
        from rag.reranker import rerank

        hits = [hit_factory(chunk_id=1), hit_factory(chunk_id=2)]
        with patch("ollama.Client") as mock_client:
            instance = mock_client.return_value
            instance.embed.return_value = {"embeddings": [[0.5], [0.7]]}
            result = rerank("q", hits, "rr-model", "http://localhost:11434")
        assert len(result) == 2
        instance.close.assert_called_once()


class TestIndexerSummarizeClientLifecycle:
    def test_client_closed_after_summarize(self, test_db, app_config):
        from rag.indexer import IncrementalIndexer

        file_ids = seed_test_files(test_db)
        seed_test_extractions(test_db, file_ids, text="x" * 600)
        with patch("ollama.Client") as mock_client:
            instance = mock_client.return_value
            instance.chat.return_value = {"message": {"content": "A summary."}}
            IncrementalIndexer(test_db, app_config)._run_summarization(file_ids[0])
        instance.close.assert_called_once()
        rows = list(test_db.query("SELECT summary_text FROM summary WHERE file_id = ?", [file_ids[0]]))
        assert rows and rows[0]["summary_text"] == "A summary."
