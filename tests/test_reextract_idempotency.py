"""Regression: re-extraction must not duplicate or stale downstream data.

`extract --reextract` appends a new extraction row per file. Chunking,
summarization, contextual augmentation, summary embedding, and GET /files/{id}
must all use only the latest successful extraction (and latest summary).
"""

import sqlite3
from unittest.mock import MagicMock, patch

import sqlite_utils
from fastapi.testclient import TestClient

from pipeline.db import migrate
from rag.phase10_chunk import run_chunk
from tests.conftest import seed_test_extractions, seed_test_files

OLD_TEXT = "Old stale content. It should never be indexed."
NEW_TEXT = "Fresh content alpha. Fresh content beta. Fresh content gamma."


def _make_config(tmp_path):
    from pipeline.config import (
        AppConfig,
        ChunkConfig,
        ContextualRetrievalConfig,
        EmbeddingConfig,
        GenerationConfig,
        ModelsConfig,
        OllamaConfig,
        PathsConfig,
        SummarizationConfig,
    )

    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    cache = corpus / ".rag-cache"
    cache.mkdir(exist_ok=True)
    return AppConfig(
        paths=PathsConfig(corpus_root=str(corpus), cache_root=str(cache)),
        ollama=OllamaConfig(host="http://localhost:11434"),
        chunk=ChunkConfig(window_size=2),
        models=ModelsConfig(
            embedding=EmbeddingConfig(name="test-embed", collection_suffix="_test"),
            summarization=SummarizationConfig(name="test-summarize"),
            contextual_retrieval=ContextualRetrievalConfig(name="test-context"),
            generation=GenerationConfig(
                name="test-gen", fallback_name="test-fb", alternate_name="test-alt"
            ),
        ),
    )


def _seed_two_extractions(db):
    file_ids = seed_test_files(db)
    seed_test_extractions(db, file_ids, text=OLD_TEXT)
    seed_test_extractions(db, file_ids, text=NEW_TEXT)
    return file_ids


class TestChunkLatestExtraction:
    def test_no_duplicate_chunks(self, tmp_path, test_db):
        cfg = _make_config(tmp_path)
        _seed_two_extractions(test_db)
        total, files = run_chunk(test_db, cfg)
        assert files == 1
        dup_groups = list(
            test_db.query(
                "SELECT file_id, chunk_index FROM chunk "
                "GROUP BY file_id, chunk_index HAVING COUNT(*) > 1"
            )
        )
        assert dup_groups == []
        assert total == len(list(test_db["chunk"].rows))

    def test_chunks_come_from_new_text(self, tmp_path, test_db):
        cfg = _make_config(tmp_path)
        _seed_two_extractions(test_db)
        run_chunk(test_db, cfg)
        for ch in test_db["chunk"].rows:
            assert "stale" not in ch["text"]
        joined = " ".join(ch["text"] for ch in test_db["chunk"].rows)
        assert "Fresh content alpha" in joined

    def test_rechunk_no_duplicates(self, tmp_path, test_db):
        cfg = _make_config(tmp_path)
        _seed_two_extractions(test_db)
        run_chunk(test_db, cfg, rechunk=True)
        count = next(iter(test_db.query("SELECT COUNT(*) AS c FROM chunk")))["c"]
        run_chunk(test_db, cfg, rechunk=True)
        count2 = next(iter(test_db.query("SELECT COUNT(*) AS c FROM chunk")))["c"]
        assert count2 == count


class TestSummarizeLatestExtraction:
    def test_summarizes_new_text_once(self, tmp_path, test_db):
        from rag.phase9_summarize import run_summarize

        cfg = _make_config(tmp_path)
        cfg.extract.size_threshold_for_summary = 10
        _seed_two_extractions(test_db)
        with patch("rag.phase9_summarize._call_ollama", return_value="A summary.") as call:
            processed, failed = run_summarize(test_db, cfg)
        assert processed == 1 and failed == 0
        assert call.call_count == 1
        prompt = call.call_args.args[0]
        assert "Fresh content alpha" in prompt
        assert "stale" not in prompt


class TestContextNoRowMultiplication:
    def test_one_llm_call_per_chunk(self, tmp_path, test_db):
        from rag.phase10_5_context import run_context

        cfg = _make_config(tmp_path)
        file_ids = _seed_two_extractions(test_db)
        run_chunk(test_db, cfg)
        for _ in range(2):
            test_db["summary"].insert(
                {"file_id": file_ids[0], "model": "test-summarize", "summary_text": "S."}
            )
        test_db.conn.commit()
        n_chunks = len(list(test_db["chunk"].rows))
        with patch("rag.phase10_5_context._call_ollama", return_value="context") as call:
            processed, failed = run_context(test_db, cfg)
        assert failed == 0
        assert processed == n_chunks
        assert call.call_count == n_chunks


class TestSummaryEmbedSkipsSuperseded:
    def test_only_latest_summary_per_model_embedded(self, tmp_path, test_db):
        from rag.phase11_embed import _embed_summaries

        file_ids = seed_test_files(test_db)
        fid = file_ids[0]
        test_db["summary"].insert(
            {"file_id": fid, "model": "m1", "summary_text": "old summary"}
        )
        test_db["summary"].insert(
            {"file_id": fid, "model": "m1", "summary_text": "new summary"}
        )
        test_db["summary"].insert(
            {"file_id": fid, "model": "m2", "summary_text": "other model summary"}
        )
        test_db.conn.commit()

        client = MagicMock()
        with (
            patch("rag.phase11_embed._get_dim_from_ollama", return_value=3),
            patch(
                "rag.phase11_embed._ollama_embed",
                side_effect=lambda host, model, texts, *a, **k: [[0.1, 0.2, 0.3]] * len(texts),
            ) as embed,
        ):
            count = _embed_summaries(
                test_db, client, "http://localhost:11434", "test-embed", "_test",
                batch_size=8, truncate_dim=0, config_hash="h", reembed=False,
            )
        assert count == 2
        embedded_texts = [t for call in embed.call_args_list for t in call.args[2]]
        assert "old summary" not in embedded_texts
        assert "new summary" in embedded_texts
        assert "other model summary" in embedded_texts


class TestGetFileLatestRows:
    def test_returns_latest_extraction_and_summary(self, tmp_path):
        from api.main import app

        conn = sqlite3.connect(str(tmp_path / "t.db"), check_same_thread=False)
        db = sqlite_utils.Database(conn)
        migrate(db)
        file_ids = seed_test_files(db)
        fid = file_ids[0]
        seed_test_extractions(db, [fid], text=OLD_TEXT)
        seed_test_extractions(db, [fid], text=NEW_TEXT)
        db["summary"].insert({"file_id": fid, "model": "m", "summary_text": "old sum"})
        db["summary"].insert({"file_id": fid, "model": "m", "summary_text": "new sum"})
        db.conn.commit()

        with patch("api.main._get_db", return_value=db):
            client = TestClient(app)
            resp = client.get(f"/files/{fid}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["summary"] == "new sum"
        assert body["extraction_path"] == NEW_TEXT
