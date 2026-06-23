"""Tests for rag/phase10_5_context.py — integration with mocked Ollama and DB."""

from unittest.mock import MagicMock, patch

from pipeline.db import migrate, open_db
from rag.phase10_5_context import _build_prompt, run_context
from tests.conftest import seed_test_extractions, seed_test_files


class TestBuildPrompt:
    def test_returns_prompt_and_hash(self):
        prompt, prompt_hash = _build_prompt("doc text", "chunk text")
        assert isinstance(prompt, str)
        assert isinstance(prompt_hash, str)
        assert len(prompt_hash) == 64  # SHA-256 hex
        assert "doc text" in prompt
        assert "chunk text" in prompt

    def test_hash_is_deterministic(self):
        _, h1 = _build_prompt("doc", "chunk")
        _, h2 = _build_prompt("doc", "chunk")
        assert h1 == h2

    def test_different_inputs_different_hash(self):
        _, h1 = _build_prompt("doc1", "chunk")
        _, h2 = _build_prompt("doc2", "chunk")
        assert h1 != h2


class TestRunContextMocked:
    """run_context with in-memory DB and mocked Ollama."""

    def _make_cfg(self, enabled=True):
        cfg = MagicMock()
        cfg.models.contextual_retrieval.enabled = enabled
        cfg.models.contextual_retrieval.name = "test-context"
        cfg.models.contextual_retrieval.temperature = 0.1
        cfg.models.contextual_retrieval.max_context_tokens = 16384
        cfg.models.contextual_retrieval.prompt_version = "v1"
        cfg.ollama.host = "http://localhost:11434"
        return cfg

    def _seed_db(self, tmp_path):
        db_path = tmp_path / "test.db"
        db = open_db(db_path)
        migrate(db)
        return db

    def test_disabled_in_config(self, tmp_path):
        db = self._seed_db(tmp_path)
        cfg = self._make_cfg(enabled=False)
        processed, failed = run_context(db, cfg)
        assert processed == 0
        assert failed == 0

    def test_no_chunks_to_process(self, tmp_path):
        db = self._seed_db(tmp_path)
        cfg = self._make_cfg()
        processed, failed = run_context(db, cfg)
        assert processed == 0
        assert failed == 0

    def test_generates_context_for_chunks(self, tmp_path):
        db = self._seed_db(tmp_path)
        cfg = self._make_cfg()
        file_ids = seed_test_files(db)
        seed_test_extractions(db, file_ids, text="Document text for context.")

        # Insert a chunk
        db["chunk"].insert({
            "file_id": file_ids[0],
            "text": "This is a chunk of text.",
            "chunk_index": 0,
            "token_count": 6,
            "metadata_json": '{"extraction_id": 1, "extraction_tool": "test"}',
        })
        db.conn.commit()

        with patch("rag.phase10_5_context._call_ollama", return_value="Context sentence for this chunk."):
            processed, failed = run_context(db, cfg)

        assert processed == 1
        assert failed == 0
        chunk = next(iter(db.query("SELECT context_text FROM chunk WHERE id = 1")))
        assert chunk["context_text"] == "Context sentence for this chunk."

    def test_skips_already_contextualized(self, tmp_path):
        db = self._seed_db(tmp_path)
        cfg = self._make_cfg()
        file_ids = seed_test_files(db)
        seed_test_extractions(db, file_ids, text="Document text.")

        db["chunk"].insert({
            "file_id": file_ids[0],
            "text": "Chunk text.",
            "chunk_index": 0,
            "token_count": 3,
            "metadata_json": "{}",
            "context_text": "Already done.",
            "context_model": "test-context",
            "context_prompt_hash": "abc123",
        })
        db.conn.commit()

        with patch("rag.phase10_5_context._call_ollama", return_value="New context.") as mock_call:
            processed, failed = run_context(db, cfg)

        assert processed == 0
        assert mock_call.call_count == 0

    def test_regenerate_flag(self, tmp_path):
        db = self._seed_db(tmp_path)
        cfg = self._make_cfg()
        file_ids = seed_test_files(db)
        seed_test_extractions(db, file_ids, text="Document text.")

        db["chunk"].insert({
            "file_id": file_ids[0],
            "text": "Chunk text.",
            "chunk_index": 0,
            "token_count": 3,
            "metadata_json": "{}",
            "context_text": "Old context.",
            "context_model": "test-context",
            "context_prompt_hash": "abc123",
        })
        db.conn.commit()

        with patch("rag.phase10_5_context._call_ollama", return_value="New context."):
            processed, failed = run_context(db, cfg, regenerate=True)

        assert processed == 1
