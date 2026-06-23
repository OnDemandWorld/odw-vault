"""Tests for rag/phase8b_transcribe.py — pure and helper functions."""

from __future__ import annotations

from unittest.mock import MagicMock

from pipeline.db import migrate, open_db
from rag.phase8b_transcribe import (
    _build_config_hash,
    _ensure_transcript_dir,
    _transcribe_file,
)
from tests.conftest import seed_test_files


class TestBuildConfigHash:
    def test_deterministic(self):
        cfg = MagicMock()
        cfg.models.transcription.model_dump.return_value = {"model": "large-v3", "language": "en"}
        h1 = _build_config_hash(cfg)
        h2 = _build_config_hash(cfg)
        assert h1 == h2

    def test_different_hashes(self):
        cfg1 = MagicMock()
        cfg1.models.transcription.model_dump.return_value = {"model": "large-v3"}
        cfg2 = MagicMock()
        cfg2.models.transcription.model_dump.return_value = {"model": "medium"}
        assert _build_config_hash(cfg1) != _build_config_hash(cfg2)

    def test_valid_sha256(self):
        cfg = MagicMock()
        cfg.models.transcription.model_dump.return_value = {"key": "value"}
        h = _build_config_hash(cfg)
        assert len(h) == 64


class TestEnsureTranscriptDir:
    def test_creates_directory(self, tmp_path):
        result = _ensure_transcript_dir(tmp_path, "abc123def")
        assert result.exists()
        assert "transcripts" in str(result)
        assert "ab" in str(result)

    def test_idempotent(self, tmp_path):
        r1 = _ensure_transcript_dir(tmp_path, "abc123")
        r2 = _ensure_transcript_dir(tmp_path, "abc456")
        assert r1.exists()
        assert r2.exists()


class TestTranscribeFile:
    def test_import_error(self, tmp_path):
        cache = tmp_path / "cache"
        cache.mkdir()
        p = tmp_path / "audio.wav"
        p.write_bytes(b"fake audio")

        succeeded, _path, err = _transcribe_file(
            str(p), "abc123", cache, model_name="large-v3"
        )
        # pywhispercpp is not installed, so this should fail with ImportError
        assert succeeded is False
        assert err is not None
        assert "pywhispercpp" in err.get("error", "")

    def test_file_not_found_error(self, tmp_path):
        cache = tmp_path / "cache"
        cache.mkdir()
        succeeded, _path, err = _transcribe_file(
            "/nonexistent/audio.wav", "abc123", cache
        )
        assert succeeded is False
        assert err is not None


class TestRunTranscribeDB:
    """run_transcribe with in-memory DB."""

    def _make_cfg(self, tmp_path):
        from pipeline.config import (
            AppConfig,
            ChunkConfig,
            ContextualRetrievalConfig,
            EmbeddingConfig,
            GenerationConfig,
            ModelsConfig,
            OllamaConfig,
            PathsConfig,
            RerankerConfig,
            RetrievalConfig,
            SummarizationConfig,
            TranscriptionConfig,
        )
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        cache = corpus / ".rag-cache"
        cache.mkdir()
        chroma = tmp_path / "chroma"
        chroma.mkdir()
        return AppConfig(
            paths=PathsConfig(corpus_root=str(corpus), cache_root=str(cache), chroma_root=str(chroma)),
            ollama=OllamaConfig(),
            models=ModelsConfig(
                embedding=EmbeddingConfig(name="test-embed", collection_suffix="test"),
                summarization=SummarizationConfig(name="test-summarize"),
                contextual_retrieval=ContextualRetrievalConfig(enabled=False, name="test-context"),
                generation=GenerationConfig(name="test-gen", fallback_name="test-fb", alternate_name="test-alt"),
                reranker=RerankerConfig(enabled=False),
                transcription=TranscriptionConfig(model="large-v3", language="auto", threads=2),
            ),
            chunk=ChunkConfig(),
            retrieval=RetrievalConfig(),
        )

    def _seed_db(self, tmp_path):
        db_path = tmp_path / "test.db"
        db = open_db(db_path)
        migrate(db)
        return db

    def test_no_eligible_files(self, tmp_path):
        from rag.phase8b_transcribe import run_transcribe
        db = self._seed_db(tmp_path)
        cfg = self._make_cfg(tmp_path)
        result = run_transcribe(db, cfg)
        assert result["files_processed"] == 0

    def test_no_matching_category(self, tmp_path):
        """Insert a document file — transcription should skip (not audio/video)."""
        from rag.phase8b_transcribe import run_transcribe
        db = self._seed_db(tmp_path)
        cfg = self._make_cfg(tmp_path)

        # seed_test_files creates a file with category="document"
        seed_test_files(db)
        db.conn.commit()

        result = run_transcribe(db, cfg)
        assert result["files_processed"] == 0
