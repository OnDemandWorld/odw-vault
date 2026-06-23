"""Tests for rag/phase8_extract.py — pure and helper functions."""

from pathlib import Path
from unittest.mock import MagicMock

from pipeline.db import migrate, open_db
from rag.phase8_extract import (
    _build_config_hash,
    _write_extraction_artifact,
)
from tests.conftest import seed_test_extractions, seed_test_files


class TestBuildConfigHash:
    def test_deterministic(self):
        cfg = MagicMock()
        cfg.extract.model_dump.return_value = {"docling_workers": 4, "tika_url": "http://localhost:9998"}
        h1 = _build_config_hash(cfg)
        h2 = _build_config_hash(cfg)
        assert h1 == h2

    def test_different_configs_different_hashes(self):
        cfg1 = MagicMock()
        cfg1.extract.model_dump.return_value = {"docling_workers": 4}
        cfg2 = MagicMock()
        cfg2.extract.model_dump.return_value = {"docling_workers": 8}
        assert _build_config_hash(cfg1) != _build_config_hash(cfg2)

    def test_hash_is_valid_sha256(self):
        cfg = MagicMock()
        cfg.extract.model_dump.return_value = {"key": "value"}
        h = _build_config_hash(cfg)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


class TestWriteExtractionArtifact:
    def test_creates_md_file(self, tmp_path):
        cache = tmp_path / "cache"
        cache.mkdir()
        sha = "abc123def456"
        result = _write_extraction_artifact(cache, sha, "Hello world", {"filename": "test.txt"}, "textutil")
        assert result is not None
        path = Path(result)
        assert path.exists()
        content = path.read_text()
        assert "# Extracted: test.txt" in content
        assert "strategy: textutil" in content
        assert "Hello world" in content

    def test_none_text_returns_none(self, tmp_path):
        cache = tmp_path / "cache"
        cache.mkdir()
        sha = "abc123"
        assert _write_extraction_artifact(cache, sha, None, {}, "test") is None

    def test_empty_string_returns_none(self, tmp_path):
        cache = tmp_path / "cache"
        cache.mkdir()
        sha = "abc123"
        assert _write_extraction_artifact(cache, sha, "", {}, "test") is None

    def test_creates_subdirectory(self, tmp_path):
        cache = tmp_path / "cache"
        cache.mkdir()
        sha = "ab12cd34"
        result = _write_extraction_artifact(cache, sha, "text", {}, "test")
        # Should create cache/extractions/ab/ directory
        assert "extractions" in str(result)
        assert "ab" in str(result)

    def test_file_content_format(self, tmp_path):
        cache = tmp_path / "cache"
        cache.mkdir()
        sha = "abc123"
        result = _write_extraction_artifact(cache, sha, "body text", {"filename": "f.txt"}, "docling")
        content = Path(result).read_text()
        assert content.startswith("# Extracted:")
        assert "---" in content
        assert "body text" in content


class TestExtractOne:
    def _make_cfg_extract(self):
        return MagicMock()

    def test_file_not_found(self, tmp_path):
        from rag.phase8_extract import _extract_one
        row = {
            "id": 1,
            "path": "/nonexistent/file.txt",
            "extract_strategy": "textutil",
            "sha256": "abc",
        }
        _file_id, succeeded, _err, _, _, _ = _extract_one(row, tmp_path, tmp_path, self._make_cfg_extract())
        assert succeeded is False
        assert "not found" in _err

    def test_no_extractor(self, tmp_path):
        from rag.phase8_extract import _extract_one
        p = tmp_path / "test.xyz"
        p.write_text("hello")
        row = {"id": 1, "path": str(p), "extract_strategy": "manual", "sha256": "abc"}
        _, succeeded, err, _, _, _ = _extract_one(row, tmp_path, tmp_path, self._make_cfg_extract())
        assert succeeded is False
        assert "no extractor" in err

    def test_extractor_exception(self, tmp_path):
        from rag.phase8_extract import EXTRACTOR_MAP, _extract_one
        p = tmp_path / "test.txt"
        p.write_text("hello")
        row = {"id": 1, "path": str(p), "extract_strategy": "textutil", "sha256": "abc"}

        orig = EXTRACTOR_MAP["textutil"]
        EXTRACTOR_MAP["textutil"] = MagicMock(side_effect=RuntimeError("fail"))
        try:
            _, succeeded, err, _, _, _ = _extract_one(row, tmp_path, tmp_path, self._make_cfg_extract())
        finally:
            EXTRACTOR_MAP["textutil"] = orig
        assert succeeded is False
        assert "fail" in err

    def test_success_returns_text(self, tmp_path):
        from rag.phase8_extract import EXTRACTOR_MAP, _extract_one
        p = tmp_path / "test.txt"
        p.write_text("hello world")
        row = {"id": 1, "path": str(p), "extract_strategy": "textutil", "sha256": "abc123"}

        orig = EXTRACTOR_MAP["textutil"]
        EXTRACTOR_MAP["textutil"] = MagicMock(return_value=("hello world", {}, True, None))
        try:
            _, succeeded, _err, artifact, char_count, _ = _extract_one(row, tmp_path, tmp_path, self._make_cfg_extract())
        finally:
            EXTRACTOR_MAP["textutil"] = orig
        assert succeeded is True
        assert char_count == 11
        assert artifact is not None


class TestRunExtractDB:
    """run_extract with in-memory DB and mocked extractors."""

    def _make_cfg(self, tmp_path):
        from pipeline.config import (
            AppConfig,
            ChunkConfig,
            ContextualRetrievalConfig,
            EmbeddingConfig,
            ExtractConfig,
            GenerationConfig,
            ModelsConfig,
            OllamaConfig,
            PathsConfig,
            RerankerConfig,
            RetrievalConfig,
            SummarizationConfig,
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
            ),
            chunk=ChunkConfig(),
            retrieval=RetrievalConfig(),
            extract=ExtractConfig(),
        )

    def _seed_db(self, tmp_path):
        db_path = tmp_path / "test.db"
        db = open_db(db_path)
        migrate(db)
        return db

    def test_no_eligible_files(self, tmp_path):
        from rag.phase8_extract import run_extract
        db = self._seed_db(tmp_path)
        cfg = self._make_cfg(tmp_path)
        result = run_extract(db, cfg)
        assert result.get("files_processed", 0) == 0

    def test_extracts_and_records(self, tmp_path):
        from rag.phase8_extract import EXTRACTOR_MAP, run_extract
        db = self._seed_db(tmp_path)
        cfg = self._make_cfg(tmp_path)

        file_ids = seed_test_files(db)
        file_path = tmp_path / "corpus" / "test.txt"
        file_path.write_text("This is the content of the test file.")
        db["file"].update(file_ids[0], {"path": str(file_path)})
        db.conn.commit()

        orig = EXTRACTOR_MAP["textutil"]
        EXTRACTOR_MAP["textutil"] = MagicMock(return_value=("Extracted content!", {}, True, None))
        try:
            result = run_extract(db, cfg, limit=1)
        finally:
            EXTRACTOR_MAP["textutil"] = orig

        assert result.get("files_processed", 0) >= 1
        exts = list(db["extraction"].rows)
        assert len(exts) >= 1
        assert "Extracted content!" in exts[0]["text_extracted"]

    def test_records_failure(self, tmp_path):
        from rag.phase8_extract import EXTRACTOR_MAP, run_extract
        db = self._seed_db(tmp_path)
        cfg = self._make_cfg(tmp_path)

        file_ids = seed_test_files(db)
        file_path = tmp_path / "corpus" / "test.txt"
        file_path.write_text("content")
        db["file"].update(file_ids[0], {"path": str(file_path)})
        db.conn.commit()

        orig = EXTRACTOR_MAP["textutil"]
        EXTRACTOR_MAP["textutil"] = MagicMock(side_effect=RuntimeError("extraction failed"))
        try:
            result = run_extract(db, cfg, limit=1)
        finally:
            EXTRACTOR_MAP["textutil"] = orig

        assert result.get("files_failed", 0) >= 1
        failures = list(db["failure"].rows)
        assert len(failures) >= 1

    def test_skips_already_extracted(self, tmp_path):
        from rag.phase8_extract import EXTRACTOR_MAP, run_extract
        db = self._seed_db(tmp_path)
        cfg = self._make_cfg(tmp_path)

        file_ids = seed_test_files(db)
        file_path = tmp_path / "corpus" / "test.txt"
        file_path.write_text("content")
        db["file"].update(file_ids[0], {"path": str(file_path)})
        # Pre-seed an extraction
        seed_test_extractions(db, file_ids, text="already extracted")

        orig = EXTRACTOR_MAP["textutil"]
        EXTRACTOR_MAP["textutil"] = MagicMock(return_value=("new content", {}, True, None))
        try:
            result = run_extract(db, cfg)
        finally:
            EXTRACTOR_MAP["textutil"] = orig

        assert result.get("files_processed", 0) == 0
