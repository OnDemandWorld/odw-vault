"""Tests for rag/phase10_chunk.py — integration with DB."""


from rag.phase10_chunk import run_chunk
from tests.conftest import seed_test_extractions, seed_test_files


def _make_config(tmp_path):
    from pipeline.config import (
        AppConfig,
        ChunkConfig,
        ContextualRetrievalConfig,
        EmbeddingConfig,
        GenerationConfig,
        ModelsConfig,
        PathsConfig,
        SummarizationConfig,
    )
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    cache = corpus / ".rag-cache"
    cache.mkdir()
    return AppConfig(
        paths=PathsConfig(corpus_root=str(corpus), cache_root=str(cache)),
        chunk=ChunkConfig(window_size=2),
        models=ModelsConfig(
            embedding=EmbeddingConfig(name="test-embed", collection_suffix="_test"),
            summarization=SummarizationConfig(name="test-summarize"),
            contextual_retrieval=ContextualRetrievalConfig(name="test-context"),
            generation=GenerationConfig(name="test-gen", fallback_name="test-fb", alternate_name="test-alt"),
        ),
    )


class TestRunChunkDB:
    def test_creates_chunk_rows(self, tmp_path, test_db):
        cfg = _make_config(tmp_path)
        file_ids = seed_test_files(test_db)
        seed_test_extractions(test_db, file_ids, text="First sentence. Second sentence. Third sentence.")
        total, files = run_chunk(test_db, cfg)
        assert total > 0
        assert files >= 1
        chunks = list(test_db["chunk"].rows)
        assert len(chunks) > 0

    def test_chunk_text_contains_sentences(self, tmp_path, test_db):
        cfg = _make_config(tmp_path)
        file_ids = seed_test_files(test_db)
        seed_test_extractions(test_db, file_ids, text="A. B. C. D. E.")
        run_chunk(test_db, cfg)
        chunks = list(test_db["chunk"].rows)
        for ch in chunks:
            # Each chunk should be empty
            assert len(ch["text"]) > 0

    def test_token_count_populated(self, tmp_path, test_db):
        cfg = _make_config(tmp_path)
        file_ids = seed_test_files(test_db)
        seed_test_extractions(test_db, file_ids, text="Hello world. Test text here.")
        run_chunk(test_db, cfg)
        chunks = list(test_db["chunk"].rows)
        for ch in chunks:
            assert ch["token_count"] > 0

    def test_rechunk_deletes_old_chunks(self, tmp_path, test_db):
        cfg = _make_config(tmp_path)
        file_ids = seed_test_files(test_db)
        seed_test_extractions(test_db, file_ids, text="A. B. C.")
        run_chunk(test_db, cfg)
        first_count = next(iter(test_db.query("SELECT COUNT(*) as c FROM chunk")))["c"]

        # Re-run with rechunk
        total, files = run_chunk(test_db, cfg, rechunk=True)
        second_count = next(iter(test_db.query("SELECT COUNT(*) as c FROM chunk")))["c"]
        # Should have same or similar count (not doubled)
        assert second_count <= first_count + 1

    def test_empty_extraction_skipped(self, tmp_path, test_db):
        cfg = _make_config(tmp_path)
        file_ids = seed_test_files(test_db)
        seed_test_extractions(test_db, file_ids, text="")
        total, files = run_chunk(test_db, cfg)
        # Empty text should be skipped (WHERE text_extracted != '')
        assert total == 0

    def test_metadata_json_valid(self, tmp_path, test_db):
        import json
        cfg = _make_config(tmp_path)
        file_ids = seed_test_files(test_db)
        seed_test_extractions(test_db, file_ids, text="Test sentence.")
        run_chunk(test_db, cfg)
        chunks = list(test_db["chunk"].rows)
        for ch in chunks:
            meta = json.loads(ch["metadata_json"])
            assert "extraction_id" in meta
            assert "extraction_tool" in meta

    def test_no_eligible_files(self, tmp_path, test_db):
        cfg = _make_config(tmp_path)
        total, files = run_chunk(test_db, cfg)
        assert total == 0
        assert files == 0

    def test_chunk_index_sequential(self, tmp_path, test_db):
        cfg = _make_config(tmp_path)
        file_ids = seed_test_files(test_db)
        seed_test_extractions(test_db, file_ids, text="A. B. C. D.")
        run_chunk(test_db, cfg)
        chunks = list(test_db.query("SELECT chunk_index FROM chunk ORDER BY chunk_index"))
        indices = [c["chunk_index"] for c in chunks]
        assert indices == list(range(len(indices)))
