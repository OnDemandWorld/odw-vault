"""Tests for rag/phase11_embed.py — mocked Chroma and Ollama."""

from unittest.mock import MagicMock, patch

import pytest

from pipeline.db import migrate, open_db
from rag.phase11_embed import (
    _ensure_collection,
    _ollama_embed,
    run_embed,
)
from tests.conftest import seed_test_extractions, seed_test_files


class TestOllamaEmbed:
    def test_returns_vectors(self):
        with patch("rag.phase11_embed.ollama.Client") as MockClient:
            instance = MockClient.return_value
            instance.embed.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
            result = _ollama_embed("http://localhost:11434", "test-model", ["hello"])
        assert result == [[0.1, 0.2, 0.3]]

    def test_empty_embeddings_raises(self):
        with patch("rag.phase11_embed.ollama.Client") as MockClient:
            instance = MockClient.return_value
            instance.embed.return_value = {"embeddings": []}
            with pytest.raises(Exception):  # tenacity wraps ValueError in RetryError
                _ollama_embed("http://localhost:11434", "test-model", ["hello"])

    def test_truncate_dim_passed(self):
        with patch("rag.phase11_embed.ollama.Client") as MockClient:
            instance = MockClient.return_value
            instance.embed.return_value = {"embeddings": [[0.1]]}
            _ollama_embed("http://localhost:11434", "test-model", ["hello"], truncate_dim=256)
            call_kwargs = instance.embed.call_args.kwargs
            assert call_kwargs["options"]["truncate_dim"] == 256

    def test_truncate_dim_zero_not_passed(self):
        with patch("rag.phase11_embed.ollama.Client") as MockClient:
            instance = MockClient.return_value
            instance.embed.return_value = {"embeddings": [[0.1]]}
            _ollama_embed("http://localhost:11434", "test-model", ["hello"], truncate_dim=0)
            call_kwargs = instance.embed.call_args.kwargs
            assert "options" not in call_kwargs


class TestEnsureCollection:
    def test_creates_new_collection(self):
        client = MagicMock()
        client.list_collections.return_value = []
        coll = MagicMock()
        client.create_collection.return_value = coll

        result = _ensure_collection(
            client,
            "chunks__test",
            embedding_model="test-embed",
            dim=4096,
            config_hash="abc123",
            source_db_path="/test.db",
        )

        assert result == coll
        client.create_collection.assert_called_once()
        call_kwargs = client.create_collection.call_args.kwargs
        assert call_kwargs["metadata"]["embedding_model"] == "test-embed"
        assert call_kwargs["metadata"]["dim"] == 4096

    def test_gets_existing_collection(self):
        coll = MagicMock()
        coll.name = "chunks__test"
        coll.metadata = {
            "embedding_model": "test-embed",
            "dim": 4096,
        }
        client = MagicMock()
        client.list_collections.return_value = [coll]
        client.get_collection.return_value = coll

        result = _ensure_collection(
            client,
            "chunks__test",
            embedding_model="test-embed",
            dim=4096,
            config_hash="abc123",
            source_db_path="/test.db",
        )

        assert result == coll
        client.create_collection.assert_not_called()

    def test_model_mismatch_raises(self):
        coll = MagicMock(spec=["name", "metadata"])
        coll.name = "chunks__test"
        coll.metadata = {"embedding_model": "old-model", "dim": 4096}
        client = MagicMock()
        client.list_collections.return_value = [coll]
        client.get_collection.return_value = coll

        with pytest.raises(ValueError, match="old-model"):
            _ensure_collection(
                client,
                "chunks__test",
                embedding_model="new-model",
                dim=4096,
                config_hash="abc123",
                source_db_path="/test.db",
            )

    def test_dim_mismatch_raises(self):
        coll = MagicMock(spec=["name", "metadata"])
        coll.name = "chunks__test"
        coll.metadata = {"embedding_model": "test-embed", "dim": 2048}
        client = MagicMock()
        client.list_collections.return_value = [coll]
        client.get_collection.return_value = coll

        with pytest.raises(ValueError, match="dim"):
            _ensure_collection(
                client,
                "chunks__test",
                embedding_model="test-embed",
                dim=4096,
                config_hash="abc123",
                source_db_path="/test.db",
            )


class TestRunEmbedMocked:
    """run_embed with in-memory DB and all mocks."""

    def _make_cfg(self, tmp_path):
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
        )
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        cache = corpus / ".rag-cache"
        cache.mkdir()
        chroma = tmp_path / "chroma"
        chroma.mkdir()

        return AppConfig(
            paths=PathsConfig(
                corpus_root=str(corpus),
                cache_root=str(cache),
                chroma_root=str(chroma),
            ),
            ollama=OllamaConfig(),
            models=ModelsConfig(
                embedding=EmbeddingConfig(name="test-embed", collection_suffix="test"),
                summarization=SummarizationConfig(name="test-summarize"),
                contextual_retrieval=ContextualRetrievalConfig(
                    enabled=False, name="test-context"
                ),
                generation=GenerationConfig(
                    name="test-gen",
                    fallback_name="test-fb",
                    alternate_name="test-alt",
                ),
                reranker=RerankerConfig(enabled=False),
            ),
            generation_runtime=GenerationRuntimeConfig(refuse_on_empty_context=True),
            chunk=ChunkConfig(),
            retrieval=RetrievalConfig(),
            extract=ExtractConfig(),
        )

    def _seed_db(self, tmp_path):
        db_path = tmp_path / "test.db"
        db = open_db(db_path)
        migrate(db)
        return db

    def test_no_chunks_returns_zero(self, tmp_path):
        db = self._seed_db(tmp_path)
        cfg = self._make_cfg(tmp_path)

        with patch("rag.phase11_embed.chromadb.PersistentClient") as MockChroma:
            result = run_embed(db, cfg)

        assert result == 0

    def test_embeds_chunks_with_mocks(self, tmp_path):
        db = self._seed_db(tmp_path)
        cfg = self._make_cfg(tmp_path)

        # Ensure folder exists for FK
        db["folder"].insert({"path": "test", "rel_path": "test", "name": "test", "depth": 0})
        db.conn.commit()

        # Set up: file + extraction + chunk
        file_ids = seed_test_files(db, files=[
            {
                "name": "test.txt",
                "folder_id": 1,
                "extract_strategy": "textutil",
                "category": "document",
            }
        ])
        seed_test_extractions(db, file_ids, text="Sample text.")

        db["chunk"].insert({
            "file_id": file_ids[0],
            "text": "This is a test chunk for embedding.",
            "chunk_index": 0,
            "token_count": 7,
            "metadata_json": '{"extraction_id": 1, "extraction_tool": "test"}',
        })
        db.conn.commit()

        mock_coll = MagicMock()
        mock_coll.name = "chunks__test"
        mock_coll.metadata = {
            "embedding_model": "test-embed",
            "dim": 3,
        }
        mock_chroma_client = MagicMock()
        mock_chroma_client.list_collections.return_value = [mock_coll]
        mock_chroma_client.get_collection.return_value = mock_coll

        with patch("rag.phase11_embed.chromadb.PersistentClient", return_value=mock_chroma_client):
            with patch("rag.phase11_embed._ollama_embed", return_value=[[0.1, 0.2, 0.3]]):
                result = run_embed(db, cfg, collections=["chunks"])

        assert result >= 1
        refs = list(db["embedding_ref"].rows)
        assert len(refs) >= 1

    def test_embeds_summaries_with_mocks(self, tmp_path):
        db = self._seed_db(tmp_path)
        cfg = self._make_cfg(tmp_path)

        file_ids = seed_test_files(db)
        seed_test_extractions(db, file_ids, text="Text to summarize.")

        # Insert a summary
        db["summary"].insert({
            "file_id": file_ids[0],
            "model": "test-summarize",
            "summary_text": "This is a summary.",
            "generated_at": "2026-01-01T00:00:00Z",
        })
        db.conn.commit()

        mock_coll = MagicMock()
        mock_coll.name = "summaries__test"
        mock_coll.metadata = {"embedding_model": "test-embed", "dim": 3}
        mock_chroma_client = MagicMock()
        mock_chroma_client.list_collections.return_value = [mock_coll]
        mock_chroma_client.get_collection.return_value = mock_coll

        with patch("rag.phase11_embed.chromadb.PersistentClient", return_value=mock_chroma_client):
            with patch("rag.phase11_embed._ollama_embed", return_value=[[0.1, 0.2, 0.3]]):
                result = run_embed(db, cfg, collections=["summaries"])

        assert result >= 1
        refs = list(db["summary_embedding_ref"].rows)
        assert len(refs) >= 1

    def test_embeds_folders_with_mocks(self, tmp_path):
        db = self._seed_db(tmp_path)
        cfg = self._make_cfg(tmp_path)

        # Insert a folder
        db["folder"].insert({
            "path": "test",
            "rel_path": "test",
            "name": "test",
            "depth": 0,
        })
        db.conn.commit()

        mock_coll = MagicMock()
        mock_coll.name = "folders__test"
        mock_coll.metadata = {"embedding_model": "test-embed", "dim": 3}
        mock_chroma_client = MagicMock()
        mock_chroma_client.list_collections.return_value = [mock_coll]
        mock_chroma_client.get_collection.return_value = mock_coll

        with patch("rag.phase11_embed.chromadb.PersistentClient", return_value=mock_chroma_client):
            with patch("rag.phase11_embed._ollama_embed", return_value=[[0.1, 0.2, 0.3]]):
                result = run_embed(db, cfg, collections=["folders"])

        assert result >= 1
        refs = list(db["folder_embedding_ref"].rows)
        assert len(refs) >= 1

    def test_reembed_flag(self, tmp_path):
        db = self._seed_db(tmp_path)
        cfg = self._make_cfg(tmp_path)

        file_ids = seed_test_files(db)
        seed_test_extractions(db, file_ids, text="Text.")
        db["chunk"].insert({
            "file_id": file_ids[0],
            "text": "Chunk text.",
            "chunk_index": 0,
            "token_count": 2,
            "metadata_json": "{}",
        })
        db.conn.commit()

        mock_coll = MagicMock()
        mock_coll.name = "chunks__test"
        mock_coll.metadata = {"embedding_model": "test-embed", "dim": 3}
        mock_chroma_client = MagicMock()
        mock_chroma_client.list_collections.return_value = [mock_coll]
        mock_chroma_client.get_collection.return_value = mock_coll

        with patch("rag.phase11_embed.chromadb.PersistentClient", return_value=mock_chroma_client):
            with patch("rag.phase11_embed._ollama_embed", return_value=[[0.1, 0.2, 0.3]]) as mock_embed:
                run_embed(db, cfg, collections=["chunks"], reembed=True)

        assert mock_embed.call_count >= 1

    def test_collection_filter(self, tmp_path):
        db = self._seed_db(tmp_path)
        cfg = self._make_cfg(tmp_path)

        with patch("rag.phase11_embed.chromadb.PersistentClient") as MockChroma:
            MockChroma.return_value.list_collections.return_value = []
            with patch("rag.phase11_embed._ollama_embed", return_value=[[0.1]]):
                result = run_embed(db, cfg, collections=[])

        assert result == 0
