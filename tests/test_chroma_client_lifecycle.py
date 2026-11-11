"""Regression tests: per-call Chroma clients must be closed.

Chroma's SharedSystemClient keeps one System per (process, persist path)
alive while any client reference remains. A leaked client pins the System
for the process lifetime, so writes from an external sync process are never
seen — dense retrieval fails with "Error finding id" or returns superseded
chunks after a CLI sync.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from rag.retrieval import retrieve


def _make_config(tmp_path):
    from pipeline.config import (
        AppConfig,
        ContextualRetrievalConfig,
        EmbeddingConfig,
        GenerationConfig,
        ModelsConfig,
        PathsConfig,
        SummarizationConfig,
    )

    return AppConfig(
        paths=PathsConfig(
            corpus_root=str(tmp_path / "corpus"),
            cache_root=str(tmp_path / "cache"),
            chroma_root=str(tmp_path / "chroma"),
        ),
        models=ModelsConfig(
            embedding=EmbeddingConfig(name="test-embed", collection_suffix="_test"),
            summarization=SummarizationConfig(name="test-summarize"),
            contextual_retrieval=ContextualRetrievalConfig(name="test-context"),
            generation=GenerationConfig(
                name="test-gen", fallback_name="test-fb", alternate_name="test-alt"
            ),
        ),
    )


class TestRetrieveClientLifecycle:
    def test_client_closed_on_success(self, tmp_path):
        with (
            patch("rag.retrieval.chromadb.PersistentClient") as pc,
            patch("rag.retrieval._retrieve_with_client", return_value=([], {})),
        ):
            retrieve("q", None, None, str(tmp_path), None)
        pc.return_value.close.assert_called_once()

    def test_client_closed_on_error(self, tmp_path):
        with (
            patch("rag.retrieval.chromadb.PersistentClient") as pc,
            patch(
                "rag.retrieval._retrieve_with_client",
                side_effect=RuntimeError("boom"),
            ),pytest.raises(RuntimeError, match="boom")
        ):
            retrieve("q", None, None, str(tmp_path), None)
        pc.return_value.close.assert_called_once()

    def test_shared_system_released_after_failed_retrieve(self, tmp_path, test_db):
        # Real Chroma: a missing collection makes retrieve raise after the
        # client is created. The shared System must not outlive the call.
        from chromadb.api.shared_system_client import SharedSystemClient

        chroma_path = str(tmp_path / "chroma")
        cfg = _make_config(tmp_path)
        with pytest.raises(RuntimeError, match="does not exist"):
            retrieve("q", test_db, None, chroma_path, cfg)
        assert chroma_path not in SharedSystemClient._identifier_to_system
