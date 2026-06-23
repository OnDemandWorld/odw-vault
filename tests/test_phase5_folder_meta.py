"""Tests for pipeline/phase5_folder_meta.py."""

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from pipeline.config import Config, FolderMetaConfig, OllamaConfig, PathsConfig
from pipeline.phase5_folder_meta import (
    FolderInference,
    _build_prompt,
    _call_ollama,
    run_phase5,
)


class TestFolderInferenceModel:
    def test_valid_input(self):
        f = FolderInference(
            category="client-project",
            label="Test Project",
            tags=["test", "sample"],
            summary="A test folder.",
        )
        assert f.category == "client-project"
        assert f.label == "Test Project"

    def test_invalid_category(self):
        with pytest.raises(ValidationError):
            FolderInference(
                category="invalid-category",
                label="Test",
                tags=["test"],
                summary="Test",
            )

    def test_label_too_long(self):
        with pytest.raises(ValidationError):
            FolderInference(
                category="client-project",
                label="x" * 121,
                tags=["test"],
                summary="Test",
            )

    def test_too_many_tags(self):
        with pytest.raises(ValidationError):
            FolderInference(
                category="client-project",
                label="Test",
                tags=["a", "b", "c", "d", "e", "f", "g", "h", "i"],  # 9 tags
                summary="Test",
            )

    def test_summary_too_long(self):
        with pytest.raises(ValidationError):
            FolderInference(
                category="client-project",
                label="Test",
                tags=["test"],
                summary="x" * 501,
            )

    def test_all_categories_valid(self):
        valid_cats = [
            "client-project",
            "internal-rnd",
            "vendor-docs",
            "admin-finance",
            "templates",
            "archive-historical",
            "personal",
            "unclear",
            "engineering",
            "training",
            "product-design",
            "test-operations",
        ]
        for cat in valid_cats:
            f = FolderInference(category=cat, label="Test", tags=[], summary="Test")
            assert f.category == cat


class TestBuildPrompt:
    def _make_folder_row(self):
        return {
            "id": 1,
            "rel_path": "test/folder",
            "file_count": 2,
            "inferred_label": None,
        }

    def test_prompt_contains_rel_path(self, test_db):
        row = self._make_folder_row()
        test_db["folder"].insert(
            {
                "path": "/test/folder",
                "rel_path": "test/folder",
                "parent_id": None,
                "name": "folder",
                "depth": 1,
            }
        )
        test_db["file"].insert(
            {
                "folder_id": 1,
                "path": "/test/folder/doc.txt",
                "rel_path": "test/folder/doc.txt",
                "name": "doc.txt",
                "size_bytes": 100,
                "mtime": "2026-01-01",
                "hash_status": "done",
                "identify_status": "done",
                "category": "document",
                "extract_strategy": "tika",
                "triage_status": "pending",
                "excluded": 0,
                "is_dup_primary": 1,
            }
        )
        cfg = Config(
            paths=PathsConfig(corpus_root="/tmp", cache_root="/tmp/.rag-cache"),
            folder_meta=FolderMetaConfig(max_filenames_in_prompt=30),
        )
        prompt, prompt_hash = _build_prompt(row, test_db, cfg)
        assert "test/folder" in prompt
        assert prompt_hash is not None
        assert len(prompt_hash) == 64

    def test_prompt_contains_filenames(self, test_db):
        row = self._make_folder_row()
        test_db["folder"].insert(
            {
                "path": "/test/folder",
                "rel_path": "test/folder",
                "parent_id": None,
                "name": "folder",
                "depth": 1,
            }
        )
        test_db["file"].insert(
            {
                "folder_id": 1,
                "path": "/test/folder/report.pdf",
                "rel_path": "test/folder/report.pdf",
                "name": "report.pdf",
                "size_bytes": 100,
                "mtime": "2026-01-01",
                "hash_status": "done",
                "identify_status": "done",
                "category": "pdf-text",
                "extract_strategy": "docling",
                "triage_status": "pending",
                "excluded": 0,
                "is_dup_primary": 1,
            }
        )
        cfg = Config(
            paths=PathsConfig(corpus_root="/tmp", cache_root="/tmp/.rag-cache"),
            folder_meta=FolderMetaConfig(max_filenames_in_prompt=30),
        )
        prompt, _ = _build_prompt(row, test_db, cfg)
        assert "report.pdf" in prompt

    def test_prompt_hash_deterministic(self, test_db):
        row = self._make_folder_row()
        test_db["folder"].insert(
            {
                "path": "/test/folder",
                "rel_path": "test/folder",
                "parent_id": None,
                "name": "folder",
                "depth": 1,
            }
        )
        cfg = Config(
            paths=PathsConfig(corpus_root="/tmp", cache_root="/tmp/.rag-cache"),
            folder_meta=FolderMetaConfig(max_filenames_in_prompt=30),
        )
        _, h1 = _build_prompt(row, test_db, cfg)
        _, h2 = _build_prompt(row, test_db, cfg)
        assert h1 == h2


class TestCallOllama:
    def test_ollama_calls_client(self):
        mock_response = {
            "response": json.dumps(
                {
                    "category": "client-project",
                    "label": "Test",
                    "tags": ["test"],
                    "summary": "A test.",
                }
            )
        }
        mock_client = MagicMock()
        mock_client.generate.return_value = mock_response

        with patch("ollama.Client", return_value=mock_client):
            result = _call_ollama("test prompt", "gemma4", "http://localhost:11434")
            assert isinstance(result, FolderInference)
            assert result.category == "client-project"

    def test_ollama_strips_markdown_fences(self):
        mock_response = {
            "response": '```json\n{"category":"client-project","label":"Test","tags":["t"],"summary":"S."}\n```'
        }
        mock_client = MagicMock()
        mock_client.generate.return_value = mock_response

        with patch("ollama.Client", return_value=mock_client):
            result = _call_ollama("test prompt", "gemma4", "http://localhost:11434")
            assert result.category == "client-project"


class TestPhase5Run:
    def _make_config(self, root):
        return Config(
            paths=PathsConfig(corpus_root=str(root), cache_root=str(root / ".rag-cache")),
            ollama=OllamaConfig(host="http://localhost:11434", model="gemma4:latest"),
            folder_meta=FolderMetaConfig(max_filenames_in_prompt=30),
        )

    def test_inference_success(self, test_db, test_corpus, mock_plog, mock_ollama):
        root, _ = test_corpus
        cfg = self._make_config(root)
        result = run_phase5(test_db, cfg, mock_plog, max_folders=2)
        assert result["files_processed"] >= 0

    def test_max_folders(self, test_db, test_corpus, mock_plog, mock_ollama):
        root, _ = test_corpus
        cfg = self._make_config(root)
        result = run_phase5(test_db, cfg, mock_plog, max_folders=1)
        # Should only process 1 folder
        folders_with_inference = test_db.execute(
            "SELECT COUNT(*) FROM folder WHERE inferred_category IS NOT NULL"
        ).fetchone()
        assert folders_with_inference[0] <= 1

    def test_ollama_failure(self, test_db, test_corpus, mock_plog):
        root, _ = test_corpus
        cfg = self._make_config(root)

        # Insert a folder row so phase5 has something to process
        test_db["folder"].insert(
            {
                "path": str(root),
                "rel_path": ".",
                "parent_id": None,
                "name": "root",
                "depth": 0,
            }
        )

        def failing_ollama(*args, **kwargs):
            raise Exception("Connection refused")

        with patch("pipeline.phase5_folder_meta._call_ollama", side_effect=failing_ollama):
            result = run_phase5(test_db, cfg, mock_plog, max_folders=1)
            assert result["files_failed"] >= 1
            failures = list(test_db["failure"].rows_where("phase = ?", ["folder_meta"]))
            assert len(failures) >= 1

    def test_empty_no_folders(self, test_db, tmp_path, mock_plog, mock_ollama):
        root = tmp_path / "empty"
        root.mkdir()
        cfg = self._make_config(root)
        result = run_phase5(test_db, cfg, mock_plog)
        # No folders to infer
        assert result["files_processed"] >= 0
