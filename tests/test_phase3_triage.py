"""Tests for pipeline/phase3_triage.py."""

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline.config import Config, PathsConfig, TriageConfig
from pipeline.phase3_triage import (
    HAS_LINGUA,
    _detect_language,
    _triage_image,
    _triage_media,
    _triage_pdf,
    run_phase3,
)


class TestTriagePdf:
    def test_text_pdf(self, test_corpus):
        root, paths = test_corpus
        cfg = Config(paths=PathsConfig(corpus_root=str(root), cache_root=str(root / ".rag-cache")))
        result = _triage_pdf(str(paths["hello.pdf"]), cfg)
        assert result["is_corrupt"] == 0
        assert result["is_encrypted"] == 0
        assert result["page_count"] == 1
        assert result["has_text_layer"] == 1
        assert result["category_override"] == "pdf-text"

    def test_scanned_pdf(self, test_corpus):
        root, paths = test_corpus
        cfg = Config(paths=PathsConfig(corpus_root=str(root), cache_root=str(root / ".rag-cache")))
        result = _triage_pdf(str(paths["scanned.pdf"]), cfg)
        assert result["is_corrupt"] == 0
        # Scanned PDF has image but no text
        assert result["has_text_layer"] == 0
        assert result["category_override"] == "pdf-scanned"

    def test_encrypted_pdf(self, test_corpus):
        root, paths = test_corpus
        cfg = Config(paths=PathsConfig(corpus_root=str(root), cache_root=str(root / ".rag-cache")))
        result = _triage_pdf(str(paths["encrypted.pdf"]), cfg)
        assert result["is_encrypted"] == 1

    def test_empty_pdf(self, test_corpus):
        root, paths = test_corpus
        cfg = Config(paths=PathsConfig(corpus_root=str(root), cache_root=str(root / ".rag-cache")))
        result = _triage_pdf(str(paths["empty.pdf"]), cfg)
        assert result["page_count"] == 1  # Minimal blank page
        assert result["has_text_layer"] == 0
        assert result["category_override"] == "pdf-scanned"

    def test_corrupt_pdf(self, tmp_path):
        f = tmp_path / "bad.pdf"
        f.write_bytes(b"not a pdf")
        cfg = Config(
            paths=PathsConfig(corpus_root=str(tmp_path), cache_root=str(tmp_path / ".rag-cache"))
        )
        result = _triage_pdf(str(f), cfg)
        assert result["is_corrupt"] == 1

    def test_triage_json_valid(self, test_corpus):
        root, paths = test_corpus
        cfg = Config(paths=PathsConfig(corpus_root=str(root), cache_root=str(root / ".rag-cache")))
        result = _triage_pdf(str(paths["hello.pdf"]), cfg)
        assert result["triage_json"] is not None
        data = json.loads(result["triage_json"])
        assert "avg_chars_per_sampled_page" in data


class TestTriageMedia:
    def test_ffprobe_not_found(self, tmp_path):
        f = tmp_path / "test.mp4"
        f.write_bytes(b"\x00" * 100)
        with patch("shutil.which", return_value=None):
            result = _triage_media(str(f))
            assert result["duration_seconds"] is None
            assert "ffprobe not found" in result["triage_json"]

    def test_corrupt_media(self, tmp_path):
        f = tmp_path / "bad.mp4"
        f.write_bytes(b"not media")
        if shutil.which("ffprobe"):
            result = _triage_media(str(f))
            assert result["is_corrupt"] == 1


class TestTriageImage:
    def test_valid_image(self, test_corpus):
        root, paths = test_corpus
        result = _triage_image(str(paths["image.png"]))
        assert result["is_corrupt"] == 0
        assert result["width"] == 1
        assert result["height"] == 1

    def test_corrupt_image(self, tmp_path):
        f = tmp_path / "bad.png"
        f.write_bytes(b"\x00" * 10)
        result = _triage_image(str(f))
        assert result["is_corrupt"] == 1


class TestDetectLanguage:
    @pytest.mark.skipif(not HAS_LINGUA, reason="lingua not installed")
    def test_detects_english(self, test_corpus):
        root, paths = test_corpus
        lang = _detect_language(str(paths["readme.txt"]), "document")
        assert lang is not None
        assert "ENGLISH" in lang

    def test_no_lingua(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello world " * 20, encoding="utf-8")
        with patch("pipeline.phase3_triage.HAS_LINGUA", False):
            result = _detect_language(str(f), "document")
            assert result is None

    def test_short_text(self, tmp_path):
        f = tmp_path / "short.txt"
        f.write_text("Hi", encoding="utf-8")
        if HAS_LINGUA:
            result = _detect_language(str(f), "document")
            assert result is None  # Too short

    def test_non_text_category(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello world " * 20, encoding="utf-8")
        if HAS_LINGUA:
            result = _detect_language(str(f), "image")
            assert result is None  # Not a text-bearing category


class TestPhase3Run:
    def _make_config(self, root):
        return Config(
            paths=PathsConfig(corpus_root=str(root), cache_root=str(root / ".rag-cache")),
            triage=TriageConfig(),
        )

    def _insert_file(self, db, path, category, db_path):
        import sqlite_utils

        main_db = sqlite_utils.Database(db_path)
        main_db.conn.execute("PRAGMA foreign_keys = ON")
        main_db["folder"].insert(
            {
                "path": str(Path(path).parent),
                "rel_path": ".",
                "parent_id": None,
                "name": Path(path).parent.name or "root",
                "depth": 0,
            },
            ignore=True,
        )
        folder_row = next(main_db["folder"].rows_where("path = ?", [str(Path(path).parent)]), None)
        folder_id = folder_row["id"] if folder_row else 1
        main_db["file"].insert(
            {
                "folder_id": folder_id,
                "path": path,
                "rel_path": Path(path).name,
                "name": Path(path).name,
                "size_bytes": 100,
                "mtime": "2026-01-01T00:00:00Z",
                "sha256": "abc123",
                "hash_status": "done",
                "identify_status": "done",
                "category": category,
                "extract_strategy": "docling",
                "triage_status": "pending",
                "excluded": 0,
                "is_dup_primary": 1,
            }
        )
        main_db.conn.commit()
        main_db.conn.close()

    def test_triage_pdf_files(self, test_db, test_corpus, mock_plog):
        root, paths = test_corpus
        cfg = self._make_config(root)
        # Use the DB path from test_db fixture
        db_path = str(test_db.path)
        pdf_path = str(paths["hello.pdf"])
        self._insert_file(test_db, pdf_path, "pdf-text", db_path)
        result = run_phase3(test_db, cfg, mock_plog, workers=1, db_path=db_path)
        assert result["files_processed"] >= 1

    def test_triage_no_files(self, test_db, test_corpus, mock_plog):
        root, _ = test_corpus
        cfg = self._make_config(root)
        db_path = str(test_db.path)
        result = run_phase3(test_db, cfg, mock_plog, workers=1, db_path=db_path)
        assert result["files_processed"] == 0
        assert result["files_failed"] == 0

    def test_triage_category_filter(self, test_db, test_corpus, mock_plog):
        root, paths = test_corpus
        cfg = self._make_config(root)
        db_path = str(test_db.path)
        pdf_path = str(paths["hello.pdf"])
        self._insert_file(test_db, pdf_path, "pdf-text", db_path)
        result = run_phase3(
            test_db, cfg, mock_plog, workers=1, categories=["pdf-text"], db_path=db_path
        )
        assert result["files_processed"] >= 1

    def test_triage_file_not_found(self, test_db, test_corpus, mock_plog):
        root, _ = test_corpus
        cfg = self._make_config(root)
        db_path = str(test_db.path)
        self._insert_file(test_db, "/nonexistent/file.pdf", "pdf-text", db_path)
        result = run_phase3(test_db, cfg, mock_plog, workers=1, db_path=db_path)
        assert result["files_failed"] >= 1
