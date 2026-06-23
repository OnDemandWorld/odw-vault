"""Tests for pipeline/phase2_identify.py."""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.config import Config, IdentifyConfig, PathsConfig
from pipeline.phase2_identify import _apply_extension_fallback, run_phase2


class TestExtensionFallback:
    def test_known_extension(self):
        puid, cat, strategy = _apply_extension_fallback("/tmp/test.log")
        assert cat == "document"
        assert strategy == "tika"

    def test_unknown_extension(self):
        puid, cat, strategy = _apply_extension_fallback("/tmp/test.xyz")
        assert cat is None
        assert strategy is None

    def test_pak_file(self):
        puid, cat, strategy = _apply_extension_fallback("/tmp/data.pak")
        assert cat == "data"
        assert strategy == "filename-only"

    def test_qm_file(self):
        puid, cat, strategy = _apply_extension_fallback("/tmp/trans.qm")
        assert cat == "data"
        assert strategy == "filename-only"

    def test_drawio_file(self):
        puid, cat, strategy = _apply_extension_fallback("/tmp/diagram.drawio")
        assert cat == "data"
        assert strategy == "filename-only"

    def test_bin_file(self):
        puid, cat, strategy = _apply_extension_fallback("/tmp/app.bin")
        assert cat == "executable"
        assert strategy == "skip"

    def test_sqlite_file(self):
        puid, cat, strategy = _apply_extension_fallback("/tmp/db.sqlite")
        assert cat == "data"
        assert strategy == "filename-only"


class TestPhase2Identify:
    def _make_config(self, root):
        return Config(
            paths=PathsConfig(corpus_root=str(root), cache_root=str(root / ".rag-cache")),
            identify=IdentifyConfig(siegfried_path="sf"),
        )

    def _mock_sf(self, file_paths):
        files = []
        for fp in file_paths:
            ext = Path(fp).suffix.lower()
            mapping = {
                ".txt": ("x-fmt/111", "Plain Text File", "text/plain"),
                ".csv": ("x-fmt/18", "Comma Separated Values", "text/csv"),
                ".json": ("fmt/817", "JSON", "application/json"),
                ".pdf": ("fmt/16", "Acrobat PDF 1.2", "application/pdf"),
                ".png": ("fmt/13", "Portable Network Graphics", "image/png"),
                ".docx": (
                    "fmt/412",
                    "Microsoft Word OOXML",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ),
                ".zip": ("fmt/189", "ZIP Format", "application/zip"),
            }
            info = mapping.get(ext, (None, None, None))
            matches = []
            if info[0]:
                matches.append(
                    {
                        "ns": "pronom",
                        "id": info[0],
                        "format": info[1],
                        "mime": info[2],
                        "version": "",
                    }
                )
            files.append(
                {
                    "filename": fp,
                    "filesize": 100,
                    "modified": "2026-01-01T00:00:00Z",
                    "matches": matches,
                }
            )
        return json.dumps({"siegfried": "v1.11.4", "files": files})

    def _ensure_file_in_db(self, db, path):
        """Insert folder and file rows for phase2 testing."""
        p = Path(path)
        db["folder"].insert(
            {
                "path": str(p.parent.resolve()),
                "rel_path": ".",
                "parent_id": None,
                "name": p.parent.name or "root",
                "depth": 0,
            },
            ignore=True,
        )
        folder_row = next(db["folder"].rows_where("path = ?", [str(p.parent.resolve())]), None)
        folder_id = folder_row["id"] if folder_row else 1
        db["file"].insert(
            {
                "folder_id": folder_id,
                "path": str(p.resolve()),
                "rel_path": p.name,
                "name": p.name,
                "size_bytes": p.stat().st_size,
                "mtime": "2026-01-01T00:00:00Z",
                "sha256": "abc123",
                "hash_status": "done",
                "identify_status": "pending",
                "triage_status": "pending",
                "excluded": 0,
            },
            ignore=True,
        )

    def test_identifies_all_files(self, test_db, test_corpus, mock_plog):
        root, paths = test_corpus
        cfg = self._make_config(root)
        all_files = [str(p.resolve()) for p in paths.values() if p.exists()]

        for fp in all_files:
            self._ensure_file_in_db(test_db, fp)

        with patch("shutil.which", return_value="/usr/bin/sf"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0, stdout=self._mock_sf(all_files), stderr=""
                )
                result = run_phase2(test_db, cfg, mock_plog)
                assert result["files_processed"] >= 5

    def test_identify_pdf(self, test_db, test_corpus, mock_plog):
        root, paths = test_corpus
        cfg = self._make_config(root)
        pdf_path = str(paths["hello.pdf"].resolve())
        self._ensure_file_in_db(test_db, pdf_path)

        with patch("shutil.which", return_value="/usr/bin/sf"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0, stdout=self._mock_sf([pdf_path]), stderr=""
                )
                run_phase2(test_db, cfg, mock_plog)
                f = next(test_db["file"].rows_where("path = ?", [pdf_path]), None)
                assert f is not None
                assert f["pronom_id"] == "fmt/16"

    def test_unknown_format_gets_fallback(self, test_db, tmp_path, mock_plog):
        root = tmp_path / "corpus"
        root.mkdir()
        f = root / "unknown.xyz"
        f.write_text("content", encoding="utf-8")
        cfg = self._make_config(root)

        # Insert folder and file rows so phase2 can find them
        test_db["folder"].insert(
            {
                "path": str(root),
                "rel_path": ".",
                "parent_id": None,
                "name": "root",
                "depth": 0,
            },
            ignore=True,
        )
        test_db["file"].insert(
            {
                "folder_id": 1,
                "path": str(f.resolve()),
                "rel_path": "unknown.xyz",
                "name": "unknown.xyz",
                "size_bytes": 7,
                "mtime": "2026-01-01T00:00:00Z",
                "sha256": "abc123",
                "hash_status": "done",
                "identify_status": "pending",
                "triage_status": "pending",
                "excluded": 0,
            },
            ignore=True,
        )

        with patch("shutil.which", return_value="/usr/bin/sf"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "siegfried": "v1.11.4",
                            "files": [
                                {
                                    "filename": str(f.resolve()),
                                    "filesize": 7,
                                    "modified": "2026-01-01",
                                    "matches": [],
                                }
                            ],
                        }
                    ),
                    stderr="",
                )
                run_phase2(test_db, cfg, mock_plog)
                row = next(test_db["file"].rows_where("path = ?", [str(f.resolve())]))
                assert row["category"] == "unknown"
                assert row["extract_strategy"] == "manual"

    def test_extension_fallback_applied(self, test_db, tmp_path, mock_plog):
        root = tmp_path / "corpus"
        root.mkdir()
        f = root / "app.pak"
        f.write_bytes(b"\x00" * 100)
        cfg = self._make_config(root)

        test_db["folder"].insert(
            {
                "path": str(root),
                "rel_path": ".",
                "parent_id": None,
                "name": "root",
                "depth": 0,
            },
            ignore=True,
        )
        test_db["file"].insert(
            {
                "folder_id": 1,
                "path": str(f.resolve()),
                "rel_path": "app.pak",
                "name": "app.pak",
                "size_bytes": 100,
                "mtime": "2026-01-01T00:00:00Z",
                "sha256": "abc123",
                "hash_status": "done",
                "identify_status": "pending",
                "triage_status": "pending",
                "excluded": 0,
            },
            ignore=True,
        )

        with patch("shutil.which", return_value="/usr/bin/sf"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "siegfried": "v1.11.4",
                            "files": [
                                {
                                    "filename": str(f.resolve()),
                                    "filesize": 100,
                                    "modified": "2026-01-01",
                                    "matches": [
                                        {
                                            "id": "UNKNOWN",
                                            "format": "Unknown",
                                            "mime": "",
                                            "version": "",
                                        }
                                    ],
                                }
                            ],
                        }
                    ),
                    stderr="",
                )
                run_phase2(test_db, cfg, mock_plog)
                row = next(test_db["file"].rows_where("path = ?", [str(f.resolve())]))
                assert row["pronom_id"] == "UNKNOWN-pak"
                assert row["category"] == "data"

    def test_sf_not_found(self, test_db, tmp_path, mock_plog):
        root = tmp_path / "corpus"
        root.mkdir()
        (root / "test.txt").write_text("hello", encoding="utf-8")
        cfg = self._make_config(root)
        # Use a name that won't exist anywhere
        cfg.identify.siegfried_path = "nonexistent-sf-binary"

        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="Siegfried not found"):
                run_phase2(test_db, cfg, mock_plog)

    def test_sf_json_parse_error(self, test_db, test_corpus, mock_plog):
        root, paths = test_corpus
        cfg = self._make_config(root)
        pdf_path = str(paths["hello.pdf"].resolve())
        self._ensure_file_in_db(test_db, pdf_path)

        with patch("shutil.which", return_value="/usr/bin/sf"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="not json", stderr="")
                with pytest.raises(RuntimeError, match="JSON parse error"):
                    run_phase2(test_db, cfg, mock_plog)

    def test_sf_nonzero_stderr(self, test_db, test_corpus, mock_plog):
        root, paths = test_corpus
        cfg = self._make_config(root)
        pdf_path = str(paths["hello.pdf"].resolve())
        self._ensure_file_in_db(test_db, pdf_path)

        with patch("shutil.which", return_value="/usr/bin/sf"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
                with pytest.raises(RuntimeError, match="Siegfried failed"):
                    run_phase2(test_db, cfg, mock_plog)

    def test_sf_timeout(self, test_db, test_corpus, mock_plog):
        root, paths = test_corpus
        cfg = self._make_config(root)
        pdf_path = str(paths["hello.pdf"].resolve())
        self._ensure_file_in_db(test_db, pdf_path)

        with patch("shutil.which", return_value="/usr/bin/sf"):
            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = subprocess.TimeoutExpired(cmd="sf", timeout=3600)
                with pytest.raises(RuntimeError, match="Siegfried timeout"):
                    run_phase2(test_db, cfg, mock_plog)
