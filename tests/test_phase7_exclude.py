"""Tests for pipeline/phase7_exclude.py."""

import csv

import pytest

from pipeline.config import Config, PathsConfig
from pipeline.phase7_exclude import run_phase7_exclude, run_phase7_exclude_batch


class TestExclude:
    def _make_config(self, root):
        return Config(paths=PathsConfig(corpus_root=str(root), cache_root=str(root / ".rag-cache")))

    def test_exclude_file(self, test_db, test_corpus):
        root, paths = test_corpus
        cfg = self._make_config(root)
        # Ensure file row exists
        fpath = str(paths["readme.txt"].resolve())
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
                "path": fpath,
                "rel_path": "readme.txt",
                "name": "readme.txt",
                "size_bytes": 100,
                "mtime": "2026-01-01",
                "hash_status": "done",
                "identify_status": "done",
                "triage_status": "done",
                "excluded": 0,
            },
            ignore=True,
        )
        row = next(test_db["file"].rows_where("path = ?", [fpath]), None)
        assert row is not None
        run_phase7_exclude(test_db, cfg, "file", row["id"], "test reason")
        updated = test_db["file"].get(row["id"])
        assert updated["excluded"] == 1
        assert updated["exclusion_reason"] == "test reason"

    def test_exclude_folder(self, test_db, test_corpus):
        root, paths = test_corpus
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
        folder = next(test_db["folder"].rows_where("path = ?", [str(root)]), None)
        assert folder is not None
        run_phase7_exclude(test_db, cfg, "folder", folder["id"], "test reason")
        updated = test_db["folder"].get(folder["id"])
        assert updated["excluded"] == 1

    def test_exclude_nonexistent(self, test_db, test_corpus):
        root, _ = test_corpus
        cfg = self._make_config(root)
        with pytest.raises(ValueError, match="not found"):
            run_phase7_exclude(test_db, cfg, "file", 99999, "test reason")


class TestExcludeBatch:
    def _make_config(self, root):
        return Config(paths=PathsConfig(corpus_root=str(root), cache_root=str(root / ".rag-cache")))

    def test_batch_csv(self, test_db, test_corpus, tmp_path):
        root, paths = test_corpus
        cfg = self._make_config(root)
        # Ensure folder and file rows exist
        test_db["folder"].insert(
            {
                "path": str(root),
                "rel_path": ".",
                "parent_id": None,
                "name": "root",
                "depth": 0,
            },
            replace=True,
        )
        folder = next(test_db["folder"].rows_where("path = ?", [str(root)]))
        fpath = str(paths["readme.txt"].resolve())
        test_db["file"].insert(
            {
                "folder_id": folder["id"],
                "path": fpath,
                "rel_path": "readme.txt",
                "name": "readme.txt",
                "size_bytes": 100,
                "mtime": "2026-01-01",
                "hash_status": "done",
                "identify_status": "done",
                "triage_status": "done",
                "excluded": 0,
            },
            replace=True,
        )
        file_row = next(test_db["file"].rows_where("path = ?", [fpath]), None)
        assert file_row is not None

        csv_path = tmp_path / "exclusions.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["target", "id", "reason"])
            writer.writerow(["file", file_row["id"], "batch exclude"])
            writer.writerow(["file", 99999, "nonexistent"])  # Should be skipped

        count, skipped = run_phase7_exclude_batch(test_db, cfg, csv_path)
        assert count == 1
        assert skipped == 1
        updated = test_db["file"].get(file_row["id"])
        assert updated["excluded"] == 1

    def test_batch_empty_csv(self, test_db, test_corpus, tmp_path):
        root, _ = test_corpus
        cfg = self._make_config(root)
        csv_path = tmp_path / "empty.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            f.write("target,id,reason\n")
        count, skipped = run_phase7_exclude_batch(test_db, cfg, csv_path)
        assert count == 0
        assert skipped == 0
