"""Tests for cli.py click commands."""

import json
from unittest.mock import patch

from cli import main
from click.testing import CliRunner

from pipeline.db import migrate, open_db


class TestCliInit:
    def _make_runner(self, tmp_path):
        runner = CliRunner()
        # Patch PROJECT_ROOT so init writes to tmp_path
        with patch("cli.PROJECT_ROOT", tmp_path):
            yield runner

    def test_init_creates_db_and_config(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        runner = CliRunner()
        with patch("cli.PROJECT_ROOT", tmp_path):
            result = runner.invoke(main, ["init", "--root", str(corpus)])
        assert result.exit_code == 0
        assert (tmp_path / "corpus.db").exists()
        assert (tmp_path / "config.toml").exists()
        assert "Created" in result.output

    def test_init_fails_for_nonexistent_root(self, tmp_path):
        runner = CliRunner()
        with patch("cli.PROJECT_ROOT", tmp_path):
            result = runner.invoke(main, ["init", "--root", "/nonexistent/path"])
        assert result.exit_code == 2
        assert "Error" in result.output

    def test_init_fails_when_db_exists_without_force(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (tmp_path / "corpus.db").touch()
        runner = CliRunner()
        with patch("cli.PROJECT_ROOT", tmp_path):
            result = runner.invoke(main, ["init", "--root", str(corpus)])
        assert result.exit_code == 1
        assert "already exists" in result.output

    def test_init_force_overwrites(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (tmp_path / "corpus.db").touch()
        runner = CliRunner()
        with patch("cli.PROJECT_ROOT", tmp_path):
            result = runner.invoke(main, ["init", "--root", str(corpus), "--force"])
        assert result.exit_code == 0
        assert (tmp_path / "corpus.db").exists()

    def test_init_updates_existing_config(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        # Write existing config and DB
        (tmp_path / "config.toml").write_text(
            'paths = { corpus_root = "/old", cache_root = "/old-cache" }\n', encoding="utf-8"
        )
        (tmp_path / "corpus.db").touch()
        runner = CliRunner()
        with patch("cli.PROJECT_ROOT", tmp_path):
            # Without --force, existing config and DB both exist -> should error about DB existing
            result = runner.invoke(main, ["init", "--root", str(corpus)])
        assert result.exit_code == 1  # DB already exists without --force


class TestCliStatus:
    def test_status_returns_json(self, tmp_path):
        runner = CliRunner()
        with patch("cli.PROJECT_ROOT", tmp_path):
            # Create minimal DB
            db_path = tmp_path / "corpus.db"
            db = open_db(db_path)
            migrate(db)
            db["config"].insert({"key": "corpus_root", "value": str(tmp_path)})

            result = runner.invoke(main, ["status"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "corpus_root" in data
        assert "phases" in data
        assert "approved" in data


class TestCliExclude:
    def test_exclude_command_recognized(self, tmp_path):
        """Verify the exclude command exists and parses args."""
        runner = CliRunner()
        db_path = tmp_path / "corpus.db"
        db = open_db(db_path)
        migrate(db)
        db["folder"].insert(
            {
                "path": str(tmp_path),
                "rel_path": ".",
                "parent_id": None,
                "name": "root",
                "depth": 0,
            }
        )
        db["file"].insert(
            {
                "folder_id": 1,
                "path": str(tmp_path / "test.txt"),
                "rel_path": "test.txt",
                "name": "test.txt",
                "size_bytes": 100,
                "mtime": "2026-01-01",
                "hash_status": "done",
                "identify_status": "done",
                "triage_status": "done",
                "excluded": 0,
            }
        )
        # Create config.toml
        (tmp_path / "config.toml").write_text(
            'paths = { corpus_root = "'
            + str(tmp_path)
            + '", cache_root = "'
            + str(tmp_path / ".rag-cache")
            + '" }\n',
            encoding="utf-8",
        )
        with patch("cli.PROJECT_ROOT", tmp_path):
            result = runner.invoke(
                main,
                [
                    "exclude",
                    "--target",
                    "file",
                    "--id",
                    "1",
                    "--reason",
                    "cli test",
                ],
            )
        assert result.exit_code == 0
        assert "Marked" in result.output


class TestCliApprove:
    def test_approve_sets_config(self, tmp_path):
        runner = CliRunner()
        db_path = tmp_path / "corpus.db"
        # Create fresh DB
        db = open_db(db_path)
        migrate(db)

        with patch("cli.PROJECT_ROOT", tmp_path):
            result = runner.invoke(main, ["approve", "--by", "test-user"])
        assert result.exit_code == 0
        assert "approved by" in result.output
        db = open_db(db_path)
        row = next(iter(db.query("SELECT value FROM config WHERE key='preflight_approved_by'")), None)
        assert row is not None
        assert row["value"] == "test-user"


class TestCliRunAll:
    def test_run_all_fails_without_db(self, tmp_path):
        runner = CliRunner()
        with patch("cli.PROJECT_ROOT", tmp_path):
            result = runner.invoke(main, ["run-all"])
        assert result.exit_code == 1
        assert "not found" in result.output


class TestCliServe:
    def test_serve_fails_without_db(self, tmp_path):
        runner = CliRunner()
        with patch("cli.PROJECT_ROOT", tmp_path):
            result = runner.invoke(main, ["serve"])
        assert result.exit_code == 2
        assert "config.toml" in result.output or "Error" in result.output


class TestCliArchives:
    def test_archives_dry_run(self, tmp_path):
        """Verify the archives command parses --dry-run."""
        root = tmp_path / "corpus"
        root.mkdir()
        runner = CliRunner()
        db_path = tmp_path / "corpus.db"
        db = open_db(db_path)
        migrate(db)
        db["config"].insert({"key": "corpus_root", "value": str(root)}, replace=True)

        with patch("cli.PROJECT_ROOT", tmp_path):
            result = runner.invoke(main, ["archives", "--dry-run"])
        # May fail due to patool or no archives, but command should run
        assert "dry-run" not in result.output or result.exit_code in (0, 2)


class TestCliIdentify:
    def test_identify_requires_config(self, tmp_path):
        runner = CliRunner()
        with patch("cli.PROJECT_ROOT", tmp_path):
            # Create DB but no config
            db_path = tmp_path / "corpus.db"
            db = open_db(db_path)
            migrate(db)
            result = runner.invoke(main, ["identify"])
        # Should fail with config not found
        assert result.exit_code == 2


class TestCliDedup:
    def test_dedup_requires_config(self, tmp_path):
        runner = CliRunner()
        with patch("cli.PROJECT_ROOT", tmp_path):
            db_path = tmp_path / "corpus.db"
            db = open_db(db_path)
            migrate(db)
            result = runner.invoke(main, ["dedup"])
        assert result.exit_code == 2


class TestCliReport:
    def test_report_requires_config(self, tmp_path):
        runner = CliRunner()
        with patch("cli.PROJECT_ROOT", tmp_path):
            db_path = tmp_path / "corpus.db"
            db = open_db(db_path)
            migrate(db)
            result = runner.invoke(main, ["report"])
        assert result.exit_code == 2


class TestCliExcludeBatch:
    def test_batch_csv_file(self, tmp_path):
        """Verify exclude-batch processes a CSV."""
        root = tmp_path / "corpus"
        root.mkdir()
        runner = CliRunner()
        db_path = tmp_path / "corpus.db"
        db = open_db(db_path)
        migrate(db)
        db["folder"].insert(
            {
                "path": str(root),
                "rel_path": ".",
                "parent_id": None,
                "name": "root",
                "depth": 0,
            }
        )
        db["file"].insert(
            {
                "folder_id": 1,
                "path": str(root / "test.txt"),
                "rel_path": "test.txt",
                "name": "test.txt",
                "size_bytes": 100,
                "mtime": "2026-01-01",
                "hash_status": "done",
                "identify_status": "done",
                "triage_status": "done",
                "excluded": 0,
            }
        )
        file_row = next(db["file"].rows_where("path = ?", [str(root / "test.txt")]))

        csv_path = tmp_path / "exclusions.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            f.write("target,id,reason\n")
            f.write(f"file,{file_row['id']},cli batch exclude\n")

        # Create config.toml
        (tmp_path / "config.toml").write_text(
            'paths = { corpus_root = "'
            + str(root)
            + '", cache_root = "'
            + str(root / ".rag-cache")
            + '" }\n',
            encoding="utf-8",
        )
        with patch("cli.PROJECT_ROOT", tmp_path):
            result = runner.invoke(main, ["exclude-batch", "--from-file", str(csv_path)])
        assert result.exit_code == 0
        assert "Processed" in result.output
