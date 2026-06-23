"""Tests for pipeline/phase0_archives.py."""

from unittest.mock import patch

from pipeline.config import Config, PathsConfig
from pipeline.phase0_archives import _find_archives, run_phase0


class TestFindArchives:
    def test_finds_zip(self, test_corpus):
        root, _ = test_corpus
        archives = _find_archives(
            root,
            Config(paths=PathsConfig(corpus_root=str(root), cache_root=str(root / ".rag-cache"))),
        )
        zip_paths = [a for a in archives if a.suffix == ".zip"]
        assert len(zip_paths) >= 2  # archive.zip and nested.zip

    def test_skips_docx(self, test_corpus):
        root, _ = test_corpus
        config = Config(
            paths=PathsConfig(corpus_root=str(root), cache_root=str(root / ".rag-cache"))
        )
        archives = _find_archives(root, config)
        docx = [a for a in archives if a.suffix.lower() == ".docx"]
        assert len(docx) == 0

    def test_skips_cache_dir(self, test_corpus):
        root, _ = test_corpus
        config = Config(
            paths=PathsConfig(corpus_root=str(root), cache_root=str(root / ".rag-cache"))
        )
        archives = _find_archives(root, config)
        cache_files = [a for a in archives if ".rag-cache" in str(a)]
        assert len(cache_files) == 0

    def test_skips_hidden(self, test_corpus):
        root, _ = test_corpus
        config = Config(
            paths=PathsConfig(corpus_root=str(root), cache_root=str(root / ".rag-cache"))
        )
        archives = _find_archives(root, config)
        hidden = [a for a in archives if a.name.startswith(".")]
        assert len(hidden) == 0


class TestPhase0:
    def _make_config(self, root):
        return Config(paths=PathsConfig(corpus_root=str(root), cache_root=str(root / ".rag-cache")))

    def test_expand_archive_success(self, test_db, test_corpus, mock_plog, mock_patool):
        root, paths = test_corpus
        cfg = self._make_config(root)
        result = run_phase0(test_db, cfg, mock_plog, max_depth=1)
        assert result["files_processed"] >= 1
        assert result["files_failed"] == 0
        # Check archive_expansion table
        rows = list(test_db["archive_expansion"].rows_where("succeeded = 1"))
        assert len(rows) >= 1

    def test_expand_dry_run(self, test_db, test_corpus, mock_plog):
        root, _ = test_corpus
        cfg = self._make_config(root)
        result = run_phase0(test_db, cfg, mock_plog, max_depth=1, dry_run=True)
        assert result["files_processed"] >= 1
        # No extraction should have happened
        rows = list(test_db["archive_expansion"].rows)
        assert len(rows) == 0

    def test_idempotent(self, test_db, test_corpus, mock_plog, mock_patool):
        root, _ = test_corpus
        cfg = self._make_config(root)
        r1 = run_phase0(test_db, cfg, mock_plog, max_depth=1)
        r2 = run_phase0(test_db, cfg, mock_plog, max_depth=1)
        # Second run should process 0 archives
        assert r2["files_processed"] == 0

    def test_no_archives(self, test_db, tmp_path, mock_plog):
        root = tmp_path / "empty"
        root.mkdir()
        cfg = self._make_config(root)
        result = run_phase0(test_db, cfg, mock_plog, max_depth=1)
        assert result["files_processed"] == 0
        assert result["files_failed"] == 0

    def test_expand_failure_recorded(self, test_db, test_corpus, mock_plog):
        """If patool raises, failure is recorded."""
        root, _ = test_corpus
        cfg = self._make_config(root)

        def failing_extract(archive_path, outdir, verbosity=-1):
            raise Exception("encrypted archive")

        with patch("patoolib.extract_archive", side_effect=failing_extract):
            result = run_phase0(test_db, cfg, mock_plog, max_depth=1)
            assert result["files_failed"] >= 1
            # Check failure table
            failures = list(test_db["failure"].rows_where("phase = ?", ["archives"]))
            assert len(failures) >= 1

    def test_ensure_file_in_db(self, test_db, test_corpus, mock_plog, mock_patool):
        """Expanded archives should have file rows with hash_status='skipped'."""
        root, paths = test_corpus
        cfg = self._make_config(root)
        run_phase0(test_db, cfg, mock_plog, max_depth=1)
        # Archive files should be in DB with hash_status='skipped'
        zip_files = list(test_db.query("SELECT * FROM file WHERE extension = '.zip'"))
        assert len(zip_files) >= 1
        for f in zip_files:
            assert f["hash_status"] == "skipped"
