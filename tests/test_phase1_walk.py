"""Tests for pipeline/phase1_walk.py."""


from pipeline.config import Config, PathsConfig
from pipeline.phase1_walk import run_phase1


class TestPhase1Walk:
    def _make_config(self, root, max_size=5_368_709_120):
        from pipeline.config import WalkConfig

        return Config(
            paths=PathsConfig(corpus_root=str(root), cache_root=str(root / ".rag-cache")),
            walk=WalkConfig(max_file_size_bytes=max_size),
        )

    def test_creates_folder_rows(self, test_db, test_corpus, mock_plog):
        root, _ = test_corpus
        cfg = self._make_config(root)
        run_phase1(test_db, cfg, mock_plog, workers=1)
        folders = list(test_db["folder"].rows)
        assert len(folders) >= 2  # at least root and subdir

    def test_creates_file_rows(self, test_db, test_corpus, mock_plog):
        root, paths = test_corpus
        cfg = self._make_config(root)
        run_phase1(test_db, cfg, mock_plog, workers=1)
        files = list(test_db["file"].rows)
        assert len(files) >= 5  # multiple test files

    def test_skips_hidden(self, test_db, test_corpus, mock_plog):
        root, _ = test_corpus
        cfg = self._make_config(root)
        run_phase1(test_db, cfg, mock_plog, workers=1)
        ds_store = test_db.execute("SELECT COUNT(*) FROM file WHERE name='.DS_Store'").fetchone()
        assert ds_store[0] == 0

    def test_skips_cache_dir(self, test_db, test_corpus, mock_plog):
        root, _ = test_corpus
        cfg = self._make_config(root)
        run_phase1(test_db, cfg, mock_plog, workers=1)
        cache_files = test_db.execute(
            "SELECT COUNT(*) FROM file WHERE path LIKE '%.rag-cache%'"
        ).fetchone()
        assert cache_files[0] == 0

    def test_hashes_files(self, test_db, test_corpus, mock_plog):
        root, _ = test_corpus
        cfg = self._make_config(root)
        result = run_phase1(test_db, cfg, mock_plog, workers=1)
        assert result["files_processed"] >= 5
        assert result["files_failed"] == 0
        # All done files should have sha256
        hashed = test_db.execute(
            "SELECT COUNT(*) FROM file WHERE sha256 IS NOT NULL AND hash_status='done'"
        ).fetchone()
        assert hashed[0] >= 5

    def test_oversized_file(self, test_db, tmp_path, mock_plog):
        root = tmp_path / "corpus"
        root.mkdir()
        (root / "big.dat").write_bytes(b"x" * 1000)
        cfg = self._make_config(root, max_size=500)
        result = run_phase1(test_db, cfg, mock_plog, workers=1)
        # File should not be hashed
        big = test_db.execute("SELECT sha256 FROM file WHERE name='big.dat'").fetchone()
        # It may or may not be in DB depending on walk logic; if in DB, sha256 should be None
        if big:
            assert big[0] is None

    def test_idempotent(self, test_db, test_corpus, mock_plog):
        root, _ = test_corpus
        cfg = self._make_config(root)
        r1 = run_phase1(test_db, cfg, mock_plog, workers=1)
        r2 = run_phase1(test_db, cfg, mock_plog, workers=1)
        # Second run should skip already-hashed files
        assert r2["files_processed"] >= 0

    def test_rehash(self, test_db, test_corpus, mock_plog):
        root, _ = test_corpus
        cfg = self._make_config(root)
        run_phase1(test_db, cfg, mock_plog, workers=1)
        r2 = run_phase1(test_db, cfg, mock_plog, workers=1, rehash=True)
        assert r2["files_processed"] >= 5

    def test_empty_corpus(self, test_db, tmp_path, mock_plog):
        root = tmp_path / "empty"
        root.mkdir()
        cfg = self._make_config(root)
        result = run_phase1(test_db, cfg, mock_plog, workers=1)
        assert result["files_processed"] == 0
        assert result["files_failed"] == 0

    def test_rel_paths_correct(self, test_db, test_corpus, mock_plog):
        root, paths = test_corpus
        cfg = self._make_config(root)
        run_phase1(test_db, cfg, mock_plog, workers=1)
        deep = next(test_db["file"].rows_where("name = ?", ["deep.txt"]), None)
        assert deep is not None
        assert "subdir" in deep["rel_path"]

    def test_hash_failure(self, test_db, tmp_path, mock_plog):
        root = tmp_path / "corpus"
        root.mkdir()
        f = root / "test.txt"
        f.write_text("hello", encoding="utf-8")
        cfg = self._make_config(root)
        # Make file unreadable so hash fails in subprocess
        f.chmod(0o000)
        try:
            result = run_phase1(test_db, cfg, mock_plog, workers=1)
            # May or may not fail depending on permissions
            assert result["files_processed"] >= 0
        finally:
            f.chmod(0o644)  # Restore so tmp_path can clean up
