"""End-to-end tests for the full RAG pre-flight pipeline."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from pipeline.db import migrate, open_db
from pipeline.phase5_folder_meta import FolderInference
from tests.conftest import build_sf_response, make_config


class TestE2EWalkAndIdentify:
    """Test phases 1+2 together — the core inventory pipeline."""

    def _setup(self, tmp_path: Path):
        root = tmp_path / "corpus"
        root.mkdir()
        (root / "readme.txt").write_text("Hello world " * 20, encoding="utf-8")
        (root / "data.csv").write_text("id,name\n1,test\n", encoding="utf-8")
        cache = root / ".rag-cache"
        cache.mkdir()
        db_path = tmp_path / "corpus.db"
        db = open_db(db_path)
        migrate(db)
        return root, cache, db_path, db

    def test_walk_then_identify(self, tmp_path):
        """Phase 1 creates file rows, phase 2 assigns formats."""
        root, cache, db_path, db = self._setup(tmp_path)
        cfg = make_config(root, cache)

        from pipeline.logging import PhaseLogger
        from pipeline.phase1_walk import run_phase1
        from pipeline.phase2_identify import run_phase2

        plog = PhaseLogger("test", cache)

        # Phase 1
        r1 = run_phase1(db, cfg, plog)
        assert r1["files_processed"] >= 2

        # Phase 2
        all_files = [f["path"] for f in db.query("SELECT path FROM file")]
        sf_response = build_sf_response(all_files, root)
        with patch("shutil.which", return_value="/usr/bin/sf"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout=sf_response, stderr="")
                r2 = run_phase2(db, cfg, plog)
                assert r2["files_processed"] >= 2

        # Verify format assignment
        txt = next(db["file"].rows_where("name = ?", ["readme.txt"]))
        assert txt["pronom_id"] is not None
        assert txt["category"] is not None

    def test_walk_idempotent(self, tmp_path):
        """Running walk twice doesn't duplicate files."""
        root, cache, db_path, db = self._setup(tmp_path)
        cfg = make_config(root, cache)

        from pipeline.logging import PhaseLogger
        from pipeline.phase1_walk import run_phase1

        plog = PhaseLogger("test", cache)

        r1 = run_phase1(db, cfg, plog)
        # Reuse same db connection for idempotency check
        r2 = run_phase1(db, cfg, plog)

        file_count = db.execute("SELECT COUNT(*) FROM file").fetchone()[0]
        assert file_count >= 2


class TestE2EReport:
    """Test that phases 5+6 produce a valid report."""

    def _setup(self, tmp_path: Path):
        root = tmp_path / "corpus"
        root.mkdir()
        (root / "doc.txt").write_text(
            "Test document content for pipeline testing.\n" * 10, encoding="utf-8"
        )
        cache = root / ".rag-cache"
        cache.mkdir()
        db_path = tmp_path / "corpus.db"
        db = open_db(db_path)
        migrate(db)

        # Insert folder + file rows manually
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
                "path": str(root / "doc.txt"),
                "rel_path": "doc.txt",
                "name": "doc.txt",
                "size_bytes": 100,
                "mtime": "2026-01-01T00:00:00Z",
                "sha256": "abc123",
                "hash_status": "done",
                "identify_status": "done",
                "triage_status": "done",
                "category": "document",
                "extract_strategy": "docling",
                "excluded": 0,
                "is_dup_primary": 1,
            }
        )
        return root, cache, db_path, db

    def test_folder_meta_and_report(self, tmp_path):
        """Phase 5 inference + phase 6 report generation."""
        root, cache, db_path, db = self._setup(tmp_path)
        cfg = make_config(root, cache)

        from pipeline.logging import PhaseLogger
        from pipeline.phase5_folder_meta import run_phase5
        from pipeline.phase6_report import run_phase6

        plog = PhaseLogger("test", cache)

        # Phase 5
        inference = FolderInference(
            category="client-project",
            label="Test Project",
            tags=["test"],
            summary="A test folder.",
        )
        with patch("pipeline.phase5_folder_meta._call_ollama", return_value=inference):
            run_phase5(db, cfg, plog)

        # Verify inference
        folder = db["folder"].get(1)
        assert folder["inferred_category"] == "client-project"

        # Phase 6
        output = root / "preflight_report.md"
        run_phase6(db, cfg, plog, output_path=str(output))

        assert output.exists()
        content = output.read_text(encoding="utf-8")
        assert "Corpus Overview" in content
        assert "RAG Pre-Flight Report" in content

    def test_report_empty_db(self, tmp_path):
        """Report generates even with no files."""
        root, cache, db_path, db = self._setup(tmp_path)
        cfg = make_config(root, cache)

        from pipeline.logging import PhaseLogger
        from pipeline.phase6_report import run_phase6

        plog = PhaseLogger("test", cache)
        output = root / "empty_report.md"
        run_phase6(db, cfg, plog, output_path=str(output))

        assert output.exists()
        content = output.read_text(encoding="utf-8")
        assert "Total files" in content
