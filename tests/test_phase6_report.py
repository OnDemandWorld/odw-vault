"""Tests for pipeline/phase6_report.py."""


from pipeline.config import Config, PathsConfig
from pipeline.phase6_report import _fmt_list, _fmt_table, run_phase6


class TestFmtTable:
    def test_basic_table(self):
        rows = [{"name": "A", "count": 1}, {"name": "B", "count": 2}]
        result = _fmt_table(rows, "Test Table")
        assert "### Test Table" in result
        assert "| name |" in result
        assert "| count |" in result
        assert "| A |" in result

    def test_empty_rows(self):
        result = _fmt_table([], "Empty Table")
        assert "*(no data)*" in result


class TestFmtList:
    def test_basic_list(self):
        items = ["item1", "item2"]
        result = _fmt_list(items, "Test List")
        assert "### Test List" in result
        assert "- item1" in result

    def test_empty_list(self):
        result = _fmt_list([], "Empty List")
        assert "*(none)*" in result


class TestPhase6Run:
    def _make_config(self, root):
        return Config(paths=PathsConfig(corpus_root=str(root), cache_root=str(root / ".rag-cache")))

    def _populate_db(self, db, root):
        """Add minimal folder/file data for report generation."""
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
                "sha256": "abc123def456",
                "hash_status": "done",
                "identify_status": "done",
                "triage_status": "done",
                "pronom_id": "x-fmt/111",
                "format_name": "Plain Text File",
                "category": "document",
                "extract_strategy": "tika",
                "excluded": 0,
                "is_dup_primary": 1,
            }
        )
        db["folder"].update(
            1,
            {
                "file_count": 1,
                "total_bytes": 100,
                "document_count": 1,
                "dominant_format": "Plain Text File",
            },
        )

    def test_report_generated(self, test_db, test_corpus, mock_plog):
        root, _ = test_corpus
        cfg = self._make_config(root)
        self._populate_db(test_db, root)
        output = root / "test_report.md"
        result = run_phase6(test_db, cfg, mock_plog, output_path=str(output))
        assert output.exists()
        content = output.read_text(encoding="utf-8")
        assert len(content) > 100

    def test_report_contains_sections(self, test_db, test_corpus, mock_plog):
        root, _ = test_corpus
        cfg = self._make_config(root)
        self._populate_db(test_db, root)
        output = root / "test_report.md"
        run_phase6(test_db, cfg, mock_plog, output_path=str(output))
        content = output.read_text(encoding="utf-8")
        assert "Corpus Overview" in content
        assert "Top" in content  # Top 30 Formats
        assert "Category Breakdown" in content

    def test_report_custom_output_path(self, test_db, test_corpus, mock_plog, tmp_path):
        root, _ = test_corpus
        cfg = self._make_config(root)
        self._populate_db(test_db, root)
        output = tmp_path / "custom_report.md"
        run_phase6(test_db, cfg, mock_plog, output_path=str(output))
        assert output.exists()

    def test_report_empty_db(self, test_db, test_corpus, mock_plog, tmp_path):
        root, _ = test_corpus
        cfg = self._make_config(root)
        output = tmp_path / "empty_report.md"
        result = run_phase6(test_db, cfg, mock_plog, output_path=str(output))
        assert output.exists()
        content = output.read_text(encoding="utf-8")
        assert "Total files" in content

    def test_report_marks_preflight_complete(self, test_db, test_corpus, mock_plog):
        root, _ = test_corpus
        cfg = self._make_config(root)
        self._populate_db(test_db, root)
        output = root / "test_report.md"
        run_phase6(test_db, cfg, mock_plog, output_path=str(output))
        val = test_db.execute(
            "SELECT value FROM config WHERE key='preflight_completed_at'"
        ).fetchone()
        assert val is not None

    def test_report_format_table(self):
        result = _fmt_table([{"a": "1", "b": "2"}], "Test")
        assert "| a | b |" in result
        assert "| --- | --- |" in result
        assert "| 1 | 2 |" in result

    def test_report_folder_taxonomy(self, test_db, test_corpus, mock_plog, tmp_path):
        root, _ = test_corpus
        cfg = self._make_config(root)
        # Ensure we have folders
        subdir = root / "subdir"
        subdir.mkdir(exist_ok=True)
        db_path = str(test_db.path) if hasattr(test_db, "path") else str(test_db.filename)
        self._populate_db(test_db, root)
        # Add a subfolder
        test_db["folder"].insert(
            {
                "path": str(subdir),
                "rel_path": "subdir",
                "parent_id": 1,
                "name": "subdir",
                "depth": 1,
            }
        )
        output = tmp_path / "report.md"
        run_phase6(test_db, cfg, mock_plog, output_path=str(output))
        content = output.read_text(encoding="utf-8")
        assert "Folder Taxonomy" in content

    def test_report_failure_summary(self, test_db, test_corpus, mock_plog, tmp_path):
        root, _ = test_corpus
        cfg = self._make_config(root)
        self._populate_db(test_db, root)
        # Insert a failure
        test_db["failure"].insert(
            {
                "file_id": None,
                "folder_id": None,
                "phase": "triage",
                "tool": "fitz",
                "error_class": "corrupt",
                "error_message": "bad pdf",
            }
        )
        output = tmp_path / "report.md"
        run_phase6(test_db, cfg, mock_plog, output_path=str(output))
        content = output.read_text(encoding="utf-8")
        assert "Failure Summary" in content
