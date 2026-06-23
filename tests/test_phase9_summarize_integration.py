"""Tests for rag/phase9_summarize.py — integration with mocked Ollama and DB."""

from unittest.mock import MagicMock, patch

from pipeline.db import migrate, open_db
from rag.phase9_summarize import _build_prompt, run_summarize
from tests.conftest import seed_test_extractions, seed_test_files


class TestBuildPrompt:
    def test_truncation_at_8000(self):
        long_text = "x" * 10000
        result = _build_prompt(long_text)
        assert len(result) < 10000

    def test_short_text_not_truncated(self):
        short = "Hello world."
        result = _build_prompt(short)
        assert "Hello world." in result

    def test_prompt_contains_document_marker(self):
        result = _build_prompt("test content")
        assert "Document:" in result
        assert "test content" in result

    def test_prompt_contains_summary_marker(self):
        result = _build_prompt("test")
        assert "Summary:" in result


class TestRunSummarizeMocked:
    """run_summarize with in-memory DB and mocked Ollama."""

    def _make_cfg(self):
        cfg = MagicMock()
        cfg.models.summarization.name = "test-summarize"
        cfg.models.summarization.temperature = 0.3
        cfg.models.summarization.max_tokens = 400
        cfg.models.summarization.prompt_version = "v1"
        cfg.extract.size_threshold_for_summary = 1
        cfg.ollama.host = "http://localhost:11434"
        return cfg

    def _seed_db(self, tmp_path):
        db_path = tmp_path / "test.db"
        db = open_db(db_path)
        migrate(db)
        return db

    def test_no_eligible_extractions(self, tmp_path):
        db = self._seed_db(tmp_path)
        cfg = self._make_cfg()
        processed, failed = run_summarize(db, cfg)
        assert processed == 0
        assert failed == 0

    def test_summarizes_extraction(self, tmp_path):
        db = self._seed_db(tmp_path)
        cfg = self._make_cfg()
        file_ids = seed_test_files(db)
        seed_test_extractions(db, file_ids, text="This is a long enough text for summarization testing purposes.")

        mock_response = {"message": {"content": "This is a summary."}}
        with patch("rag.phase9_summarize._call_ollama", return_value="This is a summary."):
            processed, failed = run_summarize(db, cfg)

        assert processed >= 1
        assert failed == 0
        summaries = list(db["summary"].rows)
        assert len(summaries) >= 1
        assert summaries[0]["summary_text"] == "This is a summary."

    def test_skips_existing_summary(self, tmp_path):
        db = self._seed_db(tmp_path)
        cfg = self._make_cfg()
        file_ids = seed_test_files(db)
        seed_test_extractions(db, file_ids, text="Some text here.")

        # First run
        with patch("rag.phase9_summarize._call_ollama", return_value="Summary text."):
            run_summarize(db, cfg)

        # Second run — should skip
        with patch("rag.phase9_summarize._call_ollama", return_value="New summary.") as mock_call:
            processed, failed = run_summarize(db, cfg)

        assert mock_call.call_count == 0
        assert processed == 0

    def test_resummarize_overwrites(self, tmp_path):
        db = self._seed_db(tmp_path)
        cfg = self._make_cfg()
        file_ids = seed_test_files(db)
        seed_test_extractions(db, file_ids, text="Some text here.")

        with patch("rag.phase9_summarize._call_ollama", return_value="First summary."):
            run_summarize(db, cfg)

        with patch("rag.phase9_summarize._call_ollama", return_value="Second summary."):
            processed, failed = run_summarize(db, cfg, resummarize=True)

        assert processed >= 1
        summaries = list(db["summary"].rows)
        assert summaries[-1]["summary_text"] == "Second summary."

    def test_limit_parameter(self, tmp_path):
        db = self._seed_db(tmp_path)
        cfg = self._make_cfg()
        # Create folder first
        db["folder"].insert({"path": "test", "rel_path": "test", "name": "test", "depth": 0})
        db.conn.commit()
        file_ids = seed_test_files(db, files=[
            {"name": "file1.txt", "folder_id": 1, "extract_strategy": "textutil", "category": "document"},
            {"name": "file2.txt", "folder_id": 1, "extract_strategy": "textutil", "category": "document"},
        ])
        seed_test_extractions(db, file_ids, text="Enough text for summary.")

        with patch("rag.phase9_summarize._call_ollama", return_value="Summary."):
            processed, failed = run_summarize(db, cfg, limit=1)

        assert processed == 1

    def test_empty_ollama_response_records_failure(self, tmp_path):
        db = self._seed_db(tmp_path)
        cfg = self._make_cfg()
        file_ids = seed_test_files(db)
        seed_test_extractions(db, file_ids, text="Some text here.")

        def raise_empty(*args, **kwargs):
            raise ValueError("Ollama returned empty response")

        with patch("rag.phase9_summarize._call_ollama", side_effect=raise_empty):
            processed, failed = run_summarize(db, cfg)

        assert failed >= 1
        failures = list(db["failure"].rows)
        assert len(failures) >= 1
