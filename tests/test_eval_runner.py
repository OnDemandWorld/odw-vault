"""Tests for eval/runner.py — pure functions and validation."""

import pytest

from eval.runner import (
    VALID_CATEGORIES,
    _format_table,
    _recall_for_hits,
    add_question,
)
from rag.retrieval import Hit


class TestAddQuestion:
    def test_add_valid_question(self, test_db):
        qid = add_question(
            test_db,
            question="What is this?",
            expected_file_ids=[1],
        )
        assert qid is not None
        row = test_db["eval_question"].get(qid)
        assert row["question"] == "What is this?"

    def test_reject_empty_file_ids(self, test_db):
        with pytest.raises(ValueError, match="must be non-empty"):
            add_question(test_db, question="test", expected_file_ids=[])

    def test_reject_invalid_category(self, test_db):
        with pytest.raises(ValueError, match="must be one of"):
            add_question(test_db, question="test", expected_file_ids=[1], category="bogus")

    def test_valid_categories(self, test_db):
        for cat in VALID_CATEGORIES:
            qid = add_question(
                test_db,
                question="test",
                expected_file_ids=[1],
                category=cat,
            )
            assert qid is not None

    def test_expected_answer_optional(self, test_db):
        qid = add_question(test_db, question="test", expected_file_ids=[1])
        row = test_db["eval_question"].get(qid)
        assert row["expected_answer"] is None


class TestRecallForHits:
    def _make_hits(self, file_ids: list[int]) -> list[Hit]:
        return [
            Hit(chunk_id=i, file_id=fid, folder_id=1, rel_path=f"f{fid}.txt", page_start=None, text="t")
            for i, fid in enumerate(file_ids, 1)
        ]

    def test_recall_hit_in_top_k(self):
        hits = self._make_hits([10, 20, 30])
        assert _recall_for_hits(hits, {20}, top_k=3) == 1

    def test_recall_miss_beyond_k(self):
        hits = self._make_hits([10, 20, 30, 40, 50])
        # Expected file at position 5, top_k=3
        assert _recall_for_hits(hits, {50}, top_k=3) == 0

    def test_recall_multiple_expected(self):
        hits = self._make_hits([10, 20, 30])
        # Either 10 or 99 in top 3 — 10 is there
        assert _recall_for_hits(hits, {10, 99}, top_k=3) == 1

    def test_recall_empty_hits(self):
        assert _recall_for_hits([], {1}, top_k=3) == 0

    def test_recall_top_k_larger_than_hits(self):
        hits = self._make_hits([10])
        assert _recall_for_hits(hits, {10}, top_k=10) == 1

    def test_recall_no_expected(self):
        hits = self._make_hits([10, 20])
        assert _recall_for_hits(hits, set(), top_k=3) == 0


class TestFormatTable:
    def test_basic_table(self):
        rows = [{"name": "Alice", "age": 30}]
        columns = [("name", "Name"), ("age", "Age")]
        result = _format_table(rows, columns)
        assert "Name" in result
        assert "Alice" in result

    def test_column_widths(self):
        rows = [{"name": "Alex", "score": "100"}]
        columns = [("name", "Name"), ("score", "Score")]
        result = _format_table(rows, columns)
        # "Score" is wider than "100", column should be padded
        assert "Score" in result

    def test_empty_rows(self):
        columns = [("a", "A"), ("b", "B")]
        result = _format_table([], columns)
        assert "A" in result
        assert "B" in result

    def test_missing_keys(self):
        rows = [{"name": "Alice"}]
        columns = [("name", "Name"), ("age", "Age")]
        result = _format_table(rows, columns)
        assert "Alice" in result
        # Missing key rendered as empty string
        assert "Name" in result
