"""Tests for rag/citations.py."""

from rag.citations import parse_citations, resolve_citations
from rag.retrieval import Hit


class TestParseCitations:
    def test_no_citations(self):
        assert parse_citations("hello world") == []

    def test_single_citation(self):
        assert parse_citations("hello [1]") == [1]

    def test_multiple_citations(self):
        assert parse_citations("see [1] and [3]") == [1, 3]

    def test_duplicate_citations(self):
        assert parse_citations("[1] then [1] again") == [1]

    def test_citation_order_preserved(self):
        assert parse_citations("[3] first, then [1]") == [3, 1]

    def test_citation_zero(self):
        assert parse_citations("[0]") == [0]

    def test_citation_in_brackets_only(self):
        assert parse_citations("no markers here") == []

    def test_citation_with_surrounding_text(self):
        assert parse_citations("As shown in [2], the results are clear.") == [2]

    def test_multiple_same_line(self):
        assert parse_citations("[1][2][3]") == [1, 2, 3]

    def test_non_numeric_brackets_ignored(self):
        assert parse_citations("[abc] not a citation [1]") == [1]


class TestResolveCitations:
    def _make_hits(self, n: int) -> list[Hit]:
        return [
            Hit(
                chunk_id=i,
                file_id=i,
                folder_id=1,
                rel_path=f"doc{i}.txt",
                page_start=i,
                text=f"content {i}" * 20,
            )
            for i in range(1, n + 1)
        ]

    def test_resolve_single(self):
        hits = self._make_hits(1)
        result = resolve_citations([1], hits)
        assert len(result) == 1
        assert result[0]["citation_number"] == 1
        assert result[0]["chunk_id"] == 1
        assert result[0]["rel_path"] == "doc1.txt"
        assert result[0]["page_start"] == 1

    def test_resolve_multiple(self):
        hits = self._make_hits(3)
        result = resolve_citations([1, 3], hits)
        assert len(result) == 2
        assert result[0]["citation_number"] == 1
        assert result[1]["citation_number"] == 3

    def test_resolve_empty_list(self):
        hits = self._make_hits(3)
        assert resolve_citations([], hits) == []

    def test_resolve_out_of_range(self):
        hits = self._make_hits(2)
        result = resolve_citations([1, 99], hits)
        # Only 1 resolves, 99 is out of range
        assert len(result) == 1
        assert result[0]["citation_number"] == 1

    def test_resolve_all_out_of_range(self):
        hits = self._make_hits(1)
        assert resolve_citations([99, 100], hits) == []

    def test_snippet_truncated(self):
        long_text = "word " * 200
        hits = [
            Hit(
                chunk_id=1,
                file_id=1,
                folder_id=1,
                rel_path="long.txt",
                page_start=None,
                text=long_text,
            )
        ]
        result = resolve_citations([1], hits)
        assert len(result) == 1
        assert len(result[0]["snippet"]) <= 303  # 300 + "..."

    def test_snippet_not_truncated(self):
        hits = [
            Hit(
                chunk_id=1,
                file_id=1,
                folder_id=1,
                rel_path="short.txt",
                page_start=None,
                text="short text",
            )
        ]
        result = resolve_citations([1], hits)
        assert result[0]["snippet"] == "short text"
