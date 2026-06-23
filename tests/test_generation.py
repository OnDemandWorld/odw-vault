"""Tests for rag/generation.py — pure functions."""

from rag.generation import DEFAULT_PROMPT, REFUSAL_TEXT, _format_chunks
from rag.retrieval import Hit


class TestFormatChunks:
    def test_single_hit(self):
        hits = [Hit(chunk_id=1, file_id=1, folder_id=1, rel_path="doc.txt", page_start=None, text="hello world")]
        result = _format_chunks(hits)
        assert "[1] doc.txt" in result
        assert "hello world" in result

    def test_multiple_hits(self):
        hits = [
            Hit(chunk_id=1, file_id=1, folder_id=1, rel_path="a.txt", page_start=None, text="one"),
            Hit(chunk_id=2, file_id=2, folder_id=1, rel_path="b.txt", page_start=None, text="two"),
            Hit(chunk_id=3, file_id=3, folder_id=1, rel_path="c.txt", page_start=None, text="three"),
        ]
        result = _format_chunks(hits)
        assert "[1] a.txt" in result
        assert "[2] b.txt" in result
        assert "[3] c.txt" in result

    def test_empty_hits(self):
        assert _format_chunks([]) == "(no context available)"

    def test_page_info_present(self):
        hits = [Hit(chunk_id=1, file_id=1, folder_id=1, rel_path="doc.pdf", text="text", page_start=5)]
        result = _format_chunks(hits)
        assert "(page 5)" in result

    def test_page_info_absent(self):
        hits = [Hit(chunk_id=1, file_id=1, folder_id=1, rel_path="doc.txt", page_start=None, text="text")]
        result = _format_chunks(hits)
        assert "(page" not in result


class TestDefaultPrompt:
    def test_prompt_contains_chunk_placeholder(self):
        assert "{numbered_chunks}" in DEFAULT_PROMPT

    def test_prompt_contains_query_placeholder(self):
        assert "{query}" in DEFAULT_PROMPT

    def test_refusal_text_nonempty(self):
        assert len(REFUSAL_TEXT) > 0
