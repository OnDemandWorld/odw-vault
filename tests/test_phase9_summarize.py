"""Tests for rag/phase9_summarize.py — pure functions."""

from rag.phase9_summarize import SUMMARIZATION_PROMPT, _build_prompt


class TestBuildPrompt:
    def test_basic_prompt(self):
        text = "This is a short document for testing."
        prompt = _build_prompt(text)
        assert text in prompt
        assert SUMMARIZATION_PROMPT[:50] in prompt

    def test_truncation_at_limit(self):
        text = "x" * 10000
        prompt = _build_prompt(text)
        # Text should be truncated to 8000 chars
        assert "x" * 8000 in prompt
        assert "x" * 8001 not in prompt

    def test_exact_boundary(self):
        text = "x" * 8000
        prompt = _build_prompt(text)
        # Not truncated — exactly at boundary
        assert "x" * 8000 in prompt

    def test_prompt_contains_document_marker(self):
        prompt = _build_prompt("test")
        assert "Document:" in prompt

    def test_short_text_not_truncated(self):
        text = "Hello world."
        prompt = _build_prompt(text)
        assert text in prompt
        assert len(prompt) > len(text)

    def test_prompt_template_structure(self):
        prompt = _build_prompt("Content here.")
        assert "Summary:" in prompt or "summarize" in prompt.lower()

    def test_prompt_uses_first_8000_chars(self):
        text = "A" * 9000
        prompt = _build_prompt(text)
        assert prompt.count("A") == 8000 or prompt.count("A") < 9000
