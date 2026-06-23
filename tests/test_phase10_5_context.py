"""Tests for rag/phase10_5_context.py — pure functions."""

from rag.phase10_5_context import _build_prompt


class TestContextBuildPrompt:
    def test_prompt_template_populated(self):
        doc_text = "This is the full document text."
        chunk_text = "This is a specific chunk."
        prompt, _ = _build_prompt(doc_text, chunk_text)
        assert doc_text in prompt
        assert chunk_text in prompt

    def test_prompt_hash_deterministic(self):
        doc_text = "Document content."
        chunk_text = "Chunk content."
        _, hash1 = _build_prompt(doc_text, chunk_text)
        _, hash2 = _build_prompt(doc_text, chunk_text)
        assert hash1 == hash2

    def test_prompt_hash_different_inputs(self):
        _, hash1 = _build_prompt("Doc A", "Chunk A")
        _, hash2 = _build_prompt("Doc B", "Chunk B")
        assert hash1 != hash2

    def test_hash_is_valid_sha256(self):
        _, hash_str = _build_prompt("doc", "chunk")
        assert len(hash_str) == 64
        assert all(c in "0123456789abcdef" for c in hash_str)

    def test_prompt_contains_both_texts(self):
        prompt, _ = _build_prompt("full text here", "specific chunk here")
        assert "full text here" in prompt
        assert "specific chunk here" in prompt

    def test_empty_inputs(self):
        prompt, hash_str = _build_prompt("", "")
        assert isinstance(prompt, str)
        assert len(hash_str) == 64
