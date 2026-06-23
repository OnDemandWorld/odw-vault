"""Tests for rag/generation.py — integration with mocked Ollama."""

from unittest.mock import MagicMock, patch

from rag.generation import (
    DEFAULT_PROMPT,
    REFUSAL_TEXT,
    _format_chunks,
    _load_prompt,
    generate_answer,
)
from rag.retrieval import Hit


class TestGenerateAnswerMocked:
    """generate_answer with mocked Ollama client."""

    def _make_cfg(self, thinking=False):
        cfg = MagicMock()
        cfg.generation_runtime.refuse_on_empty_context = True
        cfg.models.generation.name = "test-model"
        cfg.models.generation.temperature = 0.5
        cfg.models.generation.top_p = 0.95
        cfg.models.generation.top_k = 64
        cfg.models.generation.thinking = thinking
        cfg.models.generation.prompt_version = "v1"
        cfg.ollama.host = "http://localhost:11434"
        return cfg

    def _make_hits(self):
        return [
            Hit(
                chunk_id=1,
                file_id=1,
                folder_id=1,
                rel_path="doc1.txt",
                page_start=1,
                text="The project uses Python 3.11.",
            ),
            Hit(
                chunk_id=2,
                file_id=2,
                folder_id=1,
                rel_path="doc2.txt",
                page_start=None,
                text="Deployment is on AWS using ECS.",
            ),
        ]

    def test_refuse_on_empty_context(self):
        cfg = self._make_cfg()
        result = generate_answer("test query", [], cfg)
        assert result["refused"] is True
        assert result["answer"] == REFUSAL_TEXT
        assert result["citations"] == []

    def test_generate_with_mocked_ollama(self):
        cfg = self._make_cfg()
        hits = self._make_hits()

        mock_response = {"message": {"content": "The project uses Python [1] and AWS [2]."}}

        with patch("rag.generation.ollama.Client") as MockClient:
            instance = MockClient.return_value
            instance.chat.return_value = mock_response
            result = generate_answer("What tech is used?", hits, cfg)

        assert result["refused"] is False
        assert "Python" in result["answer"]
        assert result["model"] == "test-model"
        assert len(result["citations"]) == 2

    def test_empty_ollama_response_falls_back_to_refusal(self):
        cfg = self._make_cfg()
        hits = self._make_hits()

        with patch("rag.generation.ollama.Client") as MockClient:
            instance = MockClient.return_value
            instance.chat.return_value = {"message": {"content": ""}}
            result = generate_answer("test query", hits, cfg)

        assert result["answer"] == REFUSAL_TEXT

    def test_thinking_flag_adds_prefix(self):
        cfg = self._make_cfg(thinking=True)
        hits = self._make_hits()

        with patch("rag.generation.ollama.Client") as MockClient:
            instance = MockClient.return_value
            instance.chat.return_value = {"message": {"content": "Test answer."}}
            result = generate_answer("test", hits, cfg)

            # Verify the system message contains <|think|>
            call_args = instance.chat.call_args
            system_msg = call_args.kwargs["messages"][0]["content"]
            assert "<|think|>" in system_msg

    def test_generation_ms_present(self):
        cfg = self._make_cfg()
        hits = self._make_hits()

        with patch("rag.generation.ollama.Client") as MockClient:
            MockClient.return_value.chat.return_value = {"message": {"content": "Answer."}}
            result = generate_answer("test", hits, cfg)

        assert result["generation_ms"] > 0

    def test_no_hits_but_refuse_disabled(self):
        cfg = self._make_cfg()
        cfg.generation_runtime.refuse_on_empty_context = False

        with patch("rag.generation.ollama.Client") as MockClient:
            MockClient.return_value.chat.return_value = {"message": {"content": "Answer."}}
            result = generate_answer("test", [], cfg)

        assert result["refused"] is False


class TestFormatChunks:
    def test_single_hit_with_page(self):
        hit = Hit(chunk_id=1, file_id=1, folder_id=1, rel_path="doc.txt", page_start=3, text="hello")
        result = _format_chunks([hit])
        assert "[1] doc.txt (page 3)" in result
        assert "hello" in result

    def test_single_hit_without_page(self):
        hit = Hit(chunk_id=1, file_id=1, folder_id=1, rel_path="doc.txt", page_start=None, text="hello")
        result = _format_chunks([hit])
        assert "[1] doc.txt\nhello" in result

    def test_multiple_hits_separated(self):
        hits = [
            Hit(chunk_id=1, file_id=1, folder_id=1, rel_path="a.txt", page_start=None, text="first"),
            Hit(chunk_id=2, file_id=2, folder_id=1, rel_path="b.txt", page_start=None, text="second"),
        ]
        result = _format_chunks(hits)
        assert "first" in result
        assert "second" in result
        assert "\n\n" in result

    def test_empty_list(self):
        assert _format_chunks([]) == "(no context available)"


class TestLoadPrompt:
    def test_nonexistent_file_falls_back_to_default(self):
        cfg = MagicMock()
        cfg.models.generation.prompt_version = "v1"
        result = _load_prompt("/nonexistent/path.txt", cfg)
        assert result == DEFAULT_PROMPT

    def test_none_uses_default(self):
        cfg = MagicMock()
        cfg.models.generation.prompt_version = "v1"
        result = _load_prompt(None, cfg)
        assert result == DEFAULT_PROMPT
