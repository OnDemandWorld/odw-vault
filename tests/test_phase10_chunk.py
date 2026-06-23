"""Tests for rag/phase10_chunk.py — pure functions."""

from rag.phase10_chunk import _split_sentences, _token_estimate


class TestSplitSentences:
    def test_single_sentence(self):
        assert _split_sentences("Hello.") == ["Hello."]

    def test_multiple_sentences(self):
        result = _split_sentences("A. B. C.")
        assert result == ["A.", "B.", "C."]

    def test_question_exclamation(self):
        result = _split_sentences("Hi! What? Ok.")
        assert result == ["Hi!", "What?", "Ok."]

    def test_chinese_terminators(self):
        result = _split_sentences("你好。")
        assert result == ["你好。"]

    def test_newline_separator(self):
        result = _split_sentences("Line one.\nLine two.")
        assert "Line one." in result
        assert "Line two." in result

    def test_empty_string(self):
        assert _split_sentences("") == []

    def test_whitespace_only(self):
        assert _split_sentences("   \n  ") == []

    def test_leading_trailing_whitespace(self):
        result = _split_sentences("  Hello.  ")
        # Leading whitespace before sentence is preserved
        assert result == ["  Hello."]

    def test_sentence_preserves_delimiter(self):
        result = _split_sentences("End. Start!")
        assert "End." in result
        assert "Start!" in result

    def test_long_text_splits(self):
        text = "First sentence. " * 10
        result = _split_sentences(text)
        assert len(result) == 10


class TestTokenEstimate:
    def test_basic_estimate(self):
        # len("hello" * 20) = 100, // 4 = 25
        assert _token_estimate("hello" * 20) == 25

    def test_minimum_one(self):
        assert _token_estimate("") == 1

    def test_short_text(self):
        assert _token_estimate("hi") == 1

    def test_estimate_scales_with_length(self):
        t1 = _token_estimate("x" * 100)
        t2 = _token_estimate("x" * 200)
        assert t2 > t1
