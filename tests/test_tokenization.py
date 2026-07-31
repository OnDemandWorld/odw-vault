"""Tests for rag/tokenization.py (V1.2 M2 — language-aware tokenization)."""

from __future__ import annotations

import rag.tokenization as tk
from rag.tokenization import (
    bigram_tokenize,
    detect_language,
    has_cjk,
    jieba_available,
    tokenize,
    tokenize_en,
    tokenize_zh,
    zh_index_text,
)

# ---------------------------------------------------------------------------
# Z1 — English path unchanged + Chinese bigram fallback
# ---------------------------------------------------------------------------


class TestEnglishPath:
    def test_tokenize_en_whitespace(self):
        assert tokenize_en("hello world") == ["hello", "world"]

    def test_tokenize_default_is_english(self):
        assert tokenize("the quick brown fox") == ["the", "quick", "brown", "fox"]

    def test_tokenize_en_explicit(self):
        assert tokenize("a b c", "en") == ["a", "b", "c"]

    def test_non_chinese_lang_uses_english_path(self):
        assert tokenize("bonjour le monde", "fr") == ["bonjour", "le", "monde"]

    def test_empty(self):
        assert tokenize_en("") == []
        assert tokenize("", "en") == []


class TestBigramFallback:
    def test_bigrams_for_cjk_run(self):
        assert bigram_tokenize("知识库") == ["知识", "识库"]

    def test_single_char(self):
        assert bigram_tokenize("中") == ["中"]

    def test_longer_run(self):
        assert bigram_tokenize("中文检索") == ["中文", "文检", "检索"]

    def test_mixed_latin_kept_lowercase(self):
        toks = bigram_tokenize("中文ABC测试")
        assert "中文" in toks
        assert "abc" in toks
        assert "测试" in toks

    def test_pure_latin(self):
        assert bigram_tokenize("hello world") == ["hello", "world"]

    def test_empty(self):
        assert bigram_tokenize("") == []


class TestTokenizeDispatch:
    def test_zh_dispatch_uses_chinese_tokenizer(self):
        toks = tokenize("知识库检索", "zh")
        # Whether jieba or bigram, the bigram "检索" must be retrievable.
        assert "检索" in toks or "知识库" in toks

    def test_zh_uppercase_lang(self):
        # Language matching is case-insensitive; "ZH" routes to the zh tokenizer.
        assert tokenize("知识库", "ZH") == tokenize_zh("知识库")

    def test_zh_index_text_space_joined(self):
        text = zh_index_text("中文检索")
        assert text == " ".join(tokenize_zh("中文检索"))


# ---------------------------------------------------------------------------
# Z2 — jieba optional: both paths must work
# ---------------------------------------------------------------------------


class TestJiebaOptional:
    def test_jieba_available_is_bool(self):
        assert isinstance(jieba_available(), bool)

    def test_bigram_path_when_jieba_absent(self, monkeypatch):
        # Force the "jieba not installed" path.
        monkeypatch.setattr(tk, "_get_jieba", lambda: None)
        assert tokenize_zh("知识库检索") == ["知识", "识库", "库检", "检索"]

    def test_jieba_path_when_present(self, monkeypatch):
        class FakeJieba:
            @staticmethod
            def cut(text):
                return ["知识库", "检索", "系统", " "]

        monkeypatch.setattr(tk, "_get_jieba", lambda: FakeJieba())
        # Whitespace/punctuation dropped; word segments kept.
        assert tokenize_zh("知识库检索系统") == ["知识库", "检索", "系统"]

    def test_jieba_failure_falls_back_to_bigram(self, monkeypatch):
        class BrokenJieba:
            @staticmethod
            def cut(text):
                raise RuntimeError("boom")

        monkeypatch.setattr(tk, "_get_jieba", lambda: BrokenJieba())
        assert tokenize_zh("知识库") == ["知识", "识库"]


# ---------------------------------------------------------------------------
# Language detection (fasttext lid → lingua → heuristic, default en)
# ---------------------------------------------------------------------------


class TestDetectLanguage:
    def test_empty_defaults_en(self):
        assert detect_language("") == "en"
        assert detect_language("   ") == "en"

    def test_chinese_detected(self):
        assert detect_language("知识库检索系统支持中文查询").startswith("zh")

    def test_english_detected(self):
        assert detect_language("This is a plain English sentence about retrieval.") == "en"

    def test_has_cjk(self):
        assert has_cjk("中文") is True
        assert has_cjk("english only") is False
        assert has_cjk("") is False
