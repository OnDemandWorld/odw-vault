"""Regression tests: UI chat must send plain-text content to providers.

The custom frontend pushes history in Gradio messages format
(content=[{"text": ..., "type": "text"}]) and includes the current user
turn. Ollama's /api/chat rejects array content with HTTP 400, and the
current turn must not be replayed as a prior turn. Also covers the
query_log insert (embedding_model is NOT NULL).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def _array_msg(role, text):
    return {"role": role, "content": [{"text": text, "type": "text"}], "metadata": None}


class TestOnChatContentNormalization:
    def test_array_history_normalized_and_current_turn_dropped(self, hit_factory):
        import ui.gradio_app as ga

        captured: list = []

        def fake_stream(query, hits, history=None):
            captured.append(list(history or []))
            yield "Answer [1]."
            yield "Answer [1].", []

        prior = [
            _array_msg("user", "old question"),
            _array_msg("assistant", "old answer"),
            _array_msg("user", "new question"),  # current turn, pushed by JS
        ]
        with patch.object(ga, "retrieve", return_value=([hit_factory()], {})), \
             patch.object(ga, "_stream_tokens", side_effect=fake_stream), \
             patch.object(ga, "_ensure_conv_tables"), \
             patch.object(ga, "_save_conversation_message"), \
             patch.object(ga, "_record_query_log"), \
             patch.object(ga, "_get_db", return_value=MagicMock()):
            list(ga._on_chat("new question", prior, "All folders", None))

        assert len(captured) == 1
        assert captured[0] == [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
        ]

    def test_string_history_unchanged(self, hit_factory):
        import ui.gradio_app as ga

        captured: list = []

        def fake_stream(query, hits, history=None):
            captured.append(list(history or []))
            yield "Answer.", []

        prior = [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
        ]
        with patch.object(ga, "retrieve", return_value=([hit_factory()], {})), \
             patch.object(ga, "_stream_tokens", side_effect=fake_stream), \
             patch.object(ga, "_ensure_conv_tables"), \
             patch.object(ga, "_save_conversation_message"), \
             patch.object(ga, "_record_query_log"), \
             patch.object(ga, "_get_db", return_value=MagicMock()):
            list(ga._on_chat("new question", prior, "All folders", None))

        assert captured[0] == prior


class TestStreamTokensProviderPayload:
    def test_provider_messages_all_string_content(self, app_config, monkeypatch, hit_factory):
        import ui.gradio_app as ga

        monkeypatch.setattr(ga, "_cfg", app_config)
        sent: list = []

        def fake_provider_stream(entry, messages, **kwargs):
            sent.append(messages)
            yield "ok"

        history = [
            _array_msg("user", "earlier question"),
            {"role": "assistant", "content": "earlier answer"},
        ]
        with patch.object(ga, "_load_prompt", return_value="CTX:{numbered_chunks}\nQ:{query}"), \
             patch.object(ga, "_get_active_provider", return_value={"protocol": "ollama", "model": "m", "base_url": "http://localhost:11434"}), \
             patch.object(ga, "_provider_chat_stream", side_effect=fake_provider_stream):
            list(ga._stream_tokens("follow-up", [hit_factory()], history=history))

        assert len(sent) == 1
        assert all(isinstance(m["content"], str) for m in sent[0])
        assert sent[0][1] == {"role": "user", "content": "earlier question"}


class TestRecordQueryLog:
    def test_insert_satisfies_not_null_and_stores_conversation(self, test_db, app_config, monkeypatch):
        import ui.gradio_app as ga

        monkeypatch.setattr(ga, "_cfg", app_config)
        monkeypatch.setattr(ga, "_get_db", lambda: test_db)
        ga._record_query_log(
            message="q?",
            answer="a",
            hits=[],
            retrieval_ms=0.1,
            total_ms=0.5,
            conversation_id="conv-123",
        )
        rows = list(test_db.query("SELECT * FROM query_log WHERE user = 'ui'"))
        assert len(rows) == 1
        assert rows[0]["embedding_model"] == app_config.models.embedding.name
        assert rows[0]["conversation_id"] == "conv-123"
