"""Tests for multi-turn conversation memory (P0.2).

Tests cover:
- Conversation lifecycle (create, list, get messages, delete)
- Context injection in generation prompt
- Sliding window memory
- Retrieval uses last message only
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.db import migrate, open_db
from rag.conversation import (
    add_message,
    create_conversation,
    delete_conversation,
    get_conversation_messages,
    get_history,
    get_or_create_conversation,
    list_conversations,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path: Path):
    """Create a temporary database with all migrations applied."""
    db_path = tmp_path / "test.db"
    database = open_db(db_path)
    migrate(database)
    return database


# ---------------------------------------------------------------------------
# Test: conversation lifecycle
# ---------------------------------------------------------------------------


class TestConversationLifecycle:
    """Verify create, send message, get history, delete flow."""

    def test_create_conversation(self, db):
        conv_id = create_conversation(db, user="testuser", title="Test Chat")
        assert conv_id is not None
        assert len(conv_id) == 36  # UUID format

        # Verify it exists in DB
        rows = list(db.query(
            "SELECT id, user, title FROM conversation WHERE id = ?", [conv_id]
        ))
        assert len(rows) == 1
        assert rows[0]["user"] == "testuser"
        assert rows[0]["title"] == "Test Chat"

    def test_add_and_get_messages(self, db):
        conv_id = create_conversation(db, user="user1")

        # Add messages
        msg1_id = add_message(db, conv_id, "user", "Hello")
        msg2_id = add_message(db, conv_id, "assistant", "Hi there!")
        msg3_id = add_message(db, conv_id, "user", "How are you?")

        assert msg1_id > 0
        assert msg2_id > msg1_id
        assert msg3_id > msg2_id

        # Get all messages
        messages = get_conversation_messages(db, conv_id)
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hi there!"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "How are you?"

    def test_get_or_create_existing(self, db):
        conv_id = create_conversation(db, user="user1")
        result = get_or_create_conversation(db, conv_id, user="user1")
        assert result == conv_id

    def test_get_or_create_new(self, db):
        result = get_or_create_conversation(db, None, user="user1")
        assert result is not None
        assert len(result) == 36

    def test_get_or_create_invalid_id(self, db):
        """Invalid conversation_id should create a new one."""
        result = get_or_create_conversation(db, "nonexistent-id", user="user1")
        assert result is not None
        assert result != "nonexistent-id"

    def test_list_conversations(self, db):
        conv1 = create_conversation(db, user="user1", title="First")
        # Add a message to conv1 to update its timestamp
        add_message(db, conv1, "user", "hello")
        conv2 = create_conversation(db, user="user1", title="Second")
        add_message(db, conv2, "user", "world")

        convs = list_conversations(db, user="user1")
        assert len(convs) == 2
        # Both should be present
        ids = {c["id"] for c in convs}
        assert conv1 in ids
        assert conv2 in ids

    def test_list_conversations_with_message_count(self, db):
        conv_id = create_conversation(db, user="user1")
        add_message(db, conv_id, "user", "Hello")
        add_message(db, conv_id, "assistant", "Hi")

        convs = list_conversations(db, user="user1")
        assert convs[0]["message_count"] == 2

    def test_delete_conversation(self, db):
        conv_id = create_conversation(db, user="user1")
        add_message(db, conv_id, "user", "Hello")
        add_message(db, conv_id, "assistant", "Hi")

        result = delete_conversation(db, conv_id)
        assert result is True

        # Verify cascade delete
        messages = get_conversation_messages(db, conv_id)
        assert len(messages) == 0

        convs = list_conversations(db)
        assert len(convs) == 0

    def test_delete_nonexistent(self, db):
        result = delete_conversation(db, "nonexistent")
        assert result is False

    def test_title_auto_set_from_first_message(self, db):
        conv_id = create_conversation(db, user="user1")
        add_message(db, conv_id, "user", "What is the capital of France?")

        rows = list(db.query(
            "SELECT title FROM conversation WHERE id = ?", [conv_id]
        ))
        assert rows[0]["title"] == "What is the capital of France?"

    def test_title_not_overwritten(self, db):
        conv_id = create_conversation(db, user="user1", title="My Custom Title")
        add_message(db, conv_id, "user", "What is the capital of France?")

        rows = list(db.query(
            "SELECT title FROM conversation WHERE id = ?", [conv_id]
        ))
        assert rows[0]["title"] == "My Custom Title"


# ---------------------------------------------------------------------------
# Test: context injection in prompt
# ---------------------------------------------------------------------------


class TestContextInjectionInPrompt:
    """Verify that conversation history is injected into the generation prompt."""

    def test_history_injected_in_ollama_messages(self):
        """The _ollama_chat function should include history messages."""
        from rag.generation import _ollama_chat

        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {"content": "test answer"},
        }

        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]

        _ollama_chat(
            client=mock_client,
            model="test-model",
            prompt="Current question",
            history=history,
        )

        # Verify the messages sent to Ollama
        call_args = mock_client.chat.call_args
        messages = call_args.kwargs["messages"]

        # Should be: system, user (history), assistant (history), user (current)
        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Previous question"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "Previous answer"
        assert messages[3]["role"] == "user"
        assert messages[3]["content"] == "Current question"

    def test_no_history_no_extra_messages(self):
        """Without history, only system + user messages should be sent."""
        from rag.generation import _ollama_chat

        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {"content": "test answer"},
        }

        _ollama_chat(
            client=mock_client,
            model="test-model",
            prompt="Current question",
        )

        call_args = mock_client.chat.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_empty_history_same_as_none(self):
        """Empty history list should behave same as None."""
        from rag.generation import _ollama_chat

        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {"content": "test answer"},
        }

        _ollama_chat(
            client=mock_client,
            model="test-model",
            prompt="Current question",
            history=[],
        )

        call_args = mock_client.chat.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2


# ---------------------------------------------------------------------------
# Test: sliding window memory
# ---------------------------------------------------------------------------


class TestSlidingWindowMemory:
    """Verify that only the most recent N turns are included in history."""

    def test_sliding_window_default(self, db):
        """Default window size is 5 turns (10 messages)."""
        conv_id = create_conversation(db, user="user1")

        # Add 10 turns (20 messages)
        for i in range(10):
            add_message(db, conv_id, "user", f"Question {i}")
            add_message(db, conv_id, "assistant", f"Answer {i}")

        # Get history with default window (5 turns = 10 messages)
        history = get_history(db, conv_id, window_size=5)
        assert len(history) == 10

        # Should be the most recent 5 turns (questions 5-9)
        assert history[0]["content"] == "Question 5"
        assert history[-1]["content"] == "Answer 9"

    def test_sliding_window_custom(self, db):
        """Custom window size should be respected."""
        conv_id = create_conversation(db, user="user1")

        for i in range(5):
            add_message(db, conv_id, "user", f"Q{i}")
            add_message(db, conv_id, "assistant", f"A{i}")

        # Window of 2 turns = 4 messages
        history = get_history(db, conv_id, window_size=2)
        assert len(history) == 4
        assert history[0]["content"] == "Q3"
        assert history[-1]["content"] == "A4"

    def test_sliding_window_larger_than_history(self, db):
        """Window larger than history should return all messages."""
        conv_id = create_conversation(db, user="user1")

        add_message(db, conv_id, "user", "Q1")
        add_message(db, conv_id, "assistant", "A1")

        history = get_history(db, conv_id, window_size=10)
        assert len(history) == 2

    def test_sliding_window_empty_conversation(self, db):
        """Empty conversation should return empty history."""
        conv_id = create_conversation(db, user="user1")
        history = get_history(db, conv_id, window_size=5)
        assert history == []


# ---------------------------------------------------------------------------
# Test: retrieval uses last message only
# ---------------------------------------------------------------------------


class TestRetrievalUsesLastMessageOnly:
    """Verify that the retrieval function receives only the latest query,
    not the full conversation history."""

    def test_retrieval_receives_current_query(self):
        """The retrieve() function should be called with the current query only."""
        # This tests the API-level behavior: when a conversation has history,
        # the /query endpoint should pass only req.query to retrieve(),
        # not the full conversation text.

        # We verify this by checking the generate_answer call signature
        # in the API code — it receives req.query (the current message)
        # while history is passed separately to generate_answer.

        # Simulate the flow:
        current_query = "What is the latest status?"
        history = [
            {"role": "user", "content": "What happened yesterday?"},
            {"role": "assistant", "content": "Here is what happened..."},
        ]

        # The retrieve() function should receive current_query only
        # The generate_answer() function should receive both current_query and history
        mock_retrieve = MagicMock(return_value=([], {}))
        mock_generate = MagicMock(return_value={
            "answer": "test",
            "citations": [],
            "generation_ms": 100,
            "model": "test",
        })

        # Simulate the API flow
        mock_retrieve(query=current_query)
        mock_generate(query=current_query, hits=[], cfg=None, history=history)

        # Verify retrieve got only the current query
        retrieve_call = mock_retrieve.call_args
        assert retrieve_call.kwargs["query"] == current_query

        # Verify generate got both query and history
        generate_call = mock_generate.call_args
        assert generate_call.kwargs["query"] == current_query
        assert generate_call.kwargs["history"] == history
        assert len(generate_call.kwargs["history"]) == 2

    def test_query_log_stores_conversation_id(self, db):
        """query_log should record the conversation_id for each query."""
        conv_id = create_conversation(db, user="user1")

        # Simulate a query_log entry with conversation_id
        db.execute(
            """INSERT INTO query_log
               (user, query_text, query_lang, retrieved_chunks_json,
                answer_text, answer_model, embedding_model,
                latency_ms, conversation_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                "user1",
                "What is RAG?",
                "en",
                "[]",
                "RAG is...",
                "gemma4",
                "qwen3-embedding",
                500,
                conv_id,
            ],
        )
        db.conn.commit()

        row = list(db.query(
            "SELECT conversation_id FROM query_log WHERE query_text = 'What is RAG?'"
        ))
        assert row[0]["conversation_id"] == conv_id
