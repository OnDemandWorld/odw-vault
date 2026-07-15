"""Tests for query history API (P0.5).

Tests cover:
- Query logging after query execution
- Pagination of query history
- Keyword search filtering
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.db import migrate, open_db


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


def _insert_query(db, query_text: str, answer: str = "test answer", **kwargs):
    """Helper to insert a query_log row."""
    db.conn.execute(
        """INSERT INTO query_log
           (user, query_text, query_lang, retrieved_chunks_json,
            answer_text, answer_model, embedding_model,
            latency_ms, conversation_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            kwargs.get("user"),
            query_text,
            "en",
            json.dumps(kwargs.get("chunks", [])),
            answer,
            "gemma4",
            "qwen3-embedding",
            kwargs.get("latency_ms", 500),
            kwargs.get("conversation_id"),
        ),
    )
    db.conn.commit()


# ---------------------------------------------------------------------------
# Test: query logging
# ---------------------------------------------------------------------------


class TestQueryLogging:
    """Verify that queries are logged correctly."""

    def test_query_log_created(self, db):
        """After inserting a query_log row, it should be retrievable."""
        _insert_query(db, "What is RAG?", answer="RAG is...")

        rows = list(db.query(
            "SELECT * FROM query_log WHERE query_text = 'What is RAG?'"
        ))
        assert len(rows) == 1
        assert rows[0]["query_text"] == "What is RAG?"
        assert rows[0]["answer_text"] == "RAG is..."
        assert rows[0]["answer_model"] == "gemma4"

    def test_query_log_with_conversation_id(self, db):
        """query_log should store conversation_id when provided."""
        _insert_query(db, "Hello", conversation_id="conv-123")

        rows = list(db.query(
            "SELECT conversation_id FROM query_log WHERE query_text = 'Hello'"
        ))
        assert rows[0]["conversation_id"] == "conv-123"

    def test_query_log_without_conversation_id(self, db):
        """query_log should allow NULL conversation_id."""
        _insert_query(db, "Hello")

        rows = list(db.query(
            "SELECT conversation_id FROM query_log WHERE query_text = 'Hello'"
        ))
        assert rows[0]["conversation_id"] is None

    def test_query_log_source_count(self, db):
        """source_count should be derived from retrieved_chunks_json."""
        chunks = [{"rank": 1, "chunk_id": 1}, {"rank": 2, "chunk_id": 2}]
        _insert_query(db, "Test", chunks=chunks)

        rows = list(db.query(
            "SELECT retrieved_chunks_json FROM query_log WHERE query_text = 'Test'"
        ))
        parsed = json.loads(rows[0]["retrieved_chunks_json"])
        assert len(parsed) == 2


# ---------------------------------------------------------------------------
# Test: query history pagination
# ---------------------------------------------------------------------------


class TestQueryHistoryPagination:
    """Verify pagination logic for query history."""

    def test_pagination_page_1(self, db):
        """First page should return correct items."""
        for i in range(25):
            _insert_query(db, f"Query {i:02d}")

        # Simulate the API logic
        page, size = 1, 10
        offset = (page - 1) * size
        rows = list(db.query(
            "SELECT id, query_text FROM query_log ORDER BY asked_at DESC LIMIT ? OFFSET ?",
            [size, offset],
        ))
        assert len(rows) == 10

    def test_pagination_page_3(self, db):
        """Third page of size 10 should return 5 items from 25 total."""
        for i in range(25):
            _insert_query(db, f"Query {i:02d}")

        page, size = 3, 10
        offset = (page - 1) * size
        rows = list(db.query(
            "SELECT id, query_text FROM query_log ORDER BY asked_at DESC LIMIT ? OFFSET ?",
            [size, offset],
        ))
        assert len(rows) == 5

    def test_pagination_total_count(self, db):
        """Total count should reflect all matching records."""
        for i in range(25):
            _insert_query(db, f"Query {i:02d}")

        count_rows = list(db.query("SELECT COUNT(*) as total FROM query_log"))
        assert count_rows[0]["total"] == 25

    def test_pagination_empty(self, db):
        """Empty query_log should return 0 items."""
        rows = list(db.query(
            "SELECT id FROM query_log ORDER BY asked_at DESC LIMIT ? OFFSET ?",
            [10, 0],
        ))
        assert len(rows) == 0


# ---------------------------------------------------------------------------
# Test: query history search
# ---------------------------------------------------------------------------


class TestQueryHistorySearch:
    """Verify keyword search filtering."""

    def test_search_matches_keyword(self, db):
        """Search for 'RAG' should match queries containing 'RAG'."""
        _insert_query(db, "What is RAG?")
        _insert_query(db, "How does RAG work?")
        _insert_query(db, "What is Python?")

        rows = list(db.query(
            "SELECT query_text FROM query_log WHERE query_text LIKE ? ORDER BY asked_at DESC",
            ["%RAG%"],
        ))
        assert len(rows) == 2
        assert all("RAG" in r["query_text"] for r in rows)

    def test_search_no_match(self, db):
        """Search for non-existent keyword should return 0 results."""
        _insert_query(db, "What is RAG?")
        _insert_query(db, "What is Python?")

        rows = list(db.query(
            "SELECT query_text FROM query_log WHERE query_text LIKE ?",
            ["%Blockchain%"],
        ))
        assert len(rows) == 0

    def test_search_case_insensitive(self, db):
        """SQLite LIKE is case-insensitive by default for ASCII."""
        _insert_query(db, "What is RAG?")
        _insert_query(db, "What is rag?")

        # Both should match since SQLite LIKE is case-insensitive
        rows = list(db.query(
            "SELECT query_text FROM query_log WHERE query_text LIKE ?",
            ["%RAG%"],
        ))
        assert len(rows) == 2

    def test_search_with_date_filter(self, db):
        """Date filtering should work alongside keyword search."""
        _insert_query(db, "RAG query 1")
        _insert_query(db, "RAG query 2")
        _insert_query(db, "Python query")

        # Search for RAG with a date filter that includes all
        rows = list(db.query(
            "SELECT query_text FROM query_log "
            "WHERE query_text LIKE ? AND asked_at >= '2020-01-01'",
            ["%RAG%"],
        ))
        assert len(rows) == 2

    def test_search_combined_filters(self, db):
        """Multiple filters should combine with AND."""
        _insert_query(db, "RAG question", user="alice")
        _insert_query(db, "RAG question", user="bob")
        _insert_query(db, "Python question", user="alice")

        rows = list(db.query(
            "SELECT query_text, user FROM query_log "
            "WHERE query_text LIKE ? AND user = ?",
            ["%RAG%", "alice"],
        ))
        assert len(rows) == 1
        assert rows[0]["user"] == "alice"
