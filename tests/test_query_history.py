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

from pipeline.db import migrate

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path: Path):
    """Create a temporary database with all migrations applied.

    check_same_thread=False: GET /queries runs in TestClient's worker thread
    (OCR audit #61 — these tests now drive the real endpoint).
    """
    import sqlite3

    import sqlite_utils

    conn = sqlite3.connect(str(tmp_path / "test.db"), check_same_thread=False)
    database = sqlite_utils.Database(conn)
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


def _get(db, params):
    """OCR audit #61: drive the REAL GET /queries endpoint (the old tests
    wrote their own SQL and asserted SQLite returns what SQLite returns —
    the endpoint's pagination/escaping/date logic was never executed)."""
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    from api.main import app

    with patch("api.main._get_db", return_value=db):
        r = TestClient(app).get("/queries", params=params)
    assert r.status_code == 200, r.text
    return r.json()


class TestQueryHistoryPagination:
    """Verify pagination via GET /queries."""

    def test_pagination_page_1(self, db):
        for i in range(25):
            _insert_query(db, f"Query {i:02d}")
        body = _get(db, {"page": 1, "size": 10})
        assert len(body["items"]) == 10
        assert body["total"] == 25
        assert body["page"] == 1

    def test_pagination_page_3(self, db):
        for i in range(25):
            _insert_query(db, f"Query {i:02d}")
        body = _get(db, {"page": 3, "size": 10})
        assert len(body["items"]) == 5
        assert body["total"] == 25

    def test_pagination_total_count(self, db):
        for i in range(25):
            _insert_query(db, f"Query {i:02d}")
        assert _get(db, {})["total"] == 25

    def test_pagination_empty(self, db):
        body = _get(db, {"page": 1, "size": 10})
        assert body["items"] == []
        assert body["total"] == 0


class TestQueryHistorySearch:
    """Verify keyword/date filtering via GET /queries."""

    def test_search_matches_keyword(self, db):
        _insert_query(db, "What is RAG?")
        _insert_query(db, "How does RAG work?")
        _insert_query(db, "What is Python?")
        body = _get(db, {"keyword": "RAG"})
        assert body["total"] == 2
        assert all("RAG" in i["query_text"] for i in body["items"])

    def test_search_no_match(self, db):
        _insert_query(db, "What is RAG?")
        _insert_query(db, "What is Python?")
        assert _get(db, {"keyword": "Blockchain"})["total"] == 0

    def test_search_case_insensitive(self, db):
        _insert_query(db, "What is RAG?")
        _insert_query(db, "What is rag?")
        assert _get(db, {"keyword": "RAG"})["total"] == 2

    def test_search_wildcard_is_literal(self, db):
        _insert_query(db, "Coverage is 100% today")
        _insert_query(db, "Coverage is 1000 today")
        body = _get(db, {"keyword": "100%"})
        assert body["total"] == 1
        assert "100%" in body["items"][0]["query_text"]

    def test_search_with_date_filter(self, db):
        _insert_query(db, "RAG query 1")
        _insert_query(db, "RAG query 2")
        _insert_query(db, "Python query")
        body = _get(db, {"keyword": "RAG", "start_date": "2020-01-01"})
        assert body["total"] == 2

    def test_end_date_includes_final_day(self, db):
        _insert_query(db, "late RAG")
        _insert_query(db, "next day RAG")
        db.conn.execute(
            "UPDATE query_log SET asked_at = '2026-01-31 15:00:00' WHERE query_text = 'late RAG'"
        )
        db.conn.execute(
            "UPDATE query_log SET asked_at = '2026-02-01 00:00:01' WHERE query_text = 'next day RAG'"
        )
        db.conn.commit()
        body = _get(db, {"keyword": "RAG", "end_date": "2026-01-31"})
        assert body["total"] == 1
        assert body["items"][0]["query_text"] == "late RAG"

    def test_search_combined_filters(self, db):
        _insert_query(db, "RAG question", user="alice")
        _insert_query(db, "RAG question", user="bob")
        _insert_query(db, "Python question", user="alice")
        db.conn.execute(
            "UPDATE query_log SET asked_at = '2026-03-10 00:00:00' WHERE user = 'bob'"
        )
        db.conn.execute(
            "UPDATE query_log SET asked_at = '2026-01-10 00:00:00' WHERE user = 'alice'"
        )
        db.conn.commit()
        body = _get(db, {
            "keyword": "RAG", "start_date": "2026-01-01", "end_date": "2026-01-31",
        })
        assert body["total"] == 1
        assert body["items"][0]["query_text"] == "RAG question"
