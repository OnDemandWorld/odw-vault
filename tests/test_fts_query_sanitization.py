"""Regression tests: user query punctuation must never reach FTS5 MATCH as syntax.

A query like "purchases above 50,000 USD" used to raise
``sqlite3.OperationalError: fts5: syntax error near ","`` (HTTP 500) because
the token cleaner only stripped an enumerated set of special characters.
"""

from __future__ import annotations

from rag.retrieval import _bm25_retrieve
from tests.conftest import seed_test_files

DOC_TEXT = (
    "Purchases above 50,000 USD require dual sign-off from the finance "
    "director. The contract value is 1.6 million USD."
)


def _insert_chunk(db, file_id: int, text: str, index: int = 0) -> int:
    db["chunk"].insert(
        {
            "file_id": file_id,
            "chunk_index": index,
            "text": text,
            "token_count": max(1, len(text) // 4),
        }
    )
    db.conn.commit()
    return next(iter(db.query("SELECT id FROM chunk ORDER BY id DESC LIMIT 1")))["id"]


class TestFtsQueryPunctuation:
    def test_comma_query_does_not_raise_and_matches(self, test_db):
        file_ids = seed_test_files(test_db)
        chunk_id = _insert_chunk(test_db, file_ids[0], DOC_TEXT)
        hits = _bm25_retrieve(
            test_db,
            "What approvals are needed for purchases above 50,000 USD?",
            n_results=10,
            candidate_file_ids=None,
            excluded_file_ids=None,
        )
        assert any(h.chunk_id == chunk_id for h in hits)

    def test_comma_query_with_candidate_filter(self, test_db):
        # The scoped-query path (folder_filter) batches the candidate IN list;
        # it hit the same syntax error in production.
        file_ids = seed_test_files(test_db)
        chunk_id = _insert_chunk(test_db, file_ids[0], DOC_TEXT)
        hits = _bm25_retrieve(
            test_db,
            "purchases above 50,000 USD",
            n_results=10,
            candidate_file_ids=set(file_ids),
            excluded_file_ids=None,
        )
        assert any(h.chunk_id == chunk_id for h in hits)

    def test_hyphenated_query_matches_hyphenated_document(self, test_db):
        file_ids = seed_test_files(test_db)
        chunk_id = _insert_chunk(test_db, file_ids[0], DOC_TEXT)
        hits = _bm25_retrieve(
            test_db,
            "dual sign-off",
            n_results=10,
            candidate_file_ids=None,
            excluded_file_ids=None,
        )
        assert any(h.chunk_id == chunk_id for h in hits)

    def test_mixed_punctuation_does_not_raise(self, test_db):
        file_ids = seed_test_files(test_db)
        _insert_chunk(test_db, file_ids[0], DOC_TEXT)
        for q in (
            'what\'s the "finance" policy?',
            "approvals (finance); director: sign-off!",
            "cost > 50,000 & value < 2,000,000",
            "contract [value] {1.6} million* USD",
            "e-mail: finance@corp.example #policy",
        ):
            _bm25_retrieve(
                test_db, q, n_results=5, candidate_file_ids=None,
                excluded_file_ids=None,
            )

    def test_punctuation_only_query_returns_empty(self, test_db):
        file_ids = seed_test_files(test_db)
        _insert_chunk(test_db, file_ids[0], DOC_TEXT)
        hits = _bm25_retrieve(
            test_db,
            '!!! ??? ,,, "" ()',
            n_results=5,
            candidate_file_ids=None,
            excluded_file_ids=None,
        )
        assert hits == []
