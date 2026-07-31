"""Integration tests for V1.2 M2 Chinese retrieval at the BM25/FTS layer.

Verifies that a Chinese query hits a Chinese document via the additive
``chunk_fts_zh`` index, while the default (English) BM25 path is unchanged.
"""

from __future__ import annotations

from rag.retrieval import _bm25_retrieve
from rag.tokenization import index_zh_chunks
from tests.conftest import seed_test_files

ZH_TEXT = "知识库检索系统支持中文查询与分词"
EN_TEXT = "knowledge base retrieval system with english tokens"


def _insert_chunk(db, file_id: int, text: str, index: int = 0) -> int:
    """Insert a chunk row (FTS trigger populates chunk_fts) and return its id."""
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


class TestChineseBm25Retrieval:
    def test_zh_query_hits_zh_document(self, test_db):
        file_ids = seed_test_files(test_db)
        zh_chunk = _insert_chunk(test_db, file_ids[0], ZH_TEXT)
        # Populate the additive Chinese index (jieba-or-bigram tokens).
        indexed = index_zh_chunks(test_db, [(zh_chunk, ZH_TEXT)])
        assert indexed == 1

        hits = _bm25_retrieve(
            test_db,
            "中文查询",
            n_results=10,
            candidate_file_ids=None,
            excluded_file_ids=None,
            query_lang="zh",
        )
        assert any(h.chunk_id == zh_chunk for h in hits)
        assert all(h.bm25_score is not None for h in hits)

    def test_zh_query_respects_candidate_filter(self, test_db):
        file_ids = seed_test_files(
            test_db,
            files=[
                {"name": "zh1.txt", "rel_path": "zh1.txt"},
                {"name": "zh2.txt", "rel_path": "zh2.txt"},
            ],
        )
        c1 = _insert_chunk(test_db, file_ids[0], ZH_TEXT, index=0)
        c2 = _insert_chunk(test_db, file_ids[1], ZH_TEXT, index=0)
        index_zh_chunks(test_db, [(c1, ZH_TEXT), (c2, ZH_TEXT)])

        # Restrict to file 1 only — chunk 2 must not appear.
        hits = _bm25_retrieve(
            test_db,
            "中文查询",
            n_results=10,
            candidate_file_ids={file_ids[0]},
            excluded_file_ids=None,
            query_lang="zh",
        )
        hit_ids = {h.chunk_id for h in hits}
        assert c1 in hit_ids
        assert c2 not in hit_ids

    def test_zh_index_only_indexes_chinese(self, test_db):
        file_ids = seed_test_files(test_db)
        en_chunk = _insert_chunk(test_db, file_ids[0], EN_TEXT)
        # English chunk must NOT be added to chunk_fts_zh.
        indexed = index_zh_chunks(test_db, [(en_chunk, EN_TEXT)])
        assert indexed == 0
        count = next(
            iter(test_db.query("SELECT COUNT(*) AS c FROM chunk_fts_zh"))
        )["c"]
        assert count == 0


class TestEnglishPathUnchanged:
    def test_default_path_queries_chunk_fts(self, test_db):
        file_ids = seed_test_files(test_db)
        en_chunk = _insert_chunk(test_db, file_ids[0], EN_TEXT)

        # query_lang=None (default) and "en" must both use the original
        # chunk_fts index and find the English chunk.
        for lang in (None, "en"):
            hits = _bm25_retrieve(
                test_db,
                "retrieval",
                n_results=10,
                candidate_file_ids=None,
                excluded_file_ids=None,
                query_lang=lang,
            )
            assert any(h.chunk_id == en_chunk for h in hits), f"lang={lang}"

    def test_zh_query_does_not_match_english_only_index(self, test_db):
        file_ids = seed_test_files(test_db)
        _insert_chunk(test_db, file_ids[0], EN_TEXT)
        # No Chinese index entries exist → zh query returns nothing.
        hits = _bm25_retrieve(
            test_db,
            "中文查询",
            n_results=10,
            candidate_file_ids=None,
            excluded_file_ids=None,
            query_lang="zh",
        )
        assert hits == []
