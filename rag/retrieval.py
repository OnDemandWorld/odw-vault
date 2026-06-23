"""Hybrid retrieval pipeline: dense + BM25 + RRF fusion."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import chromadb

logger = logging.getLogger(__name__)


@dataclass
class Hit:
    chunk_id: int
    file_id: int
    folder_id: int
    rel_path: str
    page_start: int | None
    text: str
    dense_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None
    fused_score: float | None = None


def retrieve(
    query: str,
    db,
    chroma_client,
    chroma_path: str,
    cfg,
    folder_filter: dict | None = None,
    top_k_chunks: int | None = None,
    use_reranker: bool | None = None,
    use_augmentation: bool | None = None,
) -> tuple[list[Hit], dict]:
    """Execute the full retrieval pipeline.

    Returns (hits, metrics).
    """
    t0 = time.monotonic()
    metrics: dict = {}

    # Safety: rebuild FTS to ensure index is in sync
    try:
        db.execute('INSERT INTO chunk_fts(chunk_fts) VALUES("rebuild")')
        db.conn.commit()
    except Exception:
        pass

    # 1. Config assertion: verify Chroma collection exists and embedding model matches
    suffix = cfg.models.embedding.collection_suffix
    chunk_collection_name = f"chunks__{suffix}"

    client = chromadb.PersistentClient(path=chroma_path)
    try:
        coll = client.get_collection(chunk_collection_name)
    except Exception:
        raise RuntimeError(
            f"Chroma collection '{chunk_collection_name}' does not exist. "
            "Run the embedding phase first."
        ) from None

    stored_model = coll.metadata.get("embedding_model")
    expected_model = cfg.models.embedding.name
    if stored_model and stored_model != expected_model:
        raise RuntimeError(
            f"Embedding model mismatch: collection has '{stored_model}', "
            f"config specifies '{expected_model}'. Re-run embedding with the correct model."
        )

    # 2. Query language detection (fasttext)
    query_lang = _detect_language(query, cfg)
    metrics["query_lang"] = query_lang

    # 3. Folder filter resolution
    allowed_file_ids = None
    if folder_filter:
        from rag.filters import resolve_folder_filter

        allowed_file_ids = resolve_folder_filter(db, folder_filter)
        if allowed_file_ids is None:
            logger.info("folder_filter resolved to empty set; no results will match")
            return [], metrics

    # 4. Hierarchical narrowing
    # Embed query once for all dense searches
    import ollama as _ollama_mod

    try:
        _qc = _ollama_mod.Client(host=cfg.ollama.host)
        _qr = _qc.embed(model=cfg.models.embedding.name, input=[query])
        _query_emb = _qr["embeddings"][0]
    except Exception as _e:
        logger.error("Failed to embed query for retrieval: %s", _e)
        return [], metrics

    candidate_file_ids = None
    if cfg.retrieval.hierarchical:
        candidate_file_ids = _hierarchical_narrowing(
            client, suffix, _query_emb, query, cfg, allowed_file_ids
        )

    # Exclude derived/meta files from search (preflight_report.md
    # contains folder listings that poison keyword matching)
    _excluded_file_ids = db.query(
        "SELECT id FROM file WHERE rel_path = 'preflight_report.md'"
    )
    excluded_ids = {r["id"] for r in _excluded_file_ids}

    # Augment candidates with path-matching files for BM25
    # This ensures files with query terms in their path (e.g. "kwh")
    # are included even if hierarchical narrowing didn't pick them
    _path_match_ids = _find_files_by_path(db, query, excluded_ids)
    if candidate_file_ids is not None:
        candidate_file_ids = (candidate_file_ids | _path_match_ids) - excluded_ids
    elif _path_match_ids:
        candidate_file_ids = _path_match_ids - excluded_ids

    # 5. Dense chunk retrieval
    dense_candidates = cfg.retrieval.dense_candidates
    dense_hits = _dense_retrieve_with_embedding(
        coll,
        _query_emb,
        dense_candidates,
        candidate_file_ids,
    )
    metrics["dense_hits"] = len(dense_hits)

    # 6. BM25 chunk retrieval
    bm25_candidates = cfg.retrieval.bm25_candidates
    bm25_hits = _bm25_retrieve(db, query, bm25_candidates, candidate_file_ids, excluded_ids)

    # 6b. Path-matching BM25: find chunks from files whose path contains query tokens
    # This catches cases like "kwh" which appear in file paths but not chunk text
    path_hits = _path_match_retrieve(db, query, candidate_file_ids, excluded_ids, n_results=30)
    # Merge path_hits into bm25_hits — path_hits go FIRST so they get high RRF rank
    bm25_chunk_ids = {h.chunk_id for h in bm25_hits}
    bm25_hits = [ph for ph in path_hits if ph.chunk_id not in bm25_chunk_ids] + bm25_hits

    metrics["bm25_hits"] = len(bm25_hits)

    # 7. Reciprocal Rank Fusion
    rrf_k = cfg.retrieval.rrf_k
    fused = reciprocal_rank_fuse(dense_hits, bm25_hits, k=rrf_k)
    metrics["fused_total"] = len(fused)

    # 8. Reranking — skip if disabled
    reranker_enabled = (
        use_reranker if use_reranker is not None else getattr(cfg.models.reranker, "enabled", False)
    )
    if reranker_enabled:
        # Reranker not implemented yet; fused results pass through.
        logger.info("Reranker requested but not yet implemented; returning fused results.")

    # Truncate to top_k_chunks
    top_k = top_k_chunks if top_k_chunks is not None else cfg.retrieval.top_k_chunks
    final_hits = fused[:top_k]

    # 9. Context assembly
    _assemble_context(final_hits, db)

    elapsed = (time.monotonic() - t0) * 1000
    metrics["retrieval_ms"] = round(elapsed, 1)

    return final_hits, metrics


def reciprocal_rank_fuse(dense_hits: list[Hit], bm25_hits: list[Hit], k: int) -> list[Hit]:
    """RRF fusion of dense and BM25 results.

    Rank-based fusion: each hit gets 1/(k + rank + 1) from each list it appears in.
    """
    by_id: dict[int, Hit] = {}
    for rank, h in enumerate(dense_hits):
        h.fused_score = 1.0 / (k + rank + 1)
        by_id[h.chunk_id] = h
    for rank, h in enumerate(bm25_hits):
        c = 1.0 / (k + rank + 1)
        if h.chunk_id in by_id:
            by_id[h.chunk_id].fused_score += c
            by_id[h.chunk_id].bm25_score = h.bm25_score
        else:
            h.fused_score = c
            by_id[h.chunk_id] = h
    return sorted(by_id.values(), key=lambda x: x.fused_score, reverse=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _detect_language(query: str, cfg) -> str:
    """Detect query language using fasttext, falling back to lingua."""
    # Try fasttext first
    try:
        import fasttext

        model_path = cfg.models.language_id.model_path
        model = fasttext.load_model(model_path)
        predictions = model.predict(query.replace("\n", " "), k=1)
        lang_code = predictions[0][0].replace("__label__", "")
        return lang_code
    except Exception as e:
        logger.debug("fasttext language detection failed: %s", e)

    # Fallback to lingua
    try:
        from lingua import Language, LanguageDetectorBuilder

        detector = LanguageDetectorBuilder.from_languages(
            Language.ENGLISH, Language.CHINESE
        ).build()
        lang = detector.detect_language_of(query)
        if lang == Language.ENGLISH:
            return "en"
        elif lang == Language.CHINESE:
            return "zh"
        return lang.iso_code_639_1.name.lower() if lang else "unknown"
    except Exception:
        return "unknown"


def _hierarchical_narrowing(
    client,
    suffix: str,
    query_embedding: list[float],
    query: str,
    cfg,
    allowed_file_ids: set[int] | None,
) -> set[int] | None:
    """Narrow candidate files via folder and summary collections."""
    candidate_ids: set[int] = set()

    # Folder-level search
    folder_coll_name = f"folders__{suffix}"
    try:
        folder_coll = client.get_collection(folder_coll_name)
        folder_results = folder_coll.query(
            query_embeddings=[query_embedding],
            n_results=cfg.retrieval.top_k_folders,
        )
        if folder_results["ids"] and folder_results["ids"][0]:
            for meta in folder_results.get("metadatas", [[]])[0]:
                if meta and "folder_id" in meta:
                    fid = int(meta["folder_id"])
                    # Get file_ids belonging to this folder
                    rows = _query_db_for_files_in_folder(meta.get("_db"), fid)
                    candidate_ids.update(rows)
    except Exception:
        logger.debug("Folder collection '%s' not available, skipping", folder_coll_name)

    # Summary-level search
    summary_coll_name = f"summaries__{suffix}"
    try:
        summary_coll = client.get_collection(summary_coll_name)
        summary_results = summary_coll.query(
            query_embeddings=[query_embedding],
            n_results=cfg.retrieval.top_k_documents,
        )
        if summary_results["ids"] and summary_results["ids"][0]:
            for meta in summary_results.get("metadatas", [[]])[0]:
                if meta and "file_id" in meta:
                    candidate_ids.add(int(meta["file_id"]))
    except Exception:
        logger.debug("Summary collection '%s' not available, skipping", summary_coll_name)

    if not candidate_ids:
        return allowed_file_ids

    if allowed_file_ids is not None:
        candidate_ids &= allowed_file_ids

    return candidate_ids if candidate_ids else None


def _query_db_for_files_in_folder(db, folder_id: int) -> set[int]:
    """Get all file_ids belonging to a folder (and optionally subfolders)."""
    # If db is not a real db object, try to query from chunk metadata
    try:
        rows = db.query(
            "SELECT id FROM file WHERE folder_id = ? AND excluded = 0",
            [folder_id],
        )
        return {r["id"] for r in rows}
    except Exception:
        return set()


def _dense_retrieve_with_embedding(
    collection,
    query_embedding: list[float],
    n_results: int,
    candidate_file_ids: set[int] | None,
) -> list[Hit]:
    """Retrieve chunks via dense vector similarity from Chroma."""
    where_clause = None
    if candidate_file_ids is not None:
        where_clause = {"file_id": {"$in": list(candidate_file_ids)}}

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause,
            include=["metadatas", "documents", "distances"],
        )
    except Exception as e:
        logger.error("Dense retrieval failed: %s", e)
        return []

    hits: list[Hit] = []
    ids = results.get("ids", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for i, cid in enumerate(ids):
        meta = metadatas[i] if i < len(metadatas) else {}
        text = documents[i] if i < len(documents) else ""
        dist = distances[i] if i < len(distances) else None
        score = 1.0 - dist if dist is not None else None

        hits.append(
            Hit(
                chunk_id=int(meta.get("chunk_id", cid.removeprefix("c_"))),
                file_id=int(meta.get("file_id", 0)),
                folder_id=int(meta.get("folder_id", 0)),
                rel_path=meta.get("rel_path", ""),
                page_start=meta.get("start_page", None),
                text=text,
                dense_score=score,
            )
        )

    return hits


def _find_files_by_path(db, query: str, excluded_ids: set[int]) -> set[int]:
    """Find file_ids whose rel_path contains any non-stop-word query token."""
    STOP_WORDS = frozenset([
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
        'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
        'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
        'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
        'so', 'than', 'too', 'very', 'just', 'about', 'up', 'it', 'its',
        'this', 'that', 'these', 'those', 'what', 'which', 'who', 'whom',
        'i', 'me', 'my', 'myself', 'you', 'your', 'he', 'him', 'his',
        'she', 'her', 'we', 'our', 'they', 'them', 'their', 'and', 'or',
        'but', 'if', 'while', 'tell', 'get', 'know', 'like', 'want', 'need',
        'use', 'used', 'am', 'also', 'much', 'any', 'make', 'made',
    ])
    tokens = [t.lower() for t in query.split() if t.lower() not in STOP_WORDS and len(t) >= 2]
    if not tokens:
        return set()

    conditions = " OR ".join("f.rel_path LIKE ?" for _ in tokens)
    exc = ",".join("?" for _ in excluded_ids) if excluded_ids else ""
    exc_clause = f" AND f.id NOT IN ({exc})" if exc else ""
    params = [f"%{t}%" for t in tokens]
    if excluded_ids:
        params.extend(excluded_ids)

    sql = f"SELECT f.id FROM file f WHERE ({conditions}){exc_clause}"
    rows = db.query(sql, params)
    return {r["id"] for r in rows}


def _path_match_retrieve(
    db,
    query: str,
    candidate_file_ids: set[int] | None,
    excluded_file_ids: set[int] | None,
    n_results: int = 30,
) -> list[Hit]:
    """Retrieve chunks from files whose rel_path matches query tokens.

    Scores files by how many query tokens match the path, then returns
    top chunks from the best-matching files.
    """
    STOP_WORDS = frozenset([
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
        'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
        'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
        'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
        'so', 'than', 'too', 'very', 'just', 'about', 'up', 'it', 'its',
        'this', 'that', 'these', 'those', 'what', 'which', 'who', 'whom',
        'i', 'me', 'my', 'myself', 'you', 'your', 'he', 'him', 'his',
        'she', 'her', 'we', 'our', 'they', 'them', 'their', 'and', 'or',
        'but', 'if', 'while', 'tell', 'get', 'know', 'like', 'want', 'need',
        'use', 'used', 'am', 'also', 'much', 'any', 'make', 'made',
    ])
    tokens = [t.lower() for t in query.split() if t.lower() not in STOP_WORDS and len(t) >= 2]
    if not tokens:
        return []

    # Find all matching files and score by token count
    filter_sql = ""
    params: list = []
    if candidate_file_ids is not None:
        placeholders = ",".join("?" for _ in candidate_file_ids)
        filter_sql = f" AND f.id IN ({placeholders})"
        params.extend(candidate_file_ids)
    if excluded_file_ids:
        exc = ",".join("?" for _ in excluded_file_ids)
        filter_sql += f" AND f.id NOT IN ({exc})"
        params.extend(excluded_file_ids)

    rows = db.query(f"SELECT f.id, f.rel_path FROM file f WHERE 1=1{filter_sql}", params)

    # Score each file by how many tokens match its path
    scored: list[tuple[int, int, str]] = []  # (file_id, match_count, rel_path)
    for r in rows:
        path = r["rel_path"].lower()
        match_count = sum(1 for t in tokens if t in path)
        if match_count > 0:
            scored.append((r["id"], match_count, r["rel_path"]))

    # Sort by match count DESC, take top files
    scored.sort(key=lambda x: -x[1])
    max_matches = scored[0][1] if scored else 0
    # Keep only files with max match count (most specific matches)
    best_files = [(fid, path) for fid, cnt, path in scored if cnt == max_matches][:10]

    if not best_files:
        return []

    # Get chunks from best-matching files
    file_ids = [f[0] for f in best_files]
    placeholders = ",".join("?" for _ in file_ids)
    sql = (
        f"SELECT c.id as chunk_id, c.file_id, c.start_page, c.text, "
        f"f.folder_id, f.rel_path "
        f"FROM chunk c JOIN file f ON c.file_id = f.id "
        f"WHERE c.file_id IN ({placeholders}) "
        f"ORDER BY c.chunk_index LIMIT ?"
    )
    rows = db.query(sql, [*file_ids, n_results])

    hits = []
    for row in rows:
        hits.append(
            Hit(
                chunk_id=row["chunk_id"],
                file_id=row["file_id"],
                folder_id=row["folder_id"],
                rel_path=row["rel_path"],
                page_start=row["start_page"],
                text=row["text"],
                bm25_score=0.5,
            )
        )
    return hits


def _bm25_retrieve(
    db,
    query: str,
    n_results: int,
    candidate_file_ids: set[int] | None,
    excluded_file_ids: set[int] | None = None,
) -> list[Hit]:
    """Retrieve chunks via BM25 from chunk_fts FTS5 table."""
    import re as _re

    # Strip FTS5 special characters and filter stop words
    FTS5_SPECIAL = _re.compile(r'[\[\]\(\)\~\^\{\}\|\&\<\>\:\-\!\*\+\?\"\\\\]')
    STOP_WORDS = frozenset([
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
        'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
        'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
        'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
        'so', 'than', 'too', 'very', 'just', 'about', 'up', 'it', 'its',
        'this', 'that', 'these', 'those', 'what', 'which', 'who', 'whom',
        'i', 'me', 'my', 'myself', 'you', 'your', 'he', 'him', 'his',
        'she', 'her', 'we', 'our', 'they', 'them', 'their', 'and', 'or',
        'but', 'if', 'while', 'tell', 'get', 'know', 'like', 'want', 'need',
        'use', 'used', 'am', 'also', 'much', 'any', 'make', 'made',
    ])
    clean_tokens = []
    for t in query.split():
        if t.lower() not in STOP_WORDS:
            cleaned = FTS5_SPECIAL.sub('', t)
            if cleaned and len(cleaned) >= 2:
                clean_tokens.append(cleaned)
    if not clean_tokens:
        return []
    fts_query = " OR ".join(clean_tokens)

    if candidate_file_ids is not None:
        # candidate_file_ids already has excluded_ids removed (set subtraction above)
        placeholders = ",".join("?" for _ in candidate_file_ids)
        sql = (
            "SELECT c.id as chunk_id, c.file_id, c.start_page, c.text, "
            "f.folder_id, f.rel_path, rank "
            "FROM chunk_fts "
            "JOIN chunk c ON chunk_fts.rowid = c.id "
            "JOIN file f ON c.file_id = f.id "
            f"WHERE chunk_fts MATCH ? AND c.file_id IN ({placeholders}) "
            "ORDER BY rank LIMIT ?"
        )
        params = [fts_query, *list(candidate_file_ids), n_results]
    else:
        # No hierarchical narrowing; apply exclusion filter directly
        if excluded_file_ids:
            exc_placeholders = ",".join("?" for _ in excluded_file_ids)
            sql = (
                "SELECT c.id as chunk_id, c.file_id, c.start_page, c.text, "
                "f.folder_id, f.rel_path, rank "
                "FROM chunk_fts "
                "JOIN chunk c ON chunk_fts.rowid = c.id "
                "JOIN file f ON c.file_id = f.id "
                f"WHERE chunk_fts MATCH ? AND c.file_id NOT IN ({exc_placeholders}) "
                "ORDER BY rank LIMIT ?"
            )
            params = [fts_query, *list(excluded_file_ids), n_results]
        else:
            sql = (
                "SELECT c.id as chunk_id, c.file_id, c.start_page, c.text, "
                "f.folder_id, f.rel_path, rank "
                "FROM chunk_fts "
                "JOIN chunk c ON chunk_fts.rowid = c.id "
                "JOIN file f ON c.file_id = f.id "
                "WHERE chunk_fts MATCH ? "
                "ORDER BY rank LIMIT ?"
            )
            params = [fts_query, n_results]

    rows = db.query(sql, params)

    hits: list[Hit] = []
    for row in rows:
        # rank in FTS5: lower (more negative) = better; normalize to 0-1
        raw_rank = row.get("rank", 0)
        bm25_score = max(0.0, 1.0 + raw_rank) if raw_rank is not None else None

        hits.append(
            Hit(
                chunk_id=row["chunk_id"],
                file_id=row["file_id"],
                folder_id=row["folder_id"],
                rel_path=row["rel_path"],
                page_start=row["start_page"],
                text=row["text"],
                bm25_score=bm25_score,
            )
        )

    return hits


def _assemble_context(hits: list[Hit], db) -> None:
    """Build numbered context with file path and page reference.

    Uses chunk.text (the original chunk), never chunk.context_text.
    Orders final chunks by (file_id, ordinal).
    """
    # Sort by file_id then by chunk_id (ordinal within file)
    hits.sort(key=lambda h: (h.file_id, h.chunk_id))

    for i, hit in enumerate(hits, start=1):
        # Build a human-readable context entry
        page_info = f", page {hit.page_start}" if hit.page_start else ""
        hit.text = f"[{i}] {hit.rel_path}{page_info}\n{hit.text}"
