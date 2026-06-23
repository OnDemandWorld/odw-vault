"""FastAPI HTTP service for the RAG pipeline."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import chromadb
import ollama
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, RedirectResponse
from sse_starlette import EventSourceResponse, ServerSentEvent

from api.schemas import (
    Citation,
    FeedbackRequest,
    FileResponse,
    FolderNode,
    HealthResponse,
    Metrics,
    ModelInfo,
    QueryRequest,
    QueryResponse,
)
from pipeline.config import load_app_config
from pipeline.db import migrate, open_db
from rag.filters import resolve_folder_filter
from rag.generation import generate_answer
from rag.retrieval import Hit, retrieve

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("config.toml")
DB_NAME = "corpus.db"

# ---------------------------------------------------------------------------
# Thread-local DB connections (FastAPI runs handlers in a thread pool)
# ---------------------------------------------------------------------------

_thread_local = threading.local()


def _get_db():
    """Return a thread-local DB connection."""
    if not hasattr(_thread_local, "db"):
        db = open_db(Path(DB_NAME))
        migrate(db)
        _thread_local.db = db
    return _thread_local.db


def _load_config():
    return load_app_config(CONFIG_PATH)


app = FastAPI(title="ODW.ai Vault RAG")


@app.get("/")
def root():
    """Redirect to Swagger UI docs."""
    return RedirectResponse(url="/docs")


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health():
    cfg = _load_config()

    # Ollama check
    ollama_ok = False
    try:
        client = ollama.Client(host=cfg.ollama.host)
        client.list()
        ollama_ok = True
    except Exception:
        logger.warning("Ollama not reachable at %s", cfg.ollama.host)

    # Chroma check
    chroma_ok = False
    try:
        chroma_path = str(cfg.chroma_root_path)
        client = chromadb.PersistentClient(path=chroma_path)
        suffix = cfg.models.embedding.collection_suffix
        coll_name = f"chunks__{suffix}"
        client.get_collection(coll_name)
        chroma_ok = True
    except Exception:
        logger.warning(
            "Chroma collection '%s' not found", coll_name if "coll_name" in dir() else "unknown"
        )

    # Database check
    db_ok = False
    try:
        db = _get_db()
        db.query("SELECT 1")
        db_ok = True
    except Exception:
        logger.warning("Database not reachable")

    # fastText check
    fasttext_ok = False
    try:
        import fasttext

        model_path = cfg.models.language_id.model_path
        fasttext.load_model(model_path)
        fasttext_ok = True
    except Exception:
        logger.warning("fastText model not found at %s", model_path)

    return HealthResponse(
        ollama=ollama_ok,
        chroma=chroma_ok,
        database=db_ok,
        fasttext=fasttext_ok,
    )


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    cfg = _load_config()
    db = _get_db()

    # Check Ollama reachability
    try:
        ollama.Client(host=cfg.ollama.host).list()
    except Exception:
        raise HTTPException(status_code=503, detail="Ollama is not reachable") from None

    # Check Chroma collection
    suffix = cfg.models.embedding.collection_suffix
    coll_name = f"chunks__{suffix}"
    try:
        chroma_path = str(cfg.chroma_root_path)
        client = chromadb.PersistentClient(path=chroma_path)
        client.get_collection(coll_name)
    except Exception:
        raise HTTPException(
            status_code=503,
            detail=f"Chroma collection '{coll_name}' not found. Run embedding phase first.",
        ) from None

    # Resolve folder filter
    allowed_file_ids: set[int] | None = None
    if req.folder_filter:
        filter_dict = req.folder_filter.model_dump(exclude_none=True)
        if filter_dict:
            allowed_file_ids = resolve_folder_filter(db, filter_dict)
            if allowed_file_ids is None:
                raise HTTPException(
                    status_code=422,
                    detail="folder_filter matches no files",
                )

    # Build folder_filter dict for retrieval
    folder_filter_dict = None
    if req.folder_filter:
        folder_filter_dict = req.folder_filter.model_dump(exclude_none=True)

    # Override thinking in config if requested
    if req.thinking is not None:
        cfg.models.generation.thinking = req.thinking

    # Retrieve
    t0 = time.monotonic()
    try:
        hits, retrieval_metrics = retrieve(
            query=req.query,
            db=db,
            chroma_client=None,  # retrieve() opens its own client from chroma_path
            chroma_path=chroma_path,
            cfg=cfg,
            folder_filter=folder_filter_dict,
            top_k_chunks=req.top_k_chunks,
            use_reranker=req.use_reranker,
            use_augmentation=req.use_augmentation,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from None

    # Generate answer
    gen_result = generate_answer(
        query=req.query,
        hits=hits,
        cfg=cfg,
    )

    total_ms = round((time.monotonic() - t0) * 1000)
    retrieval_ms = retrieval_metrics.get("retrieval_ms", None)
    generation_ms = gen_result.get("generation_ms", None)

    # Build citations
    citations = []
    for c in gen_result.get("citations", []):
        citations.append(
            Citation(
                marker=f"[{c['citation_number']}]",
                file_id=c["file_id"],
                rel_path=c["rel_path"],
                page=c.get("page_start"),
                chunk_id=c["chunk_id"],
                snippet=c["snippet"],
            )
        )

    # Build retrieved_chunks
    retrieved_chunks = []
    for i, hit in enumerate(hits, start=1):
        retrieved_chunks.append(
            {
                "rank": i,
                "chunk_id": hit.chunk_id,
                "file_id": hit.file_id,
                "folder_id": hit.folder_id,
                "rel_path": hit.rel_path,
                "page_start": hit.page_start,
                "text": hit.text,
                "dense_score": hit.dense_score,
                "bm25_score": hit.bm25_score,
                "fused_score": hit.fused_score,
            }
        )

    # Log query
    reranker_model = (
        cfg.models.reranker.name if getattr(cfg.models.reranker, "enabled", False) else None
    )
    augmentation_model = (
        cfg.models.contextual_retrieval.name
        if getattr(cfg.models.contextual_retrieval, "enabled", False)
        else None
    )

    cursor = db.conn.execute(
        """INSERT INTO query_log
           (user, query_text, query_lang, folder_filter_json,
            retrieved_chunks_json, answer_text, answer_model,
            embedding_model, reranker_model,
            latency_ms, retrieval_ms, generation_ms)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            req.user,
            req.query,
            retrieval_metrics.get("query_lang", "unknown"),
            json.dumps(folder_filter_dict) if folder_filter_dict else None,
            json.dumps(retrieved_chunks),
            gen_result["answer"],
            gen_result.get("model", cfg.models.generation.name),
            cfg.models.embedding.name,
            reranker_model,
            total_ms,
            retrieval_ms,
            generation_ms,
        ),
    )
    db.conn.commit()
    query_log_id = cursor.lastrowid

    return QueryResponse(
        answer=gen_result["answer"],
        citations=citations,
        retrieved_chunks=retrieved_chunks,
        metrics=Metrics(
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            total_ms=total_ms,
        ),
        models=ModelInfo(
            embedding=cfg.models.embedding.name,
            generation=cfg.models.generation.name,
            reranker=reranker_model,
            contextual_augmentation=augmentation_model,
        ),
        query_log_id=query_log_id,
    )


# ---------------------------------------------------------------------------
# POST /query/stream
# ---------------------------------------------------------------------------


@app.post("/query/stream")
def query_stream(req: QueryRequest):
    cfg = _load_config()
    db = _get_db()

    # Check Ollama
    try:
        ollama.Client(host=cfg.ollama.host).list()
    except Exception:
        raise HTTPException(status_code=503, detail="Ollama is not reachable") from None

    # Check Chroma
    suffix = cfg.models.embedding.collection_suffix
    coll_name = f"chunks__{suffix}"
    try:
        chroma_path = str(cfg.chroma_root_path)
        client = chromadb.PersistentClient(path=chroma_path)
        client.get_collection(coll_name)
    except Exception:
        raise HTTPException(
            status_code=503,
            detail=f"Chroma collection '{coll_name}' not found",
        ) from None

    # Resolve folder filter
    folder_filter_dict = None
    if req.folder_filter:
        filter_dict = req.folder_filter.model_dump(exclude_none=True)
        if filter_dict:
            allowed_file_ids = resolve_folder_filter(db, filter_dict)
            if allowed_file_ids is None:
                raise HTTPException(
                    status_code=422,
                    detail="folder_filter matches no files",
                )
        folder_filter_dict = filter_dict

    if req.thinking is not None:
        cfg.models.generation.thinking = req.thinking

    # Retrieve
    t0 = time.monotonic()
    try:
        hits, retrieval_metrics = retrieve(
            query=req.query,
            db=db,
            chroma_client=None,
            chroma_path=chroma_path,
            cfg=cfg,
            folder_filter=folder_filter_dict,
            top_k_chunks=req.top_k_chunks,
            use_reranker=req.use_reranker,
            use_augmentation=req.use_augmentation,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from None

    retrieval_ms = retrieval_metrics.get("retrieval_ms", None)

    # Build context for generation
    numbered_chunks = _format_chunks_for_prompt(hits)

    # Build citations from hits (before streaming, so we have them)
    from rag.citations import parse_citations as _parse_citations
    from rag.citations import resolve_citations as _resolve_citations

    # Stream generation
    model_name = cfg.models.generation.name
    system_prefix = "<|think|>" if getattr(cfg.models.generation, "thinking", False) else ""
    system_content = "You are a helpful assistant."
    if system_prefix:
        system_content = f"{system_prefix}\n{system_content}"

    prompt = DEFAULT_PROMPT.format(numbered_chunks=numbered_chunks, query=req.query)

    oclient = ollama.Client(host=cfg.ollama.host)

    async def event_generator():
        nonlocal t0
        try:
            # Event: retrieval summary
            yield ServerSentEvent(
                event="retrieval",
                data=json.dumps(
                    {
                        "n_chunks": len(hits),
                        "retrieval_ms": retrieval_ms,
                        "query_lang": retrieval_metrics.get("query_lang", "unknown"),
                    }
                ),
            )

            # Event: streaming tokens
            answer_parts: list[str] = []
            stream_resp = oclient.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt},
                ],
                options={
                    "temperature": cfg.models.generation.temperature,
                    "top_p": cfg.models.generation.top_p,
                    "top_k": cfg.models.generation.top_k,
                },
                stream=True,
            )

            for chunk in stream_resp:
                token = chunk.get("message", {}).get("content", "")
                if token:
                    answer_parts.append(token)
                    yield ServerSentEvent(event="token", data=token)

            answer_text = "".join(answer_parts)
            if not answer_text:
                answer_text = "I do not have enough information in the provided context to answer this question."

            # Parse citations
            citation_numbers = _parse_citations(answer_text)
            resolved = _resolve_citations(citation_numbers, hits)
            citations_out = [
                {
                    "marker": f"[{c['citation_number']}]",
                    "file_id": c["file_id"],
                    "rel_path": c["rel_path"],
                    "page": c.get("page_start"),
                    "chunk_id": c["chunk_id"],
                    "snippet": c["snippet"],
                }
                for c in resolved
            ]

            gen_ms = round((time.monotonic() - t0) * 1000 - (retrieval_ms or 0))

            # Event: citations
            yield ServerSentEvent(
                event="citations",
                data=json.dumps(citations_out),
            )

            total_ms = round((time.monotonic() - t0) * 1000)

            # Log query
            reranker_model = (
                cfg.models.reranker.name if getattr(cfg.models.reranker, "enabled", False) else None
            )
            cursor = db.conn.execute(
                """INSERT INTO query_log
                   (user, query_text, query_lang, folder_filter_json,
                    retrieved_chunks_json, answer_text, answer_model,
                    embedding_model, reranker_model,
                    latency_ms, retrieval_ms, generation_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    req.user,
                    req.query,
                    retrieval_metrics.get("query_lang", "unknown"),
                    json.dumps(folder_filter_dict) if folder_filter_dict else None,
                    json.dumps([]),
                    answer_text,
                    model_name,
                    cfg.models.embedding.name,
                    reranker_model,
                    total_ms,
                    retrieval_ms,
                    gen_ms,
                ),
            )
            db.conn.commit()
            query_log_id = cursor.lastrowid

            # Event: done
            yield ServerSentEvent(
                event="done",
                data=json.dumps(
                    {
                        "metrics": {
                            "retrieval_ms": retrieval_ms,
                            "generation_ms": gen_ms,
                            "total_ms": total_ms,
                        },
                        "query_log_id": query_log_id,
                    }
                ),
            )

        except Exception as e:
            yield ServerSentEvent(
                event="error",
                data=json.dumps({"error": str(e)}),
            )

    return EventSourceResponse(event_generator())


# ---------------------------------------------------------------------------
# POST /feedback
# ---------------------------------------------------------------------------


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    db = _get_db()

    # Check if query_log row exists
    try:
        row = db["query_log"].get(req.query_log_id)
    except Exception:
        raise HTTPException(
            status_code=404, detail=f"query_log_id {req.query_log_id} not found"
        ) from None

    if row is None:
        raise HTTPException(status_code=404, detail=f"query_log_id {req.query_log_id} not found")

    db["query_log"].update(
        req.query_log_id,
        {
            "feedback": req.feedback,
            "feedback_note": req.note,
            "feedback_at": db.conn.execute("SELECT datetime('now')").fetchone()[0],
        },
    )
    db.conn.commit()

    return {"status": "ok", "query_log_id": req.query_log_id}


# ---------------------------------------------------------------------------
# GET /folders
# ---------------------------------------------------------------------------


@app.get("/folders", response_model=list[FolderNode])
def list_folders():
    db = _get_db()

    rows = list(db.query(
        "SELECT id, rel_path, name, inferred_category, inferred_label "
        "FROM folder WHERE excluded = 0 ORDER BY rel_path"
    ))

    # Build tree
    node_map: dict[int, FolderNode] = {}
    for r in rows:
        node_map[r["id"]] = FolderNode(
            id=r["id"],
            rel_path=r["rel_path"],
            name=r["name"],
            inferred_category=r.get("inferred_category"),
            inferred_label=r.get("inferred_label"),
        )

    # Link children to parents
    roots: list[FolderNode] = []
    for r in rows:
        node = node_map[r["id"]]
        # Determine parent by finding the folder whose rel_path is the immediate prefix
        parent_path = str(Path(r["rel_path"]).parent)
        if parent_path == "." or parent_path == "":
            roots.append(node)
        else:
            # Find parent node
            for _pid, pnode in node_map.items():
                if pnode.rel_path == parent_path:
                    pnode.children.append(node)
                    break
            else:
                roots.append(node)

    return roots


# ---------------------------------------------------------------------------
# GET /files/{file_id}
# ---------------------------------------------------------------------------


@app.get("/files/{file_id}", response_model=FileResponse)
def get_file(file_id: int):
    db = _get_db()

    row = list(db.query(
        """SELECT f.id, f.rel_path, f.name, f.category, f.format_name,
                  f.page_count, f.folder_id,
                  fo.name as parent_folder,
                  s.summary_text as summary,
                  e.text_extracted as extraction_path
           FROM file f
           LEFT JOIN folder fo ON f.folder_id = fo.id
           LEFT JOIN summary s ON s.file_id = f.id
           LEFT JOIN extraction e ON e.file_id = f.id
           WHERE f.id = ?""",
        [file_id],
    ))

    if not row:
        raise HTTPException(status_code=404, detail=f"File {file_id} not found")

    r = row[0]
    return FileResponse(
        id=r["id"],
        rel_path=r["rel_path"],
        name=r["name"],
        category=r.get("category"),
        format_name=r.get("format_name"),
        page_count=r.get("page_count"),
        folder_id=r["folder_id"],
        parent_folder=r.get("parent_folder"),
        summary=r.get("summary"),
        extraction_path=r.get("extraction_path"),
    )


# ---------------------------------------------------------------------------
# GET /files/{file_id}/text
# ---------------------------------------------------------------------------


@app.get("/files/{file_id}/text", response_class=PlainTextResponse)
def get_file_text(file_id: int):
    db = _get_db()

    row = list(db.query(
        "SELECT text_extracted FROM extraction WHERE file_id = ? AND succeeded = 1",
        [file_id],
    ))

    if not row or not row[0].get("text_extracted"):
        raise HTTPException(status_code=404, detail="No extracted text available")

    return PlainTextResponse(content=row[0]["text_extracted"])


# ---------------------------------------------------------------------------
# GET /models
# ---------------------------------------------------------------------------


@app.get("/models")
def list_models():
    cfg = _load_config()
    chroma_path = str(cfg.chroma_root_path)

    # Check which Chroma collections exist
    collections = []
    try:
        client = chromadb.PersistentClient(path=chroma_path)
        all_colls = client.list_collections()
        collections = [c.name for c in all_colls]
    except Exception:
        pass

    reranker_model = (
        cfg.models.reranker.name if getattr(cfg.models.reranker, "enabled", False) else None
    )
    augmentation_model = (
        cfg.models.contextual_retrieval.name
        if getattr(cfg.models.contextual_retrieval, "enabled", False)
        else None
    )

    return {
        "embedding": cfg.models.embedding.name,
        "generation": cfg.models.generation.name,
        "generation_fallback": cfg.models.generation.fallback_name,
        "generation_alternate": cfg.models.generation.alternate_name,
        "summarization": cfg.models.summarization.name,
        "contextual_augmentation": augmentation_model,
        "reranker": reranker_model,
        "language_id": cfg.models.language_id.backend,
        "chroma_collections": collections,
    }


# ---------------------------------------------------------------------------
# POST /eval/run
# ---------------------------------------------------------------------------


@app.post("/eval/run")
def run_eval():
    db = _get_db()
    cfg = _load_config()

    try:
        from eval.runner import run_eval as _run_eval
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="Evaluation harness not yet implemented",
        ) from None

    result = _run_eval(db, cfg)
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

DEFAULT_PROMPT = """\
You are the ODW.ai Vault internal knowledge assistant. You help staff find
information about company projects, products, deployments, and operations
by answering questions using ONLY the provided context excerpts.

You answer in the same language as the user's question (English or
Traditional Chinese). Match the user's terminology and tone.

RULES:
1. Every factual claim MUST be supported by a citation marker [N] where N
   refers to a numbered context excerpt below. Use markers inline.
2. If the context does not contain enough information to answer, say so
   explicitly. Do not guess. Do not use external knowledge about products,
   clients, robots, sites, or contracts beyond what the context says.
3. When synthesizing across multiple sources, cite each.
4. Preserve technical terminology, model numbers, robot platform names,
   client names, site names, and project names exactly as they appear in the context.
5. Never invent file names, page numbers, or citation markers that are
   not in the provided context.
6. If asked about a client or project not present in the context, state
   that you have no information about it; do not speculate.

CONTEXT EXCERPTS:
{numbered_chunks}

USER QUESTION: {query}

ANSWER:
"""


def _format_chunks_for_prompt(hits: list[Hit]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        page_info = f" (page {hit.page_start})" if hit.page_start else ""
        blocks.append(f"[{i}] {hit.rel_path}{page_info}\n{hit.text}")
    return "\n\n".join(blocks) if blocks else "(no context available)"
