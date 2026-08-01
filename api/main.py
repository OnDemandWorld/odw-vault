"""FastAPI HTTP service for the RAG pipeline."""

from __future__ import annotations

import csv
import datetime
import hmac
import io
import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path

import chromadb
import ollama
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse, Response
from sse_starlette import EventSourceResponse, ServerSentEvent

from api.audit import query_audit_records, record_audit
from api.schemas import (
    Citation,
    ConversationSummary,
    FeedbackRequest,
    FileListItem,
    FileListPage,
    FileResponse,
    FileUploadResponse,
    FolderNode,
    HealthResponse,
    MessageItem,
    Metrics,
    ModelInfo,
    QueryLogItem,
    QueryLogPage,
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

# ---------------------------------------------------------------------------
# CORS (browser clients such as the Loop canvas call Vault cross-origin)
# ---------------------------------------------------------------------------
# Origins are configurable via VAULT_CORS_ORIGINS (comma-separated list, or "*"
# to allow any origin). Defaults to "*" for local single-host deployments.
_cors_origins_env = os.environ.get("VAULT_CORS_ORIGINS", "*").strip()
if _cors_origins_env == "*":
    _cors_allow_origins = ["*"]
else:
    _cors_allow_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Optional inbound API-key authentication (additive, backward-compatible)
# ---------------------------------------------------------------------------
# Set VAULT_API_KEY to a non-empty value to require a matching
# `Authorization: Bearer <VAULT_API_KEY>` header on every endpoint except the
# health probe and the interactive API docs. When VAULT_API_KEY is unset or
# empty the service stays fully open — preserving the no-auth integration
# contract (INTEGRATION_CONTRACT.md §1) and the existing unauthenticated
# callers and tests. The variable is read per-request so it can be toggled
# without rebuilding the app (and so tests can monkeypatch it).

_AUTH_EXEMPT_PATHS = {"/", "/health", "/openapi.json"}
_AUTH_EXEMPT_PREFIXES = ("/docs", "/redoc")


def _is_auth_exempt(path: str) -> bool:
    """Return True for paths that never require the API key."""
    if path in _AUTH_EXEMPT_PATHS:
        return True
    return any(path.startswith(prefix) for prefix in _AUTH_EXEMPT_PREFIXES)


@app.middleware("http")
async def _require_api_key(request: Request, call_next):
    expected = os.environ.get("VAULT_API_KEY", "").strip()
    if expected and request.method != "OPTIONS" and not _is_auth_exempt(request.url.path):
        scheme, _, token = request.headers.get("Authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(
            token.strip().encode("utf-8"), expected.encode("utf-8")
        ):
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    return await call_next(request)


def _resolve_actor() -> str:
    """Resolve the audit actor for the current deployment (V1.3 F-Vault-1).

    When the optional V1.0 API-key auth is active (``VAULT_API_KEY`` set) the
    request has already passed the auth middleware, so we record an
    authenticated principal. Otherwise the caller is anonymous. A configured
    ``VAULT_AUDIT_ACTOR`` always wins so deployments can name their service.
    The environment is read per-call (like the auth middleware) so it can be
    toggled without rebuilding the app.
    """
    configured = os.environ.get("VAULT_AUDIT_ACTOR", "").strip()
    if configured:
        return configured
    if os.environ.get("VAULT_API_KEY", "").strip():
        return "authenticated"
    return "anonymous"


# ---------------------------------------------------------------------------
# File watcher (opt-in via config [watcher] enabled = true)
# ---------------------------------------------------------------------------

_watcher = None  # CorpusWatcher instance (started on app startup if enabled)


@app.on_event("startup")
def _startup_watcher():
    """Start the corpus file watcher if enabled in config."""
    global _watcher
    try:
        cfg = _load_config()
        if not cfg.watcher.enabled:
            return
        from rag.watcher import CorpusWatcher

        db = _get_db()
        chroma_client = chromadb.PersistentClient(path=str(cfg.chroma_root_path))
        _watcher = CorpusWatcher(cfg, db, chroma_client)
        if cfg.watcher.startup_sync:
            _watcher.indexer.sync_all()
        _watcher.start()
        logger.info("Corpus watcher started via API server")
    except Exception as exc:
        logger.warning("Failed to start corpus watcher: %s", exc)


@app.on_event("shutdown")
def _shutdown_watcher():
    """Stop the corpus file watcher."""
    global _watcher
    if _watcher is not None:
        _watcher.stop()
        _watcher = None
        logger.info("Corpus watcher stopped")


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
# GET /metrics
# ---------------------------------------------------------------------------


@app.get("/metrics")
def metrics():
    """Expose a few operational gauges derived from the database.

    Uses prometheus_client when it is importable; otherwise falls back to a
    minimal Prometheus text exposition. No heavy dependency is required either
    way.
    """
    db = _get_db()

    def _count(sql: str) -> int:
        try:
            return next(iter(db.query(sql)))["c"]
        except Exception:
            return 0

    gauges = {
        "vault_queries_total": ("Total queries logged", _count("SELECT COUNT(*) AS c FROM query_log")),
        "vault_files_total": ("Total files in the corpus", _count("SELECT COUNT(*) AS c FROM file")),
        "vault_chunks_total": ("Total chunks stored", _count("SELECT COUNT(*) AS c FROM chunk")),
    }

    try:
        from prometheus_client import (
            CONTENT_TYPE_LATEST,
            CollectorRegistry,
            Gauge,
            generate_latest,
        )

        registry = CollectorRegistry()
        for name, (help_text, value) in gauges.items():
            Gauge(name, help_text, registry=registry).set(value)
        return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        lines = []
        for name, (help_text, value) in gauges.items():
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")
        return PlainTextResponse("\n".join(lines) + "\n")


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
    # Handle conversation history
    from rag.conversation import get_or_create_conversation, get_history, add_message

    conversation_id = None
    history = None
    if req.conversation_id is not None or True:  # always support conversations
        conversation_id = get_or_create_conversation(db, req.conversation_id, user=req.user)
        # Save user message
        add_message(db, conversation_id, "user", req.query)
        # Get history for prompt injection (excluding the message we just added)
        history = get_history(db, conversation_id)
        # Remove the last message (the one we just added) from history
        if history and history[-1]["role"] == "user" and history[-1]["content"] == req.query:
            history = history[:-1]

    gen_result = generate_answer(
        query=req.query,
        hits=hits,
        cfg=cfg,
        history=history,
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
            latency_ms, retrieval_ms, generation_ms, conversation_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
            conversation_id,
        ),
    )
    db.conn.commit()
    query_log_id = cursor.lastrowid

    # Save assistant response to conversation
    if conversation_id:
        add_message(db, conversation_id, "assistant", gen_result["answer"], query_log_id=query_log_id)

    # Best-effort audit trail (V1.3 F-Vault-1) — never blocks the response.
    record_audit(
        db,
        _resolve_actor(),
        "query",
        "query_log",
        resource_id=query_log_id,
        detail=req.query,
    )

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
        conversation_id=conversation_id,
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

    # Best-effort audit trail (V1.3 F-Vault-1) — never blocks the response.
    record_audit(
        db,
        _resolve_actor(),
        "feedback",
        "query_log",
        resource_id=req.query_log_id,
        detail=req.feedback,
    )

    return {"status": "ok", "query_log_id": req.query_log_id}


# ---------------------------------------------------------------------------
# GET /folders
# ---------------------------------------------------------------------------


@app.get("/folders", response_model=list[FolderNode])
def list_folders(workspace: str | None = None):
    db = _get_db()

    # Optional workspace scoping (V1.1 M4, additive): when provided, keep only
    # folders that contain at least one file in the given workspace. Omitted =>
    # all non-excluded folders (V1.0 behavior).
    if workspace is not None:
        rows = list(db.query(
            "SELECT id, rel_path, name, inferred_category, inferred_label "
            "FROM folder WHERE excluded = 0 "
            "AND EXISTS (SELECT 1 FROM file f WHERE f.folder_id = folder.id "
            "AND f.workspace = ?) ORDER BY rel_path",
            [workspace],
        ))
    else:
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
# Conversation management endpoints
# ---------------------------------------------------------------------------


@app.get("/conversations", response_model=list[ConversationSummary])
def list_conversations_api(user: str | None = None, limit: int = 50):
    """List conversations, most recently updated first."""
    from rag.conversation import list_conversations

    db = _get_db()
    return list_conversations(db, user=user, limit=limit)


@app.get("/conversations/{conversation_id}/messages", response_model=list[MessageItem])
def get_messages_api(conversation_id: str):
    """Get all messages for a conversation."""
    from rag.conversation import get_conversation_messages

    db = _get_db()
    messages = get_conversation_messages(db, conversation_id)
    if not messages:
        # Check if conversation exists at all
        rows = list(db.query(
            "SELECT id FROM conversation WHERE id = ?", [conversation_id]
        ))
        if not rows:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation {conversation_id} not found",
            )
    return messages


@app.delete("/conversations/{conversation_id}")
def delete_conversation_api(conversation_id: str):
    """Delete a conversation and all its messages."""
    from rag.conversation import delete_conversation

    db = _get_db()
    deleted = delete_conversation(db, conversation_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation {conversation_id} not found",
        )
    return {"status": "deleted", "conversation_id": conversation_id}


# ---------------------------------------------------------------------------
# Query history endpoint
# ---------------------------------------------------------------------------


@app.get("/queries", response_model=QueryLogPage)
def query_history(
    page: int = 1,
    size: int = 20,
    keyword: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
):
    """List query history with pagination, search, and date filtering."""
    db = _get_db()

    # Build WHERE clauses
    conditions: list[str] = []
    params: list = []

    if keyword:
        conditions.append("query_text LIKE ?")
        params.append(f"%{keyword}%")

    if start_date:
        conditions.append("asked_at >= ?")
        params.append(start_date)

    if end_date:
        conditions.append("asked_at <= ?")
        params.append(end_date)

    where_clause = ""
    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)

    # Get total count
    count_rows = list(db.query(
        f"SELECT COUNT(*) as total FROM query_log {where_clause}",
        params,
    ))
    total = count_rows[0]["total"] if count_rows else 0

    # Get paginated results
    offset = (page - 1) * size
    rows = list(db.query(
        f"""SELECT id, asked_at, user, query_text, answer_text, answer_model,
                   latency_ms, feedback, conversation_id,
                   retrieved_chunks_json
            FROM query_log {where_clause}
            ORDER BY asked_at DESC
            LIMIT ? OFFSET ?""",
        params + [size, offset],
    ))

    items = []
    for r in rows:
        # Count sources from retrieved_chunks_json
        try:
            chunks = json.loads(r.get("retrieved_chunks_json", "[]") or "[]")
            source_count = len(chunks)
        except (json.JSONDecodeError, TypeError):
            source_count = 0

        items.append(QueryLogItem(
            id=r["id"],
            asked_at=r["asked_at"],
            user=r.get("user"),
            query_text=r["query_text"],
            answer_text=r["answer_text"],
            answer_model=r["answer_model"],
            latency_ms=r["latency_ms"],
            source_count=source_count,
            feedback=r.get("feedback"),
            conversation_id=r.get("conversation_id"),
        ))

    return QueryLogPage(items=items, total=total, page=page, size=size)


# ---------------------------------------------------------------------------
# POST /files/upload
# ---------------------------------------------------------------------------


@app.post("/files/upload", response_model=FileUploadResponse)
async def upload_files(
    files: list[UploadFile] = File(...),
    workspace: str | None = Form(None),
):
    """Upload files to the corpus directory.

    The optional ``workspace`` form field tags every newly created file row
    with a workspace label (defaulting to ``default`` when omitted). The
    response shape is unchanged from V1.0.
    """
    cfg = _load_config()
    workspace_label = workspace if workspace else "default"
    corpus_root = cfg.corpus_root_path
    corpus_root.mkdir(parents=True, exist_ok=True)

    db = _get_db()

    # Ensure a root folder exists
    root_folders = list(db.query(
        "SELECT id FROM folder WHERE rel_path = '.' LIMIT 1"
    ))
    if root_folders:
        folder_id = root_folders[0]["id"]
    else:
        db["folder"].insert({
            "path": str(corpus_root),
            "rel_path": ".",
            "name": ".",
            "depth": 0,
            "excluded": 0,
        })
        db.conn.commit()
        folder_id = next(iter(db.query("SELECT id FROM folder WHERE rel_path = '.'")))["id"]

    uploaded = 0
    failed: list[str] = []

    for upload_file in files:
        try:
            filename = upload_file.filename or "unnamed"
            dest = corpus_root / filename

            # Handle filename conflicts
            if dest.exists():
                stem = dest.stem
                suffix = dest.suffix
                counter = 1
                while dest.exists():
                    dest = corpus_root / f"{stem} ({counter}){suffix}"
                    counter += 1

            # Save file
            content = await upload_file.read()
            dest.write_bytes(content)

            # Get file size
            size_bytes = dest.stat().st_size

            # Determine relative path
            rel_path = str(dest.relative_to(corpus_root))

            # Get file modification time
            mtime = datetime.datetime.fromtimestamp(
                dest.stat().st_mtime, tz=datetime.timezone.utc
            ).isoformat()

            # Insert file record
            db["file"].insert({
                "folder_id": folder_id,
                "path": str(dest),
                "rel_path": rel_path,
                "name": dest.name,
                "extension": dest.suffix.lstrip("."),
                "size_bytes": size_bytes,
                "mtime": mtime,
                "sha256": "",
                "mime_type": upload_file.content_type,
                "hash_status": "pending",
                "identify_status": "pending",
                "triage_status": "pending",
                "is_dup_primary": 1,
                "excluded": 0,
                "workspace": workspace_label,
            })
            db.conn.commit()

            # Set created_at
            file_id = db["file"].last_rowid
            db.execute(
                "UPDATE file SET created_at = datetime('now') WHERE id = ?",
                [file_id],
            )
            db.conn.commit()

            uploaded += 1

            # Best-effort audit trail (V1.3 F-Vault-1) — one entry per file.
            record_audit(
                db,
                _resolve_actor(),
                "file.upload",
                "file",
                resource_id=file_id,
                detail=dest.name,
            )
        except Exception as e:
            logger.error("Failed to upload %s: %s", upload_file.filename, e)
            failed.append(upload_file.filename or "unnamed")

    return FileUploadResponse(uploaded=uploaded, failed=failed)


# ---------------------------------------------------------------------------
# POST /pipeline/sync
# ---------------------------------------------------------------------------


_sync_lock = threading.Lock()
_sync_status: dict = {"running": False, "last_result": None}


@app.post("/pipeline/sync")
def pipeline_sync():
    """Trigger a full incremental sync of the corpus.

    Runs in a background thread to avoid blocking the request.  Returns
    immediately with the current sync status.  If a sync is already running
    the caller receives a 409 Conflict.
    """
    if _sync_status["running"]:
        raise HTTPException(status_code=409, detail="A sync is already in progress")

    def _run_sync():
        try:
            cfg = _load_config()
            db = _get_db()
            chroma_path = str(cfg.chroma_root_path)
            chroma_client = chromadb.PersistentClient(path=chroma_path)

            from rag.indexer import IncrementalIndexer

            indexer = IncrementalIndexer(db, cfg, chroma_client=chroma_client)
            result = indexer.sync_all()
            _sync_status["last_result"] = result
        except Exception as exc:
            logger.error("Pipeline sync failed: %s", exc)
            _sync_status["last_result"] = {"error": str(exc)}
        finally:
            _sync_status["running"] = False

    with _sync_lock:
        if _sync_status["running"]:
            raise HTTPException(status_code=409, detail="A sync is already in progress")
        _sync_status["running"] = True

    t = threading.Thread(target=_run_sync, daemon=True)
    t.start()

    # Best-effort audit trail (V1.3 F-Vault-1): record that a sync was triggered.
    record_audit(
        _get_db(),
        _resolve_actor(),
        "pipeline.sync",
        "pipeline",
        detail="incremental sync started",
    )

    return {
        "status": "started",
        "message": "Incremental sync started in background",
        "running": True,
    }


@app.get("/pipeline/watch/status")
def watch_status():
    """Return the file watcher status."""
    if _watcher is None:
        return {"enabled": False}
    return {"enabled": True, **_watcher.status()}


# ---------------------------------------------------------------------------
# GET /files
# ---------------------------------------------------------------------------


@app.get("/files", response_model=FileListPage)
def list_files(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    folder_id: int | None = None,
    status: str | None = None,
    workspace: str | None = None,
):
    """List files with pagination and optional filtering."""
    db = _get_db()

    # Build WHERE clause
    where_clauses = []
    params = []

    if folder_id is not None:
        where_clauses.append("f.folder_id = ?")
        params.append(folder_id)

    # Optional workspace filter (V1.1 M4, additive). Omitted => all files.
    if workspace is not None:
        where_clauses.append("f.workspace = ?")
        params.append(workspace)

    if status is not None:
        if status == "indexed":
            where_clauses.append(
                "EXISTS (SELECT 1 FROM extraction e WHERE e.file_id = f.id AND e.succeeded = 1) "
                "AND EXISTS (SELECT 1 FROM chunk c JOIN embedding_ref er ON er.chunk_id = c.id "
                "WHERE c.file_id = f.id AND er.is_current = 1)"
            )
        elif status == "failed":
            where_clauses.append(
                "EXISTS (SELECT 1 FROM failure fail WHERE fail.file_id = f.id AND fail.phase = 'extraction')"
            )
        elif status == "pending":
            where_clauses.append(
                "NOT EXISTS (SELECT 1 FROM extraction e WHERE e.file_id = f.id AND e.succeeded = 1) "
                "AND NOT EXISTS (SELECT 1 FROM failure fail WHERE fail.file_id = f.id AND fail.phase = 'extraction')"
            )

    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

    # Count total
    count_sql = f"SELECT COUNT(*) as c FROM file f WHERE {where_sql}"
    total = list(db.query(count_sql, params))[0]["c"]

    # Fetch page
    offset = (page - 1) * size
    query_sql = f"""
        SELECT
            f.id,
            f.rel_path,
            f.name,
            f.size_bytes,
            f.category,
            f.mime_type,
            f.created_at,
            CASE
                WHEN EXISTS (
                    SELECT 1 FROM extraction e
                    WHERE e.file_id = f.id AND e.succeeded = 1
                ) AND EXISTS (
                    SELECT 1 FROM chunk c
                    JOIN embedding_ref er ON er.chunk_id = c.id
                    WHERE c.file_id = f.id AND er.is_current = 1
                ) THEN 'indexed'
                WHEN EXISTS (
                    SELECT 1 FROM failure fail
                    WHERE fail.file_id = f.id AND fail.phase = 'extraction'
                ) THEN 'failed'
                ELSE 'pending'
            END as status
        FROM file f
        WHERE {where_sql}
        ORDER BY f.created_at DESC
        LIMIT ? OFFSET ?
    """
    rows = list(db.query(query_sql, params + [size, offset]))

    items = [
        FileListItem(
            id=r["id"],
            rel_path=r["rel_path"],
            name=r["name"],
            size_bytes=r["size_bytes"],
            category=r.get("category"),
            mime_type=r.get("mime_type"),
            is_indexed=r["status"] == "indexed",
            status=r["status"],
            created_at=str(r["created_at"]) if r.get("created_at") else None,
        )
        for r in rows
    ]

    return FileListPage(items=items, total=total, page=page, size=size)


# ---------------------------------------------------------------------------
# GET /workspaces
# ---------------------------------------------------------------------------


@app.get("/workspaces")
def list_workspaces():
    """List distinct workspaces with their file counts (V1.1 M4, additive).

    Workspaces are a logical isolation layer over the shared corpus: every
    file row carries a ``workspace`` label (defaulting to ``default``). This
    endpoint reports which labels exist and how many files each holds.
    """
    db = _get_db()

    rows = list(db.query(
        "SELECT workspace, COUNT(*) AS file_count "
        "FROM file GROUP BY workspace ORDER BY workspace"
    ))

    workspaces = [
        {"workspace": r["workspace"], "file_count": r["file_count"]}
        for r in rows
    ]
    return {"workspaces": workspaces, "total": len(workspaces)}


# ---------------------------------------------------------------------------
# DELETE /files/{file_id}
# ---------------------------------------------------------------------------


@app.delete("/files/{file_id}")
def delete_file(file_id: int):
    """Delete a file from disk, Chroma, and database."""
    cfg = _load_config()
    db = _get_db()

    # Get file record
    file_rows = list(db.query(
        "SELECT id, rel_path, path FROM file WHERE id = ?", [file_id]
    ))
    if not file_rows:
        raise HTTPException(status_code=404, detail=f"File {file_id} not found")

    file_row = file_rows[0]
    rel_path = file_row["rel_path"]

    # Delete from disk
    corpus_root = cfg.corpus_root_path
    file_path = corpus_root / rel_path
    if file_path.exists():
        file_path.unlink()

    # Delete from Chroma
    try:
        chroma_path = str(cfg.chroma_root_path)
        chroma_client = chromadb.PersistentClient(path=chroma_path)

        # Find embedding_ref entries for this file's chunks
        embedding_refs = list(db.query(
            """SELECT er.id, er.collection, er.external_id
               FROM embedding_ref er
               JOIN chunk c ON er.chunk_id = c.id
               WHERE c.file_id = ?""",
            [file_id],
        ))

        # Group by collection and delete
        collections_to_delete: dict[str, list[str]] = {}
        for ref in embedding_refs:
            coll_name = ref.get("collection") or f"chunks__{cfg.models.embedding.collection_suffix}"
            external_id = ref.get("external_id")
            if external_id:
                collections_to_delete.setdefault(coll_name, []).append(external_id)

        for coll_name, ids in collections_to_delete.items():
            try:
                collection = chroma_client.get_collection(coll_name)
                collection.delete(ids=ids)
            except Exception as e:
                logger.warning("Failed to delete from Chroma collection %s: %s", coll_name, e)
    except Exception as e:
        logger.warning("Failed to connect to Chroma: %s", e)

    # Delete from SQLite (cascade)
    db.execute("DELETE FROM embedding_ref WHERE chunk_id IN (SELECT id FROM chunk WHERE file_id = ?)", [file_id])
    db.execute("DELETE FROM chunk WHERE file_id = ?", [file_id])
    db.execute("DELETE FROM extraction WHERE file_id = ?", [file_id])
    db.execute("DELETE FROM summary WHERE file_id = ?", [file_id])
    db.execute("DELETE FROM failure WHERE file_id = ?", [file_id])
    db.execute("DELETE FROM file WHERE id = ?", [file_id])
    db.conn.commit()

    # Best-effort audit trail (V1.3 F-Vault-1) — never blocks the response.
    record_audit(
        db,
        _resolve_actor(),
        "file.delete",
        "file",
        resource_id=file_id,
        detail=rel_path,
    )

    return {"status": "deleted", "file_id": file_id}


# ---------------------------------------------------------------------------
# GET /audit (V1.3 F-Vault-1)
# ---------------------------------------------------------------------------


@app.get("/audit")
def list_audit(
    action: str | None = None,
    limit: int = Query(100, ge=1, le=1000),
):
    """Return audit-log records, most recent first.

    Optional filters: ``action`` (exact match, e.g. ``query`` / ``file.upload``)
    and ``limit`` (max rows, default 100). When the optional V1.0 API-key auth
    is active this endpoint is protected by it via the shared auth middleware;
    otherwise it is open like the rest of the API. Additive — does not alter
    any existing endpoint.
    """
    db = _get_db()
    rows = query_audit_records(db, action=action, limit=limit)
    return {"items": rows, "total": len(rows)}


# ---------------------------------------------------------------------------
# GET /audit/export (V1.4 F-3')
# ---------------------------------------------------------------------------

# CSV column order for the compliance audit report export.
_AUDIT_EXPORT_COLUMNS = [
    "id",
    "ts",
    "actor",
    "action",
    "resource_type",
    "resource_id",
    "detail",
    "status",
]


@app.get("/audit/export")
def export_audit(
    fmt: str = Query("json", alias="format"),
    start: str | None = None,
    end: str | None = None,
    actor: str | None = None,
    action: str | None = None,
    limit: int = Query(1000, ge=1, le=10000),
):
    """Export audit-log records as a compliance report (V1.4 F-3').

    Strictly additive — ``GET /audit`` is unchanged. Query params:

    - ``format``: ``json`` (default) or ``csv``.
    - ``start`` / ``end``: inclusive ISO-8601 time bounds on ``ts``.
    - ``actor`` / ``action``: exact-match filters.
    - ``limit``: max rows (default 1000).

    JSON returns ``{items, total, filters}``. CSV is built with the stdlib
    ``csv`` module (columns ``id,ts,actor,action,resource_type,resource_id,
    detail,status``) and served as an attachment (``Content-Type: text/csv`` +
    ``Content-Disposition: attachment``). Protected by the shared API-key auth
    middleware exactly like ``GET /audit`` (protected when ``VAULT_API_KEY`` is
    active, otherwise open).
    """
    if fmt not in ("csv", "json"):
        raise HTTPException(status_code=422, detail="format must be 'csv' or 'json'")

    db = _get_db()
    rows = query_audit_records(
        db, start=start, end=end, actor=actor, action=action, limit=limit
    )
    filters = {
        "start": start,
        "end": end,
        "actor": actor,
        "action": action,
        "limit": limit,
    }

    if fmt == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(_AUDIT_EXPORT_COLUMNS)
        for row in rows:
            writer.writerow([row[col] for col in _AUDIT_EXPORT_COLUMNS])
        return Response(
            content=buf.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="audit_export.csv"'},
        )

    return {"items": rows, "total": len(rows), "filters": filters}


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
