"""Phase 14: Modern Gradio chat UI for the RAG pipeline — ODW Vault.

Design principles:
- Composer centered on empty state, docks to bottom when chat starts
- ODW.ai brand palette: warm paper (#F6F2EC), ink (#14110F), accent orange (#FF5A1F)
- Typography: Instrument Serif (display), Inter (body), JetBrains Mono (labels)
- User messages right-aligned dark bubbles, assistant messages as bare text
- Prompt starter chips, time-based greeting
- Dark sidebar matching odw.ai official design language

Implementation:
  The visible page is a single gr.HTML with full HTML/CSS/JS.
  Gradio's default container is hidden via CSS.
  A hidden gr.ChatInterface provides the /gradio_api streaming endpoint.
  The HTML page calls the Gradio API via fetch + ReadableStream for token streaming.

File organization:
- gradio_app.py          — This file (new modern UI, default)
- gradio_app_legacy.py   — Original Gradio UI (backup)
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

import gradio as gr
import ollama

from pipeline.config import load_app_config
from pipeline.db import open_db
from rag.citations import parse_citations, resolve_citations
from rag.generation import REFUSAL_TEXT, _load_prompt
from rag.retrieval import Hit, retrieve

logger = logging.getLogger(__name__)

_cfg = None
_db_path: Path | None = None
_chroma_path = ""
_ollama_host = "http://localhost:11434"
_thread_local = threading.local()


def _get_db():
    if not hasattr(_thread_local, "db"):
        from pipeline.db import migrate
        _thread_local.db = open_db(_db_path)
        migrate(_thread_local.db)
        # Return dict-like rows so column-name access (row["col"]) works
        _thread_local.db.conn.row_factory = sqlite3.Row
    return _thread_local.db


def _ensure_db(db_path: Path) -> None:
    global _db_path
    _db_path = db_path


def _get_folders() -> list[str]:
    rows = _get_db().query("SELECT rel_path FROM folder WHERE excluded = 0 ORDER BY rel_path")
    return [r["rel_path"] for r in rows]


def _make_client():
    """Create an ollama client from the generation endpoint config."""
    if _cfg is not None:
        ep = getattr(_cfg.models.generation, "endpoint", None)
        if ep:
            kwargs = {"host": ep.host}
            if getattr(ep, "api_key", ""):
                kwargs["headers"] = {"Authorization": f"Bearer {ep.api_key}"}
            return ollama.Client(**kwargs)
    return ollama.Client(host=_ollama_host)


def _check_ollama() -> bool:
    try:
        client = _make_client()
        client.list()
        return True
    except Exception:
        return False


def _check_chroma() -> tuple[bool, str]:
    try:
        import chromadb
        client = chromadb.PersistentClient(path=_chroma_path)
        suffix = _cfg.models.embedding.collection_suffix
        coll_name = f"chunks__{suffix}"
        client.get_collection(coll_name)
        return True, coll_name
    except Exception as exc:
        return False, str(exc)[:120]


def _format_chunks_for_prompt(hits: list[Hit]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        page_info = f" (page {hit.page_start})" if hit.page_start else ""
        blocks.append(f"[{i}] {hit.rel_path}{page_info}\n{hit.text}")
    return "\n\n".join(blocks) if blocks else "(no context available)"


def _stream_tokens(query: str, hits: list[Hit]):
    numbered_chunks = _format_chunks_for_prompt(hits)
    template = _load_prompt(None, _cfg)
    prompt = template.format(numbered_chunks=numbered_chunks, query=query)

    system_prefix = ""
    if getattr(_cfg.models.generation, "thinking", False):
        system_prefix = "<|think|>"
    system_content = "You are a helpful assistant."
    if system_prefix:
        system_content = f"{system_prefix}\n{system_content}"

    model_name = _cfg.models.generation.name
    client = _make_client()

    full_text = ""
    for chunk in client.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ],
        options={
            "temperature": _cfg.models.generation.temperature,
            "top_p": _cfg.models.generation.top_p,
            "top_k": _cfg.models.generation.top_k,
        },
        stream=True,
    ):
        token = chunk.get("message", {}).get("content", "")
        full_text += token
        yield full_text

    if not full_text:
        full_text = REFUSAL_TEXT

    citation_numbers = parse_citations(full_text)
    citations = resolve_citations(citation_numbers, hits)
    yield full_text, citations


def _citations_html(citations: list[dict]) -> str:
    if not citations:
        return ""
    cards = []
    for c in citations:
        page_str = f'<span class="citation-card__meta-item">\U0001f4c4 Page {c["page_start"]}</span>' if c.get("page_start") else ""
        snippet = c.get("snippet", "")
        if len(snippet) > 150:
            snippet = snippet[:150] + "..."
        cards.append(
            f'<div class="citation-card" data-cite="{c["citation_number"]}">'
            f'<div class="citation-card__relevance"></div>'
            f'<div class="citation-card__number">{c["citation_number"]}</div>'
            f'<div class="citation-card__content">'
            f'<div class="citation-card__title">{c["rel_path"]}</div>'
            f'<div class="citation-card__snippet">{snippet}</div>'
            f'<div class="citation-card__meta">{page_str}</div>'
            f'</div></div>'
        )
    return (
        f'<div class="citations-panel">'
        f'<div class="citations-panel__header"><span>\U0001f4da Sources ({len(citations)})</span></div>'
        f'<div class="citations-panel__list">{"".join(cards)}</div>'
        f'</div>'
    )


_PROMPT_CHIPS = [
    {"icon": "\U0001f50d", "text": "What documents are in the knowledge base?"},
    {"icon": "\U0001f4ca", "text": "Summarise the main topics covered in the corpus"},
    {"icon": "\U0001f9e0", "text": "What are the key themes across all folders?"},
    {"icon": "\U0001f4dd", "text": "Find policies or procedures that mention approvals"},
    {"icon": "\U0001f4cb", "text": "Compare the latest version with previous drafts"},
    {"icon": "\U0001f517", "text": "What are the dependencies between these documents?"},
    {"icon": "\U0001f4e1", "text": "Identify gaps or missing procedures"},
    {"icon": "\U0001f4b5", "text": "Extract budget or cost-related information"},
    {"icon": "\U0001f4e7", "text": "Draft a summary email for the team"},
    {"icon": "\U0001f4d1", "text": "Create a table of all action items mentioned"},
    {"icon": "\U0001f3af", "text": "What deadlines are coming up based on the documents?"},
    {"icon": "\U0001f4dd", "text": "Write a brief executive summary of the corpus"},
]


def _on_folder_change(folder_filter: str) -> str:
    if folder_filter == "All folders":
        return "Searching all folders."
    return f"Scoped to: {folder_filter}"


def _on_feedback(feedback_data: gr.LikeData) -> None:
    try:
        _get_db().conn.execute(
            "UPDATE query_log SET feedback = ?, feedback_at = datetime('now') "
            "WHERE feedback IS NULL ORDER BY asked_at DESC LIMIT 1",
            ("up" if feedback_data.liked else "down",),
        )
        _get_db().conn.commit()
    except Exception:
        logger.warning("Failed to record feedback", exc_info=True)


def _on_chat(message: str, history: list[dict], folder_filter: str, conversation_id: str | None = None):
    import time as _time
    import uuid as _uuid
    query_start = _time.time()
    logger.info(f"[QUERY] message={message[:100]!r}, folder_filter={folder_filter!r}")

    if not message or not message.strip():
        logger.warning("[QUERY] Empty message")
        yield history, _citations_html([])
        return

    # Assign or create conversation ID
    conv_id = conversation_id or str(_uuid.uuid4())

    try:
        _ensure_conv_tables()
        _save_conversation_message(conv_id, "user", message.strip())
    except Exception:
        logger.warning("Failed to save user message to conversation", exc_info=True)

    history = [*history, {"role": "user", "content": message.strip()}]
    yield history, _citations_html([])

    filter_spec = None
    if folder_filter and folder_filter != "All folders":
        filter_spec = {"path_prefix": folder_filter}
        logger.info(f"[QUERY] Filter: {filter_spec}")

    try:
        hits, _metrics = retrieve(
            query=message.strip(),
            db=_get_db(),
            chroma_client=None,
            chroma_path=_chroma_path,
            cfg=_cfg,
            folder_filter=filter_spec,
        )
    except Exception as exc:
        elapsed = _time.time() - query_start
        logger.error(f"[QUERY] Retrieval failed in {elapsed:.2f}s: {exc}", exc_info=True)
        history.append({"role": "assistant", "content": f"Retrieval failed: {exc}"})
        yield history, _citations_html([])
        return

    elapsed_retrieval = _time.time() - query_start
    logger.info(f"[QUERY] Retrieved {len(hits)} hits in {elapsed_retrieval:.2f}s, metrics={_metrics}")

    if not hits:
        history.append({"role": "assistant", "content": "No relevant context found for your question."})
        logger.warning(f"[QUERY] No hits for query: {message[:80]!r}")
        yield history, _citations_html([])
        return

    history.append({"role": "assistant", "content": ""})
    yield history, _citations_html([])

    citations = []
    full_text = ""
    try:
        token_gen = _stream_tokens(message.strip(), hits)
        for result in token_gen:
            if isinstance(result, tuple):
                full_text, citations = result
            else:
                full_text = result
            # Keep "Thinking..." placeholder until actual text arrives
            if full_text:
                history[-1]["content"] = full_text
            yield history, _citations_html([])
    except Exception as exc:
        logger.error(f"[QUERY] Generation failed: {exc}", exc_info=True)
        full_text = f"Generation failed: {exc}"
        history[-1]["content"] = full_text
        yield history, _citations_html([])

    elapsed_total = _time.time() - query_start
    logger.info(f"[QUERY] Done in {elapsed_total:.2f}s, citations={len(citations)}, tokens={len(full_text)}")

    # Save assistant response to conversation
    try:
        _save_conversation_message(conv_id, "assistant", full_text)
    except Exception:
        logger.warning("Failed to save assistant message to conversation", exc_info=True)

    yield history, _citations_html(citations)


# ---------------------------------------------------------------------------
# Conversation management
# Uses the same tables as Migration 3: `conversation` and `message`
# ---------------------------------------------------------------------------

def _ensure_conv_tables():
    """Ensure conversation tables exist (Migration 3 creates them, but this is a safety net)."""
    db = _get_db()
    db.conn.executescript("""
        CREATE TABLE IF NOT EXISTS conversation (
            id          TEXT PRIMARY KEY,
            user        TEXT,
            title       TEXT,
            created_at  TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS message (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL REFERENCES conversation(id) ON DELETE CASCADE,
            role            TEXT NOT NULL CHECK (role IN ('user','assistant','system')),
            content         TEXT NOT NULL,
            query_log_id    INTEGER REFERENCES query_log(id),
            created_at      TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_message_conversation ON message(conversation_id, created_at);
    """)
    # Safe migration: add feedback column if missing
    cols = {r["name"] for r in db.conn.execute("PRAGMA table_info(message)")}
    if "feedback" not in cols:
        db.conn.execute("ALTER TABLE message ADD COLUMN feedback TEXT")
    db.conn.commit()


def _save_conversation_message(conv_id: str, role: str, content: str, title: str | None = None):
    """Save a message to a conversation, creating the conversation if needed."""
    import uuid
    db = _get_db()
    existing = db.conn.execute("SELECT id FROM conversation WHERE id = ?", (conv_id,)).fetchone()
    if not existing:
        if not title:
            title = content[:50] if len(content) > 50 else content
        db.conn.execute(
            "INSERT INTO conversation (id, title) VALUES (?, ?)",
            (conv_id, title),
        )
    else:
        db.conn.execute(
            "UPDATE conversation SET updated_at = datetime('now') WHERE id = ?",
            (conv_id,),
        )
    db.conn.execute(
        "INSERT INTO message (conversation_id, role, content) VALUES (?, ?, ?)",
        (conv_id, role, content),
    )
    db.conn.commit()


def _list_conversations() -> list[dict]:
    """List all conversations, newest first."""
    db = _get_db()
    rows = db.conn.execute(
        "SELECT c.id, c.title, c.created_at, c.updated_at, COUNT(m.id) as message_count "
        "FROM conversation c LEFT JOIN message m ON m.conversation_id = c.id "
        "GROUP BY c.id ORDER BY c.updated_at DESC LIMIT 50"
    ).fetchall()
    return [{"id": r["id"], "title": r["title"] or "Untitled", "created_at": r["created_at"], "updated_at": r["updated_at"], "message_count": r["message_count"]} for r in rows]


def _get_conversation_messages(conv_id: str) -> list[dict]:
    """Get all messages for a conversation."""
    db = _get_db()
    rows = db.conn.execute(
        "SELECT role, content FROM message WHERE conversation_id = ? ORDER BY created_at",
        (conv_id,),
    ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def _delete_conversation(conv_id: str):
    """Delete a conversation and its messages."""
    db = _get_db()
    db.conn.execute("DELETE FROM message WHERE conversation_id = ?", (conv_id,))
    db.conn.execute("DELETE FROM conversation WHERE id = ?", (conv_id,))
    db.conn.commit()


def _record_message_feedback(conv_id: str, feedback: str):
    """Record feedback on the most recent assistant message in a conversation."""
    db = _get_db()
    db.conn.execute(
        "UPDATE message SET feedback = ? WHERE id = ("
        "  SELECT id FROM message WHERE conversation_id = ? AND role = 'assistant' "
        "  ORDER BY created_at DESC, id DESC LIMIT 1)",
        (feedback, conv_id),
    )
    db.conn.commit()


def _list_conversations_api():
    """Gradio API endpoint to list conversations."""
    return _list_conversations()


def _build_status_html() -> str:
    ollama_ok = _check_ollama()
    chroma_ok, chroma_msg = _check_chroma()
    gen_model = _cfg.models.generation.name
    emb_model = _cfg.models.embedding.name

    ollama_dot = "\U0001f7e2" if ollama_ok else "\U0001f534"
    chroma_dot = "\U0001f7e2" if chroma_ok else "\U0001f534"
    chroma_label = chroma_msg if chroma_ok else "Chroma unavailable"

    ep = getattr(_cfg.models.generation, "endpoint", None)
    gen_host = ep.host if ep else _ollama_host
    gen_label = gen_host.replace("https://", "").replace("http://", "")

    return (
        f"<div style='font-family:monospace;font-size:12px;line-height:1.8;color:#8e8e8e;'>"
        f"{ollama_dot} Generation ({gen_label}) &mdash; {'reachable' if ollama_ok else 'unreachable'}<br>"
        f"{chroma_dot} Chroma &mdash; {chroma_label}<br>"
        f"\U0001f916 Gen: <b>{gen_model}</b> &middot; "
        f"\U0001f4e1 Emb: <b>{emb_model}</b>"
        f"</div>"
    )


def _get_greeting() -> str:
    import datetime
    hour = datetime.datetime.now().hour
    if hour < 12:
        return "Good morning"
    elif hour < 17:
        return "Good afternoon"
    else:
        return "Good evening"


def _get_corpus_stats() -> dict:
    """Return corpus statistics for the hero section."""
    try:
        db = _get_db()
        file_count = db.conn.execute(
            "SELECT COUNT(*) as c FROM file WHERE excluded = 0"
        ).fetchone()["c"]
        folder_count = db.conn.execute(
            "SELECT COUNT(*) as c FROM folder WHERE excluded = 0"
        ).fetchone()["c"]
        chunk_count = db.conn.execute(
            "SELECT COUNT(*) as c FROM chunk"
        ).fetchone()["c"]
        last_indexed = db.conn.execute(
            "SELECT MAX(started_at) as t FROM pipeline_run WHERE status = 'done'"
        ).fetchone()["t"]
        gen_model = _cfg.models.generation.name if _cfg else "unknown"
        emb_model = _cfg.models.embedding.name if _cfg else "unknown"
        return {
            "file_count": file_count,
            "folder_count": folder_count,
            "chunk_count": chunk_count,
            "last_indexed": last_indexed or "never",
            "gen_model": gen_model,
            "emb_model": emb_model,
        }
    except Exception:
        return {
            "file_count": 0,
            "folder_count": 0,
            "chunk_count": 0,
            "last_indexed": "unknown",
            "gen_model": "unknown",
            "emb_model": "unknown",
        }


# ---------------------------------------------------------------------------
# Full HTML page — pure custom layout, zero Gradio interference
# ---------------------------------------------------------------------------

def launch_ui(cfg, share: bool = False, server_name: str = "127.0.0.1", server_port: int = 7860):
    """Launch the ODW Vault chat interface.

    Strategy: Gradio runs on an internal port for its API only.
    A lightweight proxy server runs on the user-facing port,
    serving our custom HTML at / and proxying /gradio_api to Gradio.
    """
    import threading
    import time
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, StreamingResponse
    from starlette.responses import Response
    import httpx

    # Ensure localhost bypasses any environment HTTP proxy. Otherwise httpx
    # (used below for the reverse proxy and by Gradio's own startup probe)
    # would route 127.0.0.1 traffic through e.g. HTTP_PROXY and get a 502.
    import os as _os
    _no_proxy = _os.environ.get("NO_PROXY", _os.environ.get("no_proxy", ""))
    _no_set = {p.strip() for p in _no_proxy.split(",") if p.strip()}
    _need = [h for h in ("127.0.0.1", "localhost") if h not in _no_set]
    if _need:
        _no_proxy = ",".join(list(_no_set) + _need)
        _os.environ["NO_PROXY"] = _no_proxy
        _os.environ["no_proxy"] = _no_proxy

    global _cfg, _chroma_path, _ollama_host

    _cfg = cfg
    _ollama_host = getattr(cfg.ollama, "host", "http://localhost:11434")
    _chroma_path = cfg.paths.chroma_root

    db_path = Path(cfg.paths.corpus_root) / ".rag-cache" / "corpus.db"
    if not db_path.exists():
        db_path = Path("corpus.db")
    _ensure_db(db_path)

    folders = _get_folders()
    folder_choices = ["All folders", *folders]

    greeting = _get_greeting()
    ollama_ok = _check_ollama()
    ollama_status = '\U0001f7e2 Ollama OK' if ollama_ok else '\U0001f534 Ollama down'

    chips_json = str([{"icon": c["icon"], "text": c["text"]} for c in _PROMPT_CHIPS[:4]]).replace("'", '"')
    folder_options = "".join(f'<option value="{f}">{f}</option>' for f in folder_choices)

    full_html = _build_full_page(greeting, chips_json, folder_options, ollama_status)

    # Step 1: Start Gradio on internal port for API only
    gradio_port = server_port + 1

    with gr.Blocks(title="ODW Vault") as gradio_app:
        chatbot = gr.Chatbot(visible=False)
        citations_out = gr.HTML(visible=False)
        msg_box = gr.Textbox(visible=False)
        folder_box = gr.Dropdown(choices=folder_choices, value="All folders", visible=False)
        conv_id_box = gr.Textbox(visible=False, value="")
        submit_btn = gr.Button(visible=False)

        submit_btn.click(
            fn=_on_chat,
            inputs=[msg_box, chatbot, folder_box, conv_id_box],
            outputs=[chatbot, citations_out],
            api_name="chat",
        )
        chatbot.like(fn=_on_feedback)

        # List conversations endpoint
        list_btn = gr.Button(visible=False)
        list_out = gr.JSON(visible=False)
        list_btn.click(
            fn=_list_conversations_api,
            inputs=[],
            outputs=[list_out],
            api_name="list_conversations",
        )

    print(f"  DB path: {db_path}")
    print(f"  Folder count: {len(folders)}")
    print(f"  Ollama host: {_ollama_host}, Chroma path: {_chroma_path}")

    # Start Gradio in background thread
    gradio_url = f"http://{server_name}:{gradio_port}"

    def run_gradio():
        # Suppress Gradio's root page by patching routes
        from fastapi.responses import HTMLResponse as HR

        async def _noop_root(request):
            return HR(content="")

        for route in gradio_app.app.routes:
            if getattr(route, "path", None) == "/":
                route.endpoint = _noop_root

        try:
            gradio_app.launch(
                server_name=server_name,
                server_port=gradio_port,
                share=False,
            )
        except Exception as exc:  # pragma: no cover - startup probe flakiness
            # Gradio's startup probe can fail under env-proxy/timeout conditions
            # even though the backend server is up; keep serving regardless.
            print(f"  WARNING: Gradio launch probe failed ({exc}); backend may still be reachable.")

    gradio_thread = threading.Thread(target=run_gradio, daemon=True)
    gradio_thread.start()

    # Wait for Gradio backend to become ready (bypass any env HTTP proxy)
    _gradio_ready = False
    for _ in range(40):
        time.sleep(0.5)
        try:
            _r = httpx.get(
                f"{gradio_url}/gradio_api/startup-events",
                trust_env=False, timeout=2.0,
            )
            if _r.is_success:
                _gradio_ready = True
                break
        except Exception:
            pass
    if not _gradio_ready:
        print("  WARNING: Gradio backend did not confirm readiness; chat may be unavailable.")
    print(f"  Gradio API backend: {gradio_url}")

    # Step 2: Create our proxy server on the user-facing port
    # Use httpx reverse proxy to forward /gradio_api/* to Gradio backend
    from starlette.middleware.base import BaseHTTPMiddleware

    proxy_app = FastAPI(title="ODW Vault")

    # Serve brand logo assets (light/dark PNGs) from resource_img/
    from fastapi.staticfiles import StaticFiles as _StaticFiles

    _resource_img_dir = Path(__file__).resolve().parent.parent / "resource_img"
    if _resource_img_dir.is_dir():
        proxy_app.mount(
            "/resource_img",
            _StaticFiles(directory=str(_resource_img_dir)),
            name="resource_img",
        )

    @proxy_app.get("/")
    async def root():
        return HTMLResponse(content=full_html)

    @proxy_app.get("/conversations/{conv_id}/messages")
    async def get_conv_messages(conv_id: str):
        try:
            _ensure_conv_tables()
            msgs = _get_conversation_messages(conv_id)
            return {"messages": msgs}
        except Exception as exc:
            return {"messages": [], "error": str(exc)}

    @proxy_app.delete("/conversations/{conv_id}")
    async def delete_conv(conv_id: str):
        try:
            _ensure_conv_tables()
            _delete_conversation(conv_id)
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @proxy_app.post("/conversations/{conv_id}/feedback")
    async def record_feedback(conv_id: str, request: Request):
        try:
            _ensure_conv_tables()
            payload = await request.json()
            fb = payload.get("feedback")
            if fb not in ("up", "down"):
                return {"ok": False, "error": "invalid feedback value"}
            _record_message_feedback(conv_id, fb)
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @proxy_app.get("/stats")
    async def get_stats():
        return _get_corpus_stats()

    # Single catch-all proxy for all Gradio API requests
    async def _do_proxy(request: Request):
        path = request.url.path[len("/gradio_api/"):]
        target = f"{gradio_url}/gradio_api/{path}"
        if request.url.query:
            target += f"?{request.url.query}"
        body = await request.body() if request.method in ("POST", "PUT", "PATCH") else None
        async with httpx.AsyncClient(trust_env=False) as client:
            # SSE streams (GET /call/chat/{event_id}) need streaming response
            if request.method == "GET" and "/call/chat/" in path:
                async with client.stream(
                    method=request.method, url=target,
                    headers={k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")},
                    timeout=None,
                ) as r:
                    headers_out = {k: v for k, v in r.headers.items() if k.lower() not in ("content-length", "transfer-encoding")}
                    async def body_iter():
                        async for chunk in r.aiter_bytes():
                            yield chunk
                    return StreamingResponse(body_iter(), status_code=r.status_code, headers=headers_out)
            else:
                r = await client.request(
                    method=request.method, url=target,
                    content=body,
                    headers={k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")},
                    timeout=None,
                )
                return Response(content=r.content, status_code=r.status_code,
                              headers={k: v for k, v in r.headers.items()})

    # Register using Starlette Route to avoid FastAPI path parameter issues
    from starlette.routing import Route as StarletteRoute
    proxy_app.router.routes.append(
        StarletteRoute("/gradio_api/{full_path:path}", _do_proxy, methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
    )
    print(f"  ODW Vault UI: http://{server_name}:{server_port}")

    uvicorn.run(
        proxy_app,
        host=server_name,
        port=server_port,
        log_level="warning",
    )


def _build_full_page(greeting: str, chips_json: str, folder_options: str, ollama_status: str) -> str:
    """Build the complete HTML page with embedded CSS and JS."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ODW Vault — Sovereign Knowledge Copilot</title>
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 512 512'%3E%3Crect fill='%23FEFEFE' width='512' height='512' rx='90'/%3E%3Cpath d='M160.66 199.59h-2.92c-2.94 0-5.33 2.39-5.33 5.33s2.39 5.33 5.33 5.33h4.9a25.4 25.4 0 01-1.98-10.66zm60.3-26.08l2.52 2.52c-.21-7.43-6.27-13.4-13.75-13.4-2.16 0-4.17.54-5.99 1.42a37.7 37.7 0 0117.22 9.46zm-27.38 72.61h-49.84c-20.88 0-37.86-16.99-37.86-37.87v-63.52c0-20.88 16.98-37.86 37.86-37.86h63.52c20.88 0 37.86 16.98 37.86 37.86v52.95l19.9 19.9v-.04l3.63 3.63c.88-4.17 1.35-8.49 1.35-12.92v-63.52c0-34.65-28.09-62.74-62.74-62.74h-63.52c-34.65 0-62.74 28.09-62.74 62.74v63.52c0 34.65 28.09 62.74 62.74 62.74h63.52c3.51 0 6.93-.36 10.29-.91l-23.97-23.97zm-38.52-69.7c0 7.62-6.18 13.79-13.79 13.79s-13.79-6.17-13.79-13.79c0-7.62 6.18-13.79 13.79-13.79s13.79 6.17 13.79 13.79z' fill='%23020303'/%3E%3Cpath d='M233.45 482.5c0-.15.01-.3.01-.45v-.05H90c-33.08 0-60-26.92-60-60V90c0-33.08 26.92-60 60-60h332c33.08 0 60 26.92 60 60v86.63c3.98-1.49 8.15-2.25 12.44-2.25 6.23 0 12.24 1.62 17.56 4.69V90c0-49.71-40.29-90-90-90H90C40.3 0 0 40.29 0 90v332c0 49.71 40.3 90 90 90h159.25c-9.67-6.4-15.8-17.35-15.8-29.5z' fill='%23020303'/%3E%3Cg transform='translate(166,168)'%3E%3Cpath d='M316.02 13.88c-3.32 1.49-6.44 3.59-9.18 6.32-11.3 11.33-11.7 29.29-1.43 41.23l10.61 10.62 7.08 7.08c1.42 1.43 1.42 3.74 0 5.16-.79.8-1.85 1.11-2.88 1.01l.03.52-.62-.63c-.62-.15-1.21-.42-1.68-.9l-1.93-1.92-39.56-39.59c-11.82-8.15-28.12-7-38.63 3.51-10.31 10.33-11.64 26.2-4.04 37.98l33.5 33.53c1.42 1.41 1.42 3.74 0 5.15-1.42 1.43-3.72 1.43-5.15.01l-14.86-14.88-31.29-31.33c-11.85-11.87-31.11-11.83-42.98.04-11.86 11.87-11.86 31.12 0 42.99l37.56 37.6c1.42 1.42 1.42 3.73 0 5.15-1.42 1.42-3.73 1.43-5.16.01l-3.03-3.04-97.93-98.03v.04L51.85 8.9c-11.86-11.87-31.1-11.87-42.95 0-11.87 11.88-11.87 31.13 0 43l51.01 51.06h.03l135.74 135.87c4.93 5.79 7.84 9.22 7.87 9.26 12.64 14.86 9.09 29.02-9.25 36H103c-16.61 0-30.09 13.33-30.36 29.9-.003.17-.014.33-.014.5-.007 14.27 9.81 26.23 23.05 29.51h160.4c49.68 0 89.95-40.31 89.95-90.04V16.82c-8.84-6.29-20.29-7.3-29.98-2.94' fill='%23CD2028'/%3E%3C/g%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script>
(function(){var t=localStorage.getItem('vault-theme');if(!t)t='light';document.documentElement.setAttribute('data-theme',t);})();
</script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root,[data-theme="light"]{
--bg-app:#F6F2EC;--bg-primary:#FFFFFF;--bg-secondary:#ECE6DC;--bg-tertiary:#E2DBCE;
--bg-user-msg:#14110F;--bg-code:#F6F2EC;--bg-citation:#FFFFFF;
--bg-sidebar:#FFFFFF;--bg-sidebar-hover:#ECE6DC;--bg-sidebar-active:#E2DBCE;
--sidebar-text:#14110F;--sidebar-text-dim:#5A544E;--sidebar-text-muted:#8A847C;--sidebar-border:rgba(20,17,15,0.10);
--text-primary:#14110F;--text-secondary:#2A2622;--text-tertiary:#5A544E;--text-disabled:#8A847C;--text-inverse:#F6F2EC;
--border-subtle:rgba(20,17,15,0.08);--border-default:#D6CFC0;--border-strong:rgba(20,17,15,0.24);
--accent:#FF5A1F;--accent-hover:#E04A12;--accent-2:#FFB199;--accent-subtle:#FFE9DF;--accent-fg:#14110F;
--citation-bg:#FFFFFF;--citation-border:#D6CFC0;--citation-marker:#FF5A1F;--citation-marker-bg:#FFE9DF;--citation-text:#5A544E;
--success:#1F7A4D;--error:#DC4B5C;--warning:#B58900;
--glow-accent:rgba(255,90,31,0.12);--glow-warm:rgba(255,177,153,0.10);
--shadow-xs:0 1px 2px rgba(20,17,15,0.05);
--shadow-sm:0 1px 3px rgba(20,17,15,0.07),0 1px 2px rgba(20,17,15,0.05);
--shadow-md:0 4px 10px -2px rgba(20,17,15,0.08),0 2px 4px -2px rgba(20,17,15,0.05);
--shadow-lg:0 12px 32px -8px rgba(20,17,15,0.14);
--shadow-composer:0 0 0 1px rgba(20,17,15,0.06),0 4px 16px rgba(20,17,15,0.07),0 12px 32px rgba(20,17,15,0.05);
--shadow-composer-focus:0 0 0 2px rgba(255,90,31,0.35),0 6px 24px rgba(255,90,31,0.10),0 16px 48px rgba(20,17,15,0.08);
--shadow-user-msg:0 2px 8px rgba(20,17,15,0.20),0 4px 16px rgba(20,17,15,0.10);
--scrollbar-thumb:rgba(20,17,15,0.14);--scrollbar-thumb-hover:rgba(20,17,15,0.26);
--radius-xs:4px;--radius-sm:6px;--radius-md:10px;--radius-lg:12px;--radius-xl:16px;--radius-2xl:22px;--radius-full:9999px;
--duration-fast:150ms;--duration-normal:240ms;--duration-slow:320ms;
--ease-default:cubic-bezier(0.4,0,0.2,1);--ease-enter:cubic-bezier(0,0,0.2,1);--ease-spring:cubic-bezier(0.34,1.56,0.64,1);
--font-display:"Instrument Serif","Times New Roman",serif;
--font-sans:"Inter",system-ui,-apple-system,"Segoe UI","Noto Sans SC","PingFang SC",sans-serif;
--font-mono:"JetBrains Mono","SF Mono","Fira Code","Cascadia Code",monospace;
}
[data-theme="dark"]{
--bg-app:#14110F;--bg-primary:#1F1B18;--bg-secondary:#2A2622;--bg-tertiary:#3A342E;
--bg-user-msg:#F6F2EC;--bg-code:#2A2622;--bg-citation:#1F1B18;
--bg-sidebar:#0F0D0B;--bg-sidebar-hover:#1F1B18;--bg-sidebar-active:#2A2622;
--sidebar-text:#F6F2EC;--sidebar-text-dim:#B8B1A5;--sidebar-text-muted:#8A847C;--sidebar-border:rgba(246,242,236,0.08);
--text-primary:#F6F2EC;--text-secondary:#E2DBCE;--text-tertiary:#B8B1A5;--text-disabled:#8A847C;--text-inverse:#14110F;
--border-subtle:rgba(246,242,236,0.08);--border-default:#3A342E;--border-strong:rgba(246,242,236,0.24);
--accent:#FF5A1F;--accent-hover:#FF7A4A;--accent-2:#FFB199;--accent-subtle:#3A1F14;--accent-fg:#14110F;
--citation-bg:#1F1B18;--citation-border:#3A342E;--citation-marker:#FF5A1F;--citation-marker-bg:#3A1F14;--citation-text:#B8B1A5;
--success:#34D399;--error:#F16577;--warning:#F5B85C;
--glow-accent:rgba(255,90,31,0.10);--glow-warm:rgba(255,177,153,0.06);
--shadow-xs:0 1px 2px rgba(0,0,0,0.30);
--shadow-sm:0 1px 3px rgba(0,0,0,0.40),0 1px 2px rgba(0,0,0,0.30);
--shadow-md:0 4px 12px -2px rgba(0,0,0,0.45),0 2px 4px -2px rgba(0,0,0,0.30);
--shadow-lg:0 16px 40px -8px rgba(0,0,0,0.55);
--shadow-composer:0 0 0 1px rgba(246,242,236,0.08),0 4px 20px rgba(0,0,0,0.40),0 12px 40px rgba(0,0,0,0.30);
--shadow-composer-focus:0 0 0 2px rgba(255,90,31,0.45),0 6px 28px rgba(255,90,31,0.12),0 16px 48px rgba(0,0,0,0.45);
--shadow-user-msg:0 2px 10px rgba(20,17,15,0.30),0 4px 16px rgba(0,0,0,0.20);
--scrollbar-thumb:rgba(246,242,236,0.12);--scrollbar-thumb-hover:rgba(246,242,236,0.24);
}
html,body{height:100dvh;overflow:hidden;background:var(--bg-app);font-family:var(--font-sans);color:var(--text-primary);font-size:15px;line-height:1.6;-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale;text-rendering:optimizeLegibility}
html.theme-transition,html.theme-transition *{transition:background-color var(--duration-slow) var(--ease-default),color var(--duration-slow) var(--ease-default),border-color var(--duration-slow) var(--ease-default),box-shadow var(--duration-slow) var(--ease-default)!important}

#app{display:flex;flex-direction:row;height:100dvh;width:100%;position:relative;background:
radial-gradient(1100px 640px at 88% -12%,var(--glow-warm),transparent 62%),
radial-gradient(950px 560px at -8% 112%,var(--glow-accent),transparent 58%),
var(--bg-app)}
#app::before{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;opacity:0.4;background-image:radial-gradient(var(--border-subtle) 1px,transparent 1px);background-size:26px 26px;mask-image:radial-gradient(ellipse 90% 80% at 50% 40%,#000 30%,transparent 100%)}
#app>*{position:relative;z-index:1}

/* Sidebar */
#sidebar{width:264px;height:100dvh;background:var(--bg-sidebar);border-right:1px solid var(--sidebar-border);display:flex;flex-direction:column;flex-shrink:0;transition:width var(--duration-slow) var(--ease-default),background var(--duration-slow) var(--ease-default);overflow:hidden;z-index:100;position:relative;color:var(--sidebar-text)}
#sidebar::after{content:'';position:absolute;top:0;right:0;bottom:0;width:1px;background:linear-gradient(to bottom,transparent,var(--sidebar-border) 20%,var(--sidebar-border) 80%,transparent);pointer-events:none}
#sidebar.collapsed{width:0;border-right:none}
#sidebar.icon-only{width:62px}
#sidebar.icon-only .sidebar-logo span,#sidebar.icon-only .sidebar-section__title,#sidebar.icon-only .conv-item__title,#sidebar.icon-only .conv-item__delete,#sidebar.icon-only .sidebar-status,#sidebar.icon-only .folder-tree-item span,#sidebar.icon-only #new-chat-btn span,#sidebar.icon-only .conv-search{display:none}
#sidebar.icon-only #new-chat-btn{padding:6px;width:38px;height:38px;justify-content:center}
#sidebar.icon-only .conv-item{padding:8px;justify-content:center}
#sidebar.icon-only .folder-tree-item{padding:6px 8px;text-align:center}
#sidebar.icon-only .sidebar-header{justify-content:center;padding:12px 8px}

.sidebar-header{display:flex;align-items:center;justify-content:space-between;padding:16px 14px 12px;min-height:60px}
.sidebar-logo{display:flex;align-items:center;text-decoration:none;padding:2px 2px;flex:1;min-width:0;transition:opacity var(--duration-fast) var(--ease-default)}
.sidebar-logo:hover{opacity:0.85}
.sidebar-logo__img{height:30px;width:auto;max-width:100%;display:block;object-fit:contain}
.sidebar-logo__img--dark{display:none}
[data-theme="dark"] .sidebar-logo__img--light{display:none}
[data-theme="dark"] .sidebar-logo__img--dark{display:block}
#new-chat-btn{display:flex;align-items:center;gap:6px;padding:8px 13px;background:var(--accent);color:#14110F;border:none;border-radius:var(--radius-sm);font-size:12.5px;font-weight:600;cursor:pointer;transition:all var(--duration-fast) var(--ease-default);font-family:var(--font-mono);letter-spacing:0.04em;text-transform:uppercase}
#new-chat-btn:hover{box-shadow:0 4px 16px var(--glow-accent);transform:translateY(-1px);filter:brightness(1.06)}
#new-chat-btn:active{transform:translateY(0);box-shadow:0 1px 4px var(--glow-accent)}

.sidebar-section{padding:12px 0;border-bottom:1px solid var(--sidebar-border)}
.sidebar-section__title{padding:0 16px 8px;font-family:var(--font-mono);font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.16em;color:var(--sidebar-text-muted);display:flex;align-items:center;gap:6px}
.sidebar-section__title::before{content:'';width:4px;height:14px;border-radius:1px;background:var(--accent)}
#conv-list{overflow-y:auto;padding:0 8px;max-height:38vh}

.conv-item{display:flex;align-items:center;justify-content:space-between;padding:9px 11px;border-radius:var(--radius-sm);cursor:pointer;font-size:13px;color:var(--sidebar-text-dim);transition:all var(--duration-fast) var(--ease-default);margin-bottom:2px;position:relative;border:1px solid transparent}
.conv-item::before{content:'';position:absolute;left:0;top:50%;transform:translateY(-50%);width:2px;height:0;background:var(--accent);border-radius:0 2px 2px 0;transition:height var(--duration-fast) var(--ease-default)}
.conv-item:hover{background:var(--bg-sidebar-hover);color:var(--sidebar-text)}
.conv-item.active{background:var(--bg-sidebar-active);color:var(--sidebar-text);font-weight:500;border-color:var(--sidebar-border)}
.conv-item.active::before{height:62%}
.conv-item__title{flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.conv-item__delete{opacity:0;background:none;border:none;cursor:pointer;color:var(--sidebar-text-muted);font-size:16px;padding:2px 6px;border-radius:var(--radius-xs);transition:all var(--duration-fast) var(--ease-default);font-family:var(--font-sans);line-height:1}
.conv-item:hover .conv-item__delete{opacity:0.7}
.conv-item__delete:hover{opacity:1!important;color:var(--error);background:rgba(220,75,92,0.10)}

.sidebar-footer{margin-top:auto;padding:12px 14px;border-top:1px solid var(--sidebar-border);display:flex;align-items:center;justify-content:space-between;gap:8px}
.sidebar-status{font-family:var(--font-mono);font-size:10px;color:var(--sidebar-text-muted);letter-spacing:0.02em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#sidebar-toggle{width:32px;height:32px;border-radius:var(--radius-sm);border:1px solid var(--sidebar-border);background:transparent;cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:15px;color:var(--sidebar-text-dim);transition:all var(--duration-fast) var(--ease-default);flex-shrink:0}
#sidebar-toggle:hover{background:var(--bg-sidebar-hover);border-color:var(--sidebar-border);color:var(--accent)}
#sidebar-toggle:active{transform:scale(0.95)}

/* Folder tree */
.folder-tree-item{display:flex;align-items:center;gap:7px;padding:6px 10px;font-size:12px;color:var(--sidebar-text-dim);cursor:pointer;border-radius:var(--radius-sm);transition:all var(--duration-fast) var(--ease-default);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;border:1px solid transparent}
.folder-tree-item:hover{background:var(--bg-sidebar-hover);color:var(--sidebar-text)}
.folder-tree-item.active{background:rgba(255,90,31,0.10);color:var(--accent);font-weight:500;border-color:rgba(255,90,31,0.15)}
.folder-tree-item__icon{font-size:13px;flex-shrink:0;opacity:0.65}
.folder-tree-item.active .folder-tree-item__icon{opacity:1}

/* Content area */
#content{flex:1;display:flex;flex-direction:column;min-width:0;height:100dvh;max-width:820px;margin:0 auto;width:100%}

/* Responsive */
@media(max-width:768px){
#sidebar{position:fixed;left:0;top:0;bottom:0;box-shadow:var(--shadow-lg)}
#sidebar.collapsed{transform:translateX(-280px);width:280px}
}

/* Topbar */
#topbar{display:flex;align-items:center;justify-content:space-between;padding:0 20px;height:54px;min-height:54px;flex-shrink:0;border-bottom:1px solid var(--border-subtle);background:transparent;backdrop-filter:blur(10px)}
.topbar-left{display:flex;align-items:center;gap:10px}
.topbar-logo{display:none;align-items:center;text-decoration:none;transition:opacity var(--duration-fast) var(--ease-default)}
.topbar-logo:hover{opacity:0.85}
.topbar-logo__img{height:26px;width:auto;display:block;object-fit:contain}
.topbar-logo__img--dark{display:none}
[data-theme="dark"] .topbar-logo__img--light{display:none}
[data-theme="dark"] .topbar-logo__img--dark{display:block}
#sidebar.collapsed ~ #content .topbar-logo{display:flex}
.topbar-right{display:flex;align-items:center;gap:10px}
.topbar-status{font-family:var(--font-mono);font-size:10px;color:var(--text-tertiary);letter-spacing:0.03em}
#theme-toggle{width:32px;height:32px;border-radius:var(--radius-full);border:1px solid var(--border-subtle);background:var(--bg-primary);cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:14px;transition:all var(--duration-fast) var(--ease-default)}
#theme-toggle:hover{background:var(--bg-tertiary);border-color:var(--border-default)}
#theme-toggle:active{transform:scale(0.95)}
#theme-toggle:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
#theme-toggle-sidebar{width:30px;height:30px;border-radius:var(--radius-full);border:1px solid var(--sidebar-border);background:transparent;cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:15px;line-height:1;color:var(--sidebar-text);transition:all var(--duration-fast) var(--ease-default);flex-shrink:0}
#theme-toggle-sidebar:hover{background:var(--bg-sidebar-hover);border-color:var(--accent);color:var(--accent)}
#theme-toggle-sidebar:active{transform:scale(0.92)}

/* Main */
#main{flex:1 1 0;min-height:0;display:flex;flex-direction:column;overflow:hidden;position:relative}

/* Hero — command deck */
#hero{display:flex;flex-direction:column;align-items:center;justify-content:flex-start;padding-top:10vh;text-align:center;padding-left:24px;padding-right:24px;animation:hero-fade-in 600ms var(--ease-enter) both}
@keyframes hero-fade-in{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}
#hero.hidden{display:none!important}
.hero-overline{display:inline-flex;align-items:center;gap:8px;padding:6px 14px;border:1px solid var(--border-default);border-radius:var(--radius-full);background:var(--bg-primary);font-family:var(--font-mono);font-size:10px;font-weight:500;letter-spacing:0.16em;text-transform:uppercase;color:var(--text-secondary);margin-bottom:22px;box-shadow:var(--shadow-xs);animation:hero-icon-in 500ms var(--ease-spring) 60ms both}
.hero-overline__dot{width:6px;height:6px;border-radius:50%;background:var(--accent);animation:pulse-dot 2.4s var(--ease-default) infinite}
@keyframes pulse-dot{0%,100%{box-shadow:0 0 0 0 var(--glow-accent)}50%{box-shadow:0 0 0 6px transparent}}
.hero-icon{width:66px;height:66px;border-radius:var(--radius-lg);background:#FFFFFF;display:flex;align-items:center;justify-content:center;margin-bottom:22px;box-shadow:0 8px 28px var(--glow-accent);animation:hero-icon-in 550ms var(--ease-spring) 120ms both;position:relative;overflow:hidden}
.hero-icon img{width:100%;height:100%;display:block}
.hero-icon::after{content:'';position:absolute;inset:-6px;border-radius:calc(var(--radius-lg) + 6px);border:1px solid var(--accent);opacity:0.2;animation:brand-ring 3.2s var(--ease-default) infinite}
@keyframes hero-icon-in{from{opacity:0;transform:scale(0.82)}to{opacity:1;transform:scale(1)}}
#hero h1{font-family:var(--font-display);font-size:clamp(36px,5vw,56px);font-weight:400;letter-spacing:-0.025em;line-height:1.05;color:var(--text-primary);margin-bottom:10px}
#hero h1::after{content:'.';color:var(--accent)}
#hero .sub{font-size:16px;color:var(--text-secondary);font-weight:400;margin-bottom:26px;letter-spacing:-0.01em}
.hero-pill{display:inline-flex;align-items:center;gap:7px;padding:7px 16px;background:var(--bg-primary);border:1px solid var(--border-subtle);border-radius:var(--radius-full);font-family:var(--font-mono);font-size:11px;color:var(--text-secondary);margin-top:4px;box-shadow:var(--shadow-xs);transition:all var(--duration-fast) var(--ease-default)}
.hero-pill:hover{border-color:var(--border-default);box-shadow:var(--shadow-sm);transform:translateY(-1px)}

/* Chips — query templates */
#chips{flex-shrink:0;padding:0 24px 12px;display:grid;grid-template-columns:repeat(2,1fr);gap:9px;animation:hero-fade-in 600ms var(--ease-enter) 240ms both;max-width:640px;margin:0 auto;width:100%}
@media(max-width:600px){#chips{grid-template-columns:1fr}}
.chip{display:flex;align-items:center;gap:11px;padding:13px 16px;border:1px solid var(--border-subtle);border-radius:var(--radius-lg);background:var(--bg-primary);cursor:pointer;font-size:13px;color:var(--text-secondary);text-align:left;font-family:var(--font-sans);line-height:1.4;transition:all var(--duration-fast) var(--ease-default);box-shadow:var(--shadow-xs);position:relative;overflow:hidden}
.chip::after{content:'\2192';position:absolute;right:14px;top:50%;transform:translateY(-50%) translateX(-4px);opacity:0;color:var(--accent);font-size:14px;transition:all var(--duration-fast) var(--ease-default)}
.chip:hover{border-color:var(--accent);transform:translateY(-2px);box-shadow:0 4px 16px var(--glow-accent);color:var(--text-primary);padding-right:34px}
.chip:hover::after{opacity:1;transform:translateY(-50%) translateX(0)}
.chip:active{transform:translateY(0);box-shadow:var(--shadow-xs)}
.chip:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.chip .i{font-size:15px;flex-shrink:0;width:30px;height:30px;display:flex;align-items:center;justify-content:center;background:var(--accent-subtle);border:1px solid var(--border-subtle);border-radius:var(--radius-sm)}
#chip-rf{display:block;margin:10px auto 0;background:none;border:none;cursor:pointer;font-size:13px;color:var(--text-tertiary);padding:4px 10px;font-family:var(--font-mono);border-radius:var(--radius-sm);transition:all var(--duration-fast) var(--ease-default)}
#chip-rf:hover{color:var(--accent);background:var(--accent-subtle)}

/* Messages */
#msgs{flex:1 1 0;min-height:0;overflow-y:auto;padding:24px 24px 8px;display:none;flex-direction:column;gap:22px}
#msgs.active{display:flex}
.msg{max-width:100%;word-wrap:break-word}
.msg.user{align-self:flex-end;max-width:74%;background:var(--bg-user-msg);color:var(--text-inverse);border-radius:var(--radius-xl) var(--radius-xl) var(--radius-xs) var(--radius-xl);padding:12px 18px;white-space:pre-wrap;animation:message-in var(--duration-normal) var(--ease-enter);box-shadow:var(--shadow-user-msg);font-size:14.5px;line-height:1.55;font-weight:450}
.msg.assistant{align-self:flex-start;background:transparent;padding:0;animation:message-in var(--duration-normal) var(--ease-enter);display:flex;gap:13px;align-items:flex-start;max-width:100%}
.msg-avatar{width:30px;height:30px;border-radius:var(--radius-sm);flex-shrink:0;background:#FFFFFF;overflow:hidden;margin-top:2px;box-shadow:0 2px 10px var(--glow-accent)}
.msg-avatar img{width:100%;height:100%;display:block}
.msg.assistant .md{flex:1;min-width:0}
@keyframes message-in{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
@media(prefers-reduced-motion:reduce){.msg{animation:none!important}.chip{transition:none!important}#composer{transition:none!important}#hero{animation:none!important}.hero-icon{animation:none!important}}

/* Markdown */
.md{line-height:1.75;color:var(--text-primary);font-size:14.5px}
.md p{margin:4px 0}
.md h1{font-family:var(--font-display);font-size:1.4em;font-weight:600;margin:18px 0 8px;color:var(--text-primary);letter-spacing:-0.02em}
.md h2{font-family:var(--font-display);font-size:1.2em;font-weight:600;margin:16px 0 6px;color:var(--text-primary);letter-spacing:-0.02em}
.md h3{font-family:var(--font-display);font-size:1.05em;font-weight:600;margin:14px 0 4px;color:var(--text-primary);letter-spacing:-0.01em}
.md ul,.md ol{margin:6px 0;padding-left:22px}
.md li{margin:3px 0}
.md a{color:var(--accent-2);text-decoration:none;border-bottom:1px solid transparent;transition:border-color var(--duration-fast) var(--ease-default)}
.md a:hover{border-bottom-color:var(--accent-2)}
.md blockquote{border-left:3px solid var(--accent);padding:6px 14px;margin:10px 0;color:var(--text-secondary);background:var(--bg-secondary);border-radius:0 var(--radius-sm) var(--radius-sm) 0}
.md hr{border:none;border-top:1px solid var(--border-default);margin:14px 0}
.md code{background:var(--bg-code);padding:2px 7px;border-radius:var(--radius-xs);font-size:12.5px;font-family:var(--font-mono);color:var(--accent);border:1px solid var(--border-subtle)}
.md pre{background:var(--bg-code);border-radius:var(--radius-md);margin:10px 0;overflow:hidden;border:1px solid var(--border-subtle)}
.md pre code{display:block;padding:14px 16px;background:none;overflow-x:auto;font-size:12.5px;line-height:1.6;border:none;color:var(--text-primary)}
.code-block{position:relative;margin:12px 0}
.code-block__bar{display:flex;align-items:center;justify-content:space-between;padding:7px 14px;background:var(--bg-tertiary);border-radius:var(--radius-md) var(--radius-md) 0 0;font-family:var(--font-mono);font-size:10px;color:var(--text-tertiary);font-weight:500;letter-spacing:0.06em;text-transform:uppercase;border:1px solid var(--border-subtle);border-bottom:none}
.code-block__copy{background:none;border:none;cursor:pointer;font-size:10px;color:var(--text-tertiary);font-family:var(--font-mono);padding:3px 8px;border-radius:var(--radius-xs);transition:all var(--duration-fast) var(--ease-default);font-weight:500;letter-spacing:0.04em}
.code-block__copy:hover{background:var(--border-subtle);color:var(--accent)}
.code-block__copy.copied{color:var(--success)}
.code-block pre{margin:0;border-radius:0 0 var(--radius-md) var(--radius-md)}

/* Streaming cursor */
.cursor .md::after{content:"\\25cf";color:var(--accent);animation:blink 700ms ease-in-out infinite;font-size:10px;margin-left:2px;vertical-align:baseline}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0.15}}

/* Citations — sources panel */
#cit{flex-shrink:0;padding:0 24px}
.citations-panel{background:var(--bg-citation);border:1px solid var(--border-subtle);border-radius:var(--radius-lg);margin:8px 0;overflow:hidden;box-shadow:var(--shadow-xs)}
.citations-panel__header{display:flex;align-items:center;justify-content:space-between;padding:11px 16px;cursor:pointer;user-select:none;font-family:var(--font-display);font-size:11px;font-weight:600;letter-spacing:0.10em;text-transform:uppercase;color:var(--text-secondary);transition:background var(--duration-fast) var(--ease-default)}
.citations-panel__header:hover{background:var(--border-subtle)}
.citations-panel__header .toggle-icon{font-size:10px;color:var(--text-tertiary);transition:transform var(--duration-normal) var(--ease-default)}
.citations-panel.collapsed .citations-panel__header .toggle-icon{transform:rotate(-90deg)}
.citations-panel__list{padding:0 12px 12px;display:flex;flex-direction:column;gap:6px;max-height:300px;overflow-y:auto;transition:max-height var(--duration-slow) var(--ease-default),opacity var(--duration-normal) var(--ease-default),padding var(--duration-slow) var(--ease-default)}
.citations-panel.collapsed .citations-panel__list{max-height:0;opacity:0;padding:0 12px;overflow:hidden}
.citation-card{display:flex;align-items:flex-start;gap:9px;padding:10px 12px;background:var(--citation-bg);border:1px solid var(--citation-border);border-radius:var(--radius-md);position:relative;overflow:hidden;transition:all var(--duration-fast) var(--ease-default);cursor:pointer}
.citation-card:hover{border-color:var(--accent-2);box-shadow:0 2px 12px var(--glow-warm);transform:translateX(2px)}
.citation-card__relevance{position:absolute;left:0;top:0;bottom:0;width:3px;background:linear-gradient(to bottom,var(--accent-2),var(--accent));border-radius:3px 0 0 3px}
.citation-card__number{width:22px;height:22px;border-radius:var(--radius-full);background:var(--citation-marker-bg);color:var(--citation-marker);font-family:var(--font-mono);font-size:10px;font-weight:600;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:1px}
.citation-card__content{flex:1;min-width:0}
.citation-card__title{font-size:12px;font-weight:600;color:var(--text-primary);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:2px}
.citation-card__snippet{font-size:11px;color:var(--citation-text);line-height:1.5;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.citation-card__meta{margin-top:4px}
.citation-card__meta-item{font-family:var(--font-mono);font-size:9.5px;color:var(--text-tertiary);letter-spacing:0.03em}

/* Filter */
#flt{flex-shrink:0;display:flex;align-items:center;justify-content:center;gap:7px;padding:4px 24px}
#flt label{font-family:var(--font-mono);font-size:10px;color:var(--text-tertiary);letter-spacing:0.06em;text-transform:uppercase}
#flt select{border:1px solid var(--border-default);border-radius:var(--radius-full);background:var(--bg-primary);padding:4px 14px;font-size:11px;font-family:var(--font-mono);color:var(--text-primary);outline:none;transition:all var(--duration-fast) var(--ease-default);cursor:pointer}
#flt select:focus{border-color:var(--accent);box-shadow:0 0 0 2px var(--accent-subtle)}

/* Composer */
#ca{flex-shrink:0;padding:8px 24px 22px}
#composer{background:var(--bg-primary);border-radius:var(--radius-2xl);border:1px solid var(--border-subtle);box-shadow:var(--shadow-composer);padding:13px 16px 12px 20px;display:flex;flex-direction:column;transition:all var(--duration-normal) var(--ease-default);backdrop-filter:blur(12px)}
#composer:focus-within{box-shadow:var(--shadow-composer-focus);border-color:var(--accent)}
#composer textarea{border:none;background:transparent;outline:none;resize:none;font-size:15px;font-family:var(--font-sans);line-height:1.55;color:var(--text-primary);width:100%;min-height:24px;max-height:200px;padding:4px 0}
#composer textarea::placeholder{color:var(--text-disabled)}
#ca-row{display:flex;align-items:center;justify-content:space-between;margin-top:5px;gap:6px}
#ca-row .scope-indicator{font-family:var(--font-mono);font-size:10px;color:var(--text-tertiary);display:flex;align-items:center;gap:5px;letter-spacing:0.03em}
#ca-row .scope-indicator__dot{width:6px;height:6px;border-radius:var(--radius-full);background:var(--accent);box-shadow:0 0 6px var(--glow-accent)}
#snd{width:37px;height:37px;border-radius:var(--radius-full);border:none;background:var(--bg-tertiary);color:var(--text-tertiary);cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:16px;font-weight:600;transition:all var(--duration-fast) var(--ease-default)}
#snd.enabled{background:var(--accent);color:#14110F;box-shadow:0 2px 12px var(--glow-accent)}
#snd.enabled:hover{box-shadow:0 4px 18px var(--glow-accent);transform:translateY(-1px);filter:brightness(1.06)}
#snd:active{transform:scale(0.95)}
#snd:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
#snd[disabled]{opacity:0.5;cursor:not-allowed}

/* Loading skeleton */
.conv-skeleton{padding:8px 12px;margin-bottom:2px}
.conv-skeleton__line{height:12px;border-radius:var(--radius-xs);background:linear-gradient(90deg,var(--bg-tertiary) 25%,var(--bg-sidebar-hover) 50%,var(--bg-tertiary) 75%);background-size:200% 100%;animation:skeleton-shimmer 1.5s ease-in-out infinite}
.conv-skeleton__line:nth-child(1){width:80%}
.conv-skeleton__line:nth-child(2){width:55%;margin-top:6px}
@keyframes skeleton-shimmer{0%{background-position:200% 0}100%{background-position:-200% 0}}

/* Scrollbar */
::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--scrollbar-thumb);border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:var(--scrollbar-thumb-hover)}

/* Typing dots animation */
.typing-dots{display:inline-flex;align-items:center;gap:4px;padding:6px 0}
.typing-dots span{width:6px;height:6px;border-radius:50%;background:var(--accent);animation:typing-bounce 1.4s ease-in-out infinite}
.typing-dots span:nth-child(2){animation-delay:0.2s}
.typing-dots span:nth-child(3){animation-delay:0.4s}
@keyframes typing-bounce{0%,60%,100%{transform:translateY(0);opacity:0.35}30%{transform:translateY(-6px);opacity:1}}

/* Stop button */
#stop-btn{width:36px;height:36px;border-radius:var(--radius-full);border:1px solid var(--border-default);background:var(--bg-primary);cursor:pointer;display:none;align-items:center;justify-content:center;font-size:16px;color:var(--text-secondary);transition:all var(--duration-fast) var(--ease-default);flex-shrink:0}
#stop-btn.visible{display:flex}
#stop-btn:hover{background:var(--error);color:#fff;border-color:var(--error)}
#stop-btn:active{transform:scale(0.95)}

/* Message action bar */
.msg-actions{display:flex;align-items:center;gap:2px;margin-top:7px;opacity:0;transition:opacity var(--duration-fast) var(--ease-default)}
.msg.assistant:hover .msg-actions{opacity:1}
.msg-action-btn{background:none;border:none;cursor:pointer;font-family:var(--font-mono);font-size:10px;color:var(--text-tertiary);padding:3px 9px;border-radius:var(--radius-xs);transition:all var(--duration-fast) var(--ease-default);display:flex;align-items:center;gap:4px;letter-spacing:0.03em}
.msg-action-btn:hover{background:var(--bg-secondary);color:var(--accent)}
.msg-action-btn.copied{color:var(--success)}
.msg-action-btn.feedback-up.liked{color:var(--success)}
.msg-action-btn.feedback-down.disliked{color:var(--error)}

/* Error state */
.msg-error{background:rgba(220,75,92,0.08)!important;border:1px solid rgba(220,75,92,0.25);border-radius:var(--radius-md);padding:12px 16px;margin:8px 0;display:flex;flex-direction:column;gap:8px}
.msg-error__text{font-size:13px;color:var(--error)}
.msg-error__retry{background:var(--error);color:#fff;border:none;padding:6px 14px;border-radius:var(--radius-sm);font-size:12px;font-weight:500;cursor:pointer;font-family:var(--font-sans);align-self:flex-start;transition:all var(--duration-fast) var(--ease-default)}
.msg-error__retry:hover{opacity:0.9;transform:translateY(-1px)}
.msg-error__retry:active{transform:translateY(0)}

/* Clickable inline citations */
.cite-link{color:var(--accent-2);text-decoration:none;font-family:var(--font-mono);font-weight:600;font-size:0.82em;cursor:pointer;padding:0 2px;transition:all var(--duration-fast) var(--ease-default);vertical-align:super}
.cite-link:hover{background:var(--citation-marker-bg);border-radius:3px;color:var(--citation-marker)}

/* Conversation search */
.conv-search{padding:0 12px 8px}
.conv-search input{width:100%;padding:6px 10px;border:1px solid var(--sidebar-border);border-radius:var(--radius-sm);background:var(--bg-sidebar-hover);font-size:12px;font-family:var(--font-sans);color:var(--sidebar-text);outline:none;transition:all var(--duration-fast) var(--ease-default)}
.conv-search input::placeholder{color:var(--sidebar-text-muted)}
.conv-search input:focus{border-color:var(--accent);box-shadow:0 0 0 2px rgba(255,90,31,0.15)}

/* Hero stats — telemetry strip */
.hero-stats{display:flex;align-items:stretch;margin-bottom:26px;border:1px solid var(--border-subtle);border-radius:var(--radius-lg);background:var(--bg-primary);box-shadow:var(--shadow-sm);overflow:hidden;animation:hero-fade-in 600ms var(--ease-enter) 180ms both}
.hero-stat{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:14px 26px;position:relative;min-width:96px}
.hero-stat + .hero-stat::before{content:'';position:absolute;left:0;top:20%;bottom:20%;width:1px;background:var(--border-subtle)}
.hero-stat__value{font-family:var(--font-mono);font-size:24px;font-weight:600;color:var(--accent);letter-spacing:-0.02em;line-height:1.1}
.hero-stat__label{font-family:var(--font-display);font-size:9.5px;color:var(--text-tertiary);text-transform:uppercase;letter-spacing:0.14em;margin-top:4px}
.hero-stat__dot{display:inline-block;width:6px;height:6px;border-radius:50%;margin-right:5px;vertical-align:middle}
.hero-stat__dot.green{background:var(--success);box-shadow:0 0 6px var(--success)}
.hero-stat__dot.yellow{background:var(--warning);box-shadow:0 0 6px var(--warning)}

/* Stop indicator during streaming */
#snd.stop-mode{background:var(--error)!important;color:#fff!important;box-shadow:0 2px 8px rgba(220,53,69,0.30)!important}
#snd.stop-mode:hover{background:#c82333!important;box-shadow:0 3px 12px rgba(220,53,69,0.40)!important}

/* Composer row with stop button */
#ca-controls{display:flex;align-items:center;gap:6px}

/* Refined mobile */
@media(max-width:768px){
#sidebar{position:fixed;left:0;top:0;bottom:0;z-index:200;box-shadow:var(--shadow-lg)}
#sidebar.collapsed{transform:translateX(-280px);width:280px}
#content{max-width:100%}
#hero{padding-top:8vh}
.hero-stats{flex-wrap:wrap}
.hero-stat{min-width:80px;padding:12px 18px}
.hero-stat__value{font-size:20px}
.hero-overline{font-size:9px;letter-spacing:0.12em}
#chips{grid-template-columns:1fr}
#ca{padding:8px 14px 14px}
#composer{padding:10px 14px 10px 16px}
.msg.user{max-width:85%}
}

/* Focus visible for keyboard nav */
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
button:focus-visible,input:focus-visible,textarea:focus-visible,select:focus-visible{outline:2px solid var(--accent);outline-offset:2px}

/* Empty state in conversations */
.conv-empty{text-align:center;padding:20px 16px;font-size:12px;color:var(--text-tertiary);line-height:1.6}

/* Message timestamp */
.msg-time{font-size:10px;color:var(--text-tertiary);margin-top:2px;opacity:0;transition:opacity var(--duration-fast) var(--ease-default)}
.msg.assistant:hover .msg-time{opacity:1}

/* Refined message spacing */
.msg.assistant .msg-body{flex:1;min-width:0}

/* Regenerate button pulse */
@keyframes btn-pulse{0%,100%{box-shadow:0 0 0 0 var(--glow-accent)}50%{box-shadow:0 0 0 6px transparent}}
.msg-action-btn.regenerate:active{animation:btn-pulse 0.6s var(--ease-default)}
</style>
</head>
<body>
<div id="app">
<!-- Sidebar -->
<div id="sidebar">
<div class="sidebar-header">
<a class="sidebar-logo" href="https://odw.ai/" target="_blank" rel="noopener"><img class="sidebar-logo__img sidebar-logo__img--light" src="/resource_img/odwai-logo-2048x651.png" alt="ODW.AI"><img class="sidebar-logo__img sidebar-logo__img--dark" src="/resource_img/odwai-logo-dark-2048x651.png" alt="ODW.AI"></a>
<button id="new-chat-btn"><span>+</span> <span>New Chat</span></button>
</div>
<div class="sidebar-section">
<div class="sidebar-section__title">Conversations</div>
<div class="conv-search"><input type="text" id="conv-search-input" placeholder="Search conversations..." autocomplete="off"></div>
<div id="conv-list"><div class="conv-skeleton"><div class="conv-skeleton__line"></div><div class="conv-skeleton__line"></div></div></div>
</div>
<div class="sidebar-section" style="flex:1;overflow:hidden;display:flex;flex-direction:column">
<div class="sidebar-section__title">Knowledge Base</div>
<div id="folder-tree" style="padding:0 8px;overflow-y:auto;flex:1"></div>
</div>
<div class="sidebar-footer">
<span class="sidebar-status">$OLLAMA_STATUS</span>
<button id="theme-toggle-sidebar" title="Toggle theme"></button>
</div>
<div style="padding:8px 14px 12px;border-top:1px solid var(--sidebar-border)">
<a href="https://odw.ai/" target="_blank" rel="noopener" style="font-family:var(--font-mono);font-size:9px;letter-spacing:0.12em;text-transform:uppercase;color:var(--sidebar-text-muted);text-decoration:none;transition:color 150ms">odw.ai &nearr;</a>
</div>
</div>

<!-- Main content -->
<div id="content">
<div id="topbar">
<div class="topbar-left">
<button id="sidebar-toggle" title="Toggle sidebar">&#9776;</button>
<a class="topbar-logo" href="https://odw.ai/" target="_blank" rel="noopener" style="margin-left:4px"><img class="topbar-logo__img topbar-logo__img--light" src="/resource_img/odwai-logo-2048x651.png" alt="ODW.AI"><img class="topbar-logo__img topbar-logo__img--dark" src="/resource_img/odwai-logo-dark-2048x651.png" alt="ODW.AI"></a>
</div>
<div class="topbar-right">
<div class="topbar-status"></div>
</div>
</div>

<div id="main">
<div id="hero">
<div class="hero-overline"><span class="hero-overline__dot"></span> ODW.AI &middot; SOVEREIGN KNOWLEDGE COPILOT</div>
<div class="hero-icon"><img src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 512 512'%3E%3Crect fill='%23FEFEFE' width='512' height='512' rx='90'/%3E%3Cpath d='M160.66 199.59h-2.92c-2.94 0-5.33 2.39-5.33 5.33s2.39 5.33 5.33 5.33h4.9a25.4 25.4 0 01-1.98-10.66zm60.3-26.08l2.52 2.52c-.21-7.43-6.27-13.4-13.75-13.4-2.16 0-4.17.54-5.99 1.42a37.7 37.7 0 0117.22 9.46zm-27.38 72.61h-49.84c-20.88 0-37.86-16.99-37.86-37.87v-63.52c0-20.88 16.98-37.86 37.86-37.86h63.52c20.88 0 37.86 16.98 37.86 37.86v52.95l19.9 19.9v-.04l3.63 3.63c.88-4.17 1.35-8.49 1.35-12.92v-63.52c0-34.65-28.09-62.74-62.74-62.74h-63.52c-34.65 0-62.74 28.09-62.74 62.74v63.52c0 34.65 28.09 62.74 62.74 62.74h63.52c3.51 0 6.93-.36 10.29-.91l-23.97-23.97zm-38.52-69.7c0 7.62-6.18 13.79-13.79 13.79s-13.79-6.17-13.79-13.79c0-7.62 6.18-13.79 13.79-13.79s13.79 6.17 13.79 13.79z' fill='%23020303'/%3E%3Cpath d='M233.45 482.5c0-.15.01-.3.01-.45v-.05H90c-33.08 0-60-26.92-60-60V90c0-33.08 26.92-60 60-60h332c33.08 0 60 26.92 60 60v86.63c3.98-1.49 8.15-2.25 12.44-2.25 6.23 0 12.24 1.62 17.56 4.69V90c0-49.71-40.29-90-90-90H90C40.3 0 0 40.29 0 90v332c0 49.71 40.3 90 90 90h159.25c-9.67-6.4-15.8-17.35-15.8-29.5z' fill='%23020303'/%3E%3Cg transform='translate(166,168)'%3E%3Cpath d='M316.02 13.88c-3.32 1.49-6.44 3.59-9.18 6.32-11.3 11.33-11.7 29.29-1.43 41.23l10.61 10.62 7.08 7.08c1.42 1.43 1.42 3.74 0 5.16-.79.8-1.85 1.11-2.88 1.01l.03.52-.62-.63c-.62-.15-1.21-.42-1.68-.9l-1.93-1.92-39.56-39.59c-11.82-8.15-28.12-7-38.63 3.51-10.31 10.33-11.64 26.2-4.04 37.98l33.5 33.53c1.42 1.41 1.42 3.74 0 5.15-1.42 1.43-3.72 1.43-5.15.01l-14.86-14.88-31.29-31.33c-11.85-11.87-31.11-11.83-42.98.04-11.86 11.87-11.86 31.12 0 42.99l37.56 37.6c1.42 1.42 1.42 3.73 0 5.15-1.42 1.42-3.73 1.43-5.16.01l-3.03-3.04-97.93-98.03v.04L51.85 8.9c-11.86-11.87-31.1-11.87-42.95 0-11.87 11.88-11.87 31.13 0 43l51.01 51.06h.03l135.74 135.87c4.93 5.79 7.84 9.22 7.87 9.26 12.64 14.86 9.09 29.02-9.25 36H103c-16.61 0-30.09 13.33-30.36 29.9-.003.17-.014.33-.014.5-.007 14.27 9.81 26.23 23.05 29.51h160.4c49.68 0 89.95-40.31 89.95-90.04V16.82c-8.84-6.29-20.29-7.3-29.98-2.94' fill='%23CD2028'/%3E%3C/g%3E%3C/svg%3E" alt="ODW.AI"></div>
<h1>$GREETING</h1>
<p class="sub">Ask anything across your indexed documents. Your data never leaves the building.</p>
<div class="hero-stats" id="hero-stats">
<div class="hero-stat"><div class="hero-stat__value" id="stat-files">--</div><div class="hero-stat__label">Files</div></div>
<div class="hero-stat"><div class="hero-stat__value" id="stat-folders">--</div><div class="hero-stat__label">Folders</div></div>
<div class="hero-stat"><div class="hero-stat__value" id="stat-chunks">--</div><div class="hero-stat__label">Chunks</div></div>
</div>
<div class="hero-pill"><span id="hero-status-dot"></span> <span id="hero-folder-count">Loading corpus...</span></div>
</div>
<div id="msgs"></div>
</div>

<div id="cit"></div>

<div id="chips"><button id="chip-rf" title="Refresh suggestions">&#x21bb;</button></div>

<div id="flt">
<label>Scope:</label>
<select id="ff">$FOLDER_OPTIONS</select>
</div>

<div id="ca">
<div id="composer">
<textarea id="inp" placeholder="Ask anything about your knowledge base..." rows="1" autofocus></textarea>
<div id="ca-row">
<div class="scope-indicator"><span class="scope-indicator__dot"></span> <span id="scope-label">All folders</span></div>
<div id="ca-controls">
<button id="stop-btn" title="Stop generating">&#x25a0;</button>
<button id="snd" title="Send">&#x2191;</button>
</div>
</div>
</div>
</div>
</div>
</div>

<script>
(function(){
  var _S = {H:[], streaming:false, abortFlag:false, lastQuery:'', lastFolder:''};
  var _allConvs = [];
  var _LOGO = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 512 512'%3E%3Crect fill='%23FEFEFE' width='512' height='512' rx='90'/%3E%3Cpath d='M160.66 199.59h-2.92c-2.94 0-5.33 2.39-5.33 5.33s2.39 5.33 5.33 5.33h4.9a25.4 25.4 0 01-1.98-10.66zm60.3-26.08l2.52 2.52c-.21-7.43-6.27-13.4-13.75-13.4-2.16 0-4.17.54-5.99 1.42a37.7 37.7 0 0117.22 9.46zm-27.38 72.61h-49.84c-20.88 0-37.86-16.99-37.86-37.87v-63.52c0-20.88 16.98-37.86 37.86-37.86h63.52c20.88 0 37.86 16.98 37.86 37.86v52.95l19.9 19.9v-.04l3.63 3.63c.88-4.17 1.35-8.49 1.35-12.92v-63.52c0-34.65-28.09-62.74-62.74-62.74h-63.52c-34.65 0-62.74 28.09-62.74 62.74v63.52c0 34.65 28.09 62.74 62.74 62.74h63.52c3.51 0 6.93-.36 10.29-.91l-23.97-23.97zm-38.52-69.7c0 7.62-6.18 13.79-13.79 13.79s-13.79-6.17-13.79-13.79c0-7.62 6.18-13.79 13.79-13.79s13.79 6.17 13.79 13.79z' fill='%23020303'/%3E%3Cpath d='M233.45 482.5c0-.15.01-.3.01-.45v-.05H90c-33.08 0-60-26.92-60-60V90c0-33.08 26.92-60 60-60h332c33.08 0 60 26.92 60 60v86.63c3.98-1.49 8.15-2.25 12.44-2.25 6.23 0 12.24 1.62 17.56 4.69V90c0-49.71-40.29-90-90-90H90C40.3 0 0 40.29 0 90v332c0 49.71 40.3 90 90 90h159.25c-9.67-6.4-15.8-17.35-15.8-29.5z' fill='%23020303'/%3E%3Cg transform='translate(166,168)'%3E%3Cpath d='M316.02 13.88c-3.32 1.49-6.44 3.59-9.18 6.32-11.3 11.33-11.7 29.29-1.43 41.23l10.61 10.62 7.08 7.08c1.42 1.43 1.42 3.74 0 5.16-.79.8-1.85 1.11-2.88 1.01l.03.52-.62-.63c-.62-.15-1.21-.42-1.68-.9l-1.93-1.92-39.56-39.59c-11.82-8.15-28.12-7-38.63 3.51-10.31 10.33-11.64 26.2-4.04 37.98l33.5 33.53c1.42 1.41 1.42 3.74 0 5.15-1.42 1.43-3.72 1.43-5.15.01l-14.86-14.88-31.29-31.33c-11.85-11.87-31.11-11.83-42.98.04-11.86 11.87-11.86 31.12 0 42.99l37.56 37.6c1.42 1.42 1.42 3.73 0 5.15-1.42 1.42-3.73 1.43-5.16.01l-3.03-3.04-97.93-98.03v.04L51.85 8.9c-11.86-11.87-31.1-11.87-42.95 0-11.87 11.88-11.87 31.13 0 43l51.01 51.06h.03l135.74 135.87c4.93 5.79 7.84 9.22 7.87 9.26 12.64 14.86 9.09 29.02-9.25 36H103c-16.61 0-30.09 13.33-30.36 29.9-.003.17-.014.33-.014.5-.007 14.27 9.81 26.23 23.05 29.51h160.4c49.68 0 89.95-40.31 89.95-90.04V16.82c-8.84-6.29-20.29-7.3-29.98-2.94' fill='%23CD2028'/%3E%3C/g%3E%3C/svg%3E";

  /* -- Theme -- */
  var _themeMode = localStorage.getItem('vault-theme') || 'light';
  var _themeIcons = {light:'\u2600\ufe0f', dark:'\\ud83c\\udf19'};
  
  function _applyTheme(){
    document.documentElement.setAttribute('data-theme', _themeMode);
    var btn = document.getElementById('theme-toggle-sidebar');
    if(btn) btn.textContent = _themeIcons[_themeMode] || '\u2600\ufe0f';
  }
  function _cycleTheme(){
    _themeMode = (_themeMode === 'light') ? 'dark' : 'light';
    localStorage.setItem('vault-theme', _themeMode);
    document.documentElement.classList.add('theme-transition');
    _applyTheme();
    setTimeout(function(){ document.documentElement.classList.remove('theme-transition'); }, 350);
  }
  _applyTheme();

  /* -- Send / Stop button state -- */
  function _updateSend(){
    var inp = document.getElementById('inp');
    var snd = document.getElementById('snd');
    var stopBtn = document.getElementById('stop-btn');
    if(!inp || !snd) return;
    var v = inp.value.trim();
    if(_S.streaming){
      snd.classList.add('enabled');
      snd.classList.add('stop-mode');
      snd.removeAttribute('disabled');
      snd.innerHTML = '\u25a0';
      if(stopBtn) stopBtn.classList.add('visible');
    } else {
      snd.classList.remove('stop-mode');
      snd.innerHTML = '\u2191';
      if(stopBtn) stopBtn.classList.remove('visible');
      if(v.length > 0){
        snd.classList.add('enabled');
        snd.removeAttribute('disabled');
      } else {
        snd.classList.remove('enabled');
        snd.setAttribute('disabled', '');
      }
    }
  }

  /* -- Scope label update -- */
  function _updateScopeLabel(){
    var sel = document.getElementById('ff');
    var lbl = document.getElementById('scope-label');
    if(sel && lbl){
      var val = sel.value;
      if(val === 'All folders'){
        lbl.textContent = 'All folders';
      } else {
        var parts = val.split('/');
        lbl.textContent = parts[parts.length - 1] || val;
      }
    }
  }

  /* -- Textarea auto-resize -- */
  function _autoResize(el){
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 200) + 'px';
  }

  /* -- Event handlers -- */
  var inp = document.getElementById('inp');
  var snd = document.getElementById('snd');
  var stopBtn = document.getElementById('stop-btn');
  if(inp){
    inp.addEventListener('input', function(){ _autoResize(this); _updateSend(); });
    inp.addEventListener('keydown', function(e){
      if(e.key === 'Enter' && !e.shiftKey){ e.preventDefault(); _send(); }
      if(e.key === 'Escape' && _S.streaming){ e.preventDefault(); _stopStreaming(); }
    });
  }
  // Keyboard shortcuts
  document.addEventListener('keydown', function(e){
    if((e.ctrlKey || e.metaKey) && e.key === 'Enter'){ e.preventDefault(); _send(); }
    if(e.key === 'Escape' && _S.streaming){ e.preventDefault(); _stopStreaming(); }
  });
  if(snd){
    snd.addEventListener('click', function(){
      if(_S.streaming){ _stopStreaming(); return; }
      _send();
    });
  }
  if(stopBtn){
    stopBtn.addEventListener('click', function(){
      if(_S.streaming) _stopStreaming();
    });
  }
  var ttBtn = document.getElementById('theme-toggle-sidebar');
  if(ttBtn) ttBtn.addEventListener('click', _cycleTheme);

  var ffSel = document.getElementById('ff');
  if(ffSel) ffSel.addEventListener('change', _updateScopeLabel);

  /* -- Chip clicks -- */
  document.getElementById('app').addEventListener('click', function(e){
    var chip = e.target.closest('.chip');
    if(chip){
      var text = chip.getAttribute('data-chip-text');
      if(text && inp){ inp.value = text; _autoResize(inp); _updateSend(); inp.focus(); }
      return;
    }
    var rf = e.target.closest('#chip-rf');
    if(rf){
      var a = window._PC;
      var o = Math.floor(Math.random() * a.length);
      var s = a.slice(o, o + 4);
      if(s.length < 4) s = s.concat(a.slice(0, 4 - s.length));
      _renderChips(s);
      return;
    }
    var citHeader = e.target.closest('.citations-panel__header');
    if(citHeader){
      var panel = citHeader.closest('.citations-panel');
      if(panel) panel.classList.toggle('collapsed');
      return;
    }
    var copyBtn = e.target.closest('.code-block__copy');
    if(copyBtn){
      var pre = copyBtn.closest('.code-block').querySelector('pre code');
      if(pre){
        navigator.clipboard.writeText(pre.textContent).then(function(){
          copyBtn.textContent = 'Copied!';
          copyBtn.classList.add('copied');
          setTimeout(function(){ copyBtn.textContent = 'Copy'; copyBtn.classList.remove('copied'); }, 2000);
        });
      }
    }
  });

  /* -- Stop streaming -- */
  function _stopStreaming(){
    _S.abortFlag = true;
    _S.streaming = false;
    _updateSend();
    var msgs = document.getElementById('msgs');
    var lastMsg = msgs ? msgs.lastElementChild : null;
    if(lastMsg && lastMsg.classList.contains('assistant') && lastMsg.classList.contains('cursor')){
      lastMsg.classList.remove('cursor');
      var mdEl = lastMsg.querySelector('.md');
      if(mdEl){
        var current = mdEl.innerHTML;
        if(!current || current.indexOf('typing-dots') !== -1){
          mdEl.innerHTML = '<p><em>Generation stopped.</em></p>';
        }
      }
    }
  }

  /* -- Send message -- */
  function _send(){
    if(!inp) return;
    var ff = document.getElementById('ff');
    var hero = document.getElementById('hero');
    var chipsEl = document.getElementById('chips');
    var t = inp.value.trim();
    if(!t || _S.streaming) return;

    _S.lastQuery = t;
    _S.lastFolder = ff ? ff.value : 'All folders';

    hero.classList.add('hidden');
    chipsEl.style.display = 'none';
    _addMsg('user', t);
    _S.H.push({role:'user', content:[{text:t, type:'text'}]});
    inp.value = ''; inp.style.height = 'auto';

    var el = _addMsg('assistant', '<div class="typing-dots"><span></span><span></span><span></span></div>', true);
    _S.streaming = true; _S.abortFlag = false;
    _updateSend();

    var _convId = window._getCurrentConvId ? window._getCurrentConvId() : null;
    if(!_convId){
      _convId = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c){
        var r = Math.random() * 16 | 0, v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
      });
      if(window._setCurrentConvId) window._setCurrentConvId(_convId);
    }
    fetch('/gradio_api/call/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({data:[t, _S.H, ff.value, _convId]})
    }).then(function(r){
      if(!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    }).then(function(resp){
      var eventId = resp.event_id;
      return fetch('/gradio_api/call/chat/' + eventId);
    }).then(function(r){
      if(!r.ok) throw new Error('HTTP ' + r.status);
      var reader = r.body.getReader();
      var buf = '';
      (function pump(){
        reader.read().then(function(res){
          if(_S.abortFlag){ reader.cancel(); _done(el); return; }
          if(res.done){ _done(el); return; }
          buf += new TextDecoder().decode(res.value);
          var lines = buf.split('\\n');
          buf = lines.pop() || '';
          for(var i = 0; i < lines.length; i++){
            var line = lines[i];
            if(line.startsWith('data:')){
              try {
                var d = JSON.parse(line.slice(5));
                if(d.error){
                  var mdEl = el.querySelector('.md');
                  if(mdEl) mdEl.innerHTML = '<div class="msg-error"><div class="msg-error__text">Error: ' + _escHtml(d.error) + '</div><button class="msg-error__retry" onclick="window._retryLast()">Retry</button></div>';
                  _done(el);
                  return;
                }
                if(d[0] && d[0].length){
                  var h = d[0];
                  var last = h[h.length - 1];
                  if(last && last.content && last.content.length){
                    var text = last.content[0].text || '';
                    var mdEl = el.querySelector('.md');
                    if(mdEl) mdEl.innerHTML = _md(text);
                  }
                  var msgs = document.getElementById('msgs');
                  msgs.scrollTop = msgs.scrollHeight;
                  if(d[1]) document.getElementById('cit').innerHTML = d[1];
                }
              } catch(e) {}
            }
          }
          pump();
        }).catch(function(err){
          if(!_S.abortFlag){
            var mdEl2 = el.querySelector('.md');
            if(mdEl2) mdEl2.innerHTML = '<div class="msg-error"><div class="msg-error__text">Error: ' + _escHtml(err.message) + '</div><button class="msg-error__retry" onclick="window._retryLast()">Retry</button></div>';
          }
          _done(el);
        });
      })();
    }).catch(function(err){
      if(!_S.abortFlag){
        var mdEl3 = el.querySelector('.md');
        if(mdEl3) mdEl3.innerHTML = '<div class="msg-error"><div class="msg-error__text">Error: ' + _escHtml(err.message) + '</div><button class="msg-error__retry" onclick="window._retryLast()">Retry</button></div>';
      }
      _done(el);
    });
  }

  function _escHtml(s){ var d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

  function _done(el){
    _S.streaming = false;
    _updateSend();
    el.classList.remove('cursor');
    if(window._loadConversations) window._loadConversations();
  }

  function _addMsg(role, text, stream){
    var d = document.createElement('div');
    d.className = 'msg ' + role;
    if(role === 'assistant'){
      var avatar = document.createElement('div');
      avatar.className = 'msg-avatar';
      avatar.innerHTML = '<img src="' + _LOGO + '" alt="ODW.AI">';
      d.appendChild(avatar);
      var body = document.createElement('div');
      body.className = 'msg-body';
      var m = document.createElement('div');
      m.className = 'md';
      m.innerHTML = _md(text);
      body.appendChild(m);
      // Action bar
      var actions = document.createElement('div');
      actions.className = 'msg-actions';
      actions.innerHTML = '<button class="msg-action-btn copy-msg" title="Copy">\u2398 Copy</button><button class="msg-action-btn regenerate" title="Regenerate">\u21bb Regenerate</button><button class="msg-action-btn feedback-up" title="Helpful">\u2191</button><button class="msg-action-btn feedback-down" title="Not helpful">\u2193</button>';
      body.appendChild(actions);
      d.appendChild(body);
      if(stream) d.classList.add('cursor');
    } else {
      d.textContent = text;
    }
    var msgs = document.getElementById('msgs');
    msgs.appendChild(d);
    msgs.classList.add('active');
    msgs.scrollTop = msgs.scrollHeight;
    return d;
  }

  /* -- Markdown renderer -- */
  function _md(t){
    if(!t) return '';
    t = t.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

    // Code blocks
    var codeBlocks = [];
    t = t.replace(/```(\\w*)\\n?([\\s\\S]*?)```/g, function(m, lang, code){
      var idx = codeBlocks.length;
      var lbl = lang || 'code';
      codeBlocks.push('<div class="code-block"><div class="code-block__bar"><span>' + lbl + '</span><button class="code-block__copy">Copy</button></div><pre><code>' + code.trim() + '</code></pre></div>');
      return '\\x00CB' + idx + '\\x00';
    });

    // Inline code
    t = t.replace(/`([^`]+)`/g, '<code>$1</code>');
    // Bold
    t = t.replace(/\\*\\*([^*]+)\\*\\*/g, '<strong>$1</strong>');
    // Italic
    t = t.replace(/(?<![\\w*])\\*([^*]+)\\*(?![\\w*])/g, '<em>$1</em>');
    // Headers
    t = t.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    t = t.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    t = t.replace(/^# (.+)$/gm, '<h1>$1</h1>');
    // Blockquotes
    t = t.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');
    // Horizontal rules
    t = t.replace(/^---$/gm, '<hr>');
    // Unordered lists
    t = t.replace(/^[\\-\\*] (.+)$/gm, '<li>$1</li>');
    t = t.replace(/((?:<li>.*<\\/li>\\n?)+)/g, '<ul>$1</ul>');
    // Ordered lists
    t = t.replace(/^\\d+\\. (.+)$/gm, '<li>$1</li>');
    // Links
    t = t.replace(/\\[([^\\]]+)\\]\\(([^)]+)\\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    // Citation markers - make clickable
    t = t.replace(/\[([\d,\s]+)\]/g, function(m, nums){
      var parts = nums.split(/[,\s]+/).filter(Boolean);
      var links = [];
      for(var ci = 0; ci < parts.length; ci++){
        links.push('<span class="cite-link" data-cite-num="' + parts[ci] + '" onclick="window._scrollToCite(' + parts[ci] + ')">[' + parts[ci] + ']</span>');
      }
      return links.join('');
    });
    // Paragraphs / line breaks
    t = t.replace(/\\n\\n/g, '</p><p>');
    t = t.replace(/\\n/g, '<br>');
    t = '<p>' + t + '</p>';
    t = t.replace(/<p><\\/p>/g, '');
    t = t.replace(/<p>(<h[123]>)/g, '$1');
    t = t.replace(/(<\\/h[123]>)<\\/p>/g, '$1');
    t = t.replace(/<p>(<ul>)/g, '$1');
    t = t.replace(/(<\\/ul>)<\\/p>/g, '$1');
    t = t.replace(/<p>(<blockquote>)/g, '$1');
    t = t.replace(/(<\\/blockquote>)<\\/p>/g, '$1');
    t = t.replace(/<p>(<hr>)<\\/p>/g, '$1');
    t = t.replace(/<p>(<div class="code-block")/g, '$1');
    t = t.replace(/(<\\/div>)<\\/p>/g, '$1');

    // Restore code blocks
    for(var i = 0; i < codeBlocks.length; i++){
      t = t.replace('\\x00CB' + i + '\\x00', codeBlocks[i]);
    }
    return t;
  }

  /* -- Chips -- */
  window._PC = $CHIPS_JSON;

  function _renderChips(s){
    var chipsEl = document.getElementById('chips');
    var rf = document.getElementById('chip-rf');
    if(!chipsEl) return;
    var h = '';
    for(var i = 0; i < s.length; i++){
      var esc = s[i].text.replace(/'/g, "\\\\'");
      h += '<button class="chip" data-chip-text="' + esc + '"><span class="i">' + s[i].icon + '</span><span>' + s[i].text + '</span></button>';
    }
    var oldChips = chipsEl.querySelectorAll('.chip');
    for(var j = 0; j < oldChips.length; j++) oldChips[j].remove();
    var tmp = document.createElement('div');
    tmp.innerHTML = h;
    while(tmp.firstChild) chipsEl.insertBefore(tmp.firstChild, rf);
  }

  _renderChips(window._PC.slice(0, 4));
  _updateSend();

  /* -- Sidebar Toggle -- */
  var sidebarToggle = document.getElementById('sidebar-toggle');
  if(sidebarToggle) sidebarToggle.addEventListener('click', function(){
    var sidebar = document.getElementById('sidebar');
    if(sidebar){
      sidebar.classList.toggle('collapsed');
    }
  });

  /* -- Conversation Management -- */
  var _currentConvId = null;

  function _loadConversations(){
    fetch('/gradio_api/call/list_conversations', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({data:[]})
    })
    .then(function(r){ return r.json(); })
    .then(function(resp){
      var eventId = resp.event_id;
      if(!eventId) return;
      return fetch('/gradio_api/call/list_conversations/' + eventId);
    })
    .then(function(r){
      if(!r) return;
      return r.text();
    })
    .then(function(text){
      if(!text) return;
      var convs = [];
      var lines = text.split('\\n');
      for(var i = 0; i < lines.length; i++){
        var line = lines[i].trim();
        if(line.indexOf('data: ') === 0){
          try {
            var parsed = JSON.parse(line.substring(6));
            if(Array.isArray(parsed) && Array.isArray(parsed[0])){
              convs = parsed[0];
            } else if(Array.isArray(parsed)){
              convs = parsed;
            }
          } catch(e){}
        }
      }
      _renderConvList(convs);
    })
    .catch(function(){});
  }

  function _renderConvList(conversations){
    var list = document.getElementById('conv-list');
    if(!list) return;
    if(!conversations || conversations.length === 0){
      list.innerHTML = '<div style="padding:16px;font-size:12px;color:var(--text-tertiary);text-align:center">No conversations yet</div>';
      return;
    }
    var html = '';
    for(var i = 0; i < conversations.length; i++){
      var c = conversations[i];
      var active = c.id === _currentConvId ? ' active' : '';
      var title = (c.title || 'Untitled').replace(/</g, '&lt;').replace(/>/g, '&gt;');
      html += '<div class="conv-item' + active + '" data-conv-id="' + c.id + '">' +
              '<span class="conv-item__title">' + title + '</span>' +
              '<button class="conv-item__delete" data-conv-id="' + c.id + '" title="Delete">&times;</button>' +
              '</div>';
    }
    list.innerHTML = html;
  }

  function _newChat(){
    _currentConvId = null;
    _S.H = [];
    _S.lastQuery = '';
    var msgs = document.getElementById('msgs');
    if(msgs){ msgs.innerHTML = ''; msgs.classList.remove('active'); }
    var hero = document.getElementById('hero');
    if(hero) hero.classList.remove('hidden');
    var chips = document.getElementById('chips');
    if(chips) chips.style.display = '';
    var cit = document.getElementById('cit');
    if(cit) cit.innerHTML = '';
    document.querySelectorAll('.conv-item').forEach(function(el){ el.classList.remove('active'); });
    if(window.innerWidth <= 768){
      var sidebar = document.getElementById('sidebar');
      if(sidebar) sidebar.classList.add('collapsed');
    }
    _fetchStats();
  }

  function _selectConversation(convId){
    _currentConvId = convId;
    fetch('/conversations/' + convId + '/messages')
    .then(function(r){ return r.json(); })
    .then(function(data){
      var messages = data.messages || data || [];
      _S.H = [];
      var msgs = document.getElementById('msgs');
      if(!msgs) return;
      msgs.innerHTML = '';
      msgs.classList.add('active');
      var hero = document.getElementById('hero');
      if(hero) hero.classList.add('hidden');
      var chips = document.getElementById('chips');
      if(chips) chips.style.display = 'none';
      for(var i = 0; i < messages.length; i++){
        var m = messages[i];
        if(m.role === 'user'){
          _addMsg('user', m.content);
          _S.H.push({role:'user', content:[{text:m.content, type:'text'}]});
        } else if(m.role === 'assistant'){
          _addMsg('assistant', m.content);
        }
      }
      msgs.scrollTop = msgs.scrollHeight;
    })
    .catch(function(){});
    document.querySelectorAll('.conv-item').forEach(function(el){
      el.classList.toggle('active', el.getAttribute('data-conv-id') === convId);
    });
    if(window.innerWidth <= 768){
      var sidebar = document.getElementById('sidebar');
      if(sidebar) sidebar.classList.add('collapsed');
    }
  }

  function _deleteConversation(convId, event){
    event.stopPropagation();
    if(!confirm('Delete this conversation?')) return;
    fetch('/conversations/' + convId, {method:'DELETE'})
    .then(function(){
      if(_currentConvId === convId) _newChat();
      _loadConversations();
    })
    .catch(function(){});
  }

  var newChatBtn = document.getElementById('new-chat-btn');
  if(newChatBtn) newChatBtn.addEventListener('click', _newChat);

  var convList = document.getElementById('conv-list');
  if(convList) convList.addEventListener('click', function(e){
    var deleteBtn = e.target.closest('.conv-item__delete');
    if(deleteBtn){
      _deleteConversation(deleteBtn.getAttribute('data-conv-id'), e);
      return;
    }
    var item = e.target.closest('.conv-item');
    if(item){
      _selectConversation(item.getAttribute('data-conv-id'));
    }
  });

  /* -- Folder Tree -- */
  function _buildFolderTree(){
    var tree = document.getElementById('folder-tree');
    var sel = document.getElementById('ff');
    if(!tree || !sel) return;
    var html = '<div class="folder-tree-item active" data-folder="All folders"><span class="folder-tree-item__icon">&#x1f4c1;</span><span>All folders</span></div>';
    for(var i = 0; i < sel.options.length; i++){
      var opt = sel.options[i];
      if(opt.value === 'All folders') continue;
      var indent = 0;
      var parts = opt.value.split('/');
      if(parts.length > 1) indent = (parts.length - 1) * 12;
      html += '<div class="folder-tree-item" data-folder="' + opt.value.replace(/"/g, '&quot;') + '" style="padding-left:' + (8 + indent) + 'px"><span class="folder-tree-item__icon">&#x1f4c4;</span><span>' + opt.text + '</span></div>';
    }
    tree.innerHTML = html;
  }

  var folderTree = document.getElementById('folder-tree');
  if(folderTree) folderTree.addEventListener('click', function(e){
    var item = e.target.closest('.folder-tree-item');
    if(!item) return;
    var folder = item.getAttribute('data-folder');
    var sel = document.getElementById('ff');
    if(sel){
      for(var i = 0; i < sel.options.length; i++){
        if(sel.options[i].value === folder){ sel.selectedIndex = i; break; }
      }
    }
    folderTree.querySelectorAll('.folder-tree-item').forEach(function(el){ el.classList.remove('active'); });
    item.classList.add('active');
    _updateScopeLabel();
  });

  _buildFolderTree();
  _loadConversations();
  _updateScopeLabel();

  /* On small screens start with the sidebar collapsed so it does not cover content */
  if(window.innerWidth <= 768){
    var _sb = document.getElementById('sidebar');
    if(_sb) _sb.classList.add('collapsed');
  }

  /* Expose _currentConvId for send function */
  window._getCurrentConvId = function(){ return _currentConvId; };
  window._setCurrentConvId = function(id){ _currentConvId = id; };
  window._loadConversations = _loadConversations;

  /* -- Action bar handlers -- */
  var msgsContainer = document.getElementById('msgs');
  if(msgsContainer) msgsContainer.addEventListener('click', function(e){
    var btn = e.target.closest('.msg-action-btn');
    if(!btn) return;
    var msgEl = e.target.closest('.msg.assistant');
    if(!msgEl) return;
    var mdEl = msgEl.querySelector('.md');
    var text = mdEl ? mdEl.textContent : '';

    if(btn.classList.contains('copy-msg')){
      navigator.clipboard.writeText(text).then(function(){
        btn.classList.add('copied');
        btn.innerHTML = '\u2713 Copied';
        setTimeout(function(){ btn.classList.remove('copied'); btn.innerHTML = '\u2398 Copy'; }, 2000);
      });
    } else if(btn.classList.contains('regenerate')){
      _retryLast();
    } else if(btn.classList.contains('feedback-up')){
      var liked = btn.classList.toggle('liked');
      var downBtn = msgEl.querySelector('.feedback-down');
      if(downBtn) downBtn.classList.remove('disliked');
      if(liked) _sendFeedback('up');
    } else if(btn.classList.contains('feedback-down')){
      var disliked = btn.classList.toggle('disliked');
      var upBtn = msgEl.querySelector('.feedback-up');
      if(upBtn) upBtn.classList.remove('liked');
      if(disliked) _sendFeedback('down');
    }
  });

  function _sendFeedback(fb){
    if(!_currentConvId) return;
    fetch('/conversations/' + _currentConvId + '/feedback', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({feedback: fb})
    }).catch(function(){});
  }

  /* -- Scroll to citation -- */
  window._scrollToCite = function(num){
    var cit = document.getElementById('cit');
    if(!cit) return;
    var card = cit.querySelector('.citation-card[data-cite="' + num + '"]');
    if(card){
      card.scrollIntoView({behavior:'smooth',block:'center'});
      card.style.boxShadow = '0 0 0 3px var(--accent)';
      setTimeout(function(){ card.style.boxShadow = ''; }, 2000);
    }
  };

  /* -- Retry last query -- */
  window._retryLast = function(){
    if(!_S.lastQuery) return;
    var inp = document.getElementById('inp');
    var ff = document.getElementById('ff');
    if(inp) inp.value = _S.lastQuery;
    if(ff && _S.lastFolder) ff.value = _S.lastFolder;
    _send();
  };

  /* -- Conversation search -- */
  var convSearchInput = document.getElementById('conv-search-input');
  if(convSearchInput) convSearchInput.addEventListener('input', function(){
    var q = this.value.toLowerCase();
    var items = document.querySelectorAll('.conv-item');
    for(var i = 0; i < items.length; i++){
      var title = (items[i].querySelector('.conv-item__title') || {}).textContent || '';
      items[i].style.display = title.toLowerCase().indexOf(q) !== -1 ? '' : 'none';
    }
  });

  /* -- Fetch corpus stats -- */
  function _fetchStats(){
    fetch('/stats')
    .then(function(r){ return r.json(); })
    .then(function(s){
      var elF = document.getElementById('stat-files');
      var elD = document.getElementById('stat-folders');
      var elC = document.getElementById('stat-chunks');
      var elS = document.getElementById('hero-folder-count');
      var elDot = document.getElementById('hero-status-dot');
      if(elF) elF.textContent = s.file_count || 0;
      if(elD) elD.textContent = s.folder_count || 0;
      if(elC) elC.textContent = s.chunk_count || 0;
      if(elS){
        var hasData = (s.file_count || 0) > 0;
        elS.textContent = hasData ? (s.file_count + ' files, ' + s.folder_count + ' folders indexed') : 'No documents indexed yet';
      }
      if(elDot){
        var hasData2 = (s.file_count || 0) > 0;
        elDot.innerHTML = '<span class="hero-stat__dot ' + (hasData2 ? 'green' : 'yellow') + '"></span>';
      }
    })
    .catch(function(){});
  }
  _fetchStats();
})();
</script>
</body>
</html>""".replace("$GREETING", greeting) \
             .replace("$CHIPS_JSON", chips_json) \
             .replace("$FOLDER_OPTIONS", folder_options) \
             .replace("$OLLAMA_STATUS", ollama_status)


if __name__ == "__main__":
    config_path = Path(__file__).resolve().parent.parent / "config.toml"
    cfg = load_app_config(config_path)
    launch_ui(cfg)
