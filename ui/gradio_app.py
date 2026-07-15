"""Phase 14: Modern Gradio chat UI for the RAG pipeline — ODW.ai Vault.

Design principles:
- Composer centered on empty state, docks to bottom when chat starts
- Neutral palette (#FAFAFA), Inter-style typography, minimal chrome
- User messages right-aligned bubbles, assistant messages as bare text
- Prompt starter chips, time-based greeting

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


def _on_chat(message: str, history: list[dict], folder_filter: str):
    import time as _time
    query_start = _time.time()
    logger.info(f"[QUERY] message={message[:100]!r}, folder_filter={folder_filter!r}")

    if not message or not message.strip():
        logger.warning("[QUERY] Empty message")
        yield history, _citations_html([])
        return

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
        history[-1]["content"] = f"Generation failed: {exc}"
        yield history, _citations_html([])

    elapsed_total = _time.time() - query_start
    logger.info(f"[QUERY] Done in {elapsed_total:.2f}s, citations={len(citations)}, tokens={len(full_text)}")

    yield history, _citations_html(citations)


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


# ---------------------------------------------------------------------------
# Full HTML page — pure custom layout, zero Gradio interference
# ---------------------------------------------------------------------------

def launch_ui(cfg, share: bool = False, server_name: str = "127.0.0.1", server_port: int = 7860):
    """Launch the modern ODW.ai Vault chat interface.

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

    with gr.Blocks(title="ODW.ai Vault") as gradio_app:
        chatbot = gr.Chatbot(visible=False)
        citations_out = gr.HTML(visible=False)
        msg_box = gr.Textbox(visible=False)
        folder_box = gr.Dropdown(choices=folder_choices, value="All folders", visible=False)
        submit_btn = gr.Button(visible=False)

        submit_btn.click(
            fn=_on_chat,
            inputs=[msg_box, chatbot, folder_box],
            outputs=[chatbot, citations_out],
            api_name="chat",
        )
        chatbot.like(fn=_on_feedback)

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

        gradio_app.launch(
            server_name=server_name,
            server_port=gradio_port,
            share=False,
        )

    gradio_thread = threading.Thread(target=run_gradio, daemon=True)
    gradio_thread.start()

    # Wait for Gradio to start
    time.sleep(2)
    print(f"  Gradio API backend: {gradio_url}")

    # Step 2: Create our proxy server on the user-facing port
    # Use httpx reverse proxy to forward /gradio_api/* to Gradio backend
    from starlette.middleware.base import BaseHTTPMiddleware

    proxy_app = FastAPI(title="ODW.ai Vault")

    @proxy_app.get("/")
    async def root():
        return HTMLResponse(content=full_html)

    # Single catch-all proxy for all Gradio API requests
    async def _do_proxy(request: Request):
        path = request.url.path[len("/gradio_api/"):]
        target = f"{gradio_url}/gradio_api/{path}"
        if request.url.query:
            target += f"?{request.url.query}"
        body = await request.body() if request.method in ("POST", "PUT", "PATCH") else None
        async with httpx.AsyncClient() as client:
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
    print(f"  ODW.ai Vault UI: http://{server_name}:{server_port}")

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
<title>ODW.ai Vault</title>
<script>
(function(){var t=localStorage.getItem('vault-theme');if(!t)t='system';var d=t;if(t==='system')d=window.matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light';document.documentElement.setAttribute('data-theme',d);document.documentElement.setAttribute('data-theme-mode',t);})();
</script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root,[data-theme="light"]{
--bg-app:#FAFAF8;--bg-primary:#FFFFFF;--bg-secondary:#F5F5F0;--bg-tertiary:#EDEDEA;
--bg-user-msg:#F0EFEB;--bg-code:#F5F4F0;--bg-citation:#F8F6F0;
--text-primary:#1A1A1A;--text-secondary:#5C5C5C;--text-tertiary:#8C8C8C;--text-disabled:#B8B8B8;--text-inverse:#FFFFFF;
--border-subtle:rgba(0,0,0,0.06);--border-default:rgba(0,0,0,0.10);--border-strong:rgba(0,0,0,0.16);
--accent:#6B5CE7;--accent-hover:#5A4BD6;--accent-subtle:#F0EEFC;--accent-fg:#FFFFFF;
--citation-bg:#F8F6F0;--citation-border:#E8E4D8;--citation-marker:#8B7E6A;--citation-marker-bg:#F0EDE4;--citation-text:#6B6355;
--success:#2D9F5E;--error:#DC3545;
--shadow-xs:0 1px 2px rgba(0,0,0,0.04);
--shadow-sm:0 1px 3px rgba(0,0,0,0.06),0 1px 2px rgba(0,0,0,0.04);
--shadow-md:0 4px 6px -1px rgba(0,0,0,0.06),0 2px 4px -2px rgba(0,0,0,0.04);
--shadow-composer:0 0 0 1px rgba(0,0,0,0.05),0 2px 12px rgba(0,0,0,0.06);
--shadow-composer-focus:0 0 0 1.5px rgba(107,92,231,0.3),0 4px 24px rgba(0,0,0,0.08);
--scrollbar-thumb:rgba(0,0,0,0.12);--scrollbar-thumb-hover:rgba(0,0,0,0.20);
--radius-xs:4px;--radius-sm:6px;--radius-md:10px;--radius-lg:14px;--radius-xl:20px;--radius-2xl:28px;--radius-full:9999px;
--duration-fast:100ms;--duration-normal:200ms;--duration-slow:300ms;
--ease-default:cubic-bezier(0.4,0,0.2,1);--ease-enter:cubic-bezier(0,0,0.2,1);
--font-sans:"Inter",-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC","PingFang SC",sans-serif;
--font-mono:"JetBrains Mono","SF Mono","Fira Code","Cascadia Code",monospace;
}
[data-theme="dark"]{
--bg-app:#141413;--bg-primary:#1A1A18;--bg-secondary:#222220;--bg-tertiary:#2C2C28;
--bg-user-msg:#2A2A26;--bg-code:#1E1E1C;--bg-citation:#201F1A;
--text-primary:#EDECEA;--text-secondary:#A09E9A;--text-tertiary:#6B6966;--text-disabled:#4A4845;--text-inverse:#1A1A1A;
--border-subtle:rgba(255,255,255,0.06);--border-default:rgba(255,255,255,0.10);--border-strong:rgba(255,255,255,0.16);
--accent:#8B7FF5;--accent-hover:#9D93F7;--accent-subtle:rgba(139,127,245,0.10);--accent-fg:#1A1A1A;
--citation-bg:#1E1D18;--citation-border:#33312A;--citation-marker:#A89B85;--citation-marker-bg:#2A2820;--citation-text:#9E9585;
--success:#3DB86E;--error:#E85565;
--shadow-xs:0 1px 2px rgba(0,0,0,0.20);
--shadow-sm:0 1px 3px rgba(0,0,0,0.30),0 1px 2px rgba(0,0,0,0.20);
--shadow-md:0 4px 6px -1px rgba(0,0,0,0.30),0 2px 4px -2px rgba(0,0,0,0.20);
--shadow-composer:0 0 0 1px rgba(255,255,255,0.06),0 2px 12px rgba(0,0,0,0.30);
--shadow-composer-focus:0 0 0 1.5px rgba(139,127,245,0.40),0 4px 24px rgba(0,0,0,0.40);
--scrollbar-thumb:rgba(255,255,255,0.10);--scrollbar-thumb-hover:rgba(255,255,255,0.18);
}
html,body{height:100dvh;overflow:hidden;background:var(--bg-app);font-family:var(--font-sans);color:var(--text-primary);font-size:15px;line-height:1.6;-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale}
html.theme-transition,html.theme-transition *{transition:background-color var(--duration-slow) var(--ease-default),color var(--duration-slow) var(--ease-default),border-color var(--duration-slow) var(--ease-default),box-shadow var(--duration-slow) var(--ease-default)!important}

#app{display:flex;flex-direction:column;height:100dvh;max-width:768px;margin:0 auto}

/* Topbar */
#topbar{display:flex;align-items:center;justify-content:space-between;padding:0 16px;height:48px;min-height:48px;flex-shrink:0;border-bottom:1px solid var(--border-subtle)}
.topbar-left{display:flex;align-items:center;gap:8px}
.topbar-logo{font-size:16px;font-weight:600;letter-spacing:-0.01em;color:var(--text-primary)}
.topbar-logo span{color:var(--text-tertiary);font-weight:400;margin-left:6px;font-size:11px}
.topbar-right{display:flex;align-items:center;gap:10px}
.topbar-status{font-size:11px;color:var(--text-tertiary)}
#theme-toggle{width:32px;height:32px;border-radius:var(--radius-full);border:1px solid var(--border-subtle);background:var(--bg-secondary);cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:14px;transition:background var(--duration-fast) var(--ease-default),border-color var(--duration-fast) var(--ease-default)}
#theme-toggle:hover{background:var(--bg-tertiary);border-color:var(--border-default)}
#theme-toggle:active{transform:scale(0.98)}
#theme-toggle:focus-visible{outline:2px solid var(--accent);outline-offset:2px}

/* Main */
#main{flex:1 1 0;min-height:0;display:flex;flex-direction:column;overflow:hidden;position:relative}

/* Hero */
#hero{display:flex;flex-direction:column;align-items:center;justify-content:flex-start;padding-top:18vh;text-align:center;padding-left:16px;padding-right:16px;transition:opacity var(--duration-normal) var(--ease-default)}
#hero.hidden{display:none!important}
#hero h1{font-size:clamp(28px,4vw,40px);font-weight:400;letter-spacing:-0.02em;line-height:1.2;color:var(--text-primary);margin-bottom:6px}
#hero .sub{font-size:16px;color:var(--text-secondary);font-weight:400;margin-bottom:16px}
.hero-pill{display:inline-flex;align-items:center;gap:6px;padding:6px 14px;background:var(--bg-secondary);border:1px solid var(--border-subtle);border-radius:var(--radius-full);font-size:12px;color:var(--text-secondary);margin-bottom:20px}

/* Chips */
#chips{flex-shrink:0;padding:0 16px 8px;display:grid;grid-template-columns:repeat(2,1fr);gap:8px}
@media(max-width:600px){#chips{grid-template-columns:1fr}}
.chip{display:flex;align-items:center;gap:8px;padding:10px 14px;border:1px solid var(--border-subtle);border-radius:var(--radius-lg);background:transparent;cursor:pointer;font-size:13px;color:var(--text-secondary);text-align:left;font-family:var(--font-sans);line-height:1.35;transition:all var(--duration-fast) var(--ease-default)}
.chip:hover{background:var(--bg-secondary);border-color:var(--border-default);transform:translateY(-1px)}
.chip:active{transform:scale(0.98)}
.chip:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.chip .i{font-size:15px;flex-shrink:0}
#chip-rf{display:block;margin:6px auto 0;background:none;border:none;cursor:pointer;font-size:12px;color:var(--text-tertiary);padding:4px;font-family:var(--font-sans);border-radius:var(--radius-sm);transition:color var(--duration-fast) var(--ease-default)}
#chip-rf:hover{color:var(--text-secondary)}

/* Messages */
#msgs{flex:1 1 0;min-height:0;overflow-y:auto;padding:16px;display:none;flex-direction:column;gap:24px}
#msgs.active{display:flex}
.msg{max-width:100%;word-wrap:break-word}
.msg.user{align-self:flex-end;max-width:70%;background:var(--bg-user-msg);border-radius:var(--radius-xl) var(--radius-xl) var(--radius-xs) var(--radius-xl);padding:10px 16px;white-space:pre-wrap;animation:message-in var(--duration-normal) var(--ease-enter)}
.msg.assistant{align-self:flex-start;background:transparent;padding:0;animation:message-in var(--duration-normal) var(--ease-enter)}
@keyframes message-in{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
@media(prefers-reduced-motion:reduce){.msg{animation:none!important}.chip{transition:none!important}#composer{transition:none!important}}

/* Markdown */
.md{line-height:1.7;color:var(--text-primary)}
.md p{margin:4px 0}
.md h1{font-size:1.4em;font-weight:600;margin:16px 0 8px;color:var(--text-primary)}
.md h2{font-size:1.2em;font-weight:600;margin:14px 0 6px;color:var(--text-primary)}
.md h3{font-size:1.05em;font-weight:600;margin:12px 0 4px;color:var(--text-primary)}
.md ul,.md ol{margin:6px 0;padding-left:22px}
.md li{margin:2px 0}
.md a{color:var(--accent);text-decoration:none}
.md a:hover{text-decoration:underline;color:var(--accent-hover)}
.md blockquote{border-left:3px solid var(--border-default);padding:4px 12px;margin:8px 0;color:var(--text-secondary);background:var(--bg-secondary);border-radius:0 var(--radius-sm) var(--radius-sm) 0}
.md hr{border:none;border-top:1px solid var(--border-default);margin:12px 0}
.md code{background:var(--bg-code);padding:2px 6px;border-radius:var(--radius-xs);font-size:13px;font-family:var(--font-mono);color:var(--text-primary)}
.md pre{background:var(--bg-code);border-radius:var(--radius-md);margin:10px 0;overflow:hidden}
.md pre code{display:block;padding:14px 16px;background:none;overflow-x:auto;font-size:13px;line-height:1.5}
.code-block{position:relative;margin:10px 0}
.code-block__bar{display:flex;align-items:center;justify-content:space-between;padding:6px 12px;background:var(--bg-tertiary);border-radius:var(--radius-md) var(--radius-md) 0 0;font-size:11px;color:var(--text-tertiary)}
.code-block__copy{background:none;border:none;cursor:pointer;font-size:11px;color:var(--text-tertiary);font-family:var(--font-sans);padding:2px 6px;border-radius:var(--radius-xs);transition:background var(--duration-fast) var(--ease-default),color var(--duration-fast) var(--ease-default)}
.code-block__copy:hover{background:var(--border-subtle);color:var(--text-secondary)}
.code-block__copy.copied{color:var(--success)}
.code-block pre{margin:0;border-radius:0 0 var(--radius-md) var(--radius-md)}

/* Streaming cursor */
.cursor .md::after{content:"\\25cd";color:var(--accent);animation:blink 800ms infinite;font-weight:300}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}

/* Citations */
#cit{flex-shrink:0;padding:0 16px}
.citations-panel{background:var(--bg-citation);border:1px solid var(--border-subtle);border-radius:var(--radius-lg);margin:8px 0;overflow:hidden}
.citations-panel__header{display:flex;align-items:center;justify-content:space-between;padding:10px 14px;cursor:pointer;user-select:none;font-size:13px;font-weight:600;color:var(--text-secondary);transition:background var(--duration-fast) var(--ease-default)}
.citations-panel__header:hover{background:var(--border-subtle)}
.citations-panel__header .toggle-icon{font-size:10px;color:var(--text-tertiary);transition:transform var(--duration-normal) var(--ease-default)}
.citations-panel.collapsed .citations-panel__header .toggle-icon{transform:rotate(-90deg)}
.citations-panel__list{padding:0 10px 10px;display:flex;flex-direction:column;gap:6px;max-height:300px;overflow-y:auto;transition:max-height var(--duration-normal) var(--ease-default),opacity var(--duration-normal) var(--ease-default),padding var(--duration-normal) var(--ease-default)}
.citations-panel.collapsed .citations-panel__list{max-height:0;opacity:0;padding:0 10px;overflow:hidden}
.citation-card{display:flex;align-items:flex-start;gap:8px;padding:8px 10px;background:var(--citation-bg);border:1px solid var(--citation-border);border-radius:var(--radius-md);position:relative;overflow:hidden;transition:border-color var(--duration-fast) var(--ease-default),box-shadow var(--duration-fast) var(--ease-default)}
.citation-card:hover{border-color:var(--accent);box-shadow:var(--shadow-xs)}
.citation-card__relevance{position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--accent);border-radius:3px 0 0 3px}
.citation-card__number{width:22px;height:22px;border-radius:var(--radius-full);background:var(--citation-marker-bg);color:var(--citation-marker);font-size:11px;font-weight:600;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:1px}
.citation-card__content{flex:1;min-width:0}
.citation-card__title{font-size:12px;font-weight:600;color:var(--text-primary);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:2px}
.citation-card__snippet{font-size:11px;color:var(--citation-text);line-height:1.4;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.citation-card__meta{margin-top:4px}
.citation-card__meta-item{font-size:10px;color:var(--text-tertiary)}

/* Filter */
#flt{flex-shrink:0;display:flex;align-items:center;justify-content:center;gap:6px;padding:4px 16px}
#flt label{font-size:11px;color:var(--text-tertiary)}
#flt select{border:1px solid var(--border-default);border-radius:var(--radius-full);background:var(--bg-primary);padding:3px 12px;font-size:11px;font-family:var(--font-sans);color:var(--text-primary);outline:none;transition:border-color var(--duration-fast) var(--ease-default)}
#flt select:focus{border-color:var(--accent)}

/* Composer */
#ca{flex-shrink:0;padding:8px 16px 16px}
#composer{background:var(--bg-primary);border-radius:var(--radius-2xl);box-shadow:var(--shadow-composer);padding:10px 14px 10px 18px;display:flex;flex-direction:column;transition:box-shadow var(--duration-normal) var(--ease-default)}
#composer:focus-within{box-shadow:var(--shadow-composer-focus)}
#composer textarea{border:none;background:transparent;outline:none;resize:none;font-size:16px;font-family:var(--font-sans);line-height:1.5;color:var(--text-primary);width:100%;min-height:24px;max-height:200px;padding:4px 0}
#composer textarea::placeholder{color:var(--text-disabled)}
#ca-row{display:flex;align-items:center;justify-content:flex-end;margin-top:2px;gap:6px}
#snd{width:34px;height:34px;border-radius:var(--radius-full);border:none;background:var(--bg-tertiary);color:var(--text-tertiary);cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:16px;transition:background var(--duration-fast) var(--ease-default),color var(--duration-fast) var(--ease-default),transform 50ms var(--ease-default)}
#snd.enabled{background:var(--accent);color:var(--accent-fg)}
#snd.enabled:hover{background:var(--accent-hover)}
#snd:active{transform:scale(0.98)}
#snd:focus-visible{outline:2px solid var(--accent);outline-offset:2px}

/* Scrollbar */
::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--scrollbar-thumb);border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:var(--scrollbar-thumb-hover)}
</style>
</head>
<body>
<div id="app">
<div id="topbar">
<div class="topbar-left">
<div class="topbar-logo">ODW.ai Vault<span>The brain</span></div>
</div>
<div class="topbar-right">
<div class="topbar-status">$OLLAMA_STATUS</div>
<button id="theme-toggle" title="Toggle theme"></button>
</div>
</div>

<div id="main">
<div id="hero">
<h1>$GREETING</h1>
<p class="sub">How can I help you today?</p>
<div class="hero-pill">\U0001f4da <span id="hero-folder-count">All folders ready</span></div>
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
<button id="snd" title="Send">&#x2191;</button>
</div>
</div>
</div>
</div>

<script>
(function(){
  var _S = {H:[], streaming:false, abortFlag:false};

  /* ── Theme ── */
  var _themeMode = localStorage.getItem('vault-theme') || 'system';
  var _themeIcons = {light:'\\u2600\\ufe0f', dark:'\\U0001f319', system:'\\U0001f504'};

  function _applyTheme(){
    var d = _themeMode;
    if(d === 'system') d = window.matchMedia('(prefers-color-scheme:dark)').matches ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', d);
    document.documentElement.setAttribute('data-theme-mode', _themeMode);
    var btn = document.getElementById('theme-toggle');
    if(btn) btn.textContent = _themeIcons[_themeMode] || '\\U0001f504';
  }
  function _cycleTheme(){
    var order = ['light','dark','system'];
    var idx = order.indexOf(_themeMode);
    _themeMode = order[(idx + 1) % 3];
    localStorage.setItem('vault-theme', _themeMode);
    document.documentElement.classList.add('theme-transition');
    _applyTheme();
    setTimeout(function(){ document.documentElement.classList.remove('theme-transition'); }, 350);
  }
  window.matchMedia('(prefers-color-scheme:dark)').addEventListener('change', function(){
    if(_themeMode === 'system') _applyTheme();
  });
  _applyTheme();

  /* ── Send button state ── */
  function _updateSend(){
    var inp = document.getElementById('inp');
    var snd = document.getElementById('snd');
    if(!inp || !snd) return;
    var v = inp.value.trim();
    if(v.length > 0 && !_S.streaming){
      snd.classList.add('enabled');
      snd.removeAttribute('disabled');
    } else {
      snd.classList.remove('enabled');
      if(!_S.streaming) snd.setAttribute('disabled', '');
    }
  }

  /* ── Textarea auto-resize ── */
  function _autoResize(el){
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 200) + 'px';
  }

  /* ── Event handlers ── */
  var inp = document.getElementById('inp');
  var snd = document.getElementById('snd');
  if(inp){
    inp.addEventListener('input', function(){ _autoResize(this); _updateSend(); });
    inp.addEventListener('keydown', function(e){
      if(e.key === 'Enter' && !e.shiftKey){ e.preventDefault(); _send(); }
    });
  }
  if(snd){
    snd.addEventListener('click', function(){
      if(_S.streaming){ _S.abortFlag = true; return; }
      _send();
    });
  }
  var ttBtn = document.getElementById('theme-toggle');
  if(ttBtn) ttBtn.addEventListener('click', _cycleTheme);

  /* ── Chip clicks ── */
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

  /* ── Send message ── */
  function _send(){
    if(!inp) return;
    var ff = document.getElementById('ff');
    var hero = document.getElementById('hero');
    var chipsEl = document.getElementById('chips');
    var t = inp.value.trim();
    if(!t || _S.streaming) return;

    hero.classList.add('hidden');
    chipsEl.style.display = 'none';
    _addMsg('user', t);
    _S.H.push({role:'user', content:[{text:t, type:'text'}]});
    inp.value = ''; inp.style.height = 'auto';
    _updateSend();

    var el = _addMsg('assistant', 'Thinking...', true);
    _S.streaming = true; _S.abortFlag = false;
    snd.innerHTML = '\\u25a0';
    snd.classList.add('enabled');
    snd.removeAttribute('disabled');

    fetch('/gradio_api/call/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({data:[t, _S.H, ff.value]})
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
                  if(mdEl) mdEl.textContent = 'Error: ' + d.error;
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
            if(mdEl2) mdEl2.textContent = 'Error: ' + err.message;
          }
          _done(el);
        });
      })();
    }).catch(function(err){
      if(!_S.abortFlag){
        var mdEl3 = el.querySelector('.md');
        if(mdEl3) mdEl3.textContent = 'Error: ' + err.message;
      }
      _done(el);
    });
  }

  function _done(el){
    _S.streaming = false;
    snd.innerHTML = '\\u2191';
    snd.classList.remove('enabled');
    el.classList.remove('cursor');
    _updateSend();
  }

  function _addMsg(role, text, stream){
    var d = document.createElement('div');
    d.className = 'msg ' + role;
    if(role === 'assistant'){
      var m = document.createElement('div');
      m.className = 'md';
      m.innerHTML = _md(text);
      d.appendChild(m);
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

  /* ── Markdown renderer ── */
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

  /* ── Chips ── */
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
