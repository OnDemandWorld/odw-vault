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
    items = []
    for c in citations:
        page_str = f", page {c['page_start']}" if c.get("page_start") else ""
        items.append(
            f"<li><b>[{c['citation_number']}]</b> "
            f"<code>{c['rel_path']}</code>{page_str}<br>"
            f"<span style='color:#666;font-size:12px'>{c['snippet']}</span>"
            f"</li>"
        )
    return (
        f"<div style='margin-top:10px;padding:8px 14px;"
        f"background:#F5F5F5;border-radius:8px;font-size:12px;'>"
        f"<b style='color:#5D5D5D'>Sources:</b><ul style='margin:4px 0 0 16px;padding:0;'>"
        f"{''.join(items)}</ul></div>"
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
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
--bg:#FAFAFA;--bg-w:#FFFFFF;--bg-u:#EEEEEE;
--t:#0D0D0D;--ts:#5D5D5D;--tm:#A0A0A0;
--b:rgba(0,0,0,0.08);
--sh:0 1px 3px rgba(0,0,0,0.06),0 4px 12px rgba(0,0,0,0.04);
--shf:0 1px 4px rgba(0,0,0,0.1),0 4px 16px rgba(0,0,0,0.06);
--r:20px;--mw:768px;
--f:"Inter",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
}
html,body{height:100dvh;overflow:hidden;background:var(--bg);font-family:var(--f);color:var(--t);font-size:15px;line-height:1.6;-webkit-font-smoothing:antialiased}

#app{display:flex;flex-direction:column;height:100dvh;max-width:var(--mw);margin:0 auto}

/* Top bar */
#topbar{display:flex;align-items:center;justify-content:space-between;padding:8px 16px;min-height:40px;flex-shrink:0;border-bottom:1px solid var(--b)}
#topbar .logo{font-size:16px;font-weight:600;letter-spacing:-.01em;color:var(--t)}
#topbar .logo span{color:var(--tm);font-weight:400;margin-left:6px;font-size:11px}
#topbar .status{font-size:11px;color:var(--tm)}

/* Main */
#main{flex:1 1 0;min-height:0;display:flex;flex-direction:column;overflow:hidden;position:relative}

/* Hero */
#hero{display:flex;flex-direction:column;align-items:center;justify-content:center;flex:1 1 0;text-align:center;padding:0 16px;transition:opacity .25s ease,transform .25s ease}
#hero.hidden{display:none!important}
#hero h1{font-size:clamp(28px,5vw,42px);font-weight:400;letter-spacing:-.02em;line-height:1.2;color:var(--t);margin-bottom:6px}
#hero .sub{font-size:16px;color:var(--ts);font-weight:400}
#hero .tag{font-size:12px;color:var(--tm);margin-top:10px}

/* Messages */
#msgs{flex:1 1 0;min-height:0;overflow-y:auto;padding:12px 16px;display:none;flex-direction:column}
#msgs.active{display:flex}
.msg{max-width:75%;padding:10px 16px;margin:4px 0;font-size:15px;line-height:1.6;word-wrap:break-word}
.msg.user{align-self:flex-end;background:var(--bg-u);border-radius:18px;margin-left:auto;white-space:pre-wrap}
.msg.assistant{align-self:flex-start;background:transparent;padding:6px 0;max-width:100%}
.md{line-height:1.7}
.md code{background:#F0F0F0;padding:2px 6px;border-radius:4px;font-size:13px;font-family:"SF Mono","Fira Code",monospace}
.md pre{background:#F5F5F5;padding:12px;border-radius:8px;overflow-x:auto;margin:8px 0}
.md pre code{background:none;padding:0}
.md p{margin:4px 0}
.md ul,.md ol{margin:4px 0;padding-left:20px}
.md h1,.md h2,.md h3{margin:12px 0 4px}
.cursor::after{content:"|";animation:blink .8s infinite;color:#A0A0A0;font-weight:300}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}

/* Citations */
#cit{flex-shrink:0;padding:0 16px}

/* Chips */
#chips{flex-shrink:0;padding:0 16px 8px;display:grid;grid-template-columns:repeat(2,1fr);gap:8px}
@media(max-width:600px){#chips{grid-template-columns:1fr}}
.chip{display:flex;align-items:center;gap:8px;padding:10px 14px;border:1px solid var(--b);border-radius:12px;background:transparent;cursor:pointer;font-size:13px;color:var(--ts);text-align:left;font-family:var(--f);line-height:1.35;transition:all .15s ease}
.chip:hover{background:#F2F2F2;border-color:rgba(0,0,0,0.12)}
.chip .i{font-size:15px;flex-shrink:0}
#chip-rf{display:block;margin:6px auto 0;background:none;border:none;cursor:pointer;font-size:12px;color:var(--tm);padding:4px;font-family:var(--f)}
#chip-rf:hover{color:var(--ts)}

/* Filter */
#flt{flex-shrink:0;display:flex;align-items:center;justify-content:center;gap:6px;padding:0 16px 4px}
#flt label{font-size:11px;color:var(--tm)}
#flt select{border:1px solid var(--b);border-radius:999px;background:#fff;padding:3px 12px;font-size:11px;font-family:var(--f);color:var(--t);outline:none}
#flt select:focus{border-color:rgba(0,0,0,0.2)}

/* Composer */
#ca{flex-shrink:0;padding:8px 16px 16px}
#composer{background:var(--bg-w);border-radius:var(--r);box-shadow:var(--sh);padding:8px 12px 8px 16px;display:flex;flex-direction:column;transition:box-shadow .15s ease}
#composer:focus-within{box-shadow:var(--shf)}
#composer textarea{border:none;background:transparent;outline:none;resize:none;font-size:15px;font-family:var(--f);line-height:1.5;color:var(--t);width:100%;min-height:24px;max-height:200px;padding:4px 0}
#composer textarea::placeholder{color:var(--tm)}
#ca-row{display:flex;align-items:center;justify-content:space-between;margin-top:2px}
#ca-row .lb{background:none;border:none;cursor:pointer;font-size:18px;color:#909090;padding:4px;border-radius:6px}
#ca-row .lb:hover{background:#F5F5F5}
#snd{width:34px;height:34px;border-radius:50%;border:none;background:var(--t);color:var(--bg-w);cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:16px;transition:background .15s ease;margin-left:auto}
#snd:disabled{background:#E0E0E0;cursor:default}
#snd:not(:disabled):hover{background:#333}

::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:#D8D8D8;border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:#B8B8B8}
</style>
</head>
<body>
<div id="app" onclick="_handleClick(event)">
<div id="topbar">
<div class="logo">ODW.ai Vault<span>The brain</span></div>
<div class="status">$OLLAMA_STATUS</div>
</div>

<div id="main">
<div id="hero">
<h1>$GREETING</h1>
<p class="sub">How can I help you today?</p>
<p class="tag">Instant, grounded answers from your organisation's entire knowledge base</p>
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
<textarea id="inp" placeholder="Ask anything about your knowledge base..." rows="1" autofocus oninput="_handleInput(this)" onkeydown="_handleKeydown(event)"></textarea>
<div id="ca-row">
<button class="lb" title="Attach">&#128206;</button>
<button id="snd" disabled title="Send">&#8593;</button>
</div>
</div>
</div>
</div>

<script>
(function(){
  var _S = {H:[], streaming:false, abortFlag:false};

  function _updateSend(){
    var inp = document.getElementById('inp');
    var snd = document.getElementById('snd');
    if(!inp || !snd) return;
    var v = inp.value.trim();
    if(v.length > 0){
      snd.removeAttribute('disabled');
      snd.style.background = '#0D0D0D';
      snd.style.cursor = 'pointer';
    } else {
      snd.setAttribute('disabled', '');
      snd.style.background = '#E0E0E0';
      snd.style.cursor = 'default';
    }
  }

  window._handleInput = function(el){
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 200) + 'px';
    _updateSend();
  };

  window._handleKeydown = function(e){
    if(e.key === 'Enter' && !e.shiftKey){
      e.preventDefault();
      _send();
    }
  };

  window._handleClick = function(e){
    var btn = e.target.closest('#snd');
    if(btn){
      if(_S.streaming){ _S.abortFlag = true; return; }
      _send();
      return;
    }
    var chip = e.target.closest('.chip');
    if(chip){
      var text = chip.getAttribute('data-chip-text');
      if(text){
        var inp = document.getElementById('inp');
        if(inp){ inp.value = text; _updateSend(); inp.focus(); }
      }
      return;
    }
    var rf = e.target.closest('#chip-rf');
    if(rf){
      var a = window._PC;
      var o = Math.floor(Math.random() * a.length);
      var s = a.slice(o, o + 4);
      if(s.length < 4) s = s.concat(a.slice(0, 4 - s.length));
      _renderChips(s);
    }
  };

  function _send(){
    var inp = document.getElementById('inp');
    var snd = document.getElementById('snd');
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
    snd.innerHTML = '■';
    snd.style.background = '#0D0D0D';
    snd.removeAttribute('disabled');

    // Gradio 6: POST /call/chat -> event_id, then GET /call/chat/{id} for SSE
    console.log('[chat] sending query:', t.substring(0, 80));
    fetch('/gradio_api/call/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({data:[t, _S.H, ff.value]})
    }).then(function(r){
      if(!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    }).then(function(resp){
      var eventId = resp.event_id;
      console.log('[chat] got event_id:', eventId);
      return fetch('/gradio_api/call/chat/' + eventId);
    }).then(function(r){
      if(!r.ok) throw new Error('HTTP ' + r.status);
      console.log('[chat] streaming response...');
      var reader = r.body.getReader();
      var buf = '';
      var tokenCount = 0;
      (function pump(){
        reader.read().then(function(res){
          if(_S.abortFlag){ reader.cancel(); _done(snd, el); return; }
          if(res.done){ console.log('[chat] done, tokens:', tokenCount); _done(snd, el); return; }
          buf += new TextDecoder().decode(res.value);
          var lines = buf.split('\\n');
          buf = lines.pop() || '';
          for(var i = 0; i < lines.length; i++){
            var line = lines[i];
            if(line.startsWith('data:')){
              try {
                var d = JSON.parse(line.slice(5));
                // Check for Gradio error event
                if(d.error) {
                  console.error('[chat] error:', d.error);
                  var mdEl = el.querySelector('.md');
                  if(mdEl) mdEl.textContent = 'Error: ' + d.error;
                  _done(snd, el);
                  return;
                }
                if(d[0] && d[0].length){
                  tokenCount++;
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
          console.error('[chat] stream error:', err.message);
          if(!_S.abortFlag){
            var mdEl2 = el.querySelector('.md');
            if(mdEl2) mdEl2.textContent = 'Error: ' + err.message;
          }
          _done(snd, el);
        });
      })();
    }).catch(function(err){
      console.error('[chat] request error:', err.message);
      if(!_S.abortFlag){
        var mdEl3 = el.querySelector('.md');
        if(mdEl3) mdEl3.textContent = 'Error: ' + err.message;
      }
      _done(snd, el);
    });
  }

  function _done(snd, el){
    _S.streaming = false;
    snd.innerHTML = '↑';
    snd.style.background = '#0D0D0D';
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

  function _md(t){
    if(!t) return '';
    return t
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/```([\\s\\S]*?)```/g, '<pre><code>$1</code></pre>')
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\\*\\*([^*]+)\\*\\*/g, '<strong>$1</strong>')
      .replace(/\\n/g, '<br>');
  }

  window._PC = $CHIPS_JSON;

  function _renderChips(s){
    var chipsEl = document.getElementById('chips');
    var rf = document.getElementById('chip-rf');
    if(!chipsEl) return;
    var h = '';
    for(var i = 0; i < s.length; i++){
      var esc = s[i].text.replace(/'/g, "\\'");
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
