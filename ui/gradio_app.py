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

import contextlib
import html
import json
import logging
import sqlite3
import threading
import time
import uuid
from pathlib import Path

import gradio as gr
import ollama
from fastapi import Request  # module-level on purpose: FastAPI resolves the string

# annotation `request: Request` (PEP 563) of routes defined inside launch_ui()
# from module globals — a local-only import makes every such route 422.
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


def _check_db_write_access():
    """Verify the database file is writable at startup; warn early if not."""
    import os
    db_path = _db_path
    if not db_path or not os.path.exists(db_path):
        return
    try:
        # Attempt a harmless write to verify access
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("CREATE TABLE IF NOT EXISTS _write_check (_id INTEGER)")
        conn.execute("DROP TABLE IF EXISTS _write_check")
        conn.close()
    except sqlite3.OperationalError as exc:
        if "readonly" in str(exc).lower():
            print(
                f"  \u26a0\ufe0f  WARNING: Database is READ-ONLY ({db_path}).\n"
                f"      Chat will fail. Fix: ensure the file and its directory are writable.\n"
                f"      On macOS, try: xattr -c {db_path} {db_path}-wal {db_path}-shm\n"
                f"      Or check file permissions: chmod u+w {db_path}"
            )


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


# ---------------------------------------------------------------------------
# Multi-provider LLM registry — user-configurable models via Settings UI
#
# Provider entries live in llm_providers.json (project root, gitignored —
# holds API keys). Three wire protocols are supported:
#   "ollama"    — native Ollama /api/chat (NDJSON streaming)
#   "openai"    — OpenAI-compatible /chat/completions (covers OpenAI, DeepSeek,
#                  Qwen/DashScope, Moonshot, GLM, Gemini-compat, Groq, Together,
#                  Mistral, xAI, vLLM, LM Studio, Ollama Cloud, ...)
#   "anthropic" — native Claude /v1/messages API
# All calls use httpx directly, so no extra SDK dependencies are needed.
# ---------------------------------------------------------------------------

_PROVIDERS_FILE = Path(__file__).resolve().parent.parent / "llm_providers.json"

_PROVIDER_PROTOCOLS = [
    {"value": "openai", "label": "OpenAI-compatible"},
    {"value": "ollama", "label": "Ollama (local)"},
    {"value": "anthropic", "label": "Anthropic (Claude)"},
]

_PROVIDER_PRESETS = [
    {"key": "openai", "label": "OpenAI", "protocol": "openai",
     "base_url": "https://api.openai.com/v1", "model": "gpt-4o"},
    {"key": "deepseek", "label": "DeepSeek", "protocol": "openai",
     "base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
    {"key": "qwen", "label": "Qwen · DashScope", "protocol": "openai",
     "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-max"},
    {"key": "moonshot", "label": "Moonshot Kimi", "protocol": "openai",
     "base_url": "https://api.moonshot.cn/v1", "model": "moonshot-v1-128k"},
    {"key": "zhipu", "label": "Zhipu GLM", "protocol": "openai",
     "base_url": "https://open.bigmodel.cn/api/paas/v4", "model": "glm-4-plus"},
    {"key": "gemini", "label": "Google Gemini", "protocol": "openai",
     "base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "model": "gemini-2.5-flash"},
    {"key": "groq", "label": "Groq", "protocol": "openai",
     "base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile"},
    {"key": "together", "label": "Together AI", "protocol": "openai",
     "base_url": "https://api.together.xyz/v1", "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct"},
    {"key": "mistral", "label": "Mistral", "protocol": "openai",
     "base_url": "https://api.mistral.ai/v1", "model": "mistral-large-latest"},
    {"key": "xai", "label": "xAI Grok", "protocol": "openai",
     "base_url": "https://api.x.ai/v1", "model": "grok-3"},
    {"key": "anthropic", "label": "Anthropic Claude", "protocol": "anthropic",
     "base_url": "https://api.anthropic.com", "model": "claude-sonnet-4-20250514"},
    {"key": "ollama", "label": "Ollama (local)", "protocol": "ollama",
     "base_url": "http://localhost:11434", "model": "gemma4:latest"},
    {"key": "lmstudio", "label": "LM Studio (local)", "protocol": "openai",
     "base_url": "http://localhost:1234/v1", "model": ""},
    {"key": "custom", "label": "Custom / Self-hosted", "protocol": "openai",
     "base_url": "", "model": ""},
]


def _load_provider_registry() -> dict:
    """Load (or seed) the provider registry from llm_providers.json."""
    if _PROVIDERS_FILE.exists():
        try:
            data = json.loads(_PROVIDERS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("providers"), list):
                return data
        except Exception:
            logger.warning("Failed to parse llm_providers.json; reseeding", exc_info=True)
    return _seed_default_registry()


def _seed_default_registry() -> dict:
    """Create the default registry from config.toml generation settings."""
    entry = {
        "id": str(uuid.uuid4())[:8],
        "name": "Local Ollama",
        "protocol": "ollama",
        "base_url": _ollama_host,
        "api_key": "",
        "model": "gemma4:latest",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if _cfg is not None:
        gen = getattr(_cfg.models, "generation", None)
        ep = getattr(gen, "endpoint", None)
        if ep and getattr(ep, "host", ""):
            entry["base_url"] = ep.host
            entry["api_key"] = getattr(ep, "api_key", "") or ""
            if "ollama.com" in ep.host:
                entry["name"] = "Ollama Cloud"
        if gen and getattr(gen, "name", ""):
            entry["model"] = gen.name
    reg = {"version": 1, "active_id": entry["id"], "providers": [entry]}
    _save_provider_registry(reg)
    return reg


def _save_provider_registry(reg: dict) -> None:
    try:
        _PROVIDERS_FILE.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    except Exception:
        logger.error("Failed to write llm_providers.json", exc_info=True)


def _mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "\u2022\u2022\u2022"
    return key[:4] + "\u2022\u2022\u2022" + key[-2:]


def _get_active_provider() -> dict | None:
    """Resolve the currently active provider entry (or None on misconfig)."""
    reg = _load_provider_registry()
    active_id = reg.get("active_id")
    for p in reg.get("providers", []):
        if p.get("id") == active_id:
            return p
    providers = reg.get("providers", [])
    return providers[0] if providers else None


def _provider_is_local(base_url: str) -> bool:
    host = (base_url or "").lower()
    return "://127.0.0.1" in host or "://localhost" in host or host.startswith("http://192.168.")


def _provider_request_args(entry: dict) -> tuple:
    """Return (base_url, headers) for registry/health calls per protocol."""
    protocol = entry.get("protocol", "ollama")
    base = (entry.get("base_url") or "").rstrip("/")
    api_key = entry.get("api_key") or ""
    headers = {"Content-Type": "application/json"}
    if protocol == "anthropic":
        if api_key:
            headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
    elif api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return base, headers


def _provider_chat_stream(
    entry: dict,
    messages: list[dict],
    *,
    temperature: float = 0.5,
    top_p: float = 0.95,
    top_k: int = 64,
):
    """Stream chat tokens from any supported protocol. Yields token strings."""
    import httpx

    protocol = entry.get("protocol", "ollama")
    base, headers = _provider_request_args(entry)
    model = entry["model"]

    if protocol == "ollama":
        url = f"{base}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature, "top_p": top_p, "top_k": top_k},
        }
    elif protocol == "anthropic":
        url = f"{base}/v1/messages"
        system_text = "\n".join(m["content"] for m in messages if m["role"] == "system")
        chat_msgs = [m for m in messages if m["role"] != "system"]
        payload = {
            "model": model,
            "messages": chat_msgs,
            "stream": True,
            "max_tokens": 8192,
            "temperature": temperature,
            "top_p": top_p,
        }
        if system_text:
            payload["system"] = system_text
    else:  # openai-compatible
        url = f"{base}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "top_p": top_p,
        }

    timeout = httpx.Timeout(connect=15.0, read=600.0, write=60.0, pool=15.0)
    with httpx.Client(trust_env=not _provider_is_local(base), timeout=timeout) as client:
        with client.stream("POST", url, headers=headers, json=payload) as resp:
            if resp.status_code >= 400:
                body = resp.read().decode("utf-8", "replace")[:300]
                raise RuntimeError(f"Provider HTTP {resp.status_code}: {body}")
            for line in resp.iter_lines():
                if not line:
                    continue
                if protocol == "ollama":
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    tok = (obj.get("message") or {}).get("content") or ""
                    if tok:
                        yield tok
                else:
                    if not line.startswith("data:"):
                        continue  # skip "event:" lines etc.
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue
                    if protocol == "anthropic":
                        if obj.get("type") == "content_block_delta":
                            tok = (obj.get("delta") or {}).get("text") or ""
                            if tok:
                                yield tok
                    else:
                        choices = obj.get("choices") or [{}]
                        tok = ((choices[0].get("delta") or {}).get("content")) or ""
                        if tok:
                            yield tok


def _provider_test(entry: dict) -> tuple[bool, str]:
    """Health-check a provider entry. Returns (ok, human-readable detail)."""
    import httpx

    protocol = entry.get("protocol", "ollama")
    base, headers = _provider_request_args(entry)
    if not base:
        return False, "Base URL is empty"
    url = {
        "ollama": f"{base}/api/tags",
        "anthropic": f"{base}/v1/models?limit=20",
    }.get(protocol, f"{base}/models")
    started = time.time()
    try:
        with httpx.Client(trust_env=not _provider_is_local(base), timeout=15.0) as client:
            r = client.get(url, headers=headers)
        latency = int((time.time() - started) * 1000)
        if r.status_code >= 400:
            return False, f"HTTP {r.status_code}: {r.text[:200]}"
        models = _parse_model_list(protocol, r.text)
        n = len(models)
        hint = f", {n} models visible" if n else ""
        return True, f"Connected in {latency} ms{hint}"
    except Exception as exc:
        return False, str(exc)[:200]


def _parse_model_list(protocol: str, body: str) -> list[str]:
    try:
        obj = json.loads(body)
    except Exception:
        return []
    if protocol == "ollama":
        return [m.get("name", "") for m in obj.get("models", []) if m.get("name")]
    data = obj.get("data", [])
    return [m.get("id", "") for m in data if m.get("id")]


def _provider_list_models(entry: dict) -> list[str]:
    """List model IDs advertised by a provider endpoint."""
    import httpx

    protocol = entry.get("protocol", "ollama")
    base, headers = _provider_request_args(entry)
    if not base:
        return []
    url = {
        "ollama": f"{base}/api/tags",
        "anthropic": f"{base}/v1/models?limit=100",
    }.get(protocol, f"{base}/models")
    try:
        with httpx.Client(trust_env=not _provider_is_local(base), timeout=15.0) as client:
            r = client.get(url, headers=headers)
        if r.status_code >= 400:
            return []
        return _parse_model_list(protocol, r.text)
    except Exception:
        return []


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

    provider = _get_active_provider()
    if provider is None:
        raise RuntimeError("No model provider configured. Open Settings to add one.")

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]

    full_text = ""
    for token in _provider_chat_stream(
        provider,
        messages,
        temperature=_cfg.models.generation.temperature,
        top_p=_cfg.models.generation.top_p,
        top_k=_cfg.models.generation.top_k,
    ):
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
        # Snippets are raw corpus text and rel_path comes from the filesystem —
        # escape both so document content can't inject HTML into the page.
        snippet = html.escape(str(c.get("snippet", ""))[:150], quote=False)
        if len(c.get("snippet", "")) > 150:
            snippet += "..."
        rel_path = html.escape(str(c.get("rel_path", "")), quote=True)
        cards.append(
            f'<div class="citation-card" data-cite="{c["citation_number"]}">'
            f'<div class="citation-card__relevance"></div>'
            f'<div class="citation-card__number">{c["citation_number"]}</div>'
            f'<div class="citation-card__content">'
            f'<div class="citation-card__title">{rel_path}</div>'
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
        err_msg = str(exc)
        if "readonly" in err_msg.lower():
            user_msg = "Database is read-only. Please restart the server with write access to corpus.db (check file permissions or macOS quarantine attributes)."
        else:
            user_msg = f"Retrieval failed: {exc}"
        history.append({"role": "assistant", "content": user_msg})
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
    # Ensure localhost bypasses any environment HTTP proxy. Otherwise httpx
    # (used below for the reverse proxy and by Gradio's own startup probe)
    # would route 127.0.0.1 traffic through e.g. HTTP_PROXY and get a 502.
    import os as _os
    import threading
    import time

    import httpx
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, StreamingResponse
    from starlette.responses import Response
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
    _check_db_write_access()

    folders = _get_folders()
    folder_choices = ["All folders", *folders]

    greeting = _get_greeting()
    ollama_ok = _check_ollama()
    ollama_status = '\U0001f7e2 Ollama OK' if ollama_ok else '\U0001f534 Ollama down'

    chips_json = str([{"icon": c["icon"], "text": c["text"]} for c in _PROMPT_CHIPS[:4]]).replace("'", '"')
    # Folder names come from the filesystem — escape so a '<' or '"' in a
    # directory name can't inject HTML into the page shell.
    folder_options = "".join(
        f'<option value="{html.escape(f, quote=True)}">{html.escape(f, quote=False)}</option>'
        for f in folder_choices
    )

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
        # no-store: the UI HTML embeds app JS; a stale cached page after an
        # upgrade keeps talking to old endpoints and breaks in confusing ways.
        return HTMLResponse(content=full_html, headers={"Cache-Control": "no-store"})

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

    # ---- Settings: multi-provider model management -------------------------
    from starlette.concurrency import run_in_threadpool

    def _public_provider(p: dict) -> dict:
        """Provider entry safe to send to the browser (keys masked)."""
        return {
            "id": p.get("id"),
            "name": p.get("name", ""),
            "protocol": p.get("protocol", "ollama"),
            "base_url": p.get("base_url", ""),
            "model": p.get("model", ""),
            "has_key": bool(p.get("api_key")),
            "key_hint": _mask_key(p.get("api_key") or ""),
            "created_at": p.get("created_at", ""),
        }

    def _settings_payload() -> dict:
        reg = _load_provider_registry()
        cfg_summary = {}
        if _cfg is not None:
            cfg_summary = {
                "corpus_root": getattr(_cfg.paths, "corpus_root", ""),
                "chroma_root": getattr(_cfg.paths, "chroma_root", ""),
                "ollama_host": getattr(_cfg.ollama, "host", ""),
                "embedding_model": getattr(_cfg.models.embedding, "name", ""),
                "generation_model": getattr(_cfg.models.generation, "name", ""),
                "temperature": getattr(_cfg.models.generation, "temperature", ""),
                "require_citations": getattr(
                    getattr(_cfg, "generation_runtime", None), "require_citations", True
                ),
            }
        return {
            "providers": [_public_provider(p) for p in reg.get("providers", [])],
            "active_id": reg.get("active_id"),
            "protocols": _PROVIDER_PROTOCOLS,
            "presets": _PROVIDER_PRESETS,
            "config": cfg_summary,
        }

    @proxy_app.get("/api/settings")
    async def get_settings():
        return _settings_payload()

    @proxy_app.post("/api/settings/providers")
    async def upsert_provider(request: Request):
        try:
            payload = await request.json()
        except Exception:
            return {"ok": False, "error": "invalid JSON body"}
        name = str(payload.get("name") or "").strip()
        protocol = str(payload.get("protocol") or "").strip()
        base_url = str(payload.get("base_url") or "").strip().rstrip("/")
        model = str(payload.get("model") or "").strip()
        api_key = str(payload.get("api_key") or "")
        if not name or not model:
            return {"ok": False, "error": "Name and model are required"}
        if protocol not in {p["value"] for p in _PROVIDER_PROTOCOLS}:
            return {"ok": False, "error": f"Unsupported protocol: {protocol!r}"}
        if not base_url:
            return {"ok": False, "error": "Base URL is required"}

        reg = _load_provider_registry()
        pid = str(payload.get("id") or "").strip()
        existing = next((p for p in reg["providers"] if p.get("id") == pid), None)
        if existing:
            existing.update({
                "name": name, "protocol": protocol, "base_url": base_url,
                "model": model,
            })
            # Empty key on edit = keep the stored one
            if api_key:
                existing["api_key"] = api_key
        else:
            reg["providers"].append({
                "id": str(uuid.uuid4())[:8],
                "name": name, "protocol": protocol, "base_url": base_url,
                "api_key": api_key, "model": model,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
        _save_provider_registry(reg)
        return {"ok": True, "settings": _settings_payload()}

    @proxy_app.delete("/api/settings/providers/{pid}")
    async def delete_provider(pid: str):
        reg = _load_provider_registry()
        before = len(reg["providers"])
        reg["providers"] = [p for p in reg["providers"] if p.get("id") != pid]
        if len(reg["providers"]) == before:
            return {"ok": False, "error": "not found"}
        if reg.get("active_id") == pid:
            reg["active_id"] = reg["providers"][0]["id"] if reg["providers"] else None
        _save_provider_registry(reg)
        return {"ok": True, "settings": _settings_payload()}

    @proxy_app.post("/api/settings/providers/{pid}/activate")
    async def activate_provider(pid: str):
        reg = _load_provider_registry()
        if not any(p.get("id") == pid for p in reg["providers"]):
            return {"ok": False, "error": "not found"}
        reg["active_id"] = pid
        _save_provider_registry(reg)
        return {"ok": True, "settings": _settings_payload()}

    @proxy_app.post("/api/settings/providers/test")
    async def test_provider(request: Request):
        try:
            payload = await request.json()
        except Exception:
            return {"ok": False, "error": "invalid JSON body"}
        entry = {
            "protocol": str(payload.get("protocol") or "ollama"),
            "base_url": str(payload.get("base_url") or "").rstrip("/"),
            "api_key": str(payload.get("api_key") or ""),
            "model": str(payload.get("model") or ""),
        }
        # Blank key with an existing id = reuse the stored key
        pid = str(payload.get("id") or "").strip()
        if pid and not entry["api_key"]:
            reg = _load_provider_registry()
            stored = next((p for p in reg["providers"] if p.get("id") == pid), None)
            if stored:
                entry["api_key"] = stored.get("api_key") or ""
        ok, detail = await run_in_threadpool(_provider_test, entry)
        return {"ok": ok, "detail": detail}

    @proxy_app.post("/api/settings/providers/models")
    async def fetch_provider_models(request: Request):
        try:
            payload = await request.json()
        except Exception:
            return {"ok": False, "error": "invalid JSON body", "models": []}
        entry = {
            "protocol": str(payload.get("protocol") or "ollama"),
            "base_url": str(payload.get("base_url") or "").rstrip("/"),
            "api_key": str(payload.get("api_key") or ""),
            "model": str(payload.get("model") or ""),
        }
        pid = str(payload.get("id") or "").strip()
        if pid and not entry["api_key"]:
            reg = _load_provider_registry()
            stored = next((p for p in reg["providers"] if p.get("id") == pid), None)
            if stored:
                entry["api_key"] = stored.get("api_key") or ""
        models = await run_in_threadpool(_provider_list_models, entry)
        return {"ok": bool(models), "models": models}

    # ---- Indexing / vectorization status -----------------------------------

    _sync_lock_ui = threading.Lock()
    _sync_status_ui: dict = {"running": False, "last_result": None, "started_at": None, "finished_at": None}

    def _get_indexing_status() -> dict:
        """Compute comprehensive indexing/vectorization status from the DB."""
        try:
            db = _get_db()

            # Total non-excluded files
            total_files = db.conn.execute(
                "SELECT COUNT(*) as c FROM file WHERE excluded = 0 AND is_dup_primary = 1"
            ).fetchone()["c"]

            # Files with embeddings (fully indexed) — takes precedence over
            # every other state, so an old failure row on a since-fixed file
            # doesn't misclassify it.
            embedded_files = db.conn.execute(
                "SELECT COUNT(DISTINCT c.file_id) as c FROM chunk c "
                "JOIN embedding_ref er ON er.chunk_id = c.id "
                "JOIN file f ON f.id = c.file_id "
                "WHERE f.excluded = 0 AND f.is_dup_primary = 1 AND er.is_current = 1"
            ).fetchone()["c"]

            # Files whose latest pipeline state is a failure — extraction,
            # embedding, or summarization can all record failures.
            failed_files = db.conn.execute(
                "SELECT COUNT(DISTINCT f.id) as c FROM file f "
                "JOIN failure fail ON fail.file_id = f.id "
                "WHERE f.excluded = 0 AND f.is_dup_primary = 1 "
                "AND fail.phase IN ('extract', 'indexer', 'embed', 'summarize') "
                "AND f.id NOT IN ("
                "  SELECT c.file_id FROM chunk c"
                "  JOIN embedding_ref er ON er.chunk_id = c.id AND er.is_current = 1"
                ")"
            ).fetchone()["c"]

            # Files with successful extraction but no embeddings yet
            extracted_files = db.conn.execute(
                "SELECT COUNT(DISTINCT f.id) as c FROM file f "
                "JOIN extraction e ON e.file_id = f.id "
                "WHERE f.excluded = 0 AND f.is_dup_primary = 1 AND e.succeeded = 1 "
                "AND f.id NOT IN ("
                "  SELECT c.file_id FROM chunk c"
                "  JOIN embedding_ref er ON er.chunk_id = c.id AND er.is_current = 1"
                ") "
                "AND f.id NOT IN ("
                "  SELECT fail.file_id FROM failure fail"
                "  WHERE fail.phase IN ('extract', 'indexer', 'embed', 'summarize')"
                ")"
            ).fetchone()["c"]

            # Total chunks
            total_chunks = db.conn.execute(
                "SELECT COUNT(*) as c FROM chunk"
            ).fetchone()["c"]

            # Embedded chunks
            embedded_chunks = db.conn.execute(
                "SELECT COUNT(DISTINCT chunk_id) as c FROM embedding_ref WHERE is_current = 1"
            ).fetchone()["c"]

            # Whatever remains is waiting to be processed
            pending_files = total_files - embedded_files - failed_files - extracted_files

            # Per-folder breakdown — "embedded" uses the same
            # embedding_ref.is_current=1 definition as the totals above.
            folder_rows = db.conn.execute(
                """SELECT fo.id, fo.rel_path, fo.name,
                          COUNT(DISTINCT f.id) AS total_files,
                          COUNT(DISTINCT CASE WHEN e.id IS NOT NULL AND e.succeeded = 1 THEN f.id END) AS extracted,
                          COUNT(DISTINCT CASE WHEN er.chunk_id IS NOT NULL THEN f.id END) AS embedded
                   FROM folder fo
                   LEFT JOIN file f ON f.folder_id = fo.id AND f.excluded = 0 AND f.is_dup_primary = 1
                   LEFT JOIN extraction e ON e.file_id = f.id AND e.succeeded = 1
                   LEFT JOIN chunk c ON c.file_id = f.id
                   LEFT JOIN embedding_ref er ON er.chunk_id = c.id AND er.is_current = 1
                   WHERE fo.excluded = 0
                   GROUP BY fo.id
                   HAVING total_files > 0
                   ORDER BY fo.rel_path"""
            ).fetchall()

            folders = []
            for r in folder_rows:
                f_total = r["total_files"]
                f_extracted = r["extracted"]
                f_embedded = r["embedded"]
                if f_total > 0:
                    status = "complete" if f_embedded >= f_total else ("partial" if f_extracted > 0 or f_embedded > 0 else "pending")
                else:
                    status = "empty"
                folders.append({
                    "id": r["id"],
                    "rel_path": r["rel_path"],
                    "name": r["name"] or r["rel_path"],
                    "total_files": f_total,
                    "extracted": f_extracted,
                    "embedded": f_embedded,
                    "status": status,
                })

            # Last UI-triggered sync (the indexer doesn't write run rows,
            # so CLI syncs are not reflected here).
            last_sync = _sync_status_ui.get("finished_at")
            last_error = None
            last_result = _sync_status_ui.get("last_result")
            if isinstance(last_result, dict) and last_result.get("error"):
                last_error = last_result["error"]

            # Overall completion percentage
            progress_pct = round((embedded_files / total_files * 100), 1) if total_files > 0 else 0

            return {
                "total_files": total_files,
                "extracted_files": extracted_files,
                "embedded_files": embedded_files,
                "failed_files": failed_files,
                "pending_files": max(0, pending_files),
                "total_chunks": total_chunks,
                "embedded_chunks": embedded_chunks,
                "progress_pct": progress_pct,
                "last_sync": last_sync,
                "last_error": last_error,
                "folders": folders,
                "sync_running": _sync_status_ui["running"],
                "sync_started_at": _sync_status_ui.get("started_at"),
            }
        except Exception as exc:
            logger.warning("Failed to get indexing status: %s", exc)
            return {
                "total_files": 0, "extracted_files": 0, "embedded_files": 0,
                "failed_files": 0, "pending_files": 0, "total_chunks": 0,
                "embedded_chunks": 0, "progress_pct": 0, "last_sync": None,
                "folders": [], "sync_running": False, "error": str(exc),
            }

    @proxy_app.get("/api/indexing/status")
    async def indexing_status():
        return await run_in_threadpool(_get_indexing_status)

    @proxy_app.post("/api/indexing/sync")
    async def trigger_indexing_sync():
        """Trigger a full incremental sync in background."""
        if _sync_status_ui["running"]:
            return {"ok": False, "error": "A sync is already in progress"}

        def _run_sync():
            db = None
            try:
                # Reuse the config this UI instance was launched with —
                # reloading config.toml from disk could target a different
                # corpus/chroma than retrieval uses.
                db = _get_db()
                import chromadb
                chroma_client = chromadb.PersistentClient(path=str(_chroma_path))
                from rag.indexer import IncrementalIndexer
                indexer = IncrementalIndexer(db, cfg, chroma_client=chroma_client)
                result = indexer.sync_all()
                _sync_status_ui["last_result"] = result
            except Exception as exc:
                logger.error("Indexing sync failed: %s", exc)
                _sync_status_ui["last_result"] = {"error": str(exc)}
            finally:
                _sync_status_ui["running"] = False
                _sync_status_ui["started_at"] = None
                _sync_status_ui["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                if db is not None:
                    with contextlib.suppress(Exception):
                        db.conn.close()

        with _sync_lock_ui:
            if _sync_status_ui["running"]:
                return {"ok": False, "error": "A sync is already in progress"}
            _sync_status_ui["running"] = True
            _sync_status_ui["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        t = threading.Thread(target=_run_sync, daemon=True)
        t.start()

        return {"ok": True, "message": "Sync started in background"}

    # Single catch-all proxy for all Gradio API requests
    # Long-lived upstream client. Shared (not per-request) so streamed
    # responses keep their connection for the whole body; per-request clients
    # would close the pool when the handler returns, killing the stream.
    _proxy_client = httpx.AsyncClient(trust_env=False, timeout=None)

    # Hop-by-hop / framing headers that must never be forwarded verbatim:
    # upstream content-length/transfer-encoding describe the UPSTREAM body,
    # and content-encoding has already been decoded by httpx. Forwarding them
    # produced malformed responses (content-length + transfer-encoding together)
    # that browsers truncate nondeterministically, cutting SSE streams mid-token.
    _REQ_DROP = ("host", "content-length", "connection", "keep-alive", "accept-encoding")
    _RESP_DROP = ("content-length", "transfer-encoding", "connection", "keep-alive", "content-encoding")

    async def _do_proxy(request: Request):
        path = request.url.path[len("/gradio_api/"):]
        target = f"{gradio_url}/gradio_api/{path}"
        if request.url.query:
            target += f"?{request.url.query}"
        body = await request.body() if request.method in ("POST", "PUT", "PATCH") else None
        req_headers = {k: v for k, v in request.headers.items() if k.lower() not in _REQ_DROP}
        # SSE streams (GET /call/chat/{event_id}) need a streaming response.
        # NOTE: `path` has the leading "/" stripped, so match without it —
        # the old check `"/call/chat/" in path` never matched and every SSE
        # stream was buffered instead, breaking incremental token streaming.
        if request.method == "GET" and path.startswith("call/chat/"):
            upstream = await _proxy_client.send(
                _proxy_client.build_request("GET", target, headers=req_headers),
                stream=True,
            )
            headers_out = {k: v for k, v in upstream.headers.items() if k.lower() not in _RESP_DROP}

            async def body_iter():
                try:
                    async for chunk in upstream.aiter_bytes():
                        yield chunk
                finally:
                    await upstream.aclose()

            return StreamingResponse(body_iter(), status_code=upstream.status_code, headers=headers_out)
        else:
            r = await _proxy_client.request(
                method=request.method, url=target,
                content=body,
                headers=req_headers,
            )
            return Response(content=r.content, status_code=r.status_code,
                            headers={k: v for k, v in r.headers.items() if k.lower() not in _RESP_DROP})

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

.sidebar-header{display:flex;flex-direction:column;gap:10px;padding:14px 14px 12px}
.sidebar-logo{display:flex;align-items:center;text-decoration:none;padding:0;transition:opacity var(--duration-fast) var(--ease-default)}
.sidebar-logo:hover{opacity:0.85}
.sidebar-logo__img{height:28px;width:auto;max-width:100%;display:block;object-fit:contain}
.sidebar-logo__img--dark{display:none}
[data-theme="dark"] .sidebar-logo__img--light{display:none}
[data-theme="dark"] .sidebar-logo__img--dark{display:block}
.sidebar-brand-row{display:flex;align-items:center;justify-content:space-between;gap:8px}
.sidebar-product-name{font-family:var(--font-sans);font-size:13px;font-weight:600;letter-spacing:-0.01em;color:var(--sidebar-text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
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
.topbar-logo{display:none;align-items:center;gap:8px;text-decoration:none;transition:opacity var(--duration-fast) var(--ease-default)}
.topbar-logo:hover{opacity:0.85}
.topbar-logo__icon{width:22px;height:22px;border-radius:4px;overflow:hidden;flex-shrink:0}
.topbar-logo__icon img{width:100%;height:100%;display:block;object-fit:contain}
.topbar-logo__name{font-family:var(--font-sans);font-size:13px;font-weight:600;letter-spacing:-0.01em;color:var(--text-primary)}
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
#hero{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding-left:24px;padding-right:24px;animation:hero-fade-in 600ms var(--ease-enter) both}
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
.msg-avatar{width:36px;height:36px;border-radius:var(--radius-sm);flex-shrink:0;background:transparent;overflow:hidden;margin-top:2px}
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

/* Thinking indicator */
.thinking-indicator{display:inline-flex;align-items:center;gap:8px;padding:4px 0}
.thinking-indicator__pulse{width:8px;height:8px;border-radius:50%;background:var(--accent);animation:think-pulse 1.8s ease-in-out infinite}
.thinking-indicator__text{font-size:13px;color:var(--text-secondary);font-style:italic;animation:think-fade 1.8s ease-in-out infinite}
@keyframes think-pulse{0%,100%{opacity:0.4;transform:scale(0.85)}50%{opacity:1;transform:scale(1.1)}}
@keyframes think-fade{0%,100%{opacity:0.5}50%{opacity:1}}

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

/* Settings button (sidebar footer) */
#settings-btn{width:30px;height:30px;border-radius:var(--radius-full);border:1px solid var(--sidebar-border);background:transparent;cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:14px;line-height:1;color:var(--sidebar-text);transition:all var(--duration-fast) var(--ease-default);flex-shrink:0}
#settings-btn:hover{background:var(--bg-sidebar-hover);border-color:var(--accent);color:var(--accent)}
#settings-btn:active{transform:scale(0.92) rotate(30deg)}

/* Model badge (composer) */
.model-badge{display:inline-flex;align-items:center;gap:6px;font-family:var(--font-mono);font-size:10px;letter-spacing:0.03em;color:var(--text-tertiary);background:var(--bg-primary);border:1px solid var(--border-subtle);border-radius:var(--radius-full);padding:4px 12px;cursor:pointer;transition:all var(--duration-fast) var(--ease-default);white-space:nowrap;max-width:260px;overflow:hidden}
.model-badge:hover{border-color:var(--accent);color:var(--accent);box-shadow:0 2px 10px var(--glow-accent)}
.model-badge__dot{width:6px;height:6px;border-radius:50%;background:var(--success);box-shadow:0 0 6px var(--success);flex-shrink:0}
.model-badge__name{overflow:hidden;text-overflow:ellipsis}
.model-badge__caret{opacity:0.55;font-size:8px}

/* Settings modal */
#settings-overlay{position:fixed;inset:0;z-index:1000;display:none;align-items:center;justify-content:center;background:rgba(20,17,15,0.48);backdrop-filter:blur(7px);padding:22px}
#settings-overlay.open{display:flex;animation:settings-fade 200ms var(--ease-enter) both}
@keyframes settings-fade{from{opacity:0}to{opacity:1}}
.settings-modal{width:100%;max-width:660px;max-height:88vh;display:flex;flex-direction:column;background:var(--bg-primary);border:1px solid var(--border-default);border-radius:var(--radius-xl);box-shadow:var(--shadow-lg);overflow:hidden;animation:settings-in 260ms var(--ease-spring) both}
@keyframes settings-in{from{opacity:0;transform:translateY(18px) scale(0.97)}to{opacity:1;transform:translateY(0) scale(1)}}
.settings-modal__header{display:flex;align-items:center;justify-content:space-between;padding:15px 20px;border-bottom:1px solid var(--border-subtle);flex-shrink:0}
.settings-modal__title{font-family:var(--font-display);font-size:19px;letter-spacing:-0.01em;color:var(--text-primary)}
.settings-modal__subtitle{font-family:var(--font-mono);font-size:9.5px;text-transform:uppercase;letter-spacing:0.14em;color:var(--text-tertiary);margin-top:2px}
.settings-modal__close{width:32px;height:32px;border-radius:var(--radius-sm);border:1px solid var(--border-subtle);background:transparent;cursor:pointer;font-size:13px;color:var(--text-tertiary);transition:all var(--duration-fast) var(--ease-default);flex-shrink:0}
.settings-modal__close:hover{background:rgba(220,75,92,0.08);color:var(--error);border-color:var(--error)}
.settings-modal__body{overflow-y:auto;padding:18px 20px 20px;display:flex;flex-direction:column;gap:24px}

.settings-block__head{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:10px}
.settings-block__title{font-family:var(--font-mono);font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.16em;color:var(--text-tertiary);display:flex;align-items:center;gap:6px}
.settings-block__title::before{content:'';width:4px;height:14px;border-radius:1px;background:var(--accent)}
.settings-hint{font-size:12px;color:var(--text-tertiary);line-height:1.55;margin:-4px 0 10px}

.settings-btn{font-family:var(--font-mono);font-size:11px;font-weight:500;letter-spacing:0.03em;padding:6px 13px;border-radius:var(--radius-sm);cursor:pointer;transition:all var(--duration-fast) var(--ease-default);border:1px solid var(--border-default);background:var(--bg-primary);color:var(--text-secondary)}
.settings-btn:hover{border-color:var(--accent);color:var(--accent)}
.settings-btn:active{transform:scale(0.97)}
.settings-btn--primary{background:var(--accent);border-color:var(--accent);color:#14110F;font-weight:600}
.settings-btn--primary:hover{filter:brightness(1.06);box-shadow:0 4px 16px var(--glow-accent);color:#14110F}
.settings-btn--danger{color:var(--error)}
.settings-btn--danger:hover{border-color:var(--error);background:rgba(220,75,92,0.08);color:var(--error)}
.settings-btn--mini{padding:4px 10px;font-size:10px}
.settings-btn[disabled]{opacity:0.5;cursor:not-allowed;pointer-events:none}

.provider-list{display:flex;flex-direction:column;gap:9px}
.provider-card{border:1px solid var(--border-default);border-radius:var(--radius-lg);padding:12px 14px;background:var(--bg-primary);display:flex;flex-direction:column;gap:7px;transition:all var(--duration-fast) var(--ease-default)}
.provider-card:hover{border-color:var(--border-strong)}
.provider-card.active{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent),0 4px 18px var(--glow-accent)}
.provider-card__row1{display:flex;align-items:center;gap:8px;min-width:0}
.provider-card__name{font-size:13.5px;font-weight:600;color:var(--text-primary);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.provider-card__badges{display:flex;align-items:center;gap:5px;flex-shrink:0;margin-left:auto}
.badge{font-family:var(--font-mono);font-size:8.5px;font-weight:600;text-transform:uppercase;letter-spacing:0.09em;padding:3px 8px;border-radius:var(--radius-full);border:1px solid var(--border-subtle);background:var(--bg-secondary);color:var(--text-tertiary);white-space:nowrap}
.badge--active{background:var(--accent);border-color:var(--accent);color:#14110F}
.badge--key{color:var(--success);border-color:rgba(31,122,77,0.3)}
[data-theme="dark"] .badge--key{border-color:rgba(52,211,153,0.3)}
.provider-card__model{font-family:var(--font-mono);font-size:11.5px;color:var(--accent);word-break:break-all}
.provider-card__url{font-family:var(--font-mono);font-size:10px;color:var(--text-tertiary);word-break:break-all;opacity:0.85}
.provider-card__actions{display:flex;align-items:center;gap:6px;margin-top:3px;flex-wrap:wrap}

.provider-form{border:1px dashed var(--border-default);border-radius:var(--radius-lg);padding:16px;display:none;flex-direction:column;gap:11px;background:var(--bg-secondary);margin-top:11px}
.provider-form.open{display:flex}
.provider-form__title{font-family:var(--font-mono);font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.14em;color:var(--text-secondary)}
.form-grid{display:grid;grid-template-columns:1fr 1fr;gap:11px}
@media(max-width:560px){.form-grid{grid-template-columns:1fr}}
.form-field{display:flex;flex-direction:column;gap:4px;min-width:0}
.form-field--full{grid-column:1 / -1}
.form-field label{font-family:var(--font-mono);font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:var(--text-tertiary)}
.form-field input,.form-field select{border:1px solid var(--border-default);border-radius:var(--radius-sm);background:var(--bg-primary);color:var(--text-primary);font-size:12.5px;font-family:var(--font-sans);padding:8px 11px;outline:none;transition:all var(--duration-fast) var(--ease-default);width:100%}
.form-field input:focus,.form-field select:focus{border-color:var(--accent);box-shadow:0 0 0 2px var(--accent-subtle)}
.form-field input::placeholder{color:var(--text-disabled)}
.form-field .field-hint{font-size:10px;color:var(--text-tertiary);font-family:var(--font-mono)}
.model-input-row{display:flex;gap:6px}
.model-input-row input{flex:1;min-width:0}
.provider-form__actions{display:flex;align-items:center;gap:8px;justify-content:flex-end;margin-top:2px}

.sysinfo-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}
@media(max-width:560px){.sysinfo-grid{grid-template-columns:1fr}}
.sysinfo-item{border:1px solid var(--border-subtle);border-radius:var(--radius-md);padding:9px 12px;background:var(--bg-secondary);display:flex;flex-direction:column;gap:2px;min-width:0}
.sysinfo-item__k{font-family:var(--font-mono);font-size:8.5px;text-transform:uppercase;letter-spacing:0.12em;color:var(--text-tertiary)}
.sysinfo-item__v{font-family:var(--font-mono);font-size:11px;color:var(--text-secondary);word-break:break-all}

#settings-toast{position:fixed;bottom:26px;left:50%;transform:translateX(-50%) translateY(16px);opacity:0;z-index:1100;font-family:var(--font-mono);font-size:11.5px;padding:10px 18px;border-radius:var(--radius-full);background:var(--bg-primary);border:1px solid var(--border-default);box-shadow:var(--shadow-lg);color:var(--text-primary);pointer-events:none;transition:all var(--duration-normal) var(--ease-spring);max-width:80vw;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#settings-toast.show{opacity:1;transform:translateX(-50%) translateY(0)}
#settings-toast.ok{border-color:var(--success)}
#settings-toast.err{border-color:var(--error)}

/* Knowledge Base Status */
.kb-status-overview{border:1px solid var(--border-default);border-radius:var(--radius-lg);padding:16px;background:var(--bg-secondary);display:flex;flex-direction:column;gap:14px}
.kb-status-row{display:flex;align-items:center;justify-content:space-between;gap:10px}
.kb-status-row__label{font-family:var(--font-mono);font-size:10px;text-transform:uppercase;letter-spacing:0.1em;color:var(--text-tertiary);font-weight:600}
.kb-status-row__value{font-family:var(--font-mono);font-size:13px;color:var(--text-primary);font-weight:600}
.kb-status-row__value--success{color:var(--success)}
.kb-status-row__value--warning{color:var(--warning)}
.kb-status-row__value--error{color:var(--error)}
.kb-progress-bar{width:100%;height:8px;border-radius:var(--radius-full);background:var(--bg-tertiary);overflow:hidden;position:relative}
.kb-progress-bar__fill{height:100%;border-radius:var(--radius-full);background:linear-gradient(90deg,var(--accent),var(--success));transition:width 600ms var(--ease-default);position:relative}
.kb-progress-bar__fill--partial{background:linear-gradient(90deg,var(--accent),var(--warning))}
.kb-progress-bar__fill--error{background:var(--error)}
.kb-status-stats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
@media(max-width:560px){.kb-status-stats{grid-template-columns:repeat(2,1fr)}}
.kb-stat{display:flex;flex-direction:column;align-items:center;padding:10px 8px;border:1px solid var(--border-subtle);border-radius:var(--radius-md);background:var(--bg-primary);gap:2px}
.kb-stat__value{font-family:var(--font-mono);font-size:18px;font-weight:700;color:var(--accent);line-height:1.1}
.kb-stat__label{font-family:var(--font-mono);font-size:8.5px;text-transform:uppercase;letter-spacing:0.1em;color:var(--text-tertiary)}
.kb-stat__value--success{color:var(--success)}
.kb-stat__value--warning{color:var(--warning)}
.kb-stat__value--error{color:var(--error)}
.kb-folder-list{display:flex;flex-direction:column;gap:6px;max-height:240px;overflow-y:auto;padding:4px 0}
.kb-folder-item{display:flex;align-items:center;gap:8px;padding:8px 10px;border:1px solid var(--border-subtle);border-radius:var(--radius-md);background:var(--bg-primary);transition:all var(--duration-fast) var(--ease-default)}
.kb-folder-item:hover{border-color:var(--border-default)}
.kb-folder-item__icon{font-size:14px;flex-shrink:0}
.kb-folder-item__name{font-size:12px;font-weight:500;color:var(--text-primary);flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.kb-folder-item__stats{font-family:var(--font-mono);font-size:10px;color:var(--text-tertiary);white-space:nowrap}
.kb-folder-item__dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.kb-folder-item__dot--complete{background:var(--success);box-shadow:0 0 4px var(--success)}
.kb-folder-item__dot--partial{background:var(--warning);box-shadow:0 0 4px var(--warning)}
.kb-folder-item__dot--pending{background:var(--text-disabled)}
.kb-folder-item__dot--syncing{background:var(--accent);animation:kb-pulse 1.2s ease-in-out infinite}
@keyframes kb-pulse{0%,100%{opacity:0.4;transform:scale(0.8)}50%{opacity:1;transform:scale(1.2)}}
.kb-status-actions{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-top:4px}
.kb-sync-indicator{display:inline-flex;align-items:center;gap:6px;font-family:var(--font-mono);font-size:10px;color:var(--accent);padding:4px 10px;border:1px solid rgba(255,90,31,0.2);border-radius:var(--radius-full);background:var(--accent-subtle)}
.kb-sync-indicator__spinner{width:10px;height:10px;border:2px solid var(--accent);border-top-color:transparent;border-radius:50%;animation:kb-spin 0.8s linear infinite}
@keyframes kb-spin{to{transform:rotate(360deg)}}
.kb-last-sync{font-family:var(--font-mono);font-size:10px;color:var(--text-tertiary);margin-top:2px}
.kb-sync-error{display:none;font-family:var(--font-mono);font-size:10.5px;color:var(--error);background:rgba(255,59,48,0.08);border:1px solid rgba(255,59,48,0.25);border-radius:var(--radius-md);padding:8px 10px;margin-top:8px;word-break:break-word}
</style>
</head>
<body>
<div id="app">
<!-- Sidebar -->
<div id="sidebar">
<div class="sidebar-header">
<a class="sidebar-logo" href="https://odw.ai/" target="_blank" rel="noopener"><img class="sidebar-logo__img sidebar-logo__img--light" src="/resource_img/odwai-logo-2048x651.png" alt="ODW.AI"><img class="sidebar-logo__img sidebar-logo__img--dark" src="/resource_img/odwai-logo-dark-2048x651.png" alt="ODW.AI"></a>
<div class="sidebar-brand-row"><span class="sidebar-product-name">ODW Vault</span><button id="new-chat-btn"><span>+</span> <span>New Chat</span></button></div>
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
<div style="display:flex;align-items:center;gap:6px">
<button id="settings-btn" title="Vault settings">&#9881;&#65038;</button>
<button id="theme-toggle-sidebar" title="Toggle theme"></button>
</div>
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
<a class="topbar-logo" href="https://odw.ai/" target="_blank" rel="noopener" style="margin-left:4px"><div class="topbar-logo__icon"><img src="/resource_img/favicon.png" alt="ODW"></div><span class="topbar-logo__name">ODW Vault</span></a>
</div>
<div class="topbar-right">
<div class="topbar-status"></div>
</div>
</div>

<div id="main">
<div id="hero">
<h1 id="greeting-text">$GREETING</h1>
</div>
<div id="msgs"></div>
</div>

<div id="cit"></div>

<div id="chips" style="display:none"><button id="chip-rf" title="Refresh suggestions">&#x21bb;</button></div>

<div id="flt" style="display:none">
<label>Scope:</label>
<select id="ff">$FOLDER_OPTIONS</select>
</div>

<div id="ca">
<div id="composer">
<textarea id="inp" placeholder="Ask anything about your knowledge base..." rows="1" autofocus></textarea>
<div id="ca-row">
<div style="display:flex;align-items:center;gap:10px;min-width:0;overflow:hidden">
<div class="scope-indicator"><span class="scope-indicator__dot"></span> <span id="scope-label">All folders</span></div>
<button id="model-badge" class="model-badge" title="Switch model"><span class="model-badge__dot"></span><span class="model-badge__name" id="model-badge-name">model</span><span class="model-badge__caret">&#9662;</span></button>
</div>
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
        if(!current || current.indexOf('thinking-indicator') !== -1){
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
    var fltEl = document.getElementById('flt');
    if(fltEl) fltEl.style.display = '';
    _addMsg('user', t);
    _S.H.push({role:'user', content:[{text:t, type:'text'}]});
    inp.value = ''; inp.style.height = 'auto';

    var el = _addMsg('assistant', '<div class="thinking-indicator"><div class="thinking-indicator__pulse"></div><span class="thinking-indicator__text">Thinking...</span></div>', true);
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
      var _handleLines = function(lines){
        for(var i = 0; i < lines.length; i++){
          var line = lines[i];
          if(line.startsWith('data:')){
            try {
              var d = JSON.parse(line.slice(5));
              // Gradio heartbeat frames are `data: null` — guard before use.
              if(d && d.error){
                var mdEl = el.querySelector('.md');
                if(mdEl) mdEl.innerHTML = '<div class="msg-error"><div class="msg-error__text">Error: ' + _escHtml(d.error) + '</div><button class="msg-error__retry" onclick="window._retryLast()">Retry</button></div>';
                _done(el);
                return true;
              }
              if(d && d[0] && d[0].length){
                var h = d[0];
                var last = h[h.length - 1];
                // Citations first: an exception later in this block (e.g. the
                // markdown renderer choking on some generated text) must
                // never swallow the citations payload of the final frames.
                if(d[1]){
                  var citEl = document.getElementById('cit');
                  if(citEl) citEl.innerHTML = d[1];
                }
                if(last && last.content && last.content.length){
                  var text = last.content[0].text || '';
                  var mdEl = el.querySelector('.md');
                  if(mdEl){
                    // Render defensively: fall back to escaped plain text
                    // instead of letting a renderer error kill the frame.
                    try {
                      mdEl.innerHTML = _md(text);
                    } catch(e){
                      mdEl.innerHTML = '<p>' + _escHtml(text).replace(/\\n/g,'<br>') + '</p>';
                    }
                  }
                }
                var msgs = document.getElementById('msgs');
                msgs.scrollTop = msgs.scrollHeight;
              }
            } catch(e) {}
          }
        }
        return false;
      };
      (function pump(){
        reader.read().then(function(res){
          if(_S.abortFlag){ reader.cancel(); _done(el); return; }
          if(res.done){
            // The stream may close without a trailing newline; flush the
            // final unterminated line or the last data: event (citations
            // panel) is silently dropped.
            if(buf){
              var tail = buf; buf = '';
              if(_handleLines([tail])) return;
            }
            _done(el); return;
          }
          buf += new TextDecoder().decode(res.value);
          var lines = buf.split('\\n');
          buf = lines.pop() || '';
          if(_handleLines(lines)) return;
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

  function _escHtml(s){ var d = document.createElement('div'); d.textContent = s; return d.innerHTML.replace(/"/g,'&quot;').replace(/'/g,'&#39;'); }

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
      avatar.innerHTML = '<img src="/resource_img/bot_new.png" alt="ODW.AI">';
      d.appendChild(avatar);
      var body = document.createElement('div');
      body.className = 'msg-body';
      var m = document.createElement('div');
      m.className = 'md';
      m.innerHTML = stream ? text : _md(text);
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
    t = t.replace(/\\[([\\d,\\s]+)\\]/g, function(m, nums){
      var parts = nums.split(/[,\\s]+/).filter(Boolean);
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
    if(chips) chips.style.display = 'none';
    var flt = document.getElementById('flt');
    if(flt) flt.style.display = 'none';
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
      var flt = document.getElementById('flt');
      if(flt) flt.style.display = '';
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

  /* ── Dynamic greeting based on system time ── */
  function _updateGreeting(){
    var el = document.getElementById('greeting-text');
    if(!el) return;
    var h = new Date().getHours();
    el.textContent = h < 12 ? 'Good morning' : (h < 17 ? 'Good afternoon' : 'Good evening');
  }
  _updateGreeting();
  setInterval(_updateGreeting, 60000);

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

<!-- Settings modal -->
<div id="settings-overlay">
<div class="settings-modal" role="dialog" aria-modal="true" aria-label="Vault settings">
<div class="settings-modal__header">
<div>
<div class="settings-modal__title">Vault Settings</div>
<div class="settings-modal__subtitle">Models, Providers &amp; Knowledge Base</div>
</div>
<button class="settings-modal__close" id="settings-close" title="Close (Esc)">\u2715</button>
</div>
<div class="settings-modal__body">

<div class="settings-block">
<div class="settings-block__head">
<span class="settings-block__title">Model Providers</span>
<button id="provider-add-btn" class="settings-btn settings-btn--primary">+ Add Provider</button>
</div>
<p class="settings-hint">The <b>active</b> model generates every answer. Supports OpenAI-compatible, Ollama and Anthropic (Claude) protocols across all major platforms. Retrieval embeddings always stay on the local embedding model.</p>
<div id="provider-list" class="provider-list"></div>

<form id="provider-form" class="provider-form" autocomplete="off">
<div class="provider-form__title" id="pf-form-title">Add Provider</div>
<div class="form-grid">
<div class="form-field">
<label for="pf-preset">Platform preset</label>
<select id="pf-preset"></select>
</div>
<div class="form-field">
<label for="pf-protocol">Protocol</label>
<select id="pf-protocol"></select>
</div>
<div class="form-field">
<label for="pf-name">Display name</label>
<input type="text" id="pf-name" placeholder="e.g. My GPT-4o" maxlength="60">
</div>
<div class="form-field">
<label for="pf-baseurl">Base URL</label>
<input type="text" id="pf-baseurl" placeholder="https://api.openai.com/v1">
</div>
<div class="form-field form-field--full">
<label for="pf-key">API key</label>
<input type="password" id="pf-key" placeholder="sk-... (leave blank for local)">
<span class="field-hint" id="pf-key-hint"></span>
</div>
<div class="form-field form-field--full">
<label for="pf-model">Model ID</label>
<div class="model-input-row">
<input type="text" id="pf-model" list="pf-model-list" placeholder="gpt-4o / deepseek-chat / gemma4:latest ...">
<button type="button" id="pf-fetch" class="settings-btn settings-btn--mini" title="Fetch available models from the endpoint">Fetch</button>
</div>
<datalist id="pf-model-list"></datalist>
</div>
</div>
<div class="provider-form__actions">
<button type="button" id="pf-test" class="settings-btn">Test connection</button>
<button type="button" id="pf-cancel" class="settings-btn">Cancel</button>
<button type="submit" id="pf-save" class="settings-btn settings-btn--primary">Save provider</button>
</div>
</form>
</div>

<div class="settings-block">
<div class="settings-block__head"><span class="settings-block__title">Knowledge Base</span>
<div style="display:flex;gap:6px">
<button id="kb-refresh-btn" class="settings-btn settings-btn--mini" title="Refresh status">&#x21bb; Refresh</button>
<button id="kb-sync-btn" class="settings-btn settings-btn--mini settings-btn--primary" title="Start incremental sync">&#x26a1; Sync Now</button>
</div>
</div>
<p class="settings-hint">Vectorization status of your corpus. Shows extraction, chunking, and embedding progress across all folders. New files added to the corpus need to be synced before they appear in search results.</p>

<div class="kb-status-overview">
<div>
<div class="kb-status-row">
<span class="kb-status-row__label">Overall Progress</span>
<span class="kb-status-row__value" id="kb-progress-text">--</span>
</div>
<div class="kb-progress-bar" style="margin-top:8px">
<div class="kb-progress-bar__fill" id="kb-progress-fill" style="width:0%"></div>
</div>
</div>

<div class="kb-status-stats">
<div class="kb-stat">
<span class="kb-stat__value" id="kb-stat-total">--</span>
<span class="kb-stat__label">Files</span>
</div>
<div class="kb-stat">
<span class="kb-stat__value kb-stat__value--success" id="kb-stat-embedded">--</span>
<span class="kb-stat__label">Embedded</span>
</div>
<div class="kb-stat">
<span class="kb-stat__value kb-stat__value--warning" id="kb-stat-pending">--</span>
<span class="kb-stat__label">Pending</span>
</div>
<div class="kb-stat">
<span class="kb-stat__value" id="kb-stat-chunks">--</span>
<span class="kb-stat__label">Chunks</span>
</div>
<div class="kb-stat">
<span class="kb-stat__value kb-stat__value--error" id="kb-stat-failed">--</span>
<span class="kb-stat__label">Failed</span>
</div>
<div class="kb-stat">
<span class="kb-stat__value" id="kb-stat-folders">--</span>
<span class="kb-stat__label">Folders</span>
</div>
</div>

<div>
<div class="kb-status-row" style="margin-bottom:6px">
<span class="kb-status-row__label">Per-Folder Status</span>
</div>
<div class="kb-folder-list" id="kb-folder-list">
<div style="padding:12px;text-align:center;font-size:11px;color:var(--text-tertiary)">Loading...</div>
</div>
</div>

<div class="kb-status-actions">
<div id="kb-sync-indicator" style="display:none" class="kb-sync-indicator">
<div class="kb-sync-indicator__spinner"></div>
<span>Syncing...</span>
</div>
<span class="kb-last-sync" id="kb-last-sync"></span>
</div>
<div id="kb-sync-error" style="display:none" class="kb-sync-error"></div>
</div>
</div>

<div class="settings-block">
<div class="settings-block__head"><span class="settings-block__title">System</span></div>
<div id="settings-sysinfo" class="sysinfo-grid"></div>
</div>

</div>
</div>
</div>
<div id="settings-toast"></div>

<script>
(function(){
  var _ST = {providers:[], activeId:null, presets:[], protocols:[], config:{}, editingId:null};
  var _toastTimer = null;

  function _q(id){ return document.getElementById(id); }
  function _esc(s){ var d = document.createElement('div'); d.textContent = (s == null ? '' : String(s)); return d.innerHTML.replace(/"/g,'&quot;').replace(/'/g,'&#39;'); }

  function _toast(msg, kind){
    var t = _q('settings-toast');
    if(!t) return;
    t.textContent = msg;
    t.className = 'show ' + (kind || '');
    clearTimeout(_toastTimer);
    _toastTimer = setTimeout(function(){ t.className = ''; }, 3200);
  }

  function _api(method, url, body){
    var opts = {method: method, headers: {}};
    if(body !== undefined){
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    }
    return fetch(url, opts).then(function(r){
      return r.json().then(function(data){
        // FastAPI validation errors arrive as {"detail": [...]} — flatten to text
        // so toasts never render "[object Object]".
        if(data && Array.isArray(data.detail)){
          data.ok = false;
          data.detail = data.detail.map(function(d){
            var loc = (d && d.loc) ? ' (' + d.loc.join('.') + ')' : '';
            var msg = (d && (d.msg || d.type)) ? (d.msg || d.type) : String(d);
            return msg + loc;
          }).join('; ');
        }
        return data;
      }).catch(function(){ return {ok: false, error: 'HTTP ' + r.status}; });
    });
  }

  function _applySettings(data){
    if(!data) return;
    if(data.providers) _ST.providers = data.providers;
    if(data.active_id !== undefined) _ST.activeId = data.active_id;
    if(data.presets) _ST.presets = data.presets;
    if(data.protocols) _ST.protocols = data.protocols;
    if(data.config) _ST.config = data.config;
    _renderProviders();
    _renderSysinfo();
    _renderFormOptions();
    _updateBadge();
  }

  function _activeProvider(){
    for(var i = 0; i < _ST.providers.length; i++){
      if(_ST.providers[i].id === _ST.activeId) return _ST.providers[i];
    }
    return _ST.providers[0] || null;
  }

  function _updateBadge(){
    var el = _q('model-badge-name');
    var p = _activeProvider();
    if(el) el.textContent = p ? (p.model || 'no model') : 'no provider';
  }

  function _protocolLabel(v){
    for(var i = 0; i < _ST.protocols.length; i++){
      if(_ST.protocols[i].value === v) return _ST.protocols[i].label;
    }
    return v;
  }

  function _renderProviders(){
    var list = _q('provider-list');
    if(!list) return;
    if(!_ST.providers.length){
      list.innerHTML = '<div class="settings-hint" style="text-align:center;padding:12px 0">No providers yet — add one to get started.</div>';
      return;
    }
    var html = '';
    for(var i = 0; i < _ST.providers.length; i++){
      var p = _ST.providers[i];
      var isActive = p.id === _ST.activeId;
      html += '<div class="provider-card' + (isActive ? ' active' : '') + '" data-pid="' + _esc(p.id) + '">' +
        '<div class="provider-card__row1">' +
        '<span class="provider-card__name">' + _esc(p.name) + '</span>' +
        '<span class="provider-card__badges">' +
        '<span class="badge">' + _esc(_protocolLabel(p.protocol)) + '</span>' +
        (p.has_key ? '<span class="badge badge--key" title="API key stored">key ' + _esc(p.key_hint) + '</span>' : '') +
        (isActive ? '<span class="badge badge--active">Active</span>' : '') +
        '</span></div>' +
        '<div class="provider-card__model">' + _esc(p.model) + '</div>' +
        '<div class="provider-card__url">' + _esc(p.base_url) + '</div>' +
        '<div class="provider-card__actions">' +
        (isActive ? '' : '<button class="settings-btn settings-btn--mini settings-btn--primary" data-act="use">Use</button>') +
        '<button class="settings-btn settings-btn--mini" data-act="test">Test</button>' +
        '<button class="settings-btn settings-btn--mini" data-act="edit">Edit</button>' +
        '<button class="settings-btn settings-btn--mini settings-btn--danger" data-act="del">Delete</button>' +
        '</div></div>';
    }
    list.innerHTML = html;
  }

  var _SYS_LABELS = [
    ['corpus_root', 'Corpus root'],
    ['chroma_root', 'Vector store'],
    ['ollama_host', 'Embedding host'],
    ['embedding_model', 'Embedding model'],
    ['generation_model', 'Config default gen'],
    ['temperature', 'Temperature'],
    ['require_citations', 'Citation-strict']
  ];
  function _renderSysinfo(){
    var el = _q('settings-sysinfo');
    if(!el) return;
    var html = '';
    for(var i = 0; i < _SYS_LABELS.length; i++){
      var k = _SYS_LABELS[i][0], lbl = _SYS_LABELS[i][1];
      var v = _ST.config[k];
      if(v === undefined || v === null || v === '') v = '\u2014';
      if(typeof v === 'boolean') v = v ? 'on' : 'off';
      html += '<div class="sysinfo-item"><span class="sysinfo-item__k">' + _esc(lbl) + '</span><span class="sysinfo-item__v">' + _esc(v) + '</span></div>';
    }
    el.innerHTML = html;
  }

  function _renderFormOptions(){
    var presetSel = _q('pf-preset');
    var protoSel = _q('pf-protocol');
    if(presetSel && !presetSel.options.length){
      var h = '';
      for(var i = 0; i < _ST.presets.length; i++){
        h += '<option value="' + _esc(_ST.presets[i].key) + '">' + _esc(_ST.presets[i].label) + '</option>';
      }
      presetSel.innerHTML = h;
    }
    if(protoSel && !protoSel.options.length){
      var h2 = '';
      for(var j = 0; j < _ST.protocols.length; j++){
        h2 += '<option value="' + _esc(_ST.protocols[j].value) + '">' + _esc(_ST.protocols[j].label) + '</option>';
      }
      protoSel.innerHTML = h2;
    }
  }

  function _openSettings(){
    _api('GET', '/api/settings').then(_applySettings).catch(function(){});
    var o = _q('settings-overlay');
    if(o){ o.classList.add('open'); }
  }
  function _closeSettings(){
    var o = _q('settings-overlay');
    if(o) o.classList.remove('open');
    _hideForm();
  }

  function _showForm(entry){
    _ST.editingId = entry ? entry.id : null;
    var f = _q('provider-form');
    _q('pf-form-title').textContent = entry ? 'Edit Provider' : 'Add Provider';
    _q('pf-name').value = entry ? entry.name : '';
    _q('pf-protocol').value = entry ? entry.protocol : 'openai';
    _q('pf-baseurl').value = entry ? entry.base_url : '';
    _q('pf-model').value = entry ? entry.model : '';
    _q('pf-key').value = '';
    _q('pf-key-hint').textContent = entry && entry.has_key ? 'A key is stored (' + entry.key_hint + ') — leave blank to keep it.' : '';
    _q('pf-preset').value = 'custom';
    if(f) f.classList.add('open');
    _q('pf-name').focus();
  }
  function _hideForm(){
    var f = _q('provider-form');
    if(f) f.classList.remove('open');
    _ST.editingId = null;
  }

  function _applyPreset(){
    var key = _q('pf-preset').value;
    for(var i = 0; i < _ST.presets.length; i++){
      if(_ST.presets[i].key === key){
        var p = _ST.presets[i];
        _q('pf-protocol').value = p.protocol;
        if(p.base_url) _q('pf-baseurl').value = p.base_url;
        if(p.model) _q('pf-model').value = p.model;
        if(!_q('pf-name').value || _q('pf-name').value === '') _q('pf-name').value = p.label;
        return;
      }
    }
  }

  function _formPayload(includeKey){
    var payload = {
      id: _ST.editingId || '',
      name: _q('pf-name').value.trim(),
      protocol: _q('pf-protocol').value,
      base_url: _q('pf-baseurl').value.trim(),
      model: _q('pf-model').value.trim()
    };
    if(includeKey) payload.api_key = _q('pf-key').value;
    return payload;
  }

  /* -- wire events -- */
  var settingsBtn = _q('settings-btn');
  // Call-through wrappers: _openSettings is re-wrapped further below to add
  // the KB status auto-fetch, and addEventListener would otherwise capture
  // the pre-wrap function value.
  if(settingsBtn) settingsBtn.addEventListener('click', function(){ _openSettings(); });
  var modelBadge = _q('model-badge');
  if(modelBadge) modelBadge.addEventListener('click', function(){ _openSettings(); });
  var closeBtn = _q('settings-close');
  if(closeBtn) closeBtn.addEventListener('click', function(){ _closeSettings(); });
  var overlay = _q('settings-overlay');
  if(overlay) overlay.addEventListener('click', function(e){ if(e.target === overlay) _closeSettings(); });
  document.addEventListener('keydown', function(e){
    if(e.key === 'Escape' && overlay && overlay.classList.contains('open')) _closeSettings();
  });

  var addBtn = _q('provider-add-btn');
  if(addBtn) addBtn.addEventListener('click', function(){ _showForm(null); });
  var cancelBtn = _q('pf-cancel');
  if(cancelBtn) cancelBtn.addEventListener('click', _hideForm);
  var presetSel2 = _q('pf-preset');
  if(presetSel2) presetSel2.addEventListener('change', _applyPreset);

  var form = _q('provider-form');
  if(form) form.addEventListener('submit', function(e){
    e.preventDefault();
    var payload = _formPayload(true);
    if(!payload.name || !payload.model || !payload.base_url){
      _toast('Name, Base URL and Model are required', 'err');
      return;
    }
    var saveBtn = _q('pf-save');
    if(saveBtn) saveBtn.setAttribute('disabled', '');
    _api('POST', '/api/settings/providers', payload)
      .then(function(resp){
        if(saveBtn) saveBtn.removeAttribute('disabled');
        if(resp.ok){
          _toast('Provider saved', 'ok');
          _hideForm();
          _applySettings(resp.settings);
        } else {
          _toast(resp.error || 'Save failed', 'err');
        }
      })
      .catch(function(){
        if(saveBtn) saveBtn.removeAttribute('disabled');
        _toast('Network error', 'err');
      });
  });

  var fetchBtn = _q('pf-fetch');
  if(fetchBtn) fetchBtn.addEventListener('click', function(){
    var btn = fetchBtn;
    btn.setAttribute('disabled', '');
    btn.textContent = '...';
    _api('POST', '/api/settings/providers/models', _formPayload(true))
      .then(function(resp){
        btn.removeAttribute('disabled');
        btn.textContent = 'Fetch';
        if(resp.ok && resp.models && resp.models.length){
          var dl = _q('pf-model-list');
          var h = '';
          for(var i = 0; i < resp.models.length; i++){
            h += '<option value="' + _esc(resp.models[i]) + '"></option>';
          }
          dl.innerHTML = h;
          _toast(resp.models.length + ' models found — pick one in the Model ID field', 'ok');
        } else {
          _toast('No models returned (check URL / key)', 'err');
        }
      })
      .catch(function(){
        btn.removeAttribute('disabled');
        btn.textContent = 'Fetch';
        _toast('Network error', 'err');
      });
  });

  var testBtn = _q('pf-test');
  if(testBtn) testBtn.addEventListener('click', function(){
    testBtn.setAttribute('disabled', '');
    testBtn.textContent = 'Testing...';
    _api('POST', '/api/settings/providers/test', _formPayload(true))
      .then(function(resp){
        testBtn.removeAttribute('disabled');
        testBtn.textContent = 'Test connection';
        _toast((resp.ok ? '\u2713 ' : '\u2717 ') + (resp.detail || (resp.ok ? 'OK' : 'failed')), resp.ok ? 'ok' : 'err');
      })
      .catch(function(){
        testBtn.removeAttribute('disabled');
        testBtn.textContent = 'Test connection';
        _toast('Network error', 'err');
      });
  });

  var plist = _q('provider-list');
  if(plist) plist.addEventListener('click', function(e){
    var btn = e.target.closest('[data-act]');
    if(!btn) return;
    var card = btn.closest('.provider-card');
    if(!card) return;
    var pid = card.getAttribute('data-pid');
    var act = btn.getAttribute('data-act');
    var entry = null;
    for(var i = 0; i < _ST.providers.length; i++){
      if(_ST.providers[i].id === pid) entry = _ST.providers[i];
    }
    if(act === 'edit'){
      _showForm(entry);
    } else if(act === 'use'){
      _api('POST', '/api/settings/providers/' + encodeURIComponent(pid) + '/activate', {})
        .then(function(resp){
          if(resp.ok){ _applySettings(resp.settings); _toast('Switched to ' + (entry ? entry.name : pid), 'ok'); }
          else _toast(resp.error || 'Failed', 'err');
        }).catch(function(){ _toast('Network error', 'err'); });
    } else if(act === 'test'){
      btn.setAttribute('disabled', ''); btn.textContent = '...';
      _api('POST', '/api/settings/providers/test', {
        id: pid, protocol: entry.protocol, base_url: entry.base_url, model: entry.model, api_key: ''
      }).then(function(resp){
        btn.removeAttribute('disabled'); btn.textContent = 'Test';
        _toast((resp.ok ? '\u2713 ' : '\u2717 ') + (resp.detail || ''), resp.ok ? 'ok' : 'err');
      }).catch(function(){ btn.removeAttribute('disabled'); btn.textContent = 'Test'; _toast('Network error', 'err'); });
    } else if(act === 'del'){
      if(!confirm('Delete provider \u201c' + (entry ? entry.name : '') + '\u201d?')) return;
      _api('DELETE', '/api/settings/providers/' + encodeURIComponent(pid))
        .then(function(resp){
          if(resp.ok){ _applySettings(resp.settings); _toast('Provider deleted', 'ok'); }
          else _toast(resp.error || 'Failed', 'err');
        }).catch(function(){ _toast('Network error', 'err'); });
    }
  });

  /* -- initial state (badge + data warm-up) -- */
  _api('GET', '/api/settings').then(_applySettings).catch(function(){});

  /* -- Knowledge Base Status -- */
  var _kbRefreshTimer = null;
  var _kbSyncPollTimer = null;

  function _renderKbStatus(data){
    if(!data) return;
    // Partial payloads (e.g. {sync_running:true}) only carry sync state —
    // skip the data sections when the status fields are absent.
    var full = (data.progress_pct != null);

    // Progress text
    var pctText = data.progress_pct + '%';
    var progressEl = _q('kb-progress-text');
    if(progressEl && full){
      if(data.sync_running){
        progressEl.textContent = 'Syncing...';
        progressEl.className = 'kb-status-row__value kb-status-row__value--warning';
      } else if(data.progress_pct >= 100){
        progressEl.textContent = 'Complete (' + pctText + ')';
        progressEl.className = 'kb-status-row__value kb-status-row__value--success';
      } else if(data.failed_files > 0){
        progressEl.textContent = pctText + ' (' + data.failed_files + ' failed)';
        progressEl.className = 'kb-status-row__value kb-status-row__value--error';
      } else if(data.pending_files > 0){
        progressEl.textContent = pctText + ' (' + data.pending_files + ' pending)';
        progressEl.className = 'kb-status-row__value kb-status-row__value--warning';
      } else {
        progressEl.textContent = pctText;
        progressEl.className = 'kb-status-row__value';
      }
    }

    // Progress bar
    var fillEl = _q('kb-progress-fill');
    if(fillEl && full){
      fillEl.style.width = Math.min(data.progress_pct, 100) + '%';
      fillEl.className = 'kb-progress-bar__fill';
      if(data.failed_files > 0 && data.progress_pct < 100) fillEl.classList.add('kb-progress-bar__fill--error');
      else if(data.pending_files > 0 && data.progress_pct < 100) fillEl.classList.add('kb-progress-bar__fill--partial');
    }

    // Stats
    if(!full) return _renderKbSyncState(data);
    var statMap = {
      'kb-stat-total': data.total_files,
      'kb-stat-embedded': data.embedded_files,
      'kb-stat-pending': data.pending_files,
      'kb-stat-chunks': data.total_chunks,
      'kb-stat-failed': data.failed_files,
      'kb-stat-folders': data.folders ? data.folders.length : 0,
    };
    for(var key in statMap){
      var el = _q(key);
      if(el) el.textContent = statMap[key] != null ? statMap[key] : '--';
    }

    // Folder list
    var listEl = _q('kb-folder-list');
    if(listEl){
      if(data.folders && !data.folders.length){
        listEl.innerHTML = '<div style="padding:12px;text-align:center;font-size:11px;color:var(--text-tertiary)">No folders found. Add documents to your corpus to get started.</div>';
      } else {
        var html = '';
        for(var i = 0; i < data.folders.length; i++){
          var f = data.folders[i];
          var dotClass = 'kb-folder-item__dot--' + f.status;
          if(data.sync_running && f.status !== 'complete') dotClass = 'kb-folder-item__dot--syncing';
          var icon = f.status === 'complete' ? '\\u2705' : (f.status === 'partial' ? '\\u23f3' : '\\u2b55');
          var statsText = f.embedded + '/' + f.total_files + ' embedded';
          html += '<div class="kb-folder-item" title="' + _esc(f.rel_path) + '">' +
            '<span class="kb-folder-item__dot ' + dotClass + '"></span>' +
            '<span class="kb-folder-item__icon">' + icon + '</span>' +
            '<span class="kb-folder-item__name">' + _esc(f.name || f.rel_path) + '</span>' +
            '<span class="kb-folder-item__stats">' + statsText + '</span>' +
            '</div>';
        }
        listEl.innerHTML = html;
      }
    }

    // Last sync
    var lastSyncEl = _q('kb-last-sync');
    if(lastSyncEl && data.last_sync != null){
      if(data.last_sync){
        lastSyncEl.textContent = 'Last sync: ' + data.last_sync;
      } else {
        lastSyncEl.textContent = 'Never synced';
      }
    }

    // Error banner from the last sync, if any
    var errEl = _q('kb-sync-error');
    if(errEl){
      if(data.last_error){
        errEl.textContent = 'Last sync failed: ' + data.last_error;
        errEl.style.display = '';
      } else {
        errEl.style.display = 'none';
      }
    }

    return _renderKbSyncState(data);
  }

  function _renderKbSyncState(data){
    // Sync indicator
    var indicatorEl = _q('kb-sync-indicator');
    if(indicatorEl) indicatorEl.style.display = data.sync_running ? '' : 'none';

    // Sync button state
    var syncBtn = _q('kb-sync-btn');
    if(syncBtn){
      if(data.sync_running){
        syncBtn.setAttribute('disabled', '');
        syncBtn.innerHTML = '\\u23f3 Syncing...';
      } else {
        syncBtn.removeAttribute('disabled');
        syncBtn.innerHTML = '\\u26a1 Sync Now';
      }
    }
  }

  function _fetchKbStatus(){
    _api('GET', '/api/indexing/status')
      .then(_renderKbStatus)
      .catch(function(){});
  }

  function _triggerKbSync(){
    _api('POST', '/api/indexing/sync', {})
      .then(function(resp){
        if(resp.ok){
          _toast('Sync started — processing new/modified files', 'ok');
          _renderKbSyncState({sync_running: true});
          _fetchKbStatus();
          // Poll for completion
          _startSyncPolling();
        } else {
          _toast(resp.error || 'Sync failed', 'err');
        }
      })
      .catch(function(){ _toast('Network error', 'err'); });
  }

  function _startSyncPolling(){
    if(_kbSyncPollTimer) clearInterval(_kbSyncPollTimer);
    var pollsLeft = 600; // 30 min cap — never poll forever
    _kbSyncPollTimer = setInterval(function(){
      if(pollsLeft-- <= 0){
        clearInterval(_kbSyncPollTimer);
        _kbSyncPollTimer = null;
        _toast('Sync is taking unusually long — refresh to check status', 'err');
        return;
      }
      _api('GET', '/api/indexing/status')
        .then(function(data){
          _renderKbStatus(data);
          if(!data.sync_running){
            clearInterval(_kbSyncPollTimer);
            _kbSyncPollTimer = null;
            if(data.last_error){
              _toast('Sync failed: ' + data.last_error, 'err');
            } else if(data.pending_files === 0 && data.failed_files === 0){
              _toast('Sync complete — all files indexed', 'ok');
            } else if(data.failed_files > 0){
              _toast('Sync finished — ' + data.failed_files + ' files failed', 'err');
            } else {
              _toast('Sync complete — ' + data.pending_files + ' files still pending', 'ok');
            }
          }
        })
        .catch(function(){});
    }, 3000);
  }

  // Wire KB status buttons
  var kbRefreshBtn = _q('kb-refresh-btn');
  if(kbRefreshBtn) kbRefreshBtn.addEventListener('click', _fetchKbStatus);
  var kbSyncBtn = _q('kb-sync-btn');
  if(kbSyncBtn) kbSyncBtn.addEventListener('click', _triggerKbSync);

  // Auto-fetch KB status when settings opens
  var _origOpenSettings = _openSettings;
  _openSettings = function(){
    _origOpenSettings();
    _fetchKbStatus();
    // Start auto-refresh while settings is open
    if(_kbRefreshTimer) clearInterval(_kbRefreshTimer);
    _kbRefreshTimer = setInterval(_fetchKbStatus, 10000);
  };
  var _origCloseSettings = _closeSettings;
  _closeSettings = function(){
    _origCloseSettings();
    if(_kbRefreshTimer){ clearInterval(_kbRefreshTimer); _kbRefreshTimer = null; }
  };
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
