"""Phase 14: Gradio chat UI for the RAG pipeline."""

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

# ---------------------------------------------------------------------------
# Global state (populated on launch)
# ---------------------------------------------------------------------------
_cfg = None
_db_path: Path | None = None
_chroma_path = ""
_ollama_host = "http://localhost:11434"

# Thread-local DB connections
_thread_local = threading.local()


def _get_db():
    """Return a thread-local DB connection."""
    if not hasattr(_thread_local, "db"):
        from pipeline.db import migrate

        _thread_local.db = open_db(_db_path)
        migrate(_thread_local.db)
    return _thread_local.db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_db(db_path: Path) -> None:
    """Store the DB path for later per-thread connection creation."""
    global _db_path
    _db_path = db_path


def _get_folders() -> list[str]:
    """Return sorted folder rel_paths from the folder table."""
    rows = _get_db().query("SELECT rel_path FROM folder WHERE excluded = 0 ORDER BY rel_path")
    return [r["rel_path"] for r in rows]


def _check_ollama() -> bool:
    """Return True if Ollama is reachable."""
    try:
        client = ollama.Client(host=_ollama_host)
        client.list()
        return True
    except Exception:
        return False


def _check_chroma() -> tuple[bool, str]:
    """Return (ok, collection_name_or_error)."""
    try:
        import chromadb

        client = chromadb.PersistentClient(path=_chroma_path)
        suffix = _cfg.models.embedding.collection_suffix
        coll_name = f"chunks__{suffix}"
        client.get_collection(coll_name)
        return True, coll_name
    except Exception as exc:
        return False, str(exc)[:120]


def _build_status_html() -> str:
    """Return an HTML status badge string."""
    ollama_ok = _check_ollama()
    chroma_ok, chroma_msg = _check_chroma()
    gen_model = _cfg.models.generation.name
    emb_model = _cfg.models.embedding.name

    ollama_dot = "&#128994;" if ollama_ok else "&#128308;"
    chroma_dot = "&#128994;" if chroma_ok else "&#128308;"
    chroma_label = chroma_msg if chroma_ok else "Chroma unavailable"

    return (
        f"<div style='font-family:monospace;font-size:13px;line-height:1.8;'>"
        f"{ollama_dot} Ollama ({_ollama_host}) &mdash; "
        f"{'reachable' if ollama_ok else 'unreachable'}<br>"
        f"{chroma_dot} Chroma &mdash; {chroma_label}<br>"
        f"&#129302; Generation: <b>{gen_model}</b><br>"
        f"&#128268; Embedding: <b>{emb_model}</b>"
        f"</div>"
    )


def _format_chunks_for_prompt(hits: list[Hit]) -> str:
    """Format hits as numbered context blocks for the LLM prompt."""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        page_info = f" (page {hit.page_start})" if hit.page_start else ""
        blocks.append(f"[{i}] {hit.rel_path}{page_info}\n{hit.text}")
    return "\n\n".join(blocks) if blocks else "(no context available)"


def _stream_tokens(
    query: str,
    hits: list[Hit],
):
    """Yield tokens as they arrive from Ollama.

    Each yield is the accumulated text so far.
    After the stream completes, returns (full_text, citations).
    """
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
    client = ollama.Client(host=_ollama_host)

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
    # Yield the final text one more time so the caller can attach citations
    yield full_text, citations


def _citations_html(citations: list[dict]) -> str:
    """Render citations as HTML."""
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
        f"<div style='margin-top:12px;padding:8px 12px;"
        f"background:#f7f7f7;border-radius:6px;font-size:13px;'>"
        f"<b>Sources:</b><ul style='margin:4px 0 0 16px;padding:0;'>"
        f"{''.join(items)}</ul></div>"
    )


# ---------------------------------------------------------------------------
# UI event handlers
# ---------------------------------------------------------------------------


def _on_chat(
    message: str,
    history: list[dict],
    folder_filter: str,
):
    """Generator that streams chat responses.

    Yields (updated_history, citations_html) tuples.
    """
    if not message or not message.strip():
        yield history, _citations_html([])
        return

    # Append user message
    history = [*history, {"role": "user", "content": message.strip()}]
    yield history, _citations_html([])

    # Resolve folder filter
    filter_spec = None
    if folder_filter and folder_filter != "All folders":
        filter_spec = {"path_prefix": folder_filter}

    # Retrieve
    try:
        hits, _metrics = retrieve(
            query=message.strip(),
            db=_get_db(),
            chroma_client=None,  # retrieve creates its own client
            chroma_path=_chroma_path,
            cfg=_cfg,
            folder_filter=filter_spec,
        )
    except Exception as exc:
        error_msg = f"Retrieval failed: {exc}"
        history.append({"role": "assistant", "content": error_msg})
        yield history, _citations_html([])
        return

    if not hits:
        history.append(
            {
                "role": "assistant",
                "content": "No relevant context found for your question.",
            }
        )
        yield history, _citations_html([])
        return

    # Add empty assistant message that will be streamed into
    history.append({"role": "assistant", "content": ""})
    yield history, _citations_html([])

    # Stream tokens from Ollama
    citations = []
    token_gen = _stream_tokens(message.strip(), hits)

    last_yielded = None
    for result in token_gen:
        if isinstance(result, tuple):
            # Final yield: (full_text, citations)
            full_text, citations = result
        else:
            # Intermediate yield: accumulated text
            full_text = result

        history[-1]["content"] = full_text
        last_yielded = full_text  # noqa: F841
        yield history, _citations_html([])

    # Final yield with citations
    yield history, _citations_html(citations)


def _on_folder_change(folder_filter: str) -> str:
    """Show a brief status message when folder filter changes."""
    if folder_filter == "All folders":
        return "Searching all folders."
    return f"Scoped to: {folder_filter}"


def _on_feedback(
    history: list[dict],
    feedback_data: gr.LikeData,
) -> None:
    """Record feedback in the query_log table."""
    if not history or feedback_data.index >= len(history):
        return

    msg = history[feedback_data.index]
    if msg.get("role") != "assistant":
        return

    # Find the most recent user query before this assistant message
    query_text = ""
    for i in range(feedback_data.index - 1, -1, -1):
        if history[i].get("role") == "user":
            query_text = history[i].get("content", "")
            break

    feedback_value = "up" if feedback_data.liked else "down"
    try:
        _get_db().conn.execute(
            "UPDATE query_log SET feedback = ?, feedback_at = datetime('now') "
            "WHERE query_text = ? AND feedback IS NULL "
            "ORDER BY asked_at DESC LIMIT 1",
            (feedback_value, query_text),
        )
        _get_db().conn.commit()
    except Exception:
        logger.warning("Failed to record feedback", exc_info=True)


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------


def launch_ui(cfg, share: bool = False, server_name: str = "127.0.0.1", server_port: int = 7860):
    """Launch the Gradio chat interface."""
    global _cfg, _chroma_path, _ollama_host

    _cfg = cfg
    _ollama_host = getattr(cfg.ollama, "host", "http://localhost:11434")
    _chroma_path = cfg.paths.chroma_root

    # Open and migrate DB
    db_path = Path(cfg.paths.corpus_root) / ".rag-cache" / "corpus.db"
    if not db_path.exists():
        db_path = Path("corpus.db")
    _ensure_db(db_path)

    # Build folder list for dropdown
    folders = _get_folders()
    folder_choices = ["All folders", *folders]

    with gr.Blocks(
        title="ODW.ai Vault Knowledge Assistant",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown("# ODW.ai Vault Knowledge Assistant")

        # Status badge
        status_html = _build_status_html()
        gr.HTML(value=status_html)

        # Folder filter
        folder_filter = gr.Dropdown(
            choices=folder_choices,
            value="All folders",
            label="Folder filter",
            interactive=True,
        )
        filter_status = gr.Markdown(value="Searching all folders.")

        # Chat interface
        chatbot = gr.Chatbot(
            label="Chat",
            height=500,
        )
        msg_input = gr.Textbox(
            label="Your question",
            placeholder="Ask something about the knowledge base...",
            lines=2,
        )

        # Citations panel
        citations_panel = gr.HTML(value="")

        # Buttons
        with gr.Row():
            submit_btn = gr.Button(value="Send", variant="primary")
            clear_btn = gr.Button(value="Clear chat", variant="stop")

        # -------------------------------------------------------------------
        # Event wiring
        # -------------------------------------------------------------------

        # Submit via button or Enter key
        submit_btn.click(
            fn=_on_chat,
            inputs=[msg_input, chatbot, folder_filter],
            outputs=[chatbot, citations_panel],
        )
        msg_input.submit(
            fn=_on_chat,
            inputs=[msg_input, chatbot, folder_filter],
            outputs=[chatbot, citations_panel],
        )

        # Clear chat
        clear_btn.click(
            fn=lambda: ([], ""),
            inputs=[],
            outputs=[chatbot, citations_panel],
        )

        # Folder filter change
        folder_filter.change(
            fn=_on_folder_change,
            inputs=[folder_filter],
            outputs=[filter_status],
        )

        # Feedback (thumbs up/down on assistant messages)
        chatbot.like(
            fn=_on_feedback,
            inputs=[chatbot, chatbot],
            outputs=[],
        )

    # Launch
    app.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
    )


if __name__ == "__main__":
    config_path = Path(__file__).resolve().parent.parent / "config.toml"
    cfg = load_app_config(config_path)
    launch_ui(cfg)
