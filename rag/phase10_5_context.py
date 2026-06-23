"""Phase 10.5: Contextual retrieval augmentation.

For each chunk, generates a short context sentence that situates the chunk
within its parent document. Uses a local LLM (Ollama) or an OpenAI-compatible
API endpoint with a prompt template from ``prompts/contextual_retrieval_v1.txt``.
Results are stored in the ``chunk`` table's context columns. Idempotent: skips
chunks that already have a context for the same prompt hash.

Post-processing validation:
After context generation, a validation pass checks for quality issues discovered
during production runs (see CONTEXT_QUALITY_AUDIT.md).  The validation pass:

1. Strips leaked `` reasoning blocks (Qwen3 models sometimes ignore the
   ``thinking: false`` flag or produce malformed closing tags).
2. Clears contexts shorter than 50 characters for reprocessing — these typically
   come from chunks with insufficient text or files without summaries, where the
   model had nothing meaningful to augment.  Re-running gives the next model
   attempt a chance to produce richer output.
"""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import ollama
from tenacity import retry, stop_after_attempt, wait_exponential

from pipeline.db import finish_model_run, heartbeat_model_run, start_model_run
from pipeline.helpers import record_failure

logger = logging.getLogger(__name__)

# Load prompt template at module import time.
_PROMPTS_DIR = Path(__file__).parent / "prompts"
_PROMPT_TEMPLATE_PATH = _PROMPTS_DIR / "contextual_retrieval_v1.txt"

_PROMPT_TEMPLATE = _PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")

# If the full extraction text exceeds this, use the summary instead.
_MAX_DOC_CHARS = 8000


def _build_prompt(document_text: str, chunk_text: str) -> tuple[str, str]:
    """Build the contextual retrieval prompt and return (prompt, prompt_hash)."""
    prompt = _PROMPT_TEMPLATE.format(
        document_text_or_summary=document_text,
        chunk_text=chunk_text,
    )
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return prompt, prompt_hash


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=30))
def _call_ollama(
    prompt: str,
    model: str,
    host: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Call Ollama or OpenAI-compatible endpoint and return stripped context text."""
    # Detect OpenAI-compatible endpoint by /v1 in the host URL
    if "/v1" in host:
        return _call_openai_compat(prompt, model, host, temperature, max_tokens)
    return _call_ollama_native(prompt, model, host, temperature, max_tokens)


def _call_openai_compat(
    prompt: str,
    model: str,
    host: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Call an OpenAI-compatible API endpoint."""
    url = f"{host}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful research assistant providing concise document context."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "thinking": False,
    }
    headers = {"Content-Type": "application/json", "Authorization": "Bearer not-needed"}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read())
    text = result["choices"][0]["message"]["content"].strip()
    # Strip reasoning tags if present (Qwen-style <think>...</think>)
    import re
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # Also catch malformed variants (malformed closing tag)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL).strip()
    if not text:
        raise ValueError("OpenAI-compatible endpoint returned empty response")
    return text


def _call_ollama_native(
    prompt: str,
    model: str,
    host: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Call Ollama native API."""
    client = ollama.Client(host=host)
    response = client.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful research assistant providing concise document context.",
            },
            {"role": "user", "content": prompt},
        ],
        options={
            "temperature": temperature,
            "num_ctx": 16384,
        },
    )
    text = response.get("message", {}).get("content", "").strip()
    if not text:
        raise ValueError("Ollama returned empty response")
    return text


def run_context(
    db,
    cfg,
    limit: int | None = None,
    regenerate: bool = False,
) -> tuple[int, int]:
    """Contextual retrieval augmentation.

    Returns (processed, failed) counts.
    """
    ctx_cfg = cfg.models.contextual_retrieval

    if not ctx_cfg.enabled:
        logger.info("Contextual retrieval is disabled in config; skipping.")
        return (0, 0)

    model = ctx_cfg.name
    temperature = ctx_cfg.temperature
    max_context_tokens = ctx_cfg.max_context_tokens
    prompt_version = ctx_cfg.prompt_version
    host = cfg.ollama.host

    # Select chunks that need context
    if regenerate:
        query = """
            SELECT c.id AS chunk_id,
                   c.file_id,
                   c.text AS chunk_text,
                   c.chunk_index,
                   f.name AS file_name,
                   e.text_extracted,
                   e.char_count,
                   s.summary_text
            FROM chunk c
            JOIN file f ON f.id = c.file_id
            LEFT JOIN extraction e ON e.file_id = f.id AND e.succeeded = 1
            LEFT JOIN summary s ON s.file_id = f.id
            WHERE f.is_dup_primary = 1 AND f.excluded = 0
            ORDER BY c.id
        """
    else:
        query = """
            SELECT c.id AS chunk_id,
                   c.file_id,
                   c.text AS chunk_text,
                   c.chunk_index,
                   f.name AS file_name,
                   e.text_extracted,
                   e.char_count,
                   s.summary_text
            FROM chunk c
            JOIN file f ON f.id = c.file_id
            LEFT JOIN extraction e ON e.file_id = f.id AND e.succeeded = 1
            LEFT JOIN summary s ON s.file_id = f.id
            WHERE c.context_text IS NULL
              AND f.is_dup_primary = 1 AND f.excluded = 0
            ORDER BY c.id
        """

    rows = list(db.query(query))

    if not rows:
        logger.info("No chunks need contextual augmentation.")
        return (0, 0)

    if limit is not None:
        rows = rows[:limit]

    run_id = start_model_run(
        db,
        role="contextual_augmentation",
        model_name=model,
        config_hash=f"{model}-{prompt_version}-t{temperature}",
        phase="context",
    )

    processed = 0
    failed = 0
    skipped = 0

    for row in rows:
        chunk_id = row["chunk_id"]
        chunk_text = row["chunk_text"]
        file_id = row["file_id"]
        file_name = row["file_name"]

        # Build document text: use summary if extraction is long, otherwise full text
        extraction_text = row["text_extracted"] or ""
        summary_text = row["summary_text"] or ""

        if len(extraction_text) > _MAX_DOC_CHARS and summary_text:
            doc_text = summary_text
        elif extraction_text:
            doc_text = extraction_text[:_MAX_DOC_CHARS]
        else:
            # No extraction available; use filename as minimal context
            doc_text = f"Document: {file_name}"

        prompt, prompt_hash = _build_prompt(doc_text, chunk_text)

        # Idempotency: check if context already exists for this prompt_hash
        if not regenerate:
            existing = list(
                db.query(
                    "SELECT 1 FROM chunk WHERE id = ? AND context_prompt_hash = ?",
                    [chunk_id, prompt_hash],
                )
            )
            if existing:
                skipped += 1
                continue

        try:
            context_text = _call_ollama(prompt, model, host, temperature, max_context_tokens)

            db["chunk"].update(
                chunk_id,
                {
                    "context_text": context_text,
                    "context_model": model,
                    "context_prompt_hash": prompt_hash,
                    "context_generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                },
            )
            db.conn.commit()

            processed += 1
            logger.debug(
                "Context generated for chunk_id=%d (file_id=%d, chunk_index=%d)",
                chunk_id,
                file_id,
                row["chunk_index"],
            )

        except Exception as e:
            record_failure(
                db,
                file_id=file_id,
                phase="context",
                tool=model,
                error_class=type(e).__name__,
                error_message=str(e),
            )
            failed += 1
            logger.warning("Failed to generate context for chunk_id=%d: %s", chunk_id, e)

        # Heartbeat every 50 chunks
        if (processed + failed + skipped) % 50 == 0:
            heartbeat_model_run(db, run_id, processed, failed)

    finish_model_run(
        db,
        run_id,
        status="done",
        items_processed=processed,
        items_failed=failed,
        notes=f"skipped={skipped}" if skipped else None,
    )

    logger.info(
        "Contextual augmentation complete: %d processed, %d failed, %d skipped",
        processed,
        failed,
        skipped,
    )
    return (processed, failed)


def validate_and_clean(db, min_context_chars: int = 50) -> dict:
    """Post-processing quality pass on all context entries.

    After context generation, this function detects and fixes known quality
    issues documented in the production audit (see CONTEXT_QUALITY_AUDIT.md):

    1. **Leaked `` reasoning blocks** — Qwen3 models sometimes ignore
       the ``thinking: false`` flag or produce malformed closing tags
       (e.g. ```` without the ``>``).  The `` block is stripped,
       leaving only the final context sentence.  If nothing remains, the
       entry is cleared for reprocessing.

    2. **Very short contexts** (< 50 chars) — These are cleared so that a
       subsequent re-run can attempt to produce richer output.  Common causes:
       chunks with only a few characters of source text, or files without
       summaries (the model had no document-level context to work with).
       After 2-3 re-runs, most improve; the remainder are genuinely
       unaugmentable (e.g. copyright footer, single-row table cells).

    Returns a dict with counts of each issue type found and fixed.
    """
    import re as _re

    rows = list(db.query(
        "SELECT id, context_text FROM chunk WHERE context_text IS NOT NULL AND context_text != ''"
    ))

    thinking_leaked = 0
    thinking_malformed = 0
    too_short = 0
    fixed = 0

    for row in rows:
        cid = row["id"]
        text = row["context_text"]
        original = text

        # Strip well-formed <think>...</think> blocks
        new_text = _re.sub(r'<think>.*?</think>', '', text, flags=_re.DOTALL).strip()
        if new_text != text:
            thinking_leaked += 1
            text = new_text

        # Strip malformed reasoning ( starts but never closes properly)
        new_text = _re.sub(r'<think>.*$', '', text, flags=_re.DOTALL).strip()
        if new_text != text:
            thinking_malformed += 1
            text = new_text

        # If after stripping nothing meaningful remains, clear for reprocessing
        if not text:
            db["chunk"].update(cid, {
                "context_text": None,
                "context_model": None,
                "context_prompt_hash": None,
                "context_generated_at": None,
            })
            fixed += 1
            continue

        # Clear very-short contexts for reprocessing
        if len(text) < min_context_chars:
            db["chunk"].update(cid, {
                "context_text": None,
                "context_model": None,
                "context_prompt_hash": None,
                "context_generated_at": None,
            })
            too_short += 1
            fixed += 1
            continue

        # Update with cleaned text if it changed
        if text != original:
            db["chunk"].update(cid, {"context_text": text})
            fixed += 1

    if thinking_leaked or thinking_malformed or too_short:
        db.conn.commit()

    total_with_context = next(iter(db.query(
        "SELECT COUNT(*) AS c FROM chunk WHERE context_text IS NOT NULL AND context_text != ''"
    )))["c"]

    logger.info(
        "Context validation: thinking_leaked=%d, thinking_malformed=%d, "
        "too_short=%d, cleared=%d, remaining=%d",
        thinking_leaked, thinking_malformed, too_short, fixed, total_with_context,
    )

    return {
        "thinking_leaked": thinking_leaked,
        "thinking_malformed": thinking_malformed,
        "too_short": too_short,
        "cleared_for_rerun": fixed,
        "remaining_with_context": total_with_context,
    }
