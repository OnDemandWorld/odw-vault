"""Phase 9: Document summarization.

Summarize extracted documents using a local LLM (Ollama). Each extraction
with sufficient text is passed to the configured summarization model, and the
result is stored in the `summary` table.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import ollama
from tenacity import retry, stop_after_attempt, wait_exponential

from pipeline.db import finish_model_run, start_model_run
from pipeline.helpers import record_failure

logger = logging.getLogger(__name__)

MAX_PROMPT_CHARS = 8000

SUMMARIZATION_PROMPT = """\
Summarize the following document excerpt in 3-5 concise paragraphs.
Focus on key facts, figures, decisions, and outcomes.
Use the same language as the document.

Document:
{text}

Summary:
"""


def _build_prompt(extracted_text: str) -> str:
    """Build summarization prompt, truncating long excerpts."""
    truncated = extracted_text[:MAX_PROMPT_CHARS]
    return SUMMARIZATION_PROMPT.format(text=truncated)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=30))
def _call_ollama(prompt: str, model: str, host: str, temperature: float, max_tokens: int) -> str:
    """Call Ollama and return stripped summary text."""
    client = ollama.Client(host=host)
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful summarization assistant."},
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


def _has_existing_summary(db, file_id: int, model: str, prompt_version: str) -> bool:
    """Check whether a summary already exists for this (file_id, model, prompt_version)."""
    rows = list(
        db.query(
            "SELECT 1 FROM summary WHERE file_id = ? AND model = ?",
            [file_id, model],
        )
    )
    return len(rows) > 0


def run_summarize(db, cfg, limit=None, resummarize=False) -> tuple[int, int]:
    """Summarize extracted documents. Returns (processed, failed) counts."""
    summarization = cfg.models.summarization
    model = summarization.name
    temperature = summarization.temperature
    max_tokens = summarization.max_tokens
    prompt_version = summarization.prompt_version
    size_threshold = cfg.extract.size_threshold_for_summary

    # Get extractions that succeeded and meet the size threshold
    rows = list(
        db.query(
            """
            SELECT e.id AS extraction_id, e.file_id, e.text_extracted, e.char_count
            FROM extraction e
            JOIN file f ON f.id = e.file_id
            WHERE e.succeeded = 1
              AND e.char_count >= ?
              AND f.is_dup_primary = 1
              AND f.excluded = 0
            ORDER BY e.file_id
            """,
            [size_threshold],
        )
    )

    if not rows:
        logger.info("No extractions eligible for summarization.")
        return (0, 0)

    # Apply limit
    if limit is not None:
        rows = rows[:limit]

    run_id = start_model_run(
        db,
        role="summarization",
        model_name=model,
        config_hash=f"{model}-{prompt_version}-t{temperature}-mt{max_tokens}",
        phase="summarize",
    )

    processed = 0
    failed = 0
    skipped = 0

    for row in rows:
        file_id = row["file_id"]
        text = row["text_extracted"]

        # Skip if summary already exists and we're not re-summarizing
        if not resummarize and _has_existing_summary(db, file_id, model, prompt_version):
            skipped += 1
            continue

        try:
            prompt = _build_prompt(text)
            summary_text = _call_ollama(prompt, model, cfg.ollama.host, temperature, max_tokens)

            # Validate: non-empty and under max_tokens (rough char check)
            if not summary_text.strip():
                raise ValueError("Summary text is empty after generation")

            db["summary"].insert(
                {
                    "file_id": file_id,
                    "model": model,
                    "summary_text": summary_text,
                    "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            )

            processed += 1
            logger.debug(
                "Summarized file_id=%d (%d chars -> %d chars)",
                file_id,
                row["char_count"],
                len(summary_text),
            )

        except Exception as e:
            record_failure(
                db,
                file_id=file_id,
                phase="summarize",
                tool=model,
                error_class=type(e).__name__,
                error_message=str(e),
            )
            failed += 1
            logger.warning("Failed to summarize file_id=%d: %s", file_id, e)

    finish_model_run(
        db,
        run_id,
        status="done",
        items_processed=processed,
        items_failed=failed,
        notes=f"skipped={skipped}" if skipped else None,
    )

    logger.info(
        "Summarization complete: %d processed, %d failed, %d skipped",
        processed,
        failed,
        skipped,
    )
    return (processed, failed)
