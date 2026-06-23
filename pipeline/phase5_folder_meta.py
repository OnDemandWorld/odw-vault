"""Phase 5: Folder semantic inference using local LLM (Ollama).

Bottom-up traversal: for each folder, builds a prompt from context
(path, parent labels, child filenames, format histogram), calls Ollama
with structured JSON output, validates via Pydantic. Cached by prompt hash.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, Field
from rich.progress import Progress, SpinnerColumn, TextColumn
from tenacity import retry, stop_after_attempt, wait_exponential

from pipeline.config import Config
from pipeline.helpers import now_iso, record_failure
from pipeline.logging import PhaseLogger


class FolderInference(BaseModel):
    category: Literal[
        "client-project",
        "internal-rnd",
        "vendor-docs",
        "admin-finance",
        "templates",
        "archive-historical",
        "personal",
        "unclear",
        "engineering",
        "training",
        "product-design",
        "test-operations",
    ]
    label: str = Field(max_length=120, description="Short human-readable label")
    tags: list[str] = Field(max_length=8, description="Up to 8 tags")
    summary: str = Field(max_length=500, description="1-3 sentence summary")


PROMPT_TEMPLATE = """You analyze folder structures in a company knowledge base.
The corpus contains mixed documents in English and Chinese (both Traditional and Simplified).

Folder path: {rel_path}
Parent labels: {parent_labels}
Number of files: {file_count}
File-type histogram: {format_histogram}
Sample filenames (up to 30):
{filenames}

Respond in JSON matching this schema exactly:
{{"category": one of [client-project, internal-rnd, vendor-docs, admin-finance, templates, archive-historical, personal, unclear, engineering, training, product-design, test-operations],
  "label": "short human-readable label, max 120 chars",
  "tags": ["up to 8 short tags"],
  "summary": "1-3 sentences describing what this folder contains, max 500 chars"}}

Only return valid JSON. No markdown fences, no explanations."""


def _build_prompt(folder_row, db, config: Config) -> tuple[str, str]:
    """Build the inference prompt and compute its hash."""
    # Get parent labels (bottom-up: children are already processed)
    parents = list(
        db.query(
            """
        WITH RECURSIVE chain(id, parent_id, label) AS (
            SELECT id, parent_id, inferred_label FROM folder WHERE id = ?
            UNION ALL
            SELECT f.id, f.parent_id, f.inferred_label FROM folder f JOIN chain c ON f.id = c.parent_id
        ) SELECT label FROM chain WHERE label IS NOT NULL
    """,
            [folder_row["id"]],
        )
    )
    parent_labels = " > ".join(p["label"] for p in reversed(parents)) or "(root)"

    # Child filenames
    files = list(
        db.query(
            "SELECT name, category FROM file WHERE folder_id = ? AND excluded = 0 LIMIT ?",
            [folder_row["id"], config.folder_meta.max_filenames_in_prompt],
        )
    )

    # File-type histogram
    histogram = {}
    for f in db.query(
        "SELECT category, COUNT(*) c FROM file WHERE folder_id=? AND excluded=0 GROUP BY category",
        [folder_row["id"]],
    ):
        histogram[f["category"] or "unknown"] = f["c"]

    filenames_str = "\n".join(f"- {f['name']}" for f in files) if files else "(empty folder)"

    prompt = PROMPT_TEMPLATE.format(
        rel_path=folder_row["rel_path"],
        parent_labels=parent_labels,
        file_count=folder_row.get("file_count", 0) or 0,
        format_histogram=json.dumps(histogram, ensure_ascii=False),
        filenames=filenames_str,
    )
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    return prompt, prompt_hash


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=30))
def _call_ollama(prompt: str, model: str, host: str) -> FolderInference:
    import ollama

    client = ollama.Client(host=host)
    response = client.generate(
        model=model,
        prompt=prompt,
        format="json",
        options={"temperature": 0.2, "num_ctx": 8192},
    )
    raw_text = response["response"].strip()
    # Strip markdown fences if present
    if raw_text.startswith("```"):
        lines = raw_text.split("\n")
        lines = [line for line in lines if not line.startswith("```")]
        raw_text = "\n".join(lines).strip()
    data = json.loads(raw_text)
    return FolderInference(**data)


def run_phase5(
    db,
    config: Config,
    plog: PhaseLogger,
    model: str | None = None,
    reinfer: bool = False,
    max_folders: int | None = None,
) -> dict:
    """Run folder semantic inference."""
    ollama_model = model or config.ollama.model
    host = config.ollama.host

    # Get folders bottom-up (depth descending), excluding extracted archives and excluded.
    # Skip folders that already have a matching prompt hash unless reinfer=True.
    folders = list(
        db.query("""
        SELECT * FROM folder WHERE excluded=0
            AND (inferred_category IS NULL OR inference_prompt_hash IS NULL)
        ORDER BY depth DESC, id ASC
    """)
    )

    if max_folders:
        folders = folders[:max_folders]

    if not folders:
        plog.info("No folders to infer.")
        return {"files_processed": 0, "files_failed": 0}

    inferred = 0
    skipped_cache = 0
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task(f"Inferring {len(folders)} folders...", total=len(folders))

        for folder in folders:
            prompt, prompt_hash = _build_prompt(folder, db, config)

            # Check cache
            if not reinfer and folder.get("inference_prompt_hash") == prompt_hash:
                skipped_cache += 1
                progress.update(task, advance=1)
                continue

            try:
                result = _call_ollama(prompt, ollama_model, host)
                db["folder"].update(
                    folder["id"],
                    {
                        "inferred_category": result.category,
                        "inferred_label": result.label,
                        "inferred_tags_json": json.dumps(result.tags, ensure_ascii=False),
                        "inferred_summary": result.summary,
                        "inference_model": ollama_model,
                        "inference_prompt_hash": prompt_hash,
                        "inferred_at": now_iso(),
                    },
                )
                inferred += 1
                plog.debug(f"Inferred: {folder['rel_path']} -> {result.category}/{result.label}")
            except Exception as e:
                record_failure(
                    db,
                    folder_id=folder["id"],
                    phase="folder_meta",
                    tool=ollama_model,
                    error_class="parse_error",
                    error_message=str(e),
                )
                failed += 1
                plog.warning(f"Failed to infer {folder['rel_path']}: {e}")

            progress.update(task, advance=1)

    plog.info(
        f"Folder inference complete: {inferred} inferred, {skipped_cache} cached, {failed} failed",
        inferred=inferred,
        cached=skipped_cache,
        failed=failed,
    )
    return {"files_processed": inferred, "files_failed": failed}
