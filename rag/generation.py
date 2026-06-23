"""Citation-strict answer generation via Ollama."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import ollama
from tenacity import retry, stop_after_attempt, wait_exponential

from rag.citations import parse_citations, resolve_citations
from rag.retrieval import Hit

CITATION_RE = re.compile(r"\[(\d+)\]")

DEFAULT_PROMPT = """\
You are the noveltebot internal knowledge assistant. You help staff find
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

REFUSAL_TEXT = (
    "I do not have enough information in the provided context to answer "
    "this question. Please rephrase or provide additional context."
)


def generate_answer(
    query: str,
    hits: list[Hit],
    cfg,
    prompt_template: str | None = None,
) -> dict:
    """Generate an answer with citations.

    Returns dict with answer, citations, generation_ms, model.
    """
    t0 = time.monotonic()

    # 1. Refuse on empty context if configured
    if cfg.generation_runtime.refuse_on_empty_context and not hits:
        return {
            "answer": REFUSAL_TEXT,
            "citations": [],
            "generation_ms": round((time.monotonic() - t0) * 1000, 1),
            "model": cfg.models.generation.name,
            "refused": True,
        }

    # 2. Assemble context from hits: numbered blocks
    numbered_chunks = _format_chunks(hits)

    # 3. Load prompt template
    template = _load_prompt(prompt_template, cfg)

    # 4. Format prompt
    prompt = template.format(numbered_chunks=numbered_chunks, query=query)

    # 5. Prepend thinking marker if enabled
    system_prefix = ""
    if getattr(cfg.models.generation, "thinking", False):
        system_prefix = "<|think|>"

    # 6. Call Ollama chat
    model_name = cfg.models.generation.name
    client = _make_client(cfg)

    response = _ollama_chat(
        client=client,
        model=model_name,
        prompt=prompt,
        system_prefix=system_prefix,
        temperature=cfg.models.generation.temperature,
        top_p=cfg.models.generation.top_p,
        top_k=cfg.models.generation.top_k,
    )

    answer_text = response.get("message", {}).get("content", "")
    if not answer_text:
        answer_text = REFUSAL_TEXT

    # 7. Parse citation markers
    citation_numbers = parse_citations(answer_text)

    # 8. Resolve citations to hit metadata
    citations = resolve_citations(citation_numbers, hits)

    elapsed = (time.monotonic() - t0) * 1000

    return {
        "answer": answer_text,
        "citations": citations,
        "generation_ms": round(elapsed, 1),
        "model": model_name,
        "refused": False,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _format_chunks(hits: list[Hit]) -> str:
    """Format hits as numbered context blocks."""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        page_info = f" (page {hit.page_start})" if hit.page_start else ""
        blocks.append(f"[{i}] {hit.rel_path}{page_info}\n{hit.text}")
    return "\n\n".join(blocks) if blocks else "(no context available)"


def _load_prompt(prompt_template: str | None, cfg) -> str:
    """Load prompt from file or use default."""
    if prompt_template and os.path.isfile(prompt_template):
        return Path(prompt_template).read_text(encoding="utf-8")

    # Try prompts/generation_v{N}.txt relative to project root
    version = getattr(cfg.models.generation, "prompt_version", "v1")
    prompt_path = Path(__file__).resolve().parent.parent / "prompts" / f"generation_{version}.txt"
    if prompt_path.is_file():
        return prompt_path.read_text(encoding="utf-8")

    return DEFAULT_PROMPT


def _make_client(cfg):
    """Create an ollama client from the generation endpoint config."""
    ep = getattr(cfg.models.generation, "endpoint", None)
    if ep:
        kwargs = {"host": ep.host}
        if getattr(ep, "api_key", ""):
            kwargs["headers"] = {"Authorization": f"Bearer {ep.api_key}"}
        return ollama.Client(**kwargs)
    return ollama.Client(host=getattr(cfg.ollama, "host", "http://localhost:11434"))


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def _ollama_chat(
    client,
    model: str,
    prompt: str,
    system_prefix: str = "",
    temperature: float = 0.5,
    top_p: float = 0.95,
    top_k: int = 64,
) -> dict:
    """Call Ollama chat endpoint with retry logic."""
    system_content = "You are a helpful assistant."
    if system_prefix:
        system_content = f"{system_prefix}\n{system_content}"

    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ],
        options={
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        },
    )
    return response
