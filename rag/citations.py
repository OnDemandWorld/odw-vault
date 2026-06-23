"""Citation parsing and resolution utilities."""

from __future__ import annotations

import re

from rag.retrieval import Hit

CITATION_RE = re.compile(r"\[(\d+)\]")


def parse_citations(answer_text: str) -> list[int]:
    """Extract citation numbers [N] from answer text, preserving order, deduplicated."""
    seen: set[int] = set()
    result: list[int] = []
    for m in CITATION_RE.finditer(answer_text):
        n = int(m.group(1))
        if n not in seen:
            seen.add(n)
            result.append(n)
    return result


def resolve_citations(citation_numbers: list[int], hits: list[Hit]) -> list[dict]:
    """Resolve citation numbers to hit metadata.

    Citation number N refers to the N-th hit (1-based) in the hits list.
    Returns list of dicts with file_id, rel_path, page, chunk_id, snippet.
    """
    resolved: list[dict] = []
    for n in citation_numbers:
        idx = n - 1  # 1-based to 0-based
        if 0 <= idx < len(hits):
            hit = hits[idx]
            resolved.append(
                {
                    "citation_number": n,
                    "chunk_id": hit.chunk_id,
                    "file_id": hit.file_id,
                    "rel_path": hit.rel_path,
                    "page_start": hit.page_start,
                    "snippet": _truncate(hit.text, 300),
                }
            )
    return resolved


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
