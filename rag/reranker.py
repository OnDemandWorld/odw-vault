"""Cross-encoder reranker for retrieval result refinement.

Uses Ollama-compatible scoring API to re-score (query, document) pairs
and re-rank fused retrieval results for improved precision.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

logger = logging.getLogger(__name__)


def rerank(
    query: str,
    hits: list[Any],
    model_name: str,
    ollama_host: str,
    top_n: int = 8,
    batch_size: int = 8,
    timeout: float = 30.0,
) -> list[Any]:
    """Re-rank hits using a cross-encoder reranker model via Ollama.

    Each hit must have a ``.text`` attribute containing the chunk text.
    The reranker scores each (query, text) pair and returns the top-N
    hits sorted by descending relevance score.

    Args:
        query: The user query string.
        hits: List of Hit objects from the fusion stage.
        model_name: Ollama model name for the reranker.
        ollama_host: Ollama server host URL.
        top_n: Number of top results to return after reranking.
        batch_size: Number of documents to score per API call.
        timeout: Per-request timeout in seconds.

    Returns:
        The top-N hits sorted by rerank_score descending.
    """
    if not hits:
        return []

    import ollama

    client = ollama.Client(host=ollama_host)

    # Build (query, document) pairs
    pairs: list[tuple[int, str]] = []
    for i, hit in enumerate(hits):
        text = hit.text or ""
        # Truncate long texts to avoid context overflow
        if len(text) > 2048:
            text = text[:2048]
        pairs.append((i, text))

    # Score in batches
    all_scores: list[float] = [0.0] * len(pairs)
    n_batches = math.ceil(len(pairs) / batch_size)

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(pairs))
        batch = pairs[start:end]

        batch_texts = [text for _, text in batch]
        scores = _score_batch(client, model_name, query, batch_texts, timeout)

        for j, score in enumerate(scores):
            all_scores[start + j] = score

    # Assign scores to hits and sort
    scored_hits: list[tuple[float, Any]] = []
    for idx, (orig_idx, _) in enumerate(pairs):
        hit = hits[orig_idx]
        hit.rerank_score = all_scores[idx]
        scored_hits.append((all_scores[idx], hit))

    # Sort by score descending
    scored_hits.sort(key=lambda x: x[0], reverse=True)

    # Return top-N
    result = [h for _, h in scored_hits[:top_n]]

    logger.info(
        "Reranked %d candidates → top %d (batches=%d, model=%s)",
        len(hits), len(result), n_batches, model_name,
    )
    return result


def _score_batch(
    client: Any,
    model_name: str,
    query: str,
    documents: list[str],
    timeout: float,
) -> list[float]:
    """Score a batch of (query, document) pairs via Ollama.

    Uses the Ollama /api/embed endpoint with the reranker model.
    Falls back to a generation-based scoring if the embed endpoint
    is not supported by the model.

    Returns a list of relevance scores in [0, 1].
    """
    t0 = time.monotonic()

    try:
        # Try the Ollama embed API — some reranker models support
        # returning relevance scores via the embed endpoint
        response = client.embed(
            model=model_name,
            input=documents,
        )
        embeddings = response.get("embeddings", [])

        if embeddings and len(embeddings) == len(documents):
            # For cross-encoder rerankers, the embedding output is
            # typically a single scalar score per document.
            # Extract the first dimension as the relevance score.
            scores: list[float] = []
            for emb in embeddings:
                if isinstance(emb, list) and len(emb) > 0:
                    # Sigmoid normalization for single-value output
                    raw = emb[0] if len(emb) == 1 else sum(emb) / len(emb)
                    scores.append(_sigmoid(raw))
                else:
                    scores.append(0.0)

            elapsed = (time.monotonic() - t0) * 1000
            logger.debug("Reranker batch scored in %.1fms", elapsed)
            return scores

    except Exception as e:
        logger.debug("Embed-based reranking failed: %s", e)

    # Fallback: generation-based scoring
    # Ask the model to score each document individually
    scores = []
    for doc in documents:
        score = _score_single(client, model_name, query, doc, timeout)
        scores.append(score)

    elapsed = (time.monotonic() - t0) * 1000
    logger.debug("Reranker generation fallback scored in %.1fms", elapsed)
    return scores


def _score_single(
    client: Any,
    model_name: str,
    query: str,
    document: str,
    timeout: float,
) -> float:
    """Score a single (query, document) pair via generation.

    Uses a prompt that asks the model to output a relevance score
    between 0 and 1.
    """
    prompt = (
        f"Rate the relevance of the following document to the query on a scale "
        f"from 0.0 to 1.0. Output ONLY the number, nothing else.\n\n"
        f"Query: {query}\n\n"
        f"Document: {document[:1024]}\n\n"
        f"Relevance score:"
    )

    try:
        response = client.generate(
            model=model_name,
            prompt=prompt,
            stream=False,
            options={
                "temperature": 0.0,
                "num_predict": 10,
            },
        )
        text = response.get("response", "").strip()
        # Extract the first float from the response
        score = _extract_score(text)
        return score
    except Exception as e:
        logger.warning("Reranker generation failed for a document: %s", e)
        return 0.0


def _extract_score(text: str) -> float:
    """Extract a float score from model output text.

    Handles formats like "0.85", "0.85/1.0", "85%", etc.
    """
    import re

    # Try direct float parse
    try:
        val = float(text.strip())
        if 0.0 <= val <= 1.0:
            return val
        # If > 1, assume it's on a 0-10 scale
        if val <= 10.0:
            return val / 10.0
    except ValueError:
        pass

    # Try extracting first float from text
    match = re.search(r"(\d+\.?\d*)", text)
    if match:
        val = float(match.group(1))
        if val > 1.0:
            val = val / 10.0 if val <= 10.0 else val / 100.0
        return max(0.0, min(1.0, val))

    return 0.0


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)
