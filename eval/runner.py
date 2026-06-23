"""Evaluation runner: add questions, run eval, generate reports."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import chromadb

from pipeline.config import AppConfig
from rag.generation import generate_answer
from rag.retrieval import Hit, retrieve

logger = logging.getLogger(__name__)

VALID_CATEGORIES = {"lookup", "synthesis", "scoped", "negative", "bilingual"}


# ---------------------------------------------------------------------------
# add_question
# ---------------------------------------------------------------------------


def add_question(
    db,
    question: str,
    expected_file_ids: list[int],
    expected_answer: str | None = None,
    category: str | None = None,
    lang: str | None = None,
) -> int:
    """Add a question to eval_question table. Returns question id."""
    if category and category not in VALID_CATEGORIES:
        raise ValueError(f"category must be one of {sorted(VALID_CATEGORIES)}, got {category!r}")
    if not expected_file_ids:
        raise ValueError("expected_file_ids must be non-empty")

    row = db["eval_question"].insert(
        {
            "question": question,
            "expected_file_ids_json": json.dumps(expected_file_ids),
            "expected_answer": expected_answer,
            "category": category,
            "lang": lang,
        }
    )
    return row.last_rowid


# ---------------------------------------------------------------------------
# run_eval
# ---------------------------------------------------------------------------


def _recall_for_hits(
    hits: list[Hit],
    expected_file_ids: set[int],
    top_k: int,
) -> int:
    """Return 1 if any expected file_id appears in top-k hits, else 0."""
    top_k_file_ids = {h.file_id for h in hits[:top_k]}
    return 1 if top_k_file_ids & expected_file_ids else 0


def run_eval(
    db,
    cfg: AppConfig,
    embedding_model: str | None = None,
    generation_model: str | None = None,
    reranker_model: str | None = None,
    no_augment: bool = False,
) -> dict:
    """Run evaluation against all questions. Returns summary dict."""
    questions = db.query("SELECT * FROM eval_question ORDER BY id")
    if not questions:
        return {
            "total_questions": 0,
            "avg_recall_at_5": 0.0,
            "avg_recall_at_10": 0.0,
            "run_count": 0,
        }

    chroma_path = str(Path(cfg.paths.chroma_root).resolve())
    chroma_client = chromadb.PersistentClient(path=chroma_path)

    embed_model = embedding_model or cfg.models.embedding.name
    gen_model = generation_model or cfg.models.generation.name
    rerank_model = reranker_model or (
        cfg.models.reranker.name if cfg.models.reranker.enabled else None
    )

    recalls_at_5: list[int] = []
    recalls_at_10: list[int] = []
    run_count = 0

    for q in questions:
        qid = q["id"]
        expected_ids = set(json.loads(q["expected_file_ids_json"]))

        try:
            # 1. Retrieve
            hits, _metrics = retrieve(
                query=q["question"],
                db=db,
                chroma_client=chroma_client,
                chroma_path=chroma_path,
                cfg=cfg,
            )

            # 2. Compute recall
            recall_5 = _recall_for_hits(hits, expected_ids, 5)
            recall_10 = _recall_for_hits(hits, expected_ids, 10)

            # 3. Generate answer
            gen_result = generate_answer(
                query=q["question"],
                hits=hits,
                cfg=cfg,
            )
            answer_text = gen_result.get("answer", "")

        except Exception as exc:
            logger.error("Question %d failed: %s", qid, exc)
            recall_5 = 0
            recall_10 = 0
            answer_text = f"ERROR: {exc}"

        # 4. Record eval_run
        db["eval_run"].insert(
            {
                "question_id": qid,
                "embedding_model": embed_model,
                "generation_model": gen_model,
                "reranker_model": rerank_model,
                "contextual_augmentation": 0 if no_augment else 1,
                "retrieval_recall_at_5": recall_5,
                "retrieval_recall_at_10": recall_10,
                "answer_text": answer_text,
                "notes": None,
            }
        )

        recalls_at_5.append(recall_5)
        recalls_at_10.append(recall_10)
        run_count += 1

    avg_r5 = sum(recalls_at_5) / len(recalls_at_5) if recalls_at_5 else 0.0
    avg_r10 = sum(recalls_at_10) / len(recalls_at_10) if recalls_at_10 else 0.0

    return {
        "total_questions": run_count,
        "avg_recall_at_5": round(avg_r5, 4),
        "avg_recall_at_10": round(avg_r10, 4),
        "run_count": run_count,
    }


# ---------------------------------------------------------------------------
# eval_report
# ---------------------------------------------------------------------------


def _format_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    """Format rows as a plain-text table (fallback when rich unavailable)."""
    headers = [c[1] for c in columns]
    keys = [c[0] for c in columns]

    # Convert values to strings
    str_rows: list[list[str]] = []
    for row in rows:
        str_rows.append([str(row.get(k, "")) for k in keys])

    # Compute column widths
    widths = [len(h) for h in headers]
    for r in str_rows:
        for i, val in enumerate(r):
            widths[i] = max(widths[i], len(val))

    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*headers), "  ".join("-" * w for w in widths)]
    for r in str_rows:
        lines.append(fmt.format(*r))
    return "\n".join(lines)


def eval_report(
    db,
    last_n: int | None = None,
    compare: bool = False,
) -> str:
    """Generate evaluation report as a string."""
    if compare:
        rows = db.query("SELECT * FROM v_eval_summary")
        if not rows:
            return "(no evaluation runs to compare)"

        columns = [
            ("embedding_model", "Embedding"),
            ("generation_model", "Generation"),
            ("contextual_augmentation", "Augment"),
            ("questions", "Questions"),
            ("recall_at_5", "Recall@5"),
            ("recall_at_10", "Recall@10"),
            ("avg_human_grade", "Avg Grade"),
        ]
        return _format_table(rows, columns)

    # Default: show individual runs
    if last_n:
        rows = db.query(
            "SELECT * FROM eval_run ORDER BY run_at DESC LIMIT ?",
            [last_n],
        )
    else:
        rows = db.query("SELECT * FROM eval_run ORDER BY run_at DESC")

    if not rows:
        return "(no evaluation runs found)"

    columns = [
        ("id", "Run"),
        ("question_id", "Q"),
        ("embedding_model", "Embed"),
        ("generation_model", "Gen"),
        ("retrieval_recall_at_5", "R@5"),
        ("retrieval_recall_at_10", "R@10"),
        ("human_grade", "Grade"),
        ("answer_text", "Answer"),
    ]
    return _format_table(rows, columns)
