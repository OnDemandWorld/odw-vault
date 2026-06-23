"""End-to-end RAG pipeline benchmark.

Measures every phase of a query from prompt submission to final response:
- UI startup (FastAPI + Gradio)
- Retrieval (language detection, dense embedding, BM25, RRF fusion)
- Generation (Ollama token streaming)
- Total end-to-end time per query

Usage:
    cd /path/to/RAG_POC
    PYTHONPATH=. .venv/bin/python ui/benchmark.py [--queries N] [--host 127.0.0.1] [--port 8888]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx
import ollama

from pipeline.config import load_app_config
from pipeline.db import open_db, migrate
from rag.retrieval import retrieve


# ── Test queries ──────────────────────────────────────────────────────────────
TEST_QUERIES = [
    "robot performance rental service",
    "what documents are in the knowledge base",
    "summarise the main topics covered in the corpus",
    "find policies or procedures that mention approvals",
    "what are the key themes across all folders",
    "compare the latest version with previous drafts",
    "what are the dependencies between these documents",
    "identify gaps or missing procedures",
    "extract budget or cost related information",
    "draft a summary email for the team",
    "create a table of all action items mentioned",
    "what deadlines are coming up based on the documents",
    "write a brief executive summary of the corpus",
    "Kettybot proposal details",
    "CDC waiter system configuration",
]


def benchmark_retrieval(cfg, queries: list[str]) -> list[dict]:
    """Measure retrieval phase directly (no HTTP overhead)."""
    from pipeline.config import load_app_config as _lac
    cfg = _lac(str(ROOT / "config.toml"))
    db_path = ROOT / "corpus.db"
    if not db_path.exists():
        db_path = ROOT / "SourceData" / ".rag-cache" / "corpus.db"
    db = open_db(str(db_path))
    migrate(db)
    chroma_path = cfg.paths.chroma_root

    results = []
    for q in queries:
        t_start = time.monotonic()
        t_embed_start = None
        t_bm25_start = None
        t_fusion_start = None

        # Patch time.monotonic to capture sub-phase timings
        _orig_mono = time.monotonic
        phase_times = {}

        def _patched_mono():
            return _orig_mono()

        try:
            hits, metrics = retrieve(
                query=q, db=db, chroma_client=None,
                chroma_path=chroma_path, cfg=cfg, folder_filter=None,
            )
        except Exception as e:
            results.append({"query": q, "error": str(e)[:200]})
            continue

        t_total = time.monotonic() - t_start
        results.append({
            "query": q,
            "retrieval_ms": metrics.get("retrieval_ms", 0),
            "dense_hits": metrics.get("dense_hits", 0),
            "bm25_hits": metrics.get("bm25_hits", 0),
            "fused_total": metrics.get("fused_total", 0),
            "hits_returned": len(hits),
            "query_lang": metrics.get("query_lang", "unknown"),
            "retrieval_s": round(t_total, 3),
        })

    return results


def benchmark_generation(cfg, queries: list[str]) -> list[dict]:
    """Measure Ollama generation phase (thinking + text output)."""
    from rag.retrieval import retrieve as _retrieve
    from rag.generation import _load_prompt, REFUSAL_TEXT
    from rag.citations import parse_citations, resolve_citations
    from rag.retrieval import Hit

    db_path = ROOT / "corpus.db"
    if not db_path.exists():
        db_path = ROOT / "SourceData" / ".rag-cache" / "corpus.db"
    db = open_db(str(db_path))
    migrate(db)

    def _format_chunks(hits: list) -> str:
        blocks = []
        for i, hit in enumerate(hits, start=1):
            page_info = f" (page {hit.page_start})" if hit.page_start else ""
            blocks.append(f"[{i}] {hit.rel_path}{page_info}\n{hit.text}")
        return "\n\n".join(blocks) if blocks else "(no context available)"

    results = []
    for q in queries:
        # First retrieve
        try:
            hits, _ = retrieve(
                query=q, db=db, chroma_client=None,
                chroma_path=cfg.paths.chroma_root, cfg=cfg, folder_filter=None,
            )
        except Exception as e:
            results.append({"query": q, "error": str(e)[:200]})
            continue

        if not hits:
            results.append({"query": q, "error": "no hits"})
            continue

        # Build prompt
        numbered_chunks = _format_chunks(hits)
        template = _load_prompt(None, cfg)
        prompt = template.format(numbered_chunks=numbered_chunks, query=q)

        system_prefix = ""
        if getattr(cfg.models.generation, "thinking", False):
            system_prefix = "<|think|>"
        system_content = "You are a helpful assistant."
        if system_prefix:
            system_content = f"{system_prefix}\n{system_content}"

        model_name = cfg.models.generation.name
        client = ollama.Client(host=cfg.ollama.host)

        t_gen_start = time.monotonic()
        t_first_token = None
        full_text = ""
        token_count = 0
        empty_tokens = 0
        nonempty_tokens = 0

        for chunk in client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": cfg.models.generation.temperature,
                "top_p": cfg.models.generation.top_p,
                "top_k": cfg.models.generation.top_k,
            },
            stream=True,
        ):
            token = chunk.get("message", {}).get("content", "")
            if token:
                if t_first_token is None:
                    t_first_token = time.monotonic() - t_gen_start
                full_text += token
                token_count += 1
                nonempty_tokens += 1
            else:
                empty_tokens += 1
                token_count += 1

        t_gen_total = time.monotonic() - t_gen_start
        t_after_first = t_gen_total - (t_first_token or 0)

        citation_numbers = parse_citations(full_text)
        citations = resolve_citations(citation_numbers, hits)

        results.append({
            "query": q[:60],
            "hits_used": len(hits),
            "prompt_length": len(prompt),
            "gen_model": model_name,
            "total_tokens": token_count,
            "empty_tokens": empty_tokens,
            "nonempty_tokens": nonempty_tokens,
            "thinking_time_s": round(t_first_token or 0, 2) if t_first_token else round(t_gen_total, 2),
            "text_output_time_s": round(t_after_first, 2),
            "gen_total_time_s": round(t_gen_total, 2),
            "output_chars": len(full_text),
            "citations": len(citations),
            "chars_per_second": round(len(full_text) / t_after_first, 1) if t_after_first > 0 else 0,
        })

    return results


def benchmark_gradio_api(queries: list[str], base_url: str) -> list[dict]:
    """Measure end-to-end time via the Gradio HTTP API (closest to user experience)."""
    results = []
    for q in queries:
        t_start = time.monotonic()

        # Step 1: POST to get event_id
        t_post_start = time.monotonic()
        try:
            resp = httpx.post(
                f"{base_url}/gradio_api/call/chat",
                json={"data": [q, [], "All folders"]},
                timeout=120,
            )
            t_post_end = time.monotonic()
            data = resp.json()
            event_id = data.get("event_id")
        except Exception as e:
            results.append({"query": q, "error": str(e)[:200]})
            continue

        # Step 2: GET SSE stream
        t_sse_start = time.monotonic()
        t_first_data = None
        t_last_data = None
        event_count = 0
        final_text = ""

        try:
            with httpx.stream("GET", f"{base_url}/gradio_api/call/chat/{event_id}", timeout=300) as r:
                for line in r.iter_lines():
                    if line.startswith("data:"):
                        event_count += 1
                        if t_first_data is None:
                            t_first_data = time.monotonic() - t_sse_start
                        t_last_data = time.monotonic() - t_sse_start
                        try:
                            d = json.loads(line[5:])
                            if d and d[0] and len(d[0]) > 0:
                                h = d[0]
                                last = h[-1]
                                if last and last.get("content"):
                                    final_text = last["content"]
                        except Exception:
                            pass
        except Exception as e:
            results.append({"query": q, "error": str(e)[:200]})
            continue

        t_sse_total = time.monotonic() - t_sse_start
        t_total = time.monotonic() - t_start

        results.append({
            "query": q[:60],
            "post_latency_ms": round((t_post_end - t_post_start) * 1000, 1),
            "sse_first_data_s": round(t_first_data or 0, 2),
            "sse_last_data_s": round(t_last_data or 0, 2),
            "sse_duration_s": round(t_sse_total, 2),
            "total_end_to_end_s": round(t_total, 2),
            "sse_events": event_count,
            "response_length": len(final_text),
        })

    return results


def print_table(title: str, rows: list[dict], columns: list[tuple[str, str, str]]):
    """Print a formatted table. columns = [(key, header, fmt)]"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

    # Header
    headers = [h for _, h, _ in columns]
    widths = [max(len(h), 10) for h in headers]
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(header_line)
    print("  ".join("-" * w for w in widths))

    # Rows
    for row in rows:
        vals = []
        for key, _, fmt in columns:
            v = row.get(key, "")
            if fmt:
                try:
                    vals.append(fmt.format(v))
                except (TypeError, ValueError):
                    vals.append(str(v)[:15])
            else:
                vals.append(str(v)[:15])
        line = "  ".join(v.ljust(w) for v, w in zip(vals, widths))
        print(line)


def print_summary(title: str, metrics: list[float], unit: str = "s"):
    """Print statistical summary."""
    if not metrics:
        print(f"  {title}: no data")
        return
    print(f"\n  {title} ({unit}):")
    print(f"    count:   {len(metrics)}")
    print(f"    min:     {min(metrics):.2f}")
    print(f"    max:     {max(metrics):.2f}")
    print(f"    mean:    {statistics.mean(metrics):.2f}")
    print(f"    median:  {statistics.median(metrics):.2f}")
    print(f"    stdev:   {statistics.stdev(metrics):.2f}" if len(metrics) > 1 else "")
    print(f"    p50:     {sorted(metrics)[len(metrics)//2]:.2f}")
    print(f"    p90:     {sorted(metrics)[int(len(metrics)*0.9)]:.2f}")
    print(f"    p95:     {sorted(metrics)[int(len(metrics)*0.95)]:.2f}")


def main():
    parser = argparse.ArgumentParser(description="RAG pipeline benchmark")
    parser.add_argument("--queries", "-n", type=int, default=15, help="Number of queries to test")
    parser.add_argument("--host", default="127.0.0.1", help="UI host")
    parser.add_argument("--port", type=int, default=8888, help="UI port")
    parser.add_argument("--skip-gradio", action="store_true", help="Skip HTTP API benchmark")
    parser.add_argument("--skip-generation", action="store_true", help="Skip generation benchmark")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    queries = TEST_QUERIES[:args.queries]
    cfg = load_app_config(str(ROOT / "config.toml"))
    base_url = f"http://{args.host}:{args.port}"

    print(f"\n  Novelte Core RAG Benchmark")
    print(f"  Queries: {len(queries)}")
    print(f"  Model: {cfg.models.generation.name}")
    print(f"  Embedding: {cfg.models.embedding.name}")
    print(f"  Ollama: {cfg.ollama.host}")
    print(f"  Chroma: {cfg.paths.chroma_root}")
    print(f"  UI: {base_url}")

    # Phase 1: Retrieval
    print("\n  Phase 1: Retrieval (direct Python call)...")
    retrieval_results = benchmark_retrieval(cfg, queries)
    if not args.json:
        print_table("Retrieval Times", retrieval_results, [
            ("query", "Query", "{:.50s}"),
            ("retrieval_s", "Total (s)", "{:.3f}"),
            ("retrieval_ms", "Dense (ms)", "{:.1f}"),
            ("dense_hits", "Dense hits", "{}"),
            ("bm25_hits", "BM25 hits", "{}"),
            ("fused_total", "Fused", "{}"),
            ("hits_returned", "Returned", "{}"),
            ("query_lang", "Lang", "{}"),
        ])
        retrieval_times = [r["retrieval_s"] for r in retrieval_results if "retrieval_s" in r]
        print_summary("Retrieval", retrieval_times, "s")

    # Phase 2: Generation (if not skipped)
    if not args.skip_generation:
        print("\n  Phase 2: Generation (Ollama)...")
        gen_results = benchmark_generation(cfg, queries)
        if not args.json:
            print_table("Generation Times", gen_results, [
                ("query", "Query", "{:.50s}"),
                ("hits_used", "Hits", "{}"),
                ("prompt_length", "Prompt len", "{}"),
                ("total_tokens", "Tokens", "{}"),
                ("empty_tokens", "Empty", "{}"),
                ("nonempty_tokens", "Non-empty", "{}"),
                ("thinking_time_s", "Think (s)", "{:.1f}"),
                ("text_output_time_s", "Text (s)", "{:.1f}"),
                ("gen_total_time_s", "Total (s)", "{:.1f}"),
                ("output_chars", "Chars", "{}"),
                ("chars_per_second", "Ch/s", "{:.1f}"),
            ])
            thinking_times = [r["thinking_time_s"] for r in gen_results if "thinking_time_s" in r]
            text_times = [r["text_output_time_s"] for r in gen_results if "text_output_time_s" in r]
            gen_totals = [r["gen_total_time_s"] for r in gen_results if "gen_total_time_s" in r]
            cps = [r["chars_per_second"] for r in gen_results if "chars_per_second" in r and r["chars_per_second"] > 0]
            print_summary("Thinking time (first text token)", thinking_times, "s")
            print_summary("Text output time", text_times, "s")
            print_summary("Generation total", gen_totals, "s")
            print_summary("Characters per second", cps, "ch/s")

    # Phase 3: End-to-end via Gradio API (if not skipped)
    if not args.skip_gradio:
        print(f"\n  Phase 3: End-to-end via Gradio API ({base_url})...")
        api_results = benchmark_gradio_api(queries, base_url)
        if not args.json:
            print_table("Gradio API End-to-End", api_results, [
                ("query", "Query", "{:.40s}"),
                ("post_latency_ms", "POST (ms)", "{:.1f}"),
                ("sse_first_data_s", "1st data (s)", "{:.2f}"),
                ("sse_last_data_s", "Last data (s)", "{:.2f}"),
                ("sse_duration_s", "SSE dur (s)", "{:.2f}"),
                ("total_end_to_end_s", "Total (s)", "{:.2f}"),
                ("sse_events", "Events", "{}"),
                ("response_length", "Chars", "{}"),
            ])
            e2e_times = [r["total_end_to_end_s"] for r in api_results if "total_end_to_end_s" in r]
            print_summary("End-to-end (user experience)", e2e_times, "s")

    # Overall summary
    if not args.json:
        print(f"\n{'='*80}")
        print(f"  OVERALL SUMMARY")
        print(f"{'='*80}")
        if retrieval_results:
            rt = [r["retrieval_s"] for r in retrieval_results if "retrieval_s" in r]
            print(f"  Retrieval avg: {statistics.mean(rt):.3f}s (median: {statistics.median(rt):.3f}s)")
        if not args.skip_generation and gen_results:
            gt = [r["gen_total_time_s"] for r in gen_results if "gen_total_time_s" in r]
            tt = [r["thinking_time_s"] for r in gen_results if "thinking_time_s" in r]
            print(f"  Generation avg: {statistics.mean(gt):.1f}s (thinking: {statistics.mean(tt):.1f}s)")
        if not args.skip_gradio and api_results:
            et = [r["total_end_to_end_s"] for r in api_results if "total_end_to_end_s" in r]
            print(f"  End-to-end avg: {statistics.mean(et):.2f}s (median: {statistics.median(et):.2f}s)")
        print()

    # JSON output
    if args.json:
        output = {
            "model": cfg.models.generation.name,
            "embedding": cfg.models.embedding.name,
            "retrieval": retrieval_results,
        }
        if not args.skip_generation:
            output["generation"] = gen_results
        if not args.skip_gradio:
            output["gradio_api"] = api_results
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
