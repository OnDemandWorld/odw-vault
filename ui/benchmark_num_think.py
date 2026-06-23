"""Benchmark gemma4 with num_think:0 vs default (thinking)."""
import sys, time, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ollama
from pipeline.config import load_app_config
from pipeline.db import open_db, migrate
from rag.retrieval import retrieve
from rag.generation import _load_prompt

cfg = load_app_config(str(ROOT / "config.toml"))

db_path = ROOT / "corpus.db"
if not db_path.exists():
    db_path = ROOT / "SourceData" / ".rag-cache" / "corpus.db"
db = open_db(str(db_path))
migrate(db)

template = _load_prompt(None, cfg)

QUERIES = [
    "robot performance rental service",
    "what are the key themes across all folders",
    "extract budget or cost related information",
    "Kettybot proposal details",
    "CDC waiter system configuration",
]


def _format_chunks(hits):
    blocks = []
    for i, h in enumerate(hits, 1):
        pg = f" (page {h.page_start})" if h.page_start else ""
        blocks.append(f"[{i}] {h.rel_path}{pg}\n{h.text}")
    return "\n\n".join(blocks) if blocks else "(no context)"


def benchmark(queries, model, client, db, cfg, template, extra_options=None):
    extra = extra_options or {}
    results = []
    for idx, q in enumerate(queries, 1):
        hits, _ = retrieve(
            query=q, db=db, chroma_client=None,
            chroma_path=cfg.paths.chroma_root, cfg=cfg, folder_filter=None,
        )
        if not hits:
            continue

        prompt = template.format(numbered_chunks=_format_chunks(hits), query=q)
        t0 = time.monotonic()
        t_ft = None
        txt = ""
        tokens = 0
        empty = 0

        options = {
            "temperature": 0.5, "top_p": 0.95, "top_k": 64,
        }
        options.update(extra)

        for chunk in client.chat(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            options=options,
            stream=True,
        ):
            tok = chunk.get("message", {}).get("content", "")
            if tok:
                if t_ft is None:
                    t_ft = time.monotonic() - t0
                txt += tok
                tokens += 1
            else:
                empty += 1
                tokens += 1

        total = time.monotonic() - t0
        text_t = total - (t_ft or 0)
        preview = txt[:70].replace("\n", " ")
        print(f"  [{idx}] TTFT={t_ft or total:.1f}s  Text={text_t:.1f}s  "
              f"Total={total:.1f}s  Tok={tokens}  Empty={empty}  "
              f"Ch/s={len(txt)/text_t:.0f}  | {preview}")
        sys.stdout.flush()

        results.append({
            "ttft": t_ft or total,
            "text": text_t,
            "total": total,
            "chars": len(txt),
            "tokens": tokens,
            "empty": empty,
            "cps": len(txt) / text_t if text_t > 0 else 0,
        })

    return results


def main():
    client = ollama.Client(host=cfg.ollama.host)

    sep = "=" * 65

    # With thinking (default)
    print(f"\n{sep}\n  MODEL: gemma4:latest  (DEFAULT - thinking ON)\n{sep}")
    sys.stdout.flush()
    r1 = benchmark(QUERIES, "gemma4:latest", client, db, cfg, template, extra_options={})
    if r1:
        print(f"\n  --- Summary ---")
        for k, l in [("ttft", "TTFT"), ("text", "Text"), ("total", "Total"), ("cps", "Chars/s")]:
            v = [r[k] for r in r1]
            print(f"  {l}: median={statistics.median(v):.1f}  mean={statistics.mean(v):.1f}")
        sys.stdout.flush()

    # Without thinking (num_think=0)
    print(f"\n{sep}\n  MODEL: gemma4:latest  (num_think=0 - thinking DISABLED)\n{sep}")
    sys.stdout.flush()
    r2 = benchmark(QUERIES, "gemma4:latest", client, db, cfg, template, extra_options={"num_think": 0})
    if r2:
        print(f"\n  --- Summary ---")
        for k, l in [("ttft", "TTFT"), ("text", "Text"), ("total", "Total"), ("cps", "Chars/s")]:
            v = [r[k] for r in r2]
            print(f"  {l}: median={statistics.median(v):.1f}  mean={statistics.mean(v):.1f}")
        sys.stdout.flush()

    # Comparison
    print(f"\n{sep}\n  COMPARISON\n{sep}")
    for k, l in [("ttft", "TTFT (s)"), ("text", "Text (s)"), ("total", "Total (s)"), ("cps", "Chars/s")]:
        w = [r[k] for r in r2]
        g = [r[k] for r in r1]
        w_med = statistics.median(w)
        g_med = statistics.median(g)
        if k == "cps":
            diff = w_med - g_med
            pct = (diff / g_med * 100) if g_med else 0
            print(f"  {l}: num_think=0: {w_med:.1f}  default: {g_med:.1f}  diff={diff:+.1f} ({pct:+.0f}%)")
        else:
            diff = g_med - w_med
            pct = (diff / g_med * 100) if g_med else 0
            print(f"  {l}: num_think=0: {w_med:.1f}  default: {g_med:.1f}  diff={diff:+.1f} ({pct:+.0f}%)")
    print()


if __name__ == "__main__":
    main()
