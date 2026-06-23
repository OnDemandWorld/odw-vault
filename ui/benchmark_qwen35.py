"""Interleaved: qwen3.5:9b num_think=0 vs gemma4:latest default."""
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
db = open_db(str(ROOT / "corpus.db"))
migrate(db)
template = _load_prompt(None, cfg)

QUERIES = [
    "robot performance rental service",
    "what are the key themes across all folders",
    "extract budget or cost related information",
    "Kettybot proposal details",
    "CDC waiter system configuration",
]


def _fmt(hits):
    blocks = []
    for i, h in enumerate(hits, 1):
        pg = f" (page {h.page_start})" if h.page_start else ""
        blocks.append(f"[{i}] {h.rel_path}{pg}\n{h.text}")
    return "\n\n".join(blocks) if blocks else "(no context)"


def run_one(model_name, prompt_text, opts):
    client = ollama.Client(host=cfg.ollama.host)
    t0 = time.monotonic()
    t_ft = None
    txt = ""
    tokens = 0
    empty = 0
    for c in client.chat(
        model=model_name,
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt_text}],
        options={"temperature": 0.5, "top_p": 0.95, "top_k": 64, **opts},
        stream=True,
    ):
        tok = c.get("message", {}).get("content", "")
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
    return {
        "ttft": t_ft or total, "text": text_t, "total": total,
        "chars": len(txt), "tokens": tokens, "empty": empty,
        "cps": len(txt) / text_t if text_t > 0 else 0,
        "preview": txt[:80].replace("\n", " "),
    }


def main():
    sep = "=" * 65
    print(f"\n{sep}\n  QWEN3.5:9B (num_think=0) vs GEMMA4:latest (default)\n{sep}")

    gem_results = []
    qwen_results = []

    for idx, q in enumerate(QUERIES, 1):
        hits, _ = retrieve(
            query=q, db=db, chroma_client=None,
            chroma_path=cfg.paths.chroma_root, cfg=cfg, folder_filter=None,
        )
        if not hits:
            continue
        prompt = template.format(numbered_chunks=_fmt(hits), query=q)

        g = run_one("gemma4:latest", prompt, {})
        qw = run_one("qwen3.5:9b", prompt, {"num_think": 0})
        gem_results.append(g)
        qwen_results.append(qw)

        diff = g["total"] - qw["total"]
        faster = "FASTER" if diff > 0 else "SLOWER"
        print(f"  [{idx}] Gemma4:  TTFT={g['ttft']:.1f}s  Total={g['total']:.1f}s  "
              f"Empty={g['empty']}  Ch={g['chars']}  | {g['preview']}")
        print(f"       Qwen3.5: TTFT={qw['ttft']:.1f}s  Total={qw['total']:.1f}s  "
              f"Empty={qw['empty']}  Ch={qw['chars']}  | {qw['preview']}")
        print(f"       => qwen3.5 is {faster} by {abs(diff):.1f}s")
        sys.stdout.flush()

    print(f"\n{sep}\n  SUMMARY\n{sep}")
    for k, l in [("ttft", "TTFT"), ("text", "Text"), ("total", "Total"), ("cps", "Chars/s")]:
        gm = [r[k] for r in gem_results]
        qm = [r[k] for r in qwen_results]
        g_med = statistics.median(gm)
        q_med = statistics.median(qm)
        if k == "cps":
            diff = q_med - g_med
            pct = (diff / g_med * 100) if g_med else 0
        else:
            diff = g_med - q_med
            pct = (diff / g_med * 100) if g_med else 0
        print(f"  {l}: gemma4={g_med:.1f}  qwen3.5={q_med:.1f}  diff={diff:+.1f}s ({pct:+.0f}%)")
    print()


if __name__ == "__main__":
    main()
