"""Interleaved benchmark: same query with both modes back-to-back."""
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
db = open_db(str(db_path))
migrate(db)
template = _load_prompt(None, cfg)
client = ollama.Client(host=cfg.ollama.host)

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


def run_one(prompt_text, extra_opts):
    t0 = time.monotonic()
    t_ft = None
    txt = ""
    tokens = 0
    empty = 0
    for c in client.chat(
        model="gemma4:latest",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt_text}],
        options={"temperature": 0.5, "top_p": 0.95, "top_k": 64, **extra_opts},
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
    }


default_results = []
no_think_results = []

sep = "=" * 65
print(f"\n{sep}\n  INTERLEAVED: gemma4:latest  (thinking vs num_think=0)\n{sep}")

for idx, q in enumerate(QUERIES, 1):
    hits, _ = retrieve(
        query=q, db=db, chroma_client=None,
        chroma_path=cfg.paths.chroma_root, cfg=cfg, folder_filter=None,
    )
    if not hits:
        continue
    prompt = template.format(numbered_chunks=_fmt(hits), query=q)

    # Run thinking first, then num_think=0 for same query
    r1 = run_one(prompt, extra_opts={})
    r2 = run_one(prompt, extra_opts={"num_think": 0})
    default_results.append(r1)
    no_think_results.append(r2)

    preview1 = r1["chars"]
    preview2 = r2["chars"]
    print(f"  [{idx}] Default: TTFT={r1['ttft']:.1f}s  Total={r1['total']:.1f}s  "
          f"Empty={r1['empty']}  Ch={r1['chars']}")
    print(f"       NoThink: TTFT={r2['ttft']:.1f}s  Total={r2['total']:.1f}s  "
          f"Empty={r2['empty']}  Ch={r2['chars']}")
    print(f"       => num_think=0 is {'FASTER' if r2['total'] < r1['total'] else 'SLOWER'} "
          f"by {abs(r1['total'] - r2['total']):.1f}s")
    sys.stdout.flush()

print(f"\n{sep}\n  SUMMARY\n{sep}")
for k, l in [("ttft", "TTFT"), ("text", "Text"), ("total", "Total"), ("cps", "Chars/s")]:
    d = [r[k] for r in default_results]
    n = [r[k] for r in no_think_results]
    d_med = statistics.median(d)
    n_med = statistics.median(n)
    if k == "cps":
        diff = n_med - d_med
        pct = (diff / d_med * 100) if d_med else 0
    else:
        diff = d_med - n_med
        pct = (diff / d_med * 100) if d_med else 0
    print(f"  {l}: default={d_med:.1f}  num_think=0={n_med:.1f}  "
          f"diff={diff:+.1f}s ({pct:+.0f}%)")
print()
