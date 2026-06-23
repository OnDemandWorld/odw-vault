"""Benchmark gpt-oss:20b on cloud vs ministral-3:3b cloud vs local gemma4."""
import sys, time, os, statistics
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

api_key = os.environ.get("OLLAMA_API_KEY")
cloud_client = ollama.Client(
    host="https://ollama.com",
    headers={"Authorization": f"Bearer {api_key}"},
)
local_client = ollama.Client(host="http://localhost:11434")

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


def run_one(client, model, prompt_text, opts, retries=3):
    for attempt in range(1, retries + 1):
        t0 = time.monotonic()
        try:
            txt = ""
            t_ft = None
            empty = 0
            for c in client.chat(
                model=model,
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
                else:
                    empty += 1
            total = time.monotonic() - t0
            text_t = total - (t_ft or 0)
            return {
                "ttft": t_ft or total, "text": text_t, "total": total,
                "chars": len(txt), "empty": empty,
                "cps": len(txt) / text_t if text_t > 0 else 0,
                "preview": txt[:80].replace("\n", " "),
            }
        except Exception as e:
            elapsed = time.monotonic() - t0
            if attempt == retries:
                return {
                    "error": str(e)[:120], "ttft": elapsed, "total": elapsed,
                    "text": 0, "chars": 0, "empty": 0, "cps": 0, "preview": f"FAIL: {e}",
                }
            time.sleep(1)
    return None


def main():
    sep = "=" * 75
    print(f"\n{sep}")
    print(f"  GPT-OSS:20B (cloud) vs MINISTRAL-3:3B (cloud) vs GEMMA4 (local)")
    print(f"{sep}")

    # Model configs: (client, model_name, label, opts)
    models = [
        (cloud_client, "gpt-oss:20b", "gpt-oss:20b (cloud)", {}),
        (cloud_client, "ministral-3:3b", "ministral-3:3b (cloud)", {}),
        (local_client, "gemma4:latest", "gemma4:latest (local)", {}),
    ]

    all_results = {m[1]: [] for m in models}

    for idx, q in enumerate(QUERIES, 1):
        hits, _ = retrieve(
            query=q, db=db, chroma_client=None,
            chroma_path=cfg.paths.chroma_root, cfg=cfg, folder_filter=None,
        )
        if not hits:
            continue
        prompt = template.format(numbered_chunks=_fmt(hits), query=q)
        prompt_len = len(prompt)

        print(f"\n  [{idx}] {q} ({prompt_len} chars, {len(hits)} hits)")

        for client, model_name, label, opts in models:
            r = run_one(client, model_name, prompt, opts)
            all_results[model_name].append(r)

            err_tag = "  [RETRY]" if r.get("error") else ""
            print(f"    {label:<30s} TTFT={r['ttft']:.1f}s  "
                  f"Total={r['total']:.1f}s  Empty={r['empty']:>4}  Ch={r['chars']:>5}  {err_tag}")
            sys.stdout.flush()

    # Summary
    print(f"\n{sep}\n  SUMMARY (medians)\n{sep}")
    print(f"{'Model':<30} {'TTFT':>8} {'Text':>8} {'Total':>8} {'Ch/s':>8} {'Empty':>6} {'OK':>4} {'Fails':>5}")
    print(f"{'-'*75}")

    for model_name, label, _ in models:
        results = all_results[model_name]
        valid = [r for r in results if not r.get("error")]
        fails = sum(1 for r in results if r.get("error"))
        if valid:
            ttft = statistics.median([r["ttft"] for r in valid])
            text = statistics.median([r["text"] for r in valid])
            total = statistics.median([r["total"] for r in valid])
            cps = statistics.median([r["cps"] for r in valid])
            empty = statistics.median([r["empty"] for r in valid])
            ok = len(valid)
            print(f"{label:<30} {ttft:>7.1f}s {text:>7.1f}s {total:>7.1f}s "
                  f"{cps:>7.0f} {empty:>6.0f} {ok:>4} {fails:>5}")
        else:
            print(f"{label:<30} {'ALL FAILED':>30} {fails:>4} {'>5':>5}")

    print()


if __name__ == "__main__":
    main()
