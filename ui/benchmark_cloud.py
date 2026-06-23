"""Benchmark Ollama Cloud vs Local — updated model list."""
import sys, time, statistics, os
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


def run_one(client, model, prompt_text, opts):
    t0 = time.monotonic()
    t_ft = None
    txt = ""
    tokens = 0
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
        "preview": txt[:90].replace("\n", " "),
    }


def main():
    api_key = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        print("ERROR: OLLAMA_API_KEY not set")
        sys.exit(1)

    cloud_client = ollama.Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    local_client = ollama.Client(host="http://localhost:11434")

    sep = "=" * 70

    # Available cloud models to test
    CLOUD_MODELS = [
        ("ministral-3:3b", "Cloud 3B — smallest"),
        ("gemma3:4b", "Cloud 4B — Gemma3"),
        ("glm-4.6", "Cloud — GLM"),
    ]
    LOCAL_MODEL = ("gemma4:latest", "Local — current production")

    print(f"\n{sep}")
    print(f"  OLLAMA CLOUD vs LOCAL BENCHMARK")
    print(f"{sep}")

    all_results = {}

    for q in QUERIES:
        hits, _ = retrieve(
            query=q, db=db, chroma_client=None,
            chroma_path=cfg.paths.chroma_root, cfg=cfg, folder_filter=None,
        )
        if not hits:
            continue
        prompt = template.format(numbered_chunks=_fmt(hits), query=q)

        print(f"\n  Query: {q}")
        sys.stdout.flush()

        # Local first
        label = f"local:{LOCAL_MODEL[0]}"
        try:
            r = run_one(local_client, LOCAL_MODEL[0], prompt, {})
            all_results.setdefault(label, []).append(r)
            print(f"    Local {LOCAL_MODEL[1]}: TTFT={r['ttft']:.1f}s  "
                  f"Total={r['total']:.1f}s  Empty={r['empty']}  Ch={r['chars']}")
        except Exception as e:
            print(f"    Local FAIL: {e}")

        # Cloud models
        for model_name, desc in CLOUD_MODELS:
            label = f"cloud:{model_name}"
            try:
                r = run_one(cloud_client, model_name, prompt, {})
                all_results.setdefault(label, []).append(r)
                print(f"    Cloud {desc}: TTFT={r['ttft']:.1f}s  "
                      f"Total={r['total']:.1f}s  Empty={r['empty']}  Ch={r['chars']}")
            except Exception as e:
                print(f"    Cloud {desc} FAIL: {e}")

        sys.stdout.flush()

    # Summary
    print(f"\n{sep}\n  SUMMARY (medians)\n{sep}")
    print(f"{'Model':<25} {'TTFT':>8} {'Text':>8} {'Total':>8} {'Ch/s':>8} {'Empty':>8} {'N':>4}")
    print(f"{'-'*70}")

    for label in all_results:
        results = all_results[label]
        if not results:
            continue
        ttft = statistics.median([r["ttft"] for r in results])
        text = statistics.median([r["text"] for r in results])
        total = statistics.median([r["total"] for r in results])
        cps = statistics.median([r["cps"] for r in results])
        empty = statistics.median([r["empty"] for r in results])
        n = len(results)
        print(f"{label:<25} {ttft:>7.1f}s {text:>7.1f}s {total:>7.1f}s {cps:>7.0f} {empty:>8.0f} {n:>4}")

    print()


if __name__ == "__main__":
    main()
