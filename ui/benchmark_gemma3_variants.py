"""Test cloud gemma3 variants to find one that handles long prompts."""
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


def test_model(model_name, retries=3):
    print(f"\n  Testing: {model_name}")
    results = []
    for idx, q in enumerate(QUERIES, 1):
        hits, _ = retrieve(
            query=q, db=db, chroma_client=None,
            chroma_path=cfg.paths.chroma_root, cfg=cfg, folder_filter=None,
        )
        if not hits:
            continue
        prompt = template.format(numbered_chunks=_fmt(hits), query=q)
        ok = False
        for attempt in range(1, retries + 1):
            t0 = time.monotonic()
            try:
                txt = ""
                t_ft = None
                empty = 0
                for c in cloud_client.chat(
                    model=model_name,
                    messages=[{"role": "system", "content": "You are a helpful assistant."},
                              {"role": "user", "content": prompt}],
                    options={"temperature": 0.5, "top_p": 0.95, "top_k": 64},
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
                preview = txt[:70].replace("\n", " ")
                print(f"    [{idx}] Attempt {attempt}: OK  "
                      f"TTFT={t_ft or total:.1f}s  Total={total:.1f}s  "
                      f"Empty={empty}  Ch={len(txt)}  | {preview}")
                results.append({"ttft": t_ft or total, "total": total,
                                "text": text_t, "chars": len(txt), "empty": empty})
                ok = True
                break
            except Exception as e:
                elapsed = time.monotonic() - t0
                if attempt == retries:
                    print(f"    [{idx}] Attempt {attempt}: FAIL after {elapsed:.1f}s  {str(e)[:100]}")
                else:
                    print(f"    [{idx}] Attempt {attempt}: FAIL  {str(e)[:80]}  retry...")
        if not ok:
            results.append(None)
    return results


def main():
    sep = "=" * 65
    print(f"\n{sep}\n  CLOUD GEMMA3 VARIANTS — WHICH HANDLES LONG PROMPTS?\n{sep}")

    for model in ["gemma3:4b", "gemma3:12b", "gemma3:27b", "ministral-3:3b"]:
        results = test_model(model)
        valid = [r for r in results if r is not None]
        fails = sum(1 for r in results if r is None)
        if valid:
            print(f"    => {model}: median={statistics.median([r['total'] for r in valid]):.1f}s  "
                  f"TTFT={statistics.median([r['ttft'] for r in valid]):.1f}s  "
                  f"Empty={int(statistics.median([r['empty'] for r in valid]))}  "
                  f"failed={fails}/{len(results)}")
        else:
            print(f"    => {model}: ALL FAILED ({fails})")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
