"""Debug: test cloud gemma3:4b to see why it fails."""
import sys, time, os
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


def main():
    api_key = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        print("ERROR: OLLAMA_API_KEY not set")
        sys.exit(1)

    cloud_client = ollama.Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {api_key}"},
    )

    print("Testing cloud gemma3:4b with 3 retries per query...")
    print()

    for idx, q in enumerate(QUERIES, 1):
        hits, _ = retrieve(
            query=q, db=db, chroma_client=None,
            chroma_path=cfg.paths.chroma_root, cfg=cfg, folder_filter=None,
        )
        if not hits:
            print(f"  [{idx}] NO HITS: {q}")
            continue

        prompt = template.format(numbered_chunks=_fmt(hits), query=q)
        prompt_len = len(prompt)

        print(f"  [{idx}] {q}")
        print(f"       Prompt length: {prompt_len} chars, {len(hits)} hits")

        for attempt in range(1, 4):
            t0 = time.monotonic()
            try:
                txt = ""
                t_ft = None
                empty = 0
                for c in cloud_client.chat(
                    model="gemma3:4b",
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
                print(f"       Attempt {attempt}: OK  TTFT={t_ft or total:.1f}s  "
                      f"Total={total:.1f}s  Empty={empty}  Ch={len(txt)}  "
                      f"| {txt[:80].replace(chr(10), ' ')}")

            except Exception as e:
                err_type = type(e).__name__
                err_msg = str(e)[:150]
                elapsed = time.monotonic() - t0
                print(f"       Attempt {attempt}: FAIL  {err_type}  {elapsed:.1f}s  {err_msg}")

        sys.stdout.flush()
        print()


if __name__ == "__main__":
    main()
