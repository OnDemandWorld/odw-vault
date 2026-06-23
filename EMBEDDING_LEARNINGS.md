# Embedding Phase — Learnings & Timing

> Date: 2026-05-04
> Model: qwen3-embedding:8b (local Ollama)
> Corpus: 8,159 chunks with context augmentation (avg 191 chars context + 981 chars text = ~1.17 KB combined)

---

## What This Phase Does

Generates 4096-dim dense embeddings for every chunk using `qwen3-embedding:8b` via
Ollama. The context text (generated in Phase 10.5) is prepended to the chunk text:
`{context}\n\n{chunk_text}`. Results are stored in ChromaDB (`chunks__qwen3emb8b`
collection) and tracked in the `embedding_ref` table.

Batch size: 32 chunks per Ollama call.

---

## Measured Timing (POC Run #1)

| Metric | Value |
|---|---|
| Total chunks | 8,159 |
| Batch size | 32 |
| Total batches | ~255 |
| Model warmup | ~60 seconds (first batch, model loading into GPU) |
| Steady-state rate | ~130-166 chunks/min (4-5 batches/min) |
| Per-batch latency | ~6-8 seconds |
| **Estimated total** | **~50-65 minutes** |

This is for a POC-scale corpus (8K chunks). For production at 100K chunks, expect
~10-12 hours with the same setup.

---

## Bugs Hit & Fixes Applied

### 1. Config mismatch — OpenAI-compat endpoint caused 404

**What happened:** After switching `config.toml` `ollama.host` to the remote
OpenAI-compatible endpoint (`http://192.168.0.72:28100/v1`) for context
generation, the embedding phase broke with:

```
POST http://192.168.0.72:28100/v1/api/embed "HTTP/1.1 404 Not Found"
```

**Root cause:** The embedding code uses `ollama.Client` which calls Ollama's
native `/api/embed` endpoint. The remote service only serves OpenAI-compatible
`/v1/embed` endpoints.

**Fix:** Reverted `ollama.host` to `http://localhost:11434` for embedding.

**Lesson:** The single `[ollama].host` config is a bottleneck when different
phases use different servers. See "Future Improvements" below.

### 2. No timeout — process hung indefinitely

**What happened:** The embedding process would process ~5,408 chunks (169 batches)
then hang indefinitely. No error, no timeout, no retry.

**Root cause:** `ollama.Client(host=host)` was created without a timeout
parameter. The `timeout_seconds = 300` value in `config.toml` was parsed but
**never passed** to the client. When a slow batch hit (likely large text or model
memory pressure), `client.embed()` hung forever. The tenacity retry decorator
only catches exceptions, not hangs.

**Fix:** Added `timeout` parameter to `_ollama_embed()` and threaded it through
from `cfg.ollama.timeout_seconds` in `run_embed()`.

```python
client = ollama.Client(host=host, timeout=timeout)  # was: Client(host=host)
```

### 3. Duplicate `embedding_ref` rows on re-embed

**What happened:** Running `--reembed` created new INSERT rows instead of
replacing old ones. After two failed runs, the count was 16,479 instead of 8,159
— exactly 2x duplicates.

**Root cause:** The `embedding_ref` table has no UNIQUE constraint on
`(chunk_id, embedding_model)`. The insert logic does a blind `INSERT` without
checking for existing rows.

**Fix:** Added `DELETE FROM embedding_ref WHERE embedding_model = ?` before
the re-embed loop, so old refs are cleared before new ones are inserted.

**Note:** This is a band-aid. A proper fix would add a UNIQUE constraint to the
schema and use `INSERT OR REPLACE` or `ON CONFLICT` upsert logic.

---

## Comparison: Context vs Embedding

| Phase | Items | Time | Items/Min | Notes |
|---|---|---|---|---|
| Context generation (Qwen3-8B remote) | 8,159 | ~11.5 hours | 5.7 | Sequential LLM calls |
| Embedding (qwen3-emb:8b local) | 8,159 | ~1 hour | 140 | Batched (32 per call) |

Context generation is ~12x slower because it's a sequential reasoning task with
longer model outputs (191 chars avg vs a 4096-dim vector). Embedding is
fast because it processes 32 items per batch.

---

## Cost Implications

| Approach | Time | Cost | Notes |
|---|---|---|---|
| Local (qwen3-embedding:8b) | ~1 hour | $0 | GPU power only |
| Ollama API (remote) | ~1 hour | $0 | Same model, different host |
| OpenAI text-embedding-3-large | ~5 min | ~$0.50 | 8K x ~1.2KB = ~10M tokens @ $0.05/MTok |
| OpenAI text-embedding-3-small | ~5 min | ~$0.10 | Same tokens @ $0.02/MTok |

For the POC, local is the right choice. At production scale (100K+ chunks), API
embedding would be ~$5-10 and complete in minutes — worth considering if time
matters more than cost.

---

## Future Improvements

### Per-Phase Endpoint Routing

Currently `[ollama].host` is a single global value. Different phases use different
models on different servers:

| Phase | Current Host | Model |
|---|---|---|
| Context generation | `192.168.0.72:28100/v1` (OpenAI-compat) | Qwen3-8B-4bit (MLX) |
| Embedding | `localhost:11434` (Ollama native) | qwen3-embedding:8b |
| Summarization | `localhost:11434` (Ollama native) | gemma4:latest |
| Generation | varies | gemma4:latest / 26b |

**Proposed config structure:**
```toml
[endpoints.context]
host = "http://192.168.0.72:28100/v1"
api_type = "openai_compat"
model = "mlx-community/Qwen3-8B-4bit"

[endpoints.embedding]
host = "http://localhost:11434"
api_type = "ollama_native"
model = "qwen3-embedding:8b"

[endpoints.summarization]
host = "http://localhost:11434"
api_type = "ollama_native"
model = "gemma4:latest"
```

This eliminates the manual config-switching workaround we had to do between
context generation and embedding.

### Schema: Add UNIQUE Constraint to `embedding_ref`

Add `UNIQUE(chunk_id, embedding_model)` or a composite primary key so that
`INSERT OR REPLACE` can handle re-embeds cleanly without the DELETE-then-INSERT
workaround.

### Parallel Batches

Currently embedding batches run sequentially. With a capable GPU server that can
handle concurrent requests, parallel batch processing could cut the ~1 hour
embedding time significantly. The Ollama API itself doesn't support concurrent
requests to the same model, but a multi-instance setup (e.g., multiple Ollama
workers) could enable this.

### Progress Reporting

The embed CLI has no `--limit` option and no progress bar during embedding. Adding
a Rich progress bar (like the pre-flight phases use) would make long runs more
visible.
