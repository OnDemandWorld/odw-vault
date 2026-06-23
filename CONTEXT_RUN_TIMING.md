# Context Augmentation — Run Timing Analysis

> Date: 2026-05-04
> Corpus: 8,159 chunks, 384 primary documents

---

## Methodology

Each chunk has a `context_generated_at` timestamp recorded when the context was
written to the database.  The actual processing time was computed by taking
`MAX(context_generated_at) - MIN(context_generated_at)` for each model group.

---

## Measured Run Times

### gemma3n:latest (first attempt — local Ollama)

| Metric | Value |
|---|---|
| Chunks | 4,237 |
| First context | 2026-05-02 15:16:04 UTC |
| Last context | 2026-05-03 04:15:23 UTC |
| Elapsed | 13.0 hours (779 minutes) |
| Rate | 5.4 chunks/min (1 every 11.1 seconds) |
| Avg context length | ~90 chars |

### mlx-community/Qwen3-8B-4bit (replacement — remote MLX)

| Metric | Value |
|---|---|
| Chunks | 3,922 |
| First context | 2026-05-03 15:01:01 UTC |
| Last context | 2026-05-04 02:32:30 UTC |
| Elapsed | 11.5 hours (692 minutes) |
| Rate | 5.7 chunks/min (1 every 10.5 seconds) |
| Avg context length | ~191 chars |

### Quality Cleanup Run (regenerated 86 entries)

| Metric | Value |
|---|---|
| Chunks regenerated | 86 |
| Time | ~27 minutes |
| Rate | 3.2 chunks/min |

### Overall Timeline

| Period | Duration | Activity |
|---|---|---|
| May 2, 15:16 – May 3, 04:15 | 13.0h | gemma3n:latest processed 4,237 chunks |
| May 3, 04:15 – 11:00 | ~7h | gemma3n stalled (too slow), killed |
| May 3, 11:00 – 11:30 | ~30m | Benchmark: gemma3n vs Qwen3-8B |
| May 3, 11:30 – 15:00 | ~3.5h | Code changes (OpenAI-compat support, thinking strip) |
| May 3, 15:01 – May 4, 02:32 | 11.5h | Qwen3-8B processed 3,922 chunks |
| May 4, 02:32 – 02:40 | ~8m | Quality audit |
| May 4, 02:40 – 03:15 | ~35m | Cleanup + regenerate 86 bad entries |
| **Total span** | **~36 hours** | Wall clock from start to clean finish |
| **Pure processing** | **~24.5 hours** | Actual model inference time |

---

## Comparison: All LLM Phases

| Phase | Items | Time | Items/Min | Notes |
|---|---|---|---|---|
| Extraction | 13 files | 3.0 min | 263 | 4-worker parallel |
| Summarization | 43 docs | 13.8 min | 187 | gemma4:latest |
| Embedding | 8,275 items | 59.6 min | 8,330 | qwen3-emb:8b, batched |
| **Context Augmentation** | **8,159** | **~24.5 hours** | **5.5** | **90% of LLM time** |

---

## Cost Implications

If this were done via API instead of local/remote MLX:

| API Model | Est. Time | Est. Cost |
|---|---|---|
| GPT-4o-mini | ~20 min | ~$1.70 |
| Haiku 4.5 | ~20 min | ~$6.30 |
| GPT-4o | ~20 min | ~$17.70 |
| Sonnet 4 | ~20 min | ~$22.70 |
| **Qwen3-8B (MLX remote)** | **11.5 hours** | **$0** |
| gemma3n (Ollama local) | 13.0 hours | $0 |

The tradeoff is clear: API is ~50x faster but costs $2-23. Local is free but takes
a full day for 8K chunks.
