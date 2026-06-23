# RAG Pipeline Performance Benchmark — 2026-05-07

**Model**: gemma4:latest (Ollama, local)
**Embedding**: qwen3-embedding:8b
**Chroma**: 8,159 chunks across 2,177 files, 71 folders
**Machine**: macOS (Apple Silicon)
**Queries tested**: 15 diverse prompts from the prompt chip set

## Executive Summary — Thinking vs Thinking Disabled

| Phase | Thinking ON (median) | Thinking OFF (median) | Difference |
|-------|---------------------|----------------------|------------|
| **Retrieval** | 0.32s | 0.39s | +0.07s (no change) |
| **Thinking (TTFT)** | 20.26s | 22.71s | +2.45s (slightly worse) |
| **Text output** | 7.72s | 6.46s | -1.26s (slightly faster) |
| **Generation total** | **28.24s** | **31.78s** | **+3.5s (worse)** |
| **End-to-end (user)** | 27.80s | N/A | — |

**`enable_thinking: false` made generation 12% slower (median), not faster.** Reverted.

## Baseline Results (Thinking ON — Current Production)

| Phase | Median | Mean | Min | Max | p90 | p95 |
|-------|--------|------|-----|-----|-----|-----|
| **Retrieval** | 0.32s | 0.84s | 0.29s | 8.10s | 0.44s | 8.10s |
| **Thinking (TTFT)** | 20.26s | 23.27s | 10.08s | 35.85s | 34.22s | 35.85s |
| **Text output** | 7.72s | 6.59s | 0.30s | 10.73s | 10.71s | 10.73s |
| **Generation total** | 28.24s | 29.86s | 11.38s | 44.96s | 43.56s | 44.96s |
| **End-to-end (user)** | 27.80s | 30.12s | 13.55s | 50.42s | 49.95s | 50.42s |

**Key takeaway**: The user waits ~28s on average for a response. Of that, **retrieval is 1% (0.3s)** and **generation is 99% (28s)**. The dominant factor is the model thinking before producing any text.

## `enable_thinking: false` Experiment — Why It Failed

Setting `"enable_thinking": false` in the Ollama API call was expected to suppress the thinking tokens entirely. On simple prompts ("What is 2+2?") it works — text appears in 0.2-0.5s with no empty tokens. But on complex RAG prompts (5k-39k chars context), the model still produces many empty tokens, just in a less structured pattern. The results:

| Metric | Thinking ON | Thinking OFF | Change |
|--------|------------|-------------|--------|
| Median generation | 28.24s | 31.78s | +12.4% |
| Mean generation | 29.86s | 35.34s | +18.3% |
| Max generation | 44.96s | 97.07s | +116% |
| Median thinking time | 20.26s | 22.71s | +12% |
| Text output speed | 170 ch/s | 145 ch/s | -15% |
| Empty token ratio | 51-97% | 53-76% | No improvement |

**Notable failures**: "what documents are in the knowledge base" took **97s** with thinking disabled vs 44s with it on. The model appears to enter an even longer pre-computation phase when thinking is disabled on large-context prompts.

**Conclusion**: `enable_thinking: false` is not a viable optimization for this workload. The model's internal computation time is not avoided — just restructured into a less efficient pattern.

## Phase 1: Retrieval

Retrieval is fast and consistent. The first query has a cold-start penalty (8.1s) from loading the embedding model into memory. After that:

| Metric | Value |
|--------|-------|
| Warm retrieval | 0.29s – 0.44s |
| Median | 0.32s |
| BM25 | Always 0 hits (FTS not populated with chunks) |
| Dense hits | Always 50 (top-k) |
| Fused total | Always 50 (dense-only since BM25=0) |
| Chunks returned | Always 8 (post-dedup/top-k cap) |
| Language detection | Always English (fasttext model not loaded) |

**Analysis**: Retrieval is not the bottleneck. The dense embedding query against Chroma with 8,159 vectors takes ~300ms consistently. BM25 returns 0 because the SQLite FTS index wasn't populated with chunk text. The `rebuild` call in retrieval.py runs every query, adding a few ms.

**Cold start**: The first query takes 8.1s because the embedding model (`qwen3-embedding:8b`) must be loaded into GPU/MLX memory. Subsequent queries reuse the loaded model.

## Phase 2: Generation

This is the dominant phase. The model generates many "thinking tokens" (empty content) before producing actual text.

### Thinking vs Text Output

| Query | Think (s) | Text (s) | Total (s) | Output chars | Chars/s | Tokens | Empty % |
|-------|-----------|----------|-----------|-------------|---------|--------|---------|
| robot performance rental service | 17.4 | 0.3 | 17.8 | 65 | 185.6 | 442 | 94% |
| what documents are in the knowledge base | 35.9 | 7.7 | 43.6 | 1,265 | 164.0 | 1,385 | 69% |
| summarise the main topics covered | 32.1 | 1.6 | 33.7 | 296 | 184.5 | 661 | 84% |
| find policies or procedures | 32.4 | 4.0 | 36.4 | 277 | 69.6 | 1,030 | 88% |
| key themes across all folders | 13.0 | 5.6 | 18.6 | 876 | 155.5 | 878 | 63% |
| compare latest version | 11.1 | 0.3 | 11.4 | 115 | 378.3 | 609 | 97% |
| dependencies between documents | 10.1 | 4.5 | 14.5 | 1,416 | 317.9 | 832 | 62% |
| identify gaps | 20.3 | 7.1 | 27.4 | 591 | 83.1 | 1,191 | 72% |
| extract budget info | 18.1 | 9.8 | 27.9 | 1,205 | 122.3 | 1,051 | 64% |
| draft summary email | 15.9 | 10.6 | 26.5 | 2,360 | 222.5 | 1,519 | 51% |
| table of action items | 32.8 | 10.7 | 43.5 | 1,547 | 144.5 | 2,440 | 70% |
| what deadlines coming up | 34.2 | 10.7 | 45.0 | 1,898 | 176.8 | 1,755 | 60% |
| executive summary | 31.1 | 8.3 | 39.4 | 1,461 | 175.6 | 1,289 | 58% |
| Kettybot proposal | 19.3 | 8.9 | 28.2 | 1,129 | 126.6 | 1,681 | 63% |
| CDC waiter system | 25.4 | 8.5 | 33.9 | 1,449 | 169.8 | 1,435 | 73% |

### Generation Summary Stats

| Metric | Value |
|--------|-------|
| Thinking time (TTFT) median | 20.26s |
| Text output median | 7.72s |
| Total generation median | 28.24s |
| Token throughput | 178 chars/s (median 170) |
| Empty tokens | 51–97% of all tokens |
| Prompt sizes | 5.6k – 38.8k chars (depends on retrieved context) |

**Analysis**:
- **Thinking dominates**: 78% of generation time is spent producing empty thinking tokens. The user sees nothing during this period.
- **Text output is fast once it starts**: 170 chars/s is reasonable for a local 8B model.
- **High variance**: Thinking time ranges from 10s to 36s depending on query complexity and context size.
- **Prompt size matters**: Larger prompts (30k+ chars) tend to have longer thinking times (34-36s).
- **Empty token ratio**: 51-97% of tokens are empty (thinking). This is the single biggest optimization opportunity.

## Phase 3: End-to-End (Gradio API)

This measures what the user actually experiences — from clicking "Send" to seeing the full response.

| Metric | Value |
|--------|-------|
| POST latency | 13.7 – 33.0ms (negligible) |
| First data event | 13.53 – 50.40s |
| SSE duration | 13.53 – 50.40s |
| Total end-to-end | 13.55 – 50.42s |
| SSE events | 441 – 1,923 |
| Median E2E | 27.80s |

**Analysis**: The end-to-end time matches the generation time closely (both ~28s median), confirming that retrieval is negligible and the HTTP/Gradio overhead is minimal (POST latency <35ms).

## Where Time Is Spent (Breakdown)

```
┌─────────────────────────────────────────────────────┐
│  Total user wait: ~28s (median)                      │
│                                                      │
│  Network/HTTP overhead:  0.03s    (0.1%)             │
│  Retrieval (embedding):   0.32s    (1.1%)            │
│  Prompt building:        ~0.01s    (0.0%)            │
│  Model thinking:         20.26s   (72.4%)  ← BIGGEST │
│  Text generation:         7.72s   (27.6%)            │
│  Citations parsing:      ~0.01s    (0.0%)            │
└─────────────────────────────────────────────────────┘
```

## Bottleneck Analysis

### 1. Model Thinking (72% of total time) — CRITICAL
Gemma 4 produces 51-97% empty "thinking" tokens before generating actual text. This is the single largest contributor to response latency. The user sees nothing during this period — just the "Thinking..." indicator.

**Potential fixes**:
- Use a model without thinking mode (e.g., `llama3.2`, `mistral`, `qwen2.5`)
- Disable thinking in config if the model supports it
- Use a smaller/faster model for faster TTFT
- Stream thinking tokens as "..." to give user feedback (already implemented)

### 2. Embedding Cold Start (8s first query) — MODERATE
The first query after server restart takes 8.1s for retrieval vs 0.3s warm. This is the embedding model loading time.

**Potential fixes**:
- Pre-warm the embedding model on startup
- Keep Ollama running persistently (don't let it unload models)
- Use `ollama keep qwen3-embedding:8b` to prevent unloading

### 3. Text Generation Speed (7.7s median) — ACCEPTABLE
Once the model starts generating text, it produces ~170 chars/s. This is reasonable for a local model. Longer responses (1,500+ chars) take 8-11s.

**Potential fixes**:
- Use a faster model (smaller parameter count)
- Limit max output tokens to keep responses concise
- Use a model optimized for speed over quality

### 4. Retrieval (0.32s median) — FINE
Dense-only retrieval (BM25 returns 0). 300ms for 8,159 vectors is acceptable.

**Potential fixes**:
- Populate the FTS index to enable BM25 hybrid retrieval
- Cache embeddings for common queries
- Reduce top-k from 50 to something smaller

## Recommendations by Impact

| Priority | Change | Expected Impact | Effort | Notes |
|----------|--------|-----------------|--------|-------|
| 1 | **Switch model entirely** (llama3.2, qwen2.5) | 15-20s faster (50-70% reduction) | Low | `enable_thinking: false` proved insufficient — need a non-thinking model |
| 2 | Limit max response tokens | 3-5s faster on long answers | Low | Reduces tail latency on verbose queries |
| 3 | Pre-warm embedding model | 8s saved on first query | Low | `ollama keep qwen3-embedding:8b` |
| 4 | Enable BM25/FTS index | Better retrieval quality | Medium | Not a speed improvement but better results |
| 5 | Reduce top-k (50→20) | 0.1s saved on retrieval | Low | Marginal but free |

**Important**: The `enable_thinking: false` experiment showed that gemma4's thinking is deeply tied to its inference pattern. Simply disabling the flag doesn't skip computation — it restructures it into a slower path. The real fix is a different model.

## Benchmark Script

Run with:
```bash
PYTHONPATH=. .venv/bin/python ui/benchmark.py [-n QUERIES] [--skip-gradio] [--skip-generation] [--json]
```

The script measures three phases independently:
- **Phase 1**: Direct Python call to `retrieve()` — measures retrieval in isolation
- **Phase 2**: Direct Python call to Ollama chat stream — measures generation with thinking vs text breakdown
- **Phase 3**: HTTP calls to the running Gradio API — measures end-to-end user experience

Each phase runs 15 diverse queries and computes min/max/mean/median/stdev/p50/p90/p95.
