# Generation Model Migration — Local gemma4 → Cloud gpt-oss:20b

**Date**: 2026-05-11
**Status**: Complete — production now uses Ollama Cloud

## Problem

The local `gemma4:latest` model on Apple Silicon produced 50–97% empty "thinking" tokens, resulting in a median response time of **24–28s**. Disabling thinking via `num_think: 0` redistributed the same computation into text generation — total latency stayed the same or worsened.

## Benchmark Results

All benchmarks used the same 5 RAG queries with identical retrieved context (8 hits, 5k–13k char prompts).

### Phase 1 — Local Models Tested

| Model | TTFT | Total | Empty tokens | Notes |
|-------|------|-------|-------------|-------|
| **gemma4:latest (local)** | 16.8s | **24.0s** | 693 | Baseline — 72% spent on thinking |
| gemma4:latest + num_think=0 | 13.7s | 27.8s | 735 | 4% worse than default |
| wizard2:latest (local) | 24.1s | 35.7s | 1 | No thinking, but 3x slower generation |
| qwen3.5:9b + num_think=0 (local) | 165.5s | 171.4s | 3,881 | num_think not supported |
| glm-4.7-flash:latest (local) | 58.9s | 60.2s | 1,168 | Slower than gemma4 |

**Conclusion**: No installed local model beats gemma4. Thinking is baked into the architecture.

### Phase 2 — Ollama Cloud Models Tested

| Model | TTFT | Total | Empty tokens | Reliability |
|-------|------|-------|-------------|-------------|
| **gpt-oss:20b (cloud)** | 1.8s | **4.6s** | 260 | 5/5 ✅ |
| ministral-3:3b (cloud) | 0.6s | **2.6s** | 2 | 5/5 ✅ |
| gemma3:4b (cloud) | 0.8s | 1.5s | 2 | 2/5 ❌ (fails on prompts >7K chars) |
| gemma3:12b (cloud) | 1.0s | 1.4s | 2 | 2/5 ❌ (same issue) |
| gemma3:27b (cloud) | 1.1s | 3.8s | 2 | 2/5 ❌ (same issue) |
| glm-4.6 (cloud) | 28.4s | 47.2s | 785 | 5/5 ✅ but worse than local |

**gemma3 family**: All variants fail with HTTP 500 on prompts >10K chars — a server-side context limit on Ollama Cloud, not fixable client-side.

**ministral-3:3b**: Fastest (2.6s) but produces shorter responses — 3B model may lack depth for complex queries.

**gpt-oss:20b**: Best balance — **5x faster** (4.6s vs 24.0s), handles all prompt sizes, produces rich responses with citations.

### Comparison: gpt-oss:20b vs gemma4

| Query | gemma4 (local) | gpt-oss:20b (cloud) | Speedup |
|-------|---------------|---------------------|---------|
| robot performance rental service | 14.8s | 2.4s | 6.2x |
| key themes across folders | 17.0s | 4.6s | 3.7x |
| budget/cost info | 29.5s | 5.1s | 5.8x |
| Kettybot proposal | 23.5s | 4.2s | 5.6x |
| CDC waiter system | 31.4s | 6.7s | 4.7x |
| **Median** | **17.0s** | **4.6s** | **~4x** |

## Architecture Changes

### New config structure: `models.generation.endpoint`

```toml
[models.generation]
name               = "gpt-oss:20b"
fallback_name      = "gemma4:latest"
alternate_name     = "gemma4:26b"
temperature        = 0.5
top_p              = 0.95
top_k              = 64
max_context_tokens = 16384
prompt_version     = "v1"
thinking           = false

# Generation endpoint. Swap host to switch between cloud and local.
# Cloud: host = "https://ollama.com", api_key = "<key>"
# Local: host = "http://localhost:11434", api_key = ""
[models.generation.endpoint]
host    = "https://ollama.com"
api_key = "<your-ollama-api-key>"
retries = 3
```

### Switching back to local

Edit `config.toml`:

```toml
[models.generation]
name = "gemma4:latest"

[models.generation.endpoint]
host    = "http://localhost:11434"
api_key = ""
retries = 3
```

Requires a server restart — config is loaded once at startup.

## Files Changed

### 1. `pipeline/config.py`

- Added `GenerationEndpointConfig` Pydantic model:
  - `host: str = "http://localhost:11434"`
  - `api_key: str = ""`
  - `retries: int = 3`
- Added `endpoint: GenerationEndpointConfig` field to `GenerationConfig`
- Updated `DEFAULT_CONFIG_TOML` with the endpoint block

### 2. `config.toml`

- Changed `[models.generation].name` from `gemma4:latest` to `gpt-oss:20b`
- Added `[models.generation.endpoint]` block with `https://ollama.com` host and API key

### 3. `rag/generation.py`

- Added `_make_client(cfg)` helper:
  - If `cfg.models.generation.endpoint` exists, creates client with `host` and optional `Authorization: Bearer` header
  - Falls back to `cfg.ollama.host` if no endpoint config
- Updated `generate_answer()` to use `_make_client(cfg)` instead of hardcoded localhost

### 4. `ui/gradio_app.py`

- Added `_make_client()` helper (same logic, uses global `_cfg`)
- Updated `_stream_tokens()` to use `_make_client()`
- Updated `_check_ollama()` to use `_make_client()`
- Status bar shows endpoint host (e.g., `Generation (ollama.com)`)

### 5. Benchmark Scripts (new files)

- `ui/benchmark.py` — Original benchmark (retrieval + generation + E2E)
- `ui/BENCHMARK_REPORT.md` — Full report with thinking ON/OFF comparison
- `ui/benchmark_compare.py` — wizard2 vs gemma4 comparison
- `ui/benchmark_num_think.py` — gemma4 num_think=0 vs default
- `ui/benchmark_interleaved.py` — Same query, both modes back-to-back
- `ui/benchmark_qwen35.py` — qwen3.5:9b vs gemma4
- `ui/benchmark_cloud.py` — Ollama Cloud vs Local (4 models)
- `ui/benchmark_gemma3_debug.py` — gemma3:4b failure analysis with retries
- `ui/benchmark_gemma3_variants.py` — All gemma3 sizes vs ministral-3
- `ui/benchmark_gpt_oss.py` — gpt-oss:20b final comparison

## Tradeoffs

| Factor | Local (gemma4) | Cloud (gpt-oss:20b) |
|--------|---------------|---------------------|
| Response time | 24s median | 4.6s median |
| Offline capable | Yes | No (needs internet) |
| Cost | Free (local GPU) | Per-token on Ollama Cloud |
| Response quality | Good | Comparable |
| Citation behavior | Works | Works |
| Data privacy | Local | Sent to Ollama Cloud |

## Key Learnings

1. **`num_think: 0` on gemma4** — Exists as an Ollama parameter but doesn't skip computation, just redistributes it. Total latency unchanged.
2. **Gemma3 on Cloud** — All sizes (4b/12b/27b) fail on prompts >10K chars with HTTP 500. Server-side context limit.
3. **qwen3.5:9b** — Doesn't support `num_think`; thinking is baked into model weights.
4. **ministral-3:3b** — Fastest but smallest; trades response depth for speed.
5. **gpt-oss:20b** — Sweet spot: 5x faster, reliable on all prompt sizes, comparable quality.
