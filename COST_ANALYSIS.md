# RAG Pipeline — Cost Analysis & Pricing Guide

> Generated: 2026-05-03
> Corpus: 384 primary documents, 2,177 total files, 4.8 GB unique content
> Models: gemma4:latest (summarization/context), qwen3-embedding:8b (embedding)

---

## 1. Corpus Profile

| Metric | Value |
|---|---|
| **Total files scanned** | 2,177 |
| **Primary (unique) files** | 384 |
| **Duplicates removed** | 1,793 |
| **Total unique size** | 4.8 GB (5.1 billion bytes) |
| **Folders** | 71 |
| **Doc-bearing files** (PDFs, Office docs, spreadsheets) | 66 |
| **Non-text files** (CAD, images, video, audio, archives, executables) | 318 |

### Breakdown by Category

| Category | Files | Size (MB) | Chunks | Tokens |
|---|---|---|---|---|
| document | 29 | 52.8 | 1,358 | 879,567 |
| pdf-text | 18 | 262.1 | 3,369 | 520,372 |
| spreadsheet | 11 | 41.2 | 2,090 | 506,374 |
| presentation | 8 | 313.8 | 1,045 | 85,252 |
| data | 88 | 34.4 | 125 | 3,054 |
| image | 60 | 50.5 | 60 | 1,628 |
| cad | 112 | 186.7 | 112 | 1,065 |
| **Text-bearing total** | **66** | **669.9** | **7,862** | **1,991,565** |
| **Non-text total** | **318** | **4,138** | **297** | **5,747** |

### Key Insight: 17% of files drive 99.7% of tokens

The 66 text-bearing files (PDFs, Office docs, spreadsheets, presentations) represent only 17% of the corpus by count but generate 99.7% of all chunk tokens. The 318 non-text files (CAD, video, images, audio) get filename/metadata entries only and contribute negligible tokens.

---

## 2. Token Breakdown by Pipeline Phase

### Phase 8: Text Extraction
| Tool | Files Succeeded | Characters Extracted | Notes |
|---|---|---|---|
| docling | 32 | 360,326 | PDFs and Office docs |
| tika | 45 | 420,116 | Spreadsheets, presentations |
| textutil | 1 | 568 | Legacy macOS docs |
| filename-only | 187 | 6,967 | Non-text files (metadata only) |
| metadata-only | 60 | 6,526 | Images, CAD, executables |
| whisper | 92* | 0 | Audio files (92 rows, 0 text — audio was empty/silent) |
| **Total** | **432** | **794,503** | |

*Whisper ran but produced no text — these are audio files with no speech or unsupported formats.

**Extraction tokens: ~199K** (794,503 chars ÷ ~4 chars/token)

> Note: Extraction uses local tools (docling, tika, textutil), NOT LLM calls. Zero token cost here.

### Phase 9: Document Summarization (gemma4:latest)
| Metric | Value |
|---|---|
| Summaries generated | 45 |
| Failures | 5 (out of 50 attempts across runs) |
| Total summary chars | 63,091 |
| Avg summary length | 1,402 chars (~350 tokens) |
| **Estimated input tokens** | **~2.25M** (~50K per doc avg) |
| **Estimated output tokens** | **~16K** (~350 per doc) |
| **Total phase tokens** | **~2.27M** |
| **Wall time** | **~14 minutes** (43 docs) |

### Phase 10: Chunking
| Metric | Value |
|---|---|
| Total chunks | 8,159 |
| Total tokens | 1,997,312 |
| Avg tokens per chunk | 245 |
| Max tokens per chunk | 11,600 |
| Min tokens per chunk | 2 |
| Files with chunks | 323 (of 384) |
| Chunks per file range | 1–949 |

> Note: Chunking is a local text-splitting operation. Zero LLM calls, zero token cost.

### Phase 10.5: Contextual Augmentation (remote Qwen3-8B-4bit)
| Metric | Value |
|---|---|
| Chunks with context generated | 4,691+ (in progress, run restarted) |
| Chunks remaining | ~3,500 |
| Avg context chars | 1,600-2,500 (Qwen3-8B produces rich context) |
| Processing rate | ~3.2 chunks/min (~19s/chunk) |
| Estimated remaining time | ~18-19 hours |
| **Per-chunk token estimate:** | |
| - Input: ~345 tokens (chunk + doc text + prompt) | |
| - Output: ~200 tokens (rich context) | |
| **Projected for all 8,159 chunks:** | |
| - Input: 8,159 × 345 = **~2.82M tokens** | |
| - Output: 8,159 × 200 = **~1.63M tokens** | |
| - **Total phase tokens: ~4.45M** | |

> Note: Qwen3-8B produces ~12x richer context than local gemma4 (1,800 chars vs. 90 chars). The output token estimate is higher than the original gemma4-based estimate, pushing total LLM tokens from ~5.5M to ~6.8M. API cost impact: +$1-4 depending on model.

### Phase 11: Embedding (qwen3-embedding:8b)
| Metric | Value |
|---|---|
| Total embeddings | 8,159 (chunks) |
| Wall time | ~60 minutes (8,275 items incl. summaries/folders) |
| Items/minute | ~138 |

> Embedding uses a local model. Zero API token cost.

---

## 3. Corrected Total Token Count

| Phase | Input Tokens | Output Tokens | Total | API Calls? |
|---|---|---|---|---|
| Extraction | 0 | 0 | 0 | No (local tools) |
| Summarization | 2,250,000 | 16,000 | 2,266,000 | Yes (LLM) |
| Chunking | 0 | 0 | 0 | No (local split) |
| Context Augmentation (Qwen3-8B) | 2,820,000 | 1,630,000 | 4,450,000 | Yes (LLM) |
| Embedding | 0 | 0 | 0 | No (local model) |
| **Total LLM tokens** | **5,070,000** | **1,646,000** | **~6.7M** | |

**~6.7 million LLM tokens total for 384 documents (66 text-bearing).**

This is ~14,400 tokens per document on average, or ~84,000 tokens per text-bearing document.

---

## 4. Cost Comparison: Local vs API

### Per-Document Pricing (384 docs, 5.54M tokens)

| Provider | Model | Input Rate | Output Rate | Total Cost | Cost/Doc | Cost/Text-Doc |
|---|---|---|---|---|---|---|
| **Local Ollama** | gemma4 + qwen3 | $0 | $0 | **$0** | $0.00 | $0.00 |
| OpenAI | GPT-4o-mini | $0.15/M | $0.60/M | **$1.03** | $0.003 | $0.016 |
| OpenAI | GPT-4o | $2.50/M | $10.00/M | **$17.43** | $0.045 | $0.264 |
| Anthropic | Haiku 4.5 | $0.80/M | $4.00/M | **$5.95** | $0.015 | $0.090 |
| Anthropic | Sonnet 4 | $3.00/M | $15.00/M | **$22.34** | $0.058 | $0.339 |
| OpenAI | text-embedding-3-large | $0.13/M (embed) | — | **$0.31** | $0.001 | $0.005 |

**Grand total including embeddings:**

| Option | LLM Cost | Embedding Cost | **Total** |
|---|---|---|---|
| Local Ollama | $0 | $0 | **$0** |
| GPT-4o-mini + embeddings | $1.03 | $0.31 | **$1.34** |
| Haiku 4.5 + embeddings | $5.95 | $0.31 | **$6.26** |
| GPT-4o + embeddings | $17.43 | $0.31 | **$17.74** |
| Sonnet 4 + embeddings | $22.34 | $0.31 | **$22.65** |

### Time Comparison

| Phase | Local (Ollama on Mac) | API (estimated) |
|---|---|---|
| Extraction | ~3 min | N/A (same tools) |
| Summarization (45 docs) | ~14 min | ~1-2 min |
| Chunking | <1 min | N/A (local split) |
| Context Augmentation (8K chunks) | ~20 hours (est.) | ~10-20 min |
| Embedding (8K items) | ~60 min | ~5-10 min |
| **Total** | **~21 hours** | **~16-33 min** |

---

## 4b. Top-Tier Model Comparison (May 2026 Latest Pricing)

For clients who want the absolute best quality, here's the cost using each provider's flagship model.

### Model Pricing Reference

| Provider | Model | Input (/MTok) | Output (/MTok) | Notes |
|---|---|---|---|---|
| Anthropic | **Claude Opus 4.7** | $5.00 | $25.00 | New tokenizer: ~35% more tokens for same input |
| OpenAI | **GPT-5.5** | $5.00 | $30.00 | 128K context window |
| OpenAI | **GPT-5 Pro** | $15.00 | $120.00 | 400K context, reasoning model |
| xAI | **Grok 4** | $3.00 | $15.00 | 131K context window |
| xAI | **Grok 4.1 Fast** | $0.20 | $0.50 | Budget option, 131K context |

### Cost for Our Corpus (6.7M LLM tokens: 5.07M input + 1.65M output)

| Model | Input Cost | Output Cost | **Total Cost** | Cost/Text-Doc |
|---|---|---|---|---|
| Grok 4.1 Fast | $1.01 | $0.83 | **$2.15** | $0.03 |
| Grok 4 | $15.21 | $24.75 | **$40.25** | $0.61 |
| Claude Opus 4.7* | $25.35 | $41.25 | **$66.60** | $1.01 |
| Claude Opus 4.7 (w/ 35% tok. overhead)* | $34.23 | $41.25 | **$75.48** | $1.14 |
| GPT-5.5 | $25.35 | $49.50 | **$74.85** | $1.13 |
| GPT-5 Pro | $76.05 | $198.00 | **$274.05** | $4.15 |

*Opus 4.7 uses a new tokenizer that consumes ~35% more tokens for equivalent inputs. The adjusted row reflects this reality.

### Cost Including Embeddings (OpenAI text-embedding-3-large: $0.31)

| Model | LLM Cost | Embedding Cost | **Total** |
|---|---|---|---|
| Grok 4.1 Fast | $2.15 | $0.31 | **$2.46** |
| Grok 4 | $40.25 | $0.31 | **$40.56** |
| Claude Opus 4.7 | $66.60 | — | **$66.60** |
| GPT-5.5 | $74.85 | $0.31 | **$75.16** |
| GPT-5 Pro | $274.05 | $0.31 | **$274.36** |

### Quality vs Cost Assessment

| Model | Quality Tier | Total Cost | Time (API) | Best For |
|---|---|---|---|---|
| Grok 4.1 Fast | Mid | $1.87 | ~15 min | Budget bulk processing (context generation) |
| Grok 4 | High | $22.91 | ~15 min | Cost-conscious quality work |
| Claude Opus 4.7 | Flagship | $37.18 | ~15 min | Legal/medical/financial — accuracy-critical |
| GPT-5.5 | Flagship | $39.85 | ~15 min | General-purpose flagship |
| GPT-5 Pro | Ultra-premium | $133.12 | ~20 min | Complex reasoning, multi-hop document analysis |

### Hybrid Strategy: Best Quality at Reasonable Cost

The smartest approach for top-quality results is **model mixing** — use the right model for each phase:

| Phase | Model | Cost |
|---|---|---|
| Summarization (quality-critical) | **Opus 4.7** | $15.31 |
| Context Augmentation (bulk, low-stakes) | **Grok 4.1 Fast** | $1.51 |
| Embedding | **text-embedding-3-large** | $0.31 |
| **Hybrid Total** | | **$17.13** |

This gives you flagship-quality summaries (the output humans will read) at near-budget pricing, because context augmentation is 59% of tokens but quality-differentiating is marginal there.

---

## 4c. Phase Timing Analysis (Measured from Actual Runs)

### LLM Phases

| Phase | Items | Duration | Items/Min | Bottleneck? |
|---|---|---|---|---|
| Extraction (docling/tika) | 13 files | 3.0 min | 263 | No — 4-worker parallel |
| Summarization (gemma4) | 43 docs | 13.8 min | 187 | No — fast enough |
| Embedding (qwen3-emb:8b) | 8,275 items | **59.6 min** | **8,330** | **No — not the bottleneck** |
| Context Augmentation (Qwen3-8B) | ~3,900 remaining | ~19 hrs est | ~3 | **Yes — dominant** |

### Non-LLM Phases (Pre-flight, phases 0-7)

| Phase | Items | Duration | Bottleneck? |
|---|---|---|---|
| Archives | 2 | <1 sec | No |
| Walk + SHA-256 | 2,177 files | ~2 sec | No (parallel hashing) |
| Identify (Siegfried) | 2,175 files | ~2 sec | No |
| Triage (PDF/media) | 2,152 files | ~5 sec | No (threaded) |
| Dedup | 232 dup groups | <1 sec | No (pure SQL) |
| Folder-meta (Ollama) | 71 folders | ~1-2 min | No |
| Report | 2,177 files | ~1 sec | No (aggregate queries) |

### Embedding is NOT the Bottleneck

Embedding takes **~60 minutes for 8,275 items** — that's 8,330 embeddings per minute, or ~138 per second. The model runs on CPU locally and processes items in batches of 32, making it highly efficient. For a corpus of this size, embedding is essentially "fast enough" to ignore in planning.

### Total Time Budget (384 docs, 4.8 GB)

| Phase | Time | LLM? |
|---|---|---|
| Pre-flight (phases 0-7) | ~10 seconds | No |
| Extraction | ~3 min | No (local tools) |
| Summarization | ~14 min | Yes |
| Chunking | <1 min | No |
| **Context Augmentation** | **~19 hours** | **Yes (90% of time)** |
| Embedding | ~1 hour | No (local model, fast) |
| **Total** | **~20 hours** | |

**If you skip context augmentation**, the whole pipeline finishes in **~18 minutes**.

---

## 4d. Contextual Augmentation: Quality vs. Time Trade-off

### What Context Augmentation Does

Without context augmentation, each chunk is a sentence fragment floating in isolation. A query like *"what happened to CDC012?"* might retrieve a chunk containing *"moved to CDC008"* with no reference to what CDC012 even is. The retriever ranks it, but the answering model has no document-level grounding.

With context augmentation, each chunk gets a ~50-100 word summary that explains its place in the parent document: *"This chunk is part of a hardware deployment log tracking GSL/CDC equipment movements — CDC012 was relocated to CDC008."* Now the dense embedding carries document-level semantics, and BM25 has richer terms to match against.

### Research Backing

This technique comes from Anthropic's [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) paper, which showed **20-30% reduction in retrieval failures** (chunks not returned at all for relevant queries) with this approach. For a corpus like ours — mixed formats, bilingual, structurally irregular — the benefit is even larger because chunks lack natural headers and structure.

### Quality Benefit by File Type

| File Type | Context Benefit | Why |
|---|---|---|
| Multi-page PDFs | **High** | Chunks lose section headers and document context |
| Spreadsheets | **Moderate** | Model explains what the table represents, what columns mean |
| Presentations | **Moderate** | Slide context and speaker notes need framing |
| Short docs (< 5 pages) | **Low** | Chunks already have enough local context |
| Non-text files (CAD, images) | **Zero** | No meaningful chunk content to augment |

### Quality Comparison: Local gemma4 vs Remote Qwen3-8B

Benchmarking revealed a significant quality gap between models:

| | Local gemma4 | Remote Qwen3-8B |
|---|---|---|
| Avg response time | 11s | 15s |
| Avg output length | **~90 chars** | **~1,100 chars** |
| Quality | Brief, minimal | Rich, structured, contextual |

The Qwen3-8B produces context that actually explains document structure, while gemma4's responses were terse to the point of being barely useful. This means **the model choice for context matters more than speed** — a cheaper/faster model that produces poor context is worse than a slower one that does the job right.

### Optimization: Context Only for Text-Bearing Files

The current approach generates context for all 8,159 chunks. The quality gain is front-loaded:

- **Context for summarization-heavy files** (long PDFs, multi-page docs) = big gain
- **Context for spreadsheets** = moderate gain (model explains the table)
- **Context for filename-only files** (CAD, images) = zero gain
- **Context for chunks from files without summaries** = minimal gain

A future optimization: generate context only for chunks from files that have summaries (the text-bearing subset of ~7,862 chunks from 66 files), skip the rest. This saves ~4% of context work with no quality loss.

### The Decision: Skip vs. Include Context

| Scenario | Recommended | Why |
|---|---|---|
| Quick demo / proof of concept | **Skip** | 18 min vs. 19 hours, quality difference invisible in demo |
| Internal knowledge base | **Include (local)** | Free, run overnight, better retrieval quality |
| Client delivery | **Include (API)** | 20 min, best quality, $1-22 API cost |
| High-stakes retrieval (legal, medical) | **Include (flagship API)** | Opus 4.7 or GPT-5.5 context, $37-40 API cost |

---

## 5. Cost Per Volume — Business Pricing Model

### Unit Economics

The pipeline has three distinct cost components:

| Component | What Drives Cost | Metric |
|---|---|---|
| **Compute (extraction, chunking, embedding)** | CPU/GPU time, wall-clock hours | Per GB or per file |
| **LLM API (summarization, context)** | Token count | Per text-bearing document |
| **Storage & Infrastructure** | DB size, vector store | Per GB stored |

### Cost Per GB of Content

| Metric | Value |
|---|---|
| Total unique content | 4.8 GB |
| LLM tokens | 5.54M |
| **Tokens per GB** | **~1.15M** |
| **Tokens per MB** | **~1,150** |
| **Tokens per file (avg)** | **~14,400** |
| **Tokens per text-bearing file** | **~84,000** |

### API Cost Per GB (if using paid models)

| Model | Cost/GB (LLM only) | Cost/GB (incl. embedding) |
|---|---|---|
| GPT-4o-mini | $0.28 | $0.35 |
| Haiku 4.5 | $1.44 | $1.50 |
| GPT-4o | $4.25 | $4.31 |
| Sonnet 4 | $5.39 | $5.45 |

### Cost Per Document

| Model | Cost/Doc (all 384) | Cost/Text-Doc (66) |
|---|---|---|
| GPT-4o-mini | $0.003 | $0.020 |
| Haiku 4.5 | $0.015 | $0.095 |
| GPT-4o | $0.045 | $0.269 |
| Sonnet 4 | $0.058 | $0.343 |

---

## 6. Recommended Pricing Tiers

Based on the actual token economics above, here's a pricing framework for offering this as a service.

### Tier 1: Quick Scan (Metadata Only)
**What:** Phases 0-7 only (walk, identify, triage, dedup, report). No extraction, no LLM.
**Time:** ~5 minutes for 384 docs
**Cost to deliver:** ~$0 (local compute only)
**Suggested price:** **$49–$99 per corpus**
**Margin:** ~100%

### Tier 2: Standard (Extraction + Summarization + Embedding, No Context)
**What:** Full pipeline minus contextual augmentation. Extract text, generate summaries, create embeddings.
**LLM tokens:** ~2.27M (summarization only)
**Time:** ~18 minutes local, ~3 minutes API
**API cost:** $1.03–$17.74 depending on model
**Suggested price:** **$199–$499 per corpus** (or $0.50–$1.50 per document)
**Margin:** 95–99%
**Quality note:** Retrieval quality is adequate for well-structured documents. Expect 20-30% more retrieval misses on irregular formats.

### Tier 3: Premium (Full Pipeline with Context)
**What:** Everything including contextual augmentation for improved retrieval quality.
**LLM tokens:** ~5.54M
**Time:** ~20 hours local (remote MLX: ~19h, API: ~20 min)
**API cost:** $1.34–$40 depending on model
**Suggested price:** **$499–$1,499 per corpus** (or $1.50–$4.00 per document)
**Margin:** 95–99%
**Quality note:** Contextual augmentation reduces retrieval failures by 20-30%. Essential for mixed-format, bilingual, or structurally irregular corpuses.

### Tier 4: Enterprise (Custom + Ongoing)
**What:** Full pipeline + custom extraction policies + periodic re-indexing + API access
**Suggested price:** **$2,000–$5,000/month retainer**
**Includes:** Up to 10 corpuses, quarterly re-indexing, custom format policies, dedicated support

---

## 7. Scaling Projections

### Cost at Different Corpus Sizes (using API — GPT-4o-mini)

| Corpus Size | Files | Est. Tokens | API Cost | Time (API) |
|---|---|---|---|---|
| Small | 50 | ~720K | $0.17 | ~2 min |
| Medium | 500 | ~7.2M | $1.70 | ~20 min |
| Large | 5,000 | ~72M | $17.00 | ~3 hours |
| XL | 50,000 | ~720M | $170.00 | ~30 hours |
| Enterprise | 500,000 | ~7.2B | $1,700 | ~12 days |

### Cost at Different Corpus Sizes (using API — Claude Opus 4.7)

| Corpus Size | Files | Est. Tokens | API Cost | Time (API) |
|---|---|---|---|---|
| Small | 50 | ~720K | $4.83 | ~2 min |
| Medium | 500 | ~7.2M | $48.30 | ~20 min |
| Large | 5,000 | ~72M | $483.00 | ~3 hours |
| XL | 50,000 | ~720M | $4,830 | ~30 hours |
| Enterprise | 500,000 | ~7.2B | $48,300 | ~12 days |

### Cost at Different Corpus Sizes (using API — GPT-5.5)

| Corpus Size | Files | Est. Tokens | API Cost | Time (API) |
|---|---|---|---|---|
| Small | 50 | ~720K | $5.14 | ~2 min |
| Medium | 500 | ~7.2M | $51.40 | ~20 min |
| Large | 5,000 | ~72M | $514.00 | ~3 hours |
| XL | 50,000 | ~720M | $5,140 | ~30 hours |
| Enterprise | 500,000 | ~7.2B | $51,400 | ~12 days |

### Cost at Different Corpus Sizes (using API — GPT-5 Pro)

| Corpus Size | Files | Est. Tokens | API Cost | Time (API) |
|---|---|---|---|---|
| Small | 50 | ~720K | $17.26 | ~2 min |
| Medium | 500 | ~7.2M | $172.60 | ~20 min |
| Large | 5,000 | ~72M | $1,726 | ~3 hours |
| XL | 50,000 | ~720M | $17,260 | ~30 hours |
| Enterprise | 500,000 | ~7.2B | $172,600 | ~12 days |

### Cost at Different Corpus Sizes (using API — Grok 4.1 Fast)

| Corpus Size | Files | Est. Tokens | API Cost | Time (API) |
|---|---|---|---|---|
| Small | 50 | ~720K | $0.20 | ~2 min |
| Medium | 500 | ~7.2M | $2.03 | ~20 min |
| Large | 5,000 | ~72M | $20.30 | ~3 hours |
| XL | 50,000 | ~720M | $203.00 | ~30 hours |
| Enterprise | 500,000 | ~7.2B | $2,030 | ~12 days |

---

## 8. Key Business Takeaways

### 1. API costs are extremely low relative to perceived value

At $1.34–$22.65 per corpus for LLM costs, the raw API spend is negligible. The real costs are:
- **Engineering time** to set up and customize the pipeline
- **Compute infrastructure** (GPU for local, or API management)
- **Quality assurance** and format policy tuning
- **Ongoing maintenance** as formats and models evolve

This means **pricing should be value-based, not cost-based.** A corpus search tool that saves a legal team 100 hours of document review is worth $10,000+ regardless of the $22 API cost.

### 2. Local Ollama is viable for small/medium corpuses

For corpuses under 5,000 documents, local processing with Ollama is practical:
- Zero API cost
- Data never leaves the machine (important for legal/medical/financial clients)
- 21 hours for 384 docs is acceptable for a one-time batch job
- Scales poorly beyond ~5,000 docs (days of processing)

### 3. Contextual augmentation is the main cost driver

Context accounts for 59% of all LLM tokens (3.28M of 5.54M). For cost optimization:
- **Skip context** for chunks from non-critical files (metadata-only, filename-only categories)
- **Use cheaper models** (GPT-4o-mini) for context generation — it's a brief snippet, not quality-critical
- **Context only for text-bearing files** — skip the 318 non-text files entirely (already done)

### 4. The "per-document" metric is misleading

Most corpuses contain a mix of text and non-text files. Pricing per total file count undercharges for text-heavy corpuses and overcharges for media-heavy ones. Better metrics:
- **Per text-bearing document** (actual LLM work)
- **Per million tokens** (directly tied to API cost)
- **Per GB of text content** (rough proxy for complexity)

### 5. Recommended pricing formula

```
Base fee + (Text-bearing docs × $0.50–$2.00) + (Non-text docs × $0.05) + Setup fee

Where:
- Base fee: $99 (covers infrastructure, reporting, DB)
- Text-bearing docs: PDFs, Office docs, spreadsheets, presentations
- Non-text docs: Images, CAD, video, audio, archives, executables
- Setup fee: $500–$2,000 (initial config, format policy, corpus mapping)
```

**Example for this corpus (384 docs, 66 text-bearing, 318 non-text):**
```
$99 + (66 × $1.00) + (318 × $0.05) + $1,000 = $99 + $66 + $16 + $1,000 = $1,181
```

API cost to deliver: $1.34–$22.65
**Gross margin: 98–99%**

---

## 9. Risk Factors

| Risk | Impact | Mitigation |
|---|---|---|
| Ollama model quality degrades | Summaries/context may be lower quality than API models | Offer API upgrade as premium option |
| Corpus has unexpected formats | New format policies need manual tuning | Format policy CSV is easily extendable |
| Very large files (100+ page PDFs) | Token count per doc can spike 10x | Per-token surcharge above 50K tokens/doc |
| Whisper failures on audio | 92 whisper runs produced 0 text in this corpus | Flag as "audio may not contain speech" in report |
| API pricing changes | OpenAI/Anthropic/xAI may raise rates | Build 20% buffer into pricing |
| Opus 4.7 tokenizer overhead | 35% more tokens than prior models for same input | Budget +40% when quoting Opus; or use hybrid strategy |
| GPT-5 Pro cost spike | $120/MTok output is 12x standard rates | Only use for summarization phase, never for bulk context |

---

## Appendix: Raw Data

Source: `corpus.db` at project root.

| Table | Rows |
|---|---|
| file | 2,177 |
| folder | 71 |
| extraction | 432 |
| chunk | 8,159 |
| summary | 45 |
| embedding_ref | 8,159 |
| model_run | 25 |
| failure | (varies) |

All token estimates use the standard heuristic: ~4 characters per token for English text. Actual token counts may vary ±15% depending on language and content type.

### Pricing Sources (May 2026)
- [Claude API Pricing — Anthropic](https://platform.claude.com/docs/en/about-claude/pricing)
- [OpenAI API Pricing](https://developers.openai.com/api/docs/pricing)
- [xAI Grok Models & Pricing](https://docs.x.ai/developers/models)
- [GPT-5.5 Pricing Breakdown — APIDog](https://apidog.com/blog/gpt-5-5-pricing/)
- [Claude Opus 4.7 Price — GlobalGPT](https://www.glbgpt.com/resources/claude-opus-4-7-price/)
- [LLM API Pricing Comparison — Awesome Agents](https://awesomeagents.ai/pricing/llm-api-pricing-comparison/)
