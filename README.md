# ODW Vault

> The sovereign knowledge core of the ODW.ai suite — a self-hosted, open-source Retrieval-Augmented Generation (RAG) platform that turns internal documents, wikis, and structured data into an AI-queryable knowledge base without any data leaving your infrastructure.

ODW Vault is a **fully offline** pre-flight pipeline + end-to-end RAG system for mixed-format document corpora. It analyzes a hierarchical folder of documents, identifies formats, deduplicates, extracts text, generates embeddings, and provides query access — all running on-premises with no outbound network calls during inference.

## Status

⚠️ **Early release.** ODW [Name] is an early, functional release — core features work, but it is not yet hardened for production. We are refining every module toward a first full public release in **Q3 2026**. Until then, it is best used as a foundation to build on with AI coding agents (see below).

** Build Status:** Part 1 (pre-flight) and Part 2 (RAG pipeline) both implemented and operational. 167 tests passing.

## Features

### Part 1: Pre-Flight (phases 0–7)
- **Format identification** via Siegfried (PRONOM signatures) — 87 format policies
- **Archive expansion** — recursive nested extraction (ZIP, TAR, 7z, RAR)
- **Deduplication** via SHA-256 hash grouping (pure SQL)
- **Content triage** — PDF text-vs-scanned, media duration, image dimensions
- **Language detection** — English + Chinese via lingua
- **Semantic folder inference** — Ollama/gemma4 generates folder labels and categories
- **Interactive exploration** — Datasette server for DB browsing

### Part 2: RAG Pipeline (phases 8–14)
- **Text extraction** — 8 extractors (Docling, Tika, RapidOCR, textutil, etc.)
- **Document summarization** — Ollama/gemma4 summaries for large documents
- **Sentence-window chunking** — configurable window with char offset tracking
- **Contextual retrieval** — chunk-level context augmentation (MLX/Qwen3-8B-4bit)
- **Embedding** — qwen3-embedding:8b via Chroma persistent collections
- **Hybrid retrieval** — dense vector + BM25 + Reciprocal Rank Fusion
- **Citation-strict generation** — gemma4 answers with numbered chunk citations
- **HTTP API** — FastAPI with 10 endpoints (query, stream, feedback, eval)
- **Gradio UI** — chat interface with folder filtering and citation display
- **Evaluation framework** — question bank, run eval, accuracy reporting

## Requirements

### Python
- Python 3.11+ (managed via `uv`, venv at `.venv/`)

### External tools (macOS)
```bash
brew install ffmpeg unar ollama
ollama pull gemma4:latest
ollama pull qwen3-embedding:8b
```
- **Siegfried**: Download from [GitHub releases](https://github.com/richardlehane/siegfried/releases), extract the binary, and place it as `./sf` in the project root.

## Quick Start

### 1. Setup
```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 2. Configure
```bash
cp config.example.toml config.toml
# Edit config.toml — set your Ollama API key if using the cloud endpoint
```

### 3. Install external dependencies (macOS)
```bash
brew install ffmpeg unar ollama
ollama pull gemma4:latest
ollama pull qwen3-embedding:8b
# Siegfried: download from https://github.com/richardlehane/siegfried/releases, place as ./sf
```

### 4. Place a corpus
```bash
# Put your documents in a folder, e.g. data/my-corpus
vault init --root ./data/my-corpus
```
Creates `corpus.db` and `.rag-cache/` in the project directory.

### 5. Run Pre-Flight (Part 1)
```bash
vault run-all
```

### 6. Run RAG Pipeline (Part 2)
```bash
# Extract, summarize, chunk, embed
vault extract
vault summarize
vault chunk
vault context      # optional, slow
vault embed
```

### 7. Query
```bash
# CLI
vault query "What is this corpus about?" --top-k 5

# JSON output
vault query "What formats are in the corpus?" --json

# API server
vault serve --port 8001 &
curl -X POST http://127.0.0.1:8001/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What CAD files exist?"}'

# Gradio UI
vault ui
```

## Architecture

### Phase Pipeline

| Phase | Command | What It Does | Output |
|-------|---------|-------------|--------|
| **Part 1: Pre-Flight** |
| 0 | `archives` | Expands nested archives | `archive_expansion` table |
| 1 | `walk` | Walks tree, computes SHA-256 | `folder` + `file` tables |
| 2 | `identify` | Siegfried PRONOM format ID | Categories, extract strategies |
| 3 | `triage` | PDF/media/image inspection | Text layer, duration, language |
| 4 | `dedup` | SHA-256 grouping | `dup_group_id`, `is_dup_primary` |
| 5 | `folder-meta` | Ollama folder inference | `inferred_category`, `inferred_label` |
| 6 | `report` | Aggregate statistics | `preflight_report.md` |
| 7 | `exclude` | Manual exclusion marking | `excluded` flag |
| **Part 2: RAG Pipeline** |
| 8 | `extract` | Text extraction by format | `extraction` table |
| 8b | `transcribe` | Audio/video transcription | Opt-in by folder globs |
| 9 | `summarize` | Ollama document summaries | `summary` table |
| 10 | `chunk` | Sentence-window chunking | `chunk` table + `chunk_fts` |
| 10.5 | `context` | Contextual augmentation | `chunk.context_text` |
| 11 | `embed` | Chroma vector store | `embedding_ref` tables |
| 12 | `query` | Hybrid retrieval + generation | Answer with citations |
| 13 | `eval` | Evaluation framework | `eval_run` results |
| 14 | `serve`/`ui` | API + Gradio UI | HTTP endpoints / chat |

### Guiding Principles

- **Originals are never modified** — all derived artifacts go under `.rag-cache/`
- **Every phase is idempotent and resumable** — safe to re-run at any point
- **SQLite is the single source of truth** — `corpus.db` holds everything
- **All failures go to the `failure` table** — silent failures are defects
- **No outbound HTTP during runtime** — air-gap capable

### Technology Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| Database | SQLite 3 via `sqlite-utils` | Single-file, no server |
| Format ID | Siegfried 1.11.4 | PRONOM format signatures |
| LLM | Ollama + `gpt-oss:20b` | Answer generation |
| LLM (fallback) | Ollama + `gemma4:latest` | Fallback generation |
| LLM (summarization) | Ollama + `gemma4:latest` | Document summarization |
| LLM (context) | MLX + `Qwen3-8B-4bit` | Contextual augmentation |
| Embedding | Ollama + `qwen3-embedding:8b` | 4096-dim vectors |
| Vector store | Chroma (PersistentClient) | Dense vector storage |
| OCR | RapidOCR (ONNX) | Scanned PDF/image text |
| Extraction | IBM Docling | Office document text |
| Language detection | `lingua-language-detector` | English + Chinese |
| PDF triage | PyMuPDF (fitz) | Text layer detection |
| API | FastAPI + sse-starlette | Query endpoint with streaming |
| UI | Gradio 6.x | Chat interface |
| CLI | click | Subcommand surface |
| Progress bars | rich | Terminal UI |
| Testing | pytest + pytest-cov | 167 tests, 86% coverage |

### Database Schema

24 tables (18 user + FTS5 internals), 28 indexes, 12 views. Key tables:

| Table | Purpose |
|-------|---------|
| `folder` | Directory tree with semantic labels |
| `file` | File inventory: hash, format, category, triage, dedup |
| `format_policy` | PRONOM ID to category + extract strategy |
| `extraction` | Extracted text with provenance |
| `summary` | Document summaries |
| `chunk` | Sentence-window chunks + FTS5 |
| `embedding_ref` | Chunk embedding references |
| `summary_embedding_ref` | Summary embedding references |
| `folder_embedding_ref` | Folder embedding references |
| `model_run` | Per-model-run history |
| `query_log` | Query tracking with feedback |
| `failure` | Error tracking with classification |

## Configuration

Settings live in `config.toml` with multiple Pydantic sub-configs:

```toml
[paths]
corpus_root = "/path/to/corpus"
cache_root  = "/path/to/corpus/.rag-cache"
chroma_root = "./chroma"

[ollama]
host = "http://localhost:11434"

[models.embedding]
name              = "qwen3-embedding:8b"
collection_suffix = "qwen3emb8b"
batch_size        = 32
truncate_dim      = 0

[models.summarization]
name = "gemma4:latest"
temperature = 0.3

[models.generation]
name = "gpt-oss:20b"
temperature = 0.5

[models.contextual_retrieval]
name = "mlx-community/Qwen3-8B-4bit"

[chunk]
chunker = "sentence-window"
window_size = 5

[retrieval]
top_k_chunks = 8
dense_candidates = 50
bm25_candidates = 50
rrf_k = 60
```

## CLI Reference

```bash
# Part 1: Pre-Flight
vault init --root "/path/to/corpus" [--force]
vault run-all
vault archives [--max-depth N] [--dry-run]
vault walk [--workers N] [--rehash]
vault identify [--reidentify]
vault triage [--workers N] [--categories CAT1,CAT2,...]
vault dedup
vault folder-meta [--model NAME] [--reinfer]
vault report [--output PATH]
vault exclude --target {file,folder} --id N --reason TEXT
vault exclude-batch --from-file exclusions.csv
vault approve --by NAME
vault status              # JSON per-phase status
vault serve --port 8001   # Launch Datasette

# Part 2: RAG Pipeline
vault extract [--workers N] [--reextract]
vault summarize [--resummarize]
vault chunk [--window-size N] [--rechunk]
vault context [--regenerate]
vault embed [--model NAME] [--reembed]
vault embed-switch-to --model NAME
vault embed-gc
vault embed-list
vault query "question" [--top-k N] [--json]
vault serve --port 8001           # API server
vault ui --port 7860                # Gradio UI
vault eval add/run/report
vault models list/pull/check

# Tests
pytest tests/ --cov=pipeline --cov=cli --cov-report=term-missing -v
```

## Documentation

- **BUILD_STATUS.md** — Complete build status, all 28 bugs resolved, pending issues
- **PART_2_STATUS.md** — Part 2 (phases 8–14) detailed status, architecture, query results
- **TEST_REPORT.md** — Test suite report (167 tests, 86% coverage)
- **CLAUDE.md** — AI assistant context for this project
- **Technical Specification Document- Local RAG Pre-Flight Pipeline.md** — Original Part 1 spec
- **Technical Specification Document- Local RAG Pipeline (Phases 8–14).md** — Part 2 spec

## Pending Work

### Critical
1. **Part 2 tests** — no coverage for rag/, api/, eval/, ui/ modules.

### Quality
2. **Chinese FTS5 tokenizer** — current Porter stemmer only handles English.
3. **Reranker implementation** — configured but not wired up.
4. **Evaluation benchmarks** — eval framework exists but no questions loaded.
5. **Whisper extractor** — implement for audio/video.

## Working with AI agents
This repository is built to be extended with AI coding agents. Rather than a turnkey product, ODW [Name] is a working, well-structured codebase you can clone and adapt to your own needs with an agent like Claude Code. The repo includes agent context files (e.g. `CLAUDE.md`) and clear architecture docs so an agent can quickly understand the structure and help you customise, integrate, and extend it. To get started: clone the repo, open it with your coding agent, point it at this README and the docs, and describe what you want to build.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

