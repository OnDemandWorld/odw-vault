# ODW Vault

> The sovereign knowledge core of the ODW.ai suite — a self-hosted, open-source Retrieval-Augmented Generation (RAG) platform that turns internal documents, wikis, and structured data into an AI-queryable knowledge base without any data leaving your infrastructure.

ODW Vault is a **fully offline** pre-flight pipeline + end-to-end RAG system for mixed-format document corpora. It analyzes a hierarchical folder of documents, identifies formats, deduplicates, extracts text, generates embeddings, and provides query access — all running on-premises with no outbound network calls during inference.

## Status

⚠️ **Early release.** ODW Vault is an early, functional release — core features work, but it is not yet hardened for production. We are refining every module toward a first full public release in **Q3 2026**. Until then, it is best used as a foundation to build on with AI coding agents (see below).

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

| Tool | Purpose | Install method |
|------|---------|---------------|
| `uv` | Python package manager | `brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| `ffmpeg` | Media duration detection | `brew install ffmpeg` |
| `unar` | Archive extraction | `brew install unar` |
| `ollama` | Local LLM server | **Official script** (see below) — ⚠️ Homebrew version is outdated (0.13.x), the latest release (0.31.x+) is required |
| `sf` (Siegfried) | PRONOM format identification | Manual download (see below) |
| `LibreOffice` *(optional)* | DOCX → PDF conversion for text extraction | `brew install --cask libreoffice` |

> **⚠️ Ollama version notice:** `brew install ollama` installs an outdated version (0.13.x) that does **not** support the latest models (e.g. `gemma4:latest`). Use the official install script instead:
> ```bash
> curl -fsSL https://ollama.com/install.sh | sh
> ```

## Quick Start

### Option A — Step-by-step (full control)

#### 1. Install `uv` (Python package manager)
```bash
brew install uv
# or: curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 2. Setup Python environment
```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

#### 3. Configure
```bash
cp config.example.toml config.toml
# Edit config.toml — set your Ollama API key if using the cloud endpoint
```

#### 4. Install external dependencies (macOS)
```bash
# System tools (use brew for these)
brew install ffmpeg unar

# Ollama — use the OFFICIAL script, NOT brew (brew version is outdated)
curl -fsSL https://ollama.com/install.sh | sh

# Start the Ollama service, then pull models
ollama serve &            # start server in background (first time only)
ollama pull gemma4:latest
ollama pull qwen3-embedding:8b
```

> **Note:** After installing Ollama via the official script, you need to start the server **before** pulling models. Run `ollama serve` in a separate terminal or background it with `&`. Once the server is running, `ollama pull` will work.

#### 5. Install Siegfried (manual step)
```bash
# 1. Download from https://github.com/richardlehane/siegfried/releases
#    Choose the macOS ARM64 (Apple Silicon) or AMD64 (Intel) build
# 2. Extract the zip, rename the binary to "sf", and place it in the project root
# 3. Grant execute permission (macOS Gatekeeper requires this on first run)
chmod +x ./sf
./sf                     # first run — macOS will prompt for permission, allow it
```

> **⚠️ macOS permission:** The first time you run `./sf`, macOS may block it with a security warning. Go to **System Settings → Privacy & Security** and click **Allow Anyway**, then run `./sf` again.

#### 6. Place a corpus
```bash
# Create the corpus directory and put your documents in it
mkdir -p data/my-corpus
# Copy your documents into data/my-corpus/, then:
vault init --root ./data/my-corpus
```
Creates `corpus.db` and `.rag-cache/` in the project directory.

> **Note:** The corpus directory **must exist and contain files** before running `vault init`. The default `config.example.toml` points to `./SourceData` — if you use that path, create it first: `mkdir -p SourceData`.

#### 7. Run Pre-Flight (Part 1)
```bash
vault run-all
```

#### 8. Run RAG Pipeline (Part 2)
```bash
# Extract, summarize, chunk, embed
vault extract
vault summarize
vault chunk
vault context      # optional, slow
vault embed
```

#### 9. Query
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

---

### Option B — Quick install (one-shot script)

For users who want to get up and running as fast as possible. Copy and paste the entire block:

```bash
# ── 1. System dependencies ──────────────────────────────────
brew install uv ffmpeg unar

# ── 2. Ollama (official script — NOT brew) ──────────────────
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &                          # start server in background
sleep 3                                 # wait for server to be ready
ollama pull gemma4:latest
ollama pull qwen3-embedding:8b

# ── 3. Python environment ───────────────────────────────────
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

# ── 4. Configuration ────────────────────────────────────────
cp config.example.toml config.toml

# ── 5. Siegfried (manual — download, extract, place as ./sf) 
#    Download: https://github.com/richardlehane/siegfried/releases
#    Then:
chmod +x ./sf && ./sf                   # grant permission on first run

# ── 6. Initialize corpus ────────────────────────────────────
mkdir -p data/my-corpus
# → Put your documents in data/my-corpus/ before continuing
vault init --root ./data/my-corpus

# ── 7. Run pipeline ─────────────────────────────────────────
vault run-all                           # Part 1: pre-flight
vault extract && vault summarize && vault chunk && vault embed  # Part 2: RAG

# ── 8. Launch UI ────────────────────────────────────────────
vault ui                                # Gradio chat interface at http://localhost:7860
```

---

### Troubleshooting common issues

| Problem | Solution |
|---------|----------|
| `ollama: command not found` after brew install | Use `curl -fsSL https://ollama.com/install.sh \| sh` instead of brew |
| `pull model manifest: 412` when pulling models | Ollama version is too old. Reinstall with the official script above |
| `ollama pull` hangs or connection refused | Start the server first: `ollama serve` |
| `sf` crashes with "error opening signature file" | Run `chmod +x ./sf` and execute `./sf` once to grant macOS permission |
| `No such file or directory: 'SourceData/...'` | Ensure the corpus directory exists and contains files before running pipeline |
| `ModuleNotFoundError: No module named 'ui'` | Re-run `uv pip install -e ".[dev]"` to ensure editable install registers all packages |
| DOCX files fail extraction with "no DOCX to PDF converters" | Install LibreOffice: `brew install --cask libreoffice` |
| `RapidOCR returned empty result` | Normal for some image-based PDFs — the pipeline logs a warning and continues |
| Folder inference fails with `ValidationError` | Ensure Ollama server is running and `gemma4:latest` model is pulled |

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
This repository is built to be extended with AI coding agents. Rather than a turnkey product, ODW Vault is a working, well-structured codebase you can clone and adapt to your own needs with an agent like Claude Code. The repo includes agent context files (e.g. `CLAUDE.md`) and clear architecture docs so an agent can quickly understand the structure and help you customise, integrate, and extend it. To get started: clone the repo, open it with your coding agent, point it at this README and the docs, and describe what you want to build.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

