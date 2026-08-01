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
- **Multi-workspace knowledge bases** — optional `workspace` tag for logical isolation of uploads and retrieval (V1.1, fully backward-compatible)
- **Gradio UI** — chat interface with folder filtering and citation display
- **Evaluation framework** — question bank, run eval, accuracy reporting

## Requirements

### Python
- Python 3.11+ (managed via `uv`, venv at `.venv/`)

### Platform Support

| Platform | Status (v0.2.x) | Status (v0.3.0) |
|----------|-----------------|------------------|
| **macOS** (Apple Silicon / Intel) | ✅ Fully supported | ✅ Native `.dmg` installer |
| **Linux** (Ubuntu 22.04+ / Debian 12+) | ✅ Supported (manual setup) | ✅ `.deb` + `.AppImage` installer |
| **Windows** (10/11, x86_64) | ⚠️ WSL2 recommended | ✅ Native `.exe` / `.msi` installer |

> **v0.3.0 note:** The next release will bundle all dependencies into native installers — no Python, no terminal, no manual setup required. See [Next Version Roadmap](#next-version-v030--roadmap) below.

### External tools

#### macOS

| Tool | Purpose | Install method |
|------|---------|---------------|
| `uv` | Python package manager | `brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| `ffmpeg` | Media duration detection | `brew install ffmpeg` |
| `unar` | Archive extraction | `brew install unar` |
| `ollama` | Local LLM server | **Official script** (see below) — ⚠️ Homebrew version is outdated (0.13.x), the latest release (0.31.x+) is required |
| `sf` (Siegfried) | PRONOM format identification | Manual download (see below) |
| `LibreOffice` *(optional)* | DOCX → PDF conversion for text extraction | `brew install --cask libreoffice` |

#### Linux (Ubuntu / Debian)

| Tool | Purpose | Install method |
|------|---------|---------------|
| `uv` | Python package manager | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| `ffmpeg` | Media duration detection | `sudo apt install ffmpeg` |
| `unar` | Archive extraction | `sudo apt install unar` |
| `ollama` | Local LLM server | `curl -fsSL https://ollama.com/install.sh \| sh` |
| `sf` (Siegfried) | PRONOM format identification | Download Linux binary from [releases](https://github.com/richardlehane/siegfried/releases) |
| `LibreOffice` *(optional)* | DOCX → PDF conversion | `sudo apt install libreoffice` |

#### Windows

| Tool | Purpose | Install method |
|------|---------|---------------|
| `uv` | Python package manager | `powershell -c "irm https://astral.sh/uv/install.ps1 \| iex"` |
| `ffmpeg` | Media duration detection | `winget install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org/download.html) |
| `7-Zip` | Archive extraction | `winget install 7zip` |
| `ollama` | Local LLM server | Download from [ollama.com/download](https://ollama.com/download) |
| `sf` (Siegfried) | PRONOM format identification | Download Windows binary from [releases](https://github.com/richardlehane/siegfried/releases) |
| `LibreOffice` *(optional)* | DOCX → PDF conversion | `winget install LibreOffice` |

> **Windows note (v0.2.x):** Native Windows support is experimental. WSL2 (Ubuntu) is recommended for the best experience. Full native Windows support arrives with the v0.3.0 installer.

> **⚠️ Ollama version notice:** `brew install ollama` installs an outdated version (0.13.x) that does **not** support the latest models (e.g. `gemma4:latest`). Use the official install script instead:
> ```bash
> curl -fsSL https://ollama.com/install.sh | sh
> ```

> **Cloud LLM alternative:** Don't want to install Ollama? v0.3.0 supports cloud providers (OpenAI, DeepSeek, Qwen, Gemini, etc.) via API key — no local GPU needed. See [Cloud LLM Provider Support](#1-cloud-llm-provider-support).

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

### Option B — Quick install (one-shot script, macOS)

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

### Option C — Quick install (Linux)

```bash
# ── 1. System dependencies ──────────────────────────────────
sudo apt update && sudo apt install -y ffmpeg unar
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or restart terminal

# ── 2. Ollama ───────────────────────────────────────────────
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &                          # start server in background
sleep 3
ollama pull gemma4:latest
ollama pull qwen3-embedding:8b

# ── 3. Python environment ───────────────────────────────────
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

# ── 4. Configuration ────────────────────────────────────────
cp config.example.toml config.toml

# ── 5. Siegfried ────────────────────────────────────────────
# Download Linux binary from https://github.com/richardlehane/siegfried/releases
# Extract and place as ./sf in project root
chmod +x ./sf

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

### Option D — Cloud LLM (no local GPU needed)

If you don't have a powerful GPU or prefer cloud models, skip Ollama entirely:

```bash
# ── 1. Set your API key ─────────────────────────────────────
export OPENAI_API_KEY="sk-..."          # or DEEPSEEK_API_KEY, DASHSCOPE_API_KEY, etc.

# ── 2. Python environment (same as above) ───────────────────
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

# ── 3. Configure for cloud ──────────────────────────────────
cp config.example.toml config.toml
# Edit config.toml:
#   [llm_provider]
#   backend = "openai_compatible"
#   [llm_provider.openai_compatible]
#   base_url = "https://api.openai.com/v1"
#   api_key_env = "OPENAI_API_KEY"
#   [llm_provider.models]
#   embedding = "text-embedding-3-small"
#   generation = "gpt-4o"

# ── 4. Initialize and run ───────────────────────────────────
mkdir -p data/my-corpus
vault init --root ./data/my-corpus
vault run-all
vault extract && vault summarize && vault chunk && vault embed
vault ui
```

> **Note:** Cloud LLM support requires v0.3.0. In v0.2.x, Ollama is required for embedding and generation.

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

## Multi-Workspace Knowledge Bases (V1.1)

Vault supports **multiple logical knowledge bases** ("workspaces") over a single
shared corpus. Every file carries a `workspace` label; uploads and retrieval can
be scoped to one workspace so unrelated knowledge bases do not bleed into each
other's answers.

The feature is **strictly additive and backward-compatible**: every workspace
parameter is optional, and when omitted Vault behaves exactly as in V1.0 (the
whole corpus is one implicit `default` workspace). Existing databases gain the
`file.workspace` column automatically on startup (idempotent migration; existing
rows backfill to `default`).

**Tag uploads with a workspace** (`POST /files/upload`, optional form field):

```bash
curl -X POST http://127.0.0.1:8765/files/upload \
  -F "files=@meeting-notes.md" \
  -F "workspace=team-a"
```

**Query a single workspace** (`POST /query` / `POST /query/stream`, optional
`folder_filter.workspace`, composes with `path_prefix` / `folder_id`):

```bash
curl -X POST http://127.0.0.1:8765/query \
  -H "Content-Type: application/json" \
  -d '{"query": "what did we decide?", "folder_filter": {"workspace": "team-a"}}'
```

Omit `folder_filter.workspace` to search the entire corpus (V1.0 behavior).

**List / filter by workspace:**

| Endpoint | Purpose |
|----------|---------|
| `GET /workspaces` | Distinct workspaces with per-workspace file counts |
| `GET /files?workspace=team-a` | Files in one workspace |
| `GET /folders?workspace=team-a` | Folders that contain files in one workspace |

> **⚠️ Logical isolation only.** Workspaces are a filtering layer over one
> shared SQLite database and one shared Chroma vector store — they are **not** a
> security boundary or a hard multi-tenancy guarantee. There is no per-workspace
> authentication, quota, or separate vector collection; a caller that omits the
> `workspace` parameter (or has DB access) can still see every workspace. Use
> workspaces to keep unrelated knowledge bases from polluting each other's
> retrieval, not to enforce access control. Per-workspace collections /
> databases / RBAC are deferred to a future release.

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

## Next Version (v0.3.0) — Roadmap

The next major release focuses on two pillars: **cloud LLM flexibility** and **cross-platform distribution**.

### 1. Cloud LLM Provider Support

Vault v0.3.0 introduces a unified LLM integration layer based on the **OpenAI-compatible API standard**, allowing users to swap between local Ollama models and mainstream cloud providers by simply editing `config.toml` and setting an API key.

#### Supported Providers (out of the box)

| Provider | `base_url` | Recommended Models | Env Variable |
|----------|-----------|-------------------|--------------|
| **OpenAI** | `https://api.openai.com/v1` | `gpt-4o`, `text-embedding-3-small` | `OPENAI_API_KEY` |
| **DeepSeek** | `https://api.deepseek.com/v1` | `deepseek-chat`, `deepseek-reasoner` | `DEEPSEEK_API_KEY` |
| **Qwen (DashScope)** | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-max`, `text-embedding-v3` | `DASHSCOPE_API_KEY` |
| **Moonshot** | `https://api.moonshot.cn/v1` | `moonshot-v1-128k` | `MOONSHOT_API_KEY` |
| **Google Gemini** | `https://generativelanguage.googleapis.com/v1beta/openai` | `gemini-2.5-flash` | `GOOGLE_API_KEY` |
| **Together AI** | `https://api.together.xyz/v1` | `meta-llama/Llama-4-Scout-17B` | `TOGETHER_API_KEY` |
| **Groq** | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| **Ollama (local)** | `http://localhost:11434` | `gemma4:latest`, `qwen3-embedding:8b` | *(none)* |

#### Configuration Example

```toml
# config.toml — Cloud LLM configuration

[llm_provider]
# Backend selector: "ollama" (local) | "openai_compatible" (cloud)
backend = "openai_compatible"

[llm_provider.openai_compatible]
base_url = "https://api.openai.com/v1"     # Any OpenAI-compatible endpoint
api_key_env = "OPENAI_API_KEY"             # Read key from environment variable
# api_key = "sk-..."                       # Or set directly (not recommended)

# Map each pipeline role to a cloud model
[llm_provider.models]
embedding = "text-embedding-3-small"       # Embedding model
generation = "gpt-4o"                      # Answer generation
summarization = "gpt-4o-mini"              # Document summarization
contextual_retrieval = "gpt-4o-mini"       # Context augmentation
reranker = ""                              # Empty = disabled
```

#### Key Design Decisions

- **OpenAI SDK as unified adapter** — all providers expose an OpenAI-compatible `/v1/chat/completions` and `/v1/embeddings` endpoint; the `openai` Python SDK handles auth, retries, and streaming uniformly.
- **Per-role model mapping** — embedding, generation, summarization, and contextual retrieval can each use a different provider/model.
- **Environment variable for secrets** — API keys are read from env vars by default (`api_key_env`), never committed to config files.
- **Backward compatible** — setting `backend = "ollama"` preserves the existing local-only behavior with zero changes.
- **Embedding dimension awareness** — switching embedding models requires re-running `vault embed` to rebuild the Chroma vector store (different models produce different vector dimensions).

---

### 2. Cross-Platform Desktop Packaging

Vault v0.3.0 will ship as **native installers** for all three major desktop platforms, so users can install and run Vault without Python, terminal, or any manual setup.

#### Target Platforms

| Platform | Installer Format | Architecture | Minimum OS |
|----------|-----------------|--------------|------------|
| **macOS** | `.dmg` (drag-to-install) | Apple Silicon (ARM64) + Intel (x86_64) | macOS 13 Ventura |
| **Windows** | `.exe` (NSIS installer) + `.msi` | x86_64 | Windows 10 (1809+) |
| **Linux** | `.deb` + `.AppImage` | x86_64 + ARM64 | Ubuntu 22.04 / Debian 12 |

#### What's Bundled

Each installer packages the complete runtime so users need **zero external dependencies**:

```
ODW Vault.app / ODW Vault.exe / odv-vault.AppImage
├── Python 3.11 runtime (embedded, no system Python needed)
├── All Python dependencies (pip packages frozen)
├── Siegfried binary (format identification)
├── ffmpeg / ffprobe (media processing)
├── SQLite + Chroma (data layer)
├── Gradio UI server (auto-launches browser)
└── config.toml (first-run wizard generates this)
```

#### User Experience

1. **Download** the installer for your platform from the releases page
2. **Install** — standard OS installer flow (drag to Applications / Next-Next-Finish / `dpkg -i`)
3. **First launch** — a setup wizard guides you through:
   - Choose corpus folder (your document directory)
   - Choose LLM backend: **Local (Ollama)** or **Cloud (API key)**
   - If cloud: select provider, paste API key, pick models
   - If local: auto-detect Ollama or offer to download it
4. **Pipeline runs automatically** — documents are indexed in the background with a progress indicator
5. **Chat UI opens in browser** — `http://localhost:7860`

#### Technical Approach

| Concern | Solution |
|---------|----------|
| Python bundling | [PyInstaller](https://pyinstaller.org) or [Briefcase](https://beeware.org/project/projects/tools/briefcase/) (BeeWare) |
| macOS code signing | Apple Developer ID + notarization (required for Gatekeeper) |
| Windows signing | EV code-signing certificate (avoids SmartScreen warnings) |
| Auto-update | [Sparkle](https://sparkle-project.org/) (macOS) / [NSIS + GitHub Releases](https://nsis.sourceforge.io/) (Win) / AppImage self-update (Linux) |
| Ollama bundling | Optional — installer detects existing Ollama; if absent, offers to download the official binary |
| Data directory | `~/Library/Application Support/ODVVault/` (macOS), `%APPDATA%/ODVVault/` (Win), `~/.local/share/odv-vault/` (Linux) |

#### Build Pipeline (CI/CD)

```
GitHub Actions workflow:
├── macOS job   → build .dmg (ARM64 + x86_64 universal binary)
├── Windows job → build .exe + .msi (x86_64)
├── Linux job   → build .deb + .AppImage (x86_64 + ARM64)
└── Release job → upload to GitHub Releases + auto-update manifest
```

---

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

---

## V1.2 Feature Expansion (M1 + M2)

> Strictly additive. The default chunking strategy and the default (English)
> retrieval path are unchanged; existing indices and behaviour are unaffected.

### M1 — Multi-chunking strategies (F-Vault-1)

Chunking is now pluggable via a strategy registry (`rag/chunk_strategies.py`):

| Strategy | Behaviour |
|----------|-----------|
| `sentence_window` | Focal sentence ± N surrounding sentences (**default**, identical to the legacy chunker) |
| `recursive` | Split by a separator hierarchy (`\n\n` → `\n` → sentence terminators → space) respecting `chunk_size` / `chunk_overlap` |
| `paragraph` | Split on blank lines; over-long paragraphs are further split/merged by sentence |

Select a strategy in `config.toml`:

```toml
[chunk]
strategy = "sentence_window"   # sentence_window | recursive | paragraph
chunk_size = 2000              # target chars for recursive/paragraph
chunk_overlap = 200            # overlap chars for recursive
# category_strategies = { document = "paragraph" }   # optional per-category override
```

The registry is extensible: implement `ChunkStrategy` (`name` + `chunk(text, **opts)`)
and call `get_registry().register(...)`. The CLI `vault chunk --chunker sentence-window`
alias still maps to the default strategy.

### M2 — Chinese retrieval (F-Vault-2)

BM25/FTS retrieval is now language-aware (`rag/tokenization.py`):

- `tokenize(text, lang)` abstraction. **English (`lang="en"`, default) is unchanged.**
- Chinese (`lang="zh"`) uses **jieba** when installed, otherwise a dependency-free
  **character-bigram** tokenizer (always runnable/testable offline).
- jieba is an **optional** dependency: `pip install -e ".[chinese]"`. Without it, the
  bigram fallback is used automatically.
- A separate, additive FTS5 index `chunk_fts_zh` (migration 6) stores Chinese
  segment/bigram tokens; Chinese queries are routed there, English queries keep using
  `chunk_fts`. Language detection reuses the existing fasttext `lid` model, falling back
  to lingua and then a CJK heuristic (default `en` on failure).

## V1.3 Audit Logging (F-Vault-1)

Vault now keeps a **best-effort compliance audit trail** (who / when / what) for
sensitive operations. It is strictly additive: no existing endpoint, response shape,
or status code changes.

- **`audit_log` table** (idempotent migration 7): `id, ts, actor, action,
  resource_type, resource_id, detail, status`.
- **Audited operations** (`api/audit.py` → `record_audit`): `POST /query`
  (`query`), `POST /files/upload` (`file.upload`), `DELETE /files/{id}`
  (`file.delete`), `POST /pipeline/sync` (`pipeline.sync`), and `POST /feedback`
  (`feedback`).
- **`GET /audit`** returns records most-recent-first, with optional `action`
  (exact match) and `limit` (default 100) query filters. When the optional V1.0
  API-key auth is active (`VAULT_API_KEY` set) this endpoint is protected by it;
  otherwise it is open like the rest of the API.
- **Actor resolution**: a configured `VAULT_AUDIT_ACTOR` wins; otherwise the actor
  is `authenticated` when `VAULT_API_KEY` is active, else `anonymous`.
- **Best-effort guarantee**: audit writes catch all errors and only log a warning —
  a failing audit write never blocks or alters the main request (the operation still
  returns its normal response).

## V1.4 Audit Report Export (F-3')

Vault can now **export the compliance audit trail** as a machine-readable report
for SOC2 / GDPR evidence. The feature is strictly additive: `GET /audit` is
unchanged, and the export reuses the same audit read path (same columns and
most-recent-first ordering), extended with time-range and actor filtering.

- **`GET /audit/export`** query parameters:
  - `format=csv|json` (default `json`)
  - `start` / `end` — inclusive ISO-8601 time bounds on `ts`
    (e.g. `2026-01-01` or `2026-01-01T00:00:00`)
  - `actor` — exact-match actor filter
  - `action` — exact-match action filter (e.g. `query`, `file.upload`)
  - `limit` — max rows (default `1000`)
- **JSON** returns `{items, total, filters}` where `filters` echoes the applied
  filters.
- **CSV** is built with the Python standard library (`csv` / `io.StringIO`) with
  columns `id,ts,actor,action,resource_type,resource_id,detail,status`, served as
  a download (`Content-Type: text/csv` + `Content-Disposition: attachment`). No new
  dependencies are introduced.
- **Auth**: protected by the same shared API-key middleware as `GET /audit` —
  required when `VAULT_API_KEY` is active, otherwise open.

```bash
# JSON report for January 2026, actor=alice
curl "http://127.0.0.1:8765/audit/export?format=json&start=2026-01-01&end=2026-02-01&actor=alice"

# CSV download of all file.upload events (saved to audit.csv)
curl -o audit.csv "http://127.0.0.1:8765/audit/export?format=csv&action=file.upload"

# With API-key auth active
curl -H "Authorization: Bearer $VAULT_API_KEY" \
  "http://127.0.0.1:8765/audit/export?format=csv"
```

