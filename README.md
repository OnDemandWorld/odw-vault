# rag-preflight + RAG Pipeline

A **fully offline** pre-flight pipeline + end-to-end RAG system for mixed-format document corpora. Analyzes a hierarchical folder of documents, identifies formats, deduplicates, extracts text, generates embeddings, and provides query access — all running on-premises with no outbound network calls during inference.

**Status:** Part 1 (pre-flight) and Part 2 (RAG pipeline) both implemented and operational. 167 tests passing. Full corpus processed: 384 unique files, 325 extractions, 8,159 chunks (all context-augmented), 8,275 embeddings.

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

### 4. Initialize
```bash
PYTHONPATH=. python cli.py init --root "/path/to/corpus"
```
Creates `corpus.db` and `.rag-cache/` in the project directory.

### 5. Run Pre-Flight (Part 1)
```bash
PYTHONPATH=. python cli.py run-all
```

### 6. Run RAG Pipeline (Part 2)
```bash
# Extract, summarize, chunk, embed
PYTHONPATH=. python cli.py extract
PYTHONPATH=. python cli.py summarize
PYTHONPATH=. python cli.py chunk
PYTHONPATH=. python cli.py context      # optional, slow
PYTHONPATH=. python cli.py embed
```

### 7. Query
```bash
# CLI
PYTHONPATH=. python cli.py query "What is this corpus about?" --top-k 5

# JSON output
PYTHONPATH=. python cli.py query "What formats are in the corpus?" --json

# API server
PYTHONPATH=. python cli.py serve --port 8001 &
curl -X POST http://127.0.0.1:8001/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What CAD files exist?"}'

# Gradio UI
PYTHONPATH=. python cli.py ui
```

## Verified Results

### Pre-Flight Results

| Metric | Value |
|--------|-------|
| Total files | 2,177 |
| Unique files (after dedup) | 384 |
| Duplicate copies removed | 1,793 |
| Total folders | 71 |
| Corpus size | 7.4 GB |
| Languages | English (63), Chinese (5) |
| Video duration | 1.41 hours (36 files) |
| Audio duration | 0.42 hours (10 files) |
| Scanned PDFs | 0 |
| Unknown formats | 0 |

### RAG Pipeline Results

| Metric | Value |
|--------|-------|
| Successful extractions | 325 / 372 eligible |
| Document summaries | 45 |
| Text chunks | 8,159 |
| Chunks with context | 8,159 / 8,159 (complete) |
| Chunk embeddings | 8,159 |
| Summary embeddings | 45 |
| Folder embeddings | 71 |
| Query response time | ~30s |

### Top Formats

| Format | Count | Category |
|--------|-------|----------|
| Windows PE (fmt/899) | 112 | cad |
| Unknown (UNKNOWN-pak) | 55 | data |
| JPEG (fmt/43) | 31 | image |
| HEIC (fmt/1101) | 23 | image |
| MP4 (fmt/199) | 21 | video |
| Unknown (UNKNOWN-qm) | 16 | data |
| QuickTime (x-fmt/384) | 15 | video |
| XLSX (fmt/214) | 11 | spreadsheet |
| CSV (x-fmt/18) | 11 | data |
| MP3 (fmt/134) | 10 | audio |

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
PYTHONPATH=. python cli.py init --root "/path/to/corpus" [--force]
PYTHONPATH=. python cli.py run-all
PYTHONPATH=. python cli.py archives [--max-depth N] [--dry-run]
PYTHONPATH=. python cli.py walk [--workers N] [--rehash]
PYTHONPATH=. python cli.py identify [--reidentify]
PYTHONPATH=. python cli.py triage [--workers N] [--categories CAT1,...]
PYTHONPATH=. python cli.py dedup
PYTHONPATH=. python cli.py folder-meta [--model NAME] [--reinfer]
PYTHONPATH=. python cli.py report [--output PATH]
PYTHONPATH=. python cli.py exclude --target {file,folder} --id N --reason TEXT
PYTHONPATH=. python cli.py exclude-batch --from-file exclusions.csv
PYTHONPATH=. python cli.py approve --by NAME
PYTHONPATH=. python cli.py status
PYTHONPATH=. python cli.py serve --port 8001           # API server

# Part 2: RAG Pipeline
PYTHONPATH=. python cli.py extract [--workers N] [--reextract]
PYTHONPATH=. python cli.py summarize [--resummarize]
PYTHONPATH=. python cli.py chunk [--window-size N] [--rechunk]
PYTHONPATH=. python cli.py context [--regenerate]
PYTHONPATH=. python cli.py embed [--model NAME] [--reembed]
PYTHONPATH=. python cli.py embed-switch-to --model NAME
PYTHONPATH=. python cli.py embed-gc
PYTHONPATH=. python cli.py embed-list
PYTHONPATH=. python cli.py query "question" [--top-k N] [--json]
PYTHONPATH=. python cli.py serve --port 8001           # API server
PYTHONPATH=. python cli.py ui --port 7860              # Gradio UI
PYTHONPATH=. python cli.py eval add/run/report
PYTHONPATH=. python cli.py models list/pull/check

# Tests
PYTHONPATH=. pytest tests/ --cov=pipeline --cov=cli --cov-report=term-missing -v
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
4. **Evaluation benchmarks** — load questions, run eval.
5. **Whisper extractor** — implement for audio/video (92 files).

## License

MIT
