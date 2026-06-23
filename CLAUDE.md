# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ODW.ai Vault** is a self-hosted, open-source Retrieval-Augmented Generation (RAG) platform — the sovereign knowledge core of the ODW.ai suite. It provides a `vault` CLI that runs a local, fully offline pre-flight pipeline (`rag-preflight`) and an end-to-end RAG system over a hierarchical folder of mixed-format documents.

The pre-flight phase analyzes documents, produces a structured, queryable manifest (`corpus.db`), and prepares the corpus for downstream extraction, chunking, embedding, and retrieval. The RAG pipeline then extracts text, summarizes, chunks, augments context, generates embeddings, and serves a queryable knowledge base through a FastAPI backend and a Gradio UI.

The system runs entirely on-premises with no outbound network calls during inference. All format identification, triage, and LLM inference use locally hosted resources (Ollama, Siegfried, etc.).

**Current state:** All 7 pre-flight phases and all RAG pipeline phases (8–14) are implemented and covered by 167 unit tests at 86% line coverage. The codebase is distributed without any bundled corpus, database, vector store, or cached artifacts; users provide their own documents and run the pipeline locally.

## Key Documents

- **README.md** — Project documentation, quick start, CLI reference, architecture overview
- **BUILD_STATUS.md** — Current status, all 16 bugs encountered/resolved, pending issues, next steps
- **TEST_REPORT.md** — Full test suite report: 167 tests, coverage gaps, bugs found during testing, future improvements
- **Technical Specification Document- Local RAG Pre-Flight Pipeline.md** — Original technical spec (schema, CLI, phase logic)
- **Pre-Flight Plan- Local RAG for Project Knowledge Base.md** — High-level phase overview
- **Pre-Flight Question Checklist for RAG Projects.md** — Design decision checklist

## Source Corpus

This repository ships without a sample corpus. Users place their own documents under a directory (conventionally `data/`) and run `vault init --root ./data/my-corpus` to point the pipeline at them. Supported formats include PDF, Office documents, images, video, audio, archives, CAD files, and executables.

## Architecture

### Technology Stack
- **Python 3.11** (managed via `uv`, venv at `.venv/`)
- **SQLite 3** via `sqlite-utils` as the database (`corpus.db`)
- **Siegfried 1.11.4** (binary at project root as `sf`) for PRONOM format identification
- **Ollama + `gemma4:latest`** for local LLM folder semantic inference
- **Ollama + `gpt-oss:20b`** (via Ollama.com API) for answer generation
- **MLX + `Qwen3-8B-4bit`** for contextual augmentation
- **Ollama + `qwen3-embedding:8b`** for embeddings
- **Chroma** (PersistentClient) for vector store
- **PyMuPDF (fitz)** for PDF triage and image dimension reading
- **patool + unar** for archive expansion
- **lingua-language-detector** for English/Chinese language detection
- **Pydantic 2** for config parsing and structured LLM output validation
- **click** for CLI surface, **rich** for terminal progress bars
- **tenacity** for retry logic on Ollama calls
- **ffprobe** (via Homebrew ffmpeg) for audio/video duration
- **Docling + RapidOCR** for text extraction
- **FastAPI + sse-starlette** for HTTP API
- **Gradio** for chat UI
- **pytest + pytest-cov** for testing (167 tests, 86% coverage)

### Project Layout
All paths relative to project root.

```
.
├── cli.py                          # 30+ click commands (pre-flight + RAG pipeline)
├── config.toml                     # Configuration
├── pyproject.toml                  # Dependencies + pytest config
├── sf                              # Siegfried binary (v1.11.4), user-provided
├── README.md
├── BUILD_STATUS.md
├── CLAUDE.md
├── TEST_REPORT.md                  # Test suite report (167 tests, 86% coverage)
├── PART_2_STATUS.md                # Part 2 build status (phases 8-14)
├── data/
│   └── README.md                   # Instructions for placing a corpus
├── seeds/
│   └── format_policy.csv           # 87 PRONOM policy entries (incl. UNKNOWN-* fallbacks)
├── pipeline/
│   ├── __init__.py
│   ├── config.py                   # Pydantic config models (15+ sub-configs)
│   ├── db.py                       # Schema (24 tables, 28 indexes, 12 views), migrations
│   ├── helpers.py                  # SHA-256, archive detection, error classification
│   ├── logging.py                  # JSON structured logging + rich console
│   ├── phase0_archives.py          # Archive expansion via patool
│   ├── phase1_walk.py              # File/folder walk + parallel SHA-256
│   ├── phase2_identify.py          # Siegfried PRONOM format ID + extension fallback
│   ├── phase3_triage.py            # PDF/media triage + language detection (threaded)
│   ├── phase4_dedup.py             # SQL-only exact deduplication
│   ├── phase5_folder_meta.py       # Ollama/gemma4 folder inference (bottom-up)
│   ├── phase6_report.py            # Aggregate markdown report
│   └── phase7_exclude.py           # Exclusion marking (single + batch CSV)
├── tests/                          # 15 test files, 167 tests
│   ├── conftest.py                 # Shared fixtures: corpus factory, DB, config, mocks
│   ├── test_helpers.py             # 20 tests
│   ├── test_config.py              # 15 tests
│   ├── test_db.py                  # 12 tests
│   ├── test_phase0_archives.py     # 10 tests
│   ├── test_phase1_walk.py         # 11 tests
│   ├── test_phase2_identify.py     # 13 tests
│   ├── test_phase3_triage.py       # 16 tests
│   ├── test_phase4_dedup.py        # 8 tests
│   ├── test_phase5_folder_meta.py  # 15 tests
│   ├── test_phase6_report.py       # 10 tests
│   ├── test_phase7_exclude.py      # 8 tests
│   ├── test_cli.py                 # 14 tests
│   └── test_e2e_pipeline.py        # 6 tests
├── rag/                            # Part 2: extraction through generation
│   ├── phase8_extract.py           # Extraction dispatcher
│   ├── phase8b_transcribe.py     # Audio/video transcription (opt-in)
│   ├── phase9_summarize.py         # Document summarization
│   ├── phase10_chunk.py            # Sentence-window chunking
│   ├── phase10_5_context.py        # Contextual retrieval augmentation
│   ├── phase11_embed.py            # Embedding + Chroma management
│   ├── retrieval.py                # Hybrid retrieval (dense + BM25 + RRF)
│   ├── generation.py               # Citation-strict answer generation
│   ├── filters.py                  # Folder filter resolution
│   ├── citations.py                # Citation parsing utilities
│   └── extractors/                 # Text extraction backends
├── api/                            # Part 2: HTTP API
│   ├── main.py                     # FastAPI app (10 endpoints + root redirect)
│   └── schemas.py                  # Pydantic request/response models
├── eval/                           # Part 2: Evaluation
│   └── runner.py                   # add_question, run_eval, eval_report
└── ui/                             # Part 2: User interface
    ├── gradio_app.py               # Gradio chat interface
    └── gradio_app_legacy.py        # Original Gradio UI (backup)
```

### Phase Pipeline (execute in order)
1. **Phase 0 — `archives`**: Recursively expand archives. Creates sibling `<archive>.extracted/` folders. Records provenance in `archive_expansion` table. Idempotent — skips already-expanded archives. Uses patool (supports ZIP, RAR, 7z, TAR, etc.). Excludes DOC_ARCHIVES (docx, xlsx, pptx, epub, jar, etc.) since these are compound documents, not archives to expand.

2. **Phase 1 — `walk`**: Walk tree, populate `folder`/`file` tables, compute SHA-256 in parallel (ProcessPoolExecutor). Skips hidden/system files (dot-prefixed, Thumbs.db, __MACOSX), cache directory. Skips oversized files (configurable, default 5 GiB). Reuses file rows from Phase 0 instead of INSERT OR REPLACE (avoids FK cascade on self-referencing `extracted_from_archive_id`).

3. **Phase 2 — `identify`**: Run Siegfried `sf -json -multi N` on all files with `hash_status='done'` and `identify_status='pending'`. Parse `matches` field (not `identification`), map `id` (not `puid`) and `format` (not `name`) to category/extract_strategy via `format_policy` table. Extension fallback (`EXT_FALLBACK` dict) handles files Siegfried can't identify (.pak, .qm, .drawio, .bin, etc.). Binary resolution: check PATH first, then project root fallback.

4. **Phase 3 — `triage`**: PDF text-vs-scanned detection (PyMuPDF page sampling), media duration (ffprobe), image dimensions (PyMuPDF or fitz), language detection (lingua). Thread-safe: each worker opens its own DB connection with explicit `conn.commit()` after UPDATEs. Categories with text-bearing content get language detection; PDFs get `page_count`, `has_text_layer`, `is_encrypted`, `is_corrupt`.

5. **Phase 4 — `dedup`**: Pure SQL. Group by SHA-256 where `hash_status='done'` and `excluded=0`. Assign `dup_group_id`, mark canonical copy as `is_dup_primary=1` (shortest path, oldest mtime tiebreaker). Only groups with 2+ files get a `dup_group_id`. Idempotent — safe to re-run.

6. **Phase 5 — `folder-meta`**: Bottom-up traversal (depth DESC). For each folder, build a prompt containing: path, parent labels, file count, format histogram, sample filenames. Call Ollama with `format="json"`, validate response via Pydantic `FolderInference` model. Cache by prompt hash to avoid redundant LLM calls. Retry via tenacity (3 attempts, exponential backoff).

7. **Phase 6 — `report`**: Compute folder-level aggregates (file_count, total_bytes, document_count, dominant_format). Generate `preflight_report.md` with: corpus overview, format histogram (top 30), category breakdown, OCR workload, transcription workload, duplicate summary, problem files, unknown formats, language distribution, folder taxonomy tree, failure summary. Marks `preflight_completed_at` in config table.

8. **Phase 7 — `exclude` + `approve`**: Mark files/folders as `excluded=1` with reason. Batch CSV import (`target,id,reason`). `db[table].get(id)` raises `NotFoundError` if row doesn't exist — catch and convert to `ValueError` in single exclusion, skip in batch. Final sign-off records `preflight_approved_by` and `preflight_completed_at`.

### Guiding Principles
- **Originals are never modified** — all derived artifacts go under `.rag-cache/`
- **Every phase is idempotent and resumable** — safe to re-run at any point
- **SQLite is the single source of truth** — `corpus.db` holds everything
- **All failures go to the `failure` table** — silent failures are defects
- **No outbound HTTP during pre-flight** — phases 0–7 are air-gap capable. Phase 12 (generation) uses Ollama.com API by default, can be switched to local Ollama.

### Commands

All commands run from project root.

```bash
# Setup
source .venv/bin/activate
uv pip install -e ".[dev]"

# Initialize a new corpus
vault init --root "/path/to/corpus" [--force]

# Run all phases
vault run-all

# Individual phases
vault archives [--max-depth N] [--dry-run]
vault walk [--workers N] [--rehash]
vault identify [--reidentify]
vault triage [--workers N] [--categories CAT1,CAT2,...]
vault dedup
vault folder-meta [--model NAME] [--reinfer] [--max-folders N]
vault report [--output PATH]

# Exclusion
vault exclude --target {file,folder} --id N --reason TEXT
vault exclude-batch --from-file exclusions.csv

# Sign-off
vault approve --by NAME

# Diagnostics
vault status              # JSON per-phase status
vault serve --port 8001   # Launch Datasette

# Tests
pytest tests/ --cov=pipeline --cov=cli --cov-report=term-missing -v
```

### External Dependencies (macOS)

```bash
brew install ffmpeg unar ollama
ollama pull gemma4:latest
# Siegfried: download from GitHub releases, place as ./sf
```

### Important Implementation Notes

#### sqlite-utils API Patterns
- `db.query("SELECT ...")` — returns list of dicts. Use for all SELECT queries.
- `db.execute("UPDATE ...", [params])` — returns Cursor, does NOT auto-commit. Must call `db.conn.commit()` explicitly for data-modifying queries outside of `with db.conn:` context.
- `db["table"].rows_where("col = ?", [val])` — returns iterator. Use `next(...)` for single row lookup.
- `db["table"].get(id)` — returns dict by primary key. **Raises `NotFoundError`** if row doesn't exist (does NOT return None).
- `db["table"].insert(row, ignore=True)` — skips if PK conflict.
- `db["table"].insert(row, replace=True)` — **dangerous** if table has FK constraints with ON DELETE CASCADE — deleting and re-inserting can cascade-delete child rows.
- `db["table"].update(id, updates)` — safe way to modify existing rows.
- All connections: `PRAGMA foreign_keys = ON`, `PRAGMA journal_mode = WAL`, `PRAGMA synchronous = NORMAL`.

#### Phase-Specific Gotchas
- **Phase 1 walk** must skip files already in DB from Phase 0 (check `hash_status in ("done", "skipped")`). Uses UPDATE for existing rows, not INSERT OR REPLACE.
- **Phase 2 identify** only processes rows already in `file` table with `identify_status='pending'` and `hash_status='done'`. Creating files on disk without DB rows results in 0 files processed.
- **Phase 3 triage** workers need per-thread DB connections with explicit `conn.commit()` after UPDATEs. Shared connections cause `database is locked` errors.
- **Phase 5 folder-meta** requires `folder` rows in DB. Empty DB = 0 folders = 0 inferences. Uses bottom-up traversal so children are processed before parents.
- **Phase 7 exclude** — `db[table].get(id)` raises `NotFoundError`, not `None`. The pipeline code catches this and converts to `ValueError` for single exclusion, or skips silently in batch mode.

#### Category Values
`document`, `spreadsheet`, `presentation`, `pdf-text`, `pdf-scanned`, `image`, `image-with-text`, `video`, `audio`, `archive`, `code`, `executable`, `data`, `email`, `ebook`, `cad`, `ros-bag`, `unknown`

#### Extract Strategy Values
`docling`, `tika`, `ocr`, `whisper`, `textutil`, `filename-only`, `metadata-only`, `skip`, `manual`, `unsupported`

### Database Schema
24 tables (18 user + FTS5 internals), 28 indexes, 12 views. See `pipeline/db.py` for the full DDL.

Key tables:
- `folder` — directory tree with semantic labels (inferred_category, inferred_label, etc.)
- `file` — file inventory: hash, format, category, triage, dedup status
- `format_policy` — PRONOM ID to category + extract strategy mapping (seeded from CSV)
- `extraction` — extracted text with provenance
- `summary` — document summaries
- `chunk` — sentence-window chunks with FTS5 + context_text
- `embedding_ref` — chunk embedding references
- `summary_embedding_ref` — summary embedding references
- `folder_embedding_ref` — folder embedding references
- `archive_expansion` — archive extraction provenance
- `pipeline_run` — per-phase run history
- `query_log` — query tracking with feedback
- `eval_question` / `eval_run` — evaluation framework
- `model_run` — per-model-run history
- `failure` — error tracking with classification

Key views: `v_format_histogram`, `v_category_summary`, `v_ocr_workload`, `v_transcription_workload`, `v_duplicate_summary`, `v_problem_files`, `v_unknown_formats`, `v_extraction_status`, `v_embedding_coverage`, `v_context_coverage`, `v_query_volume`, `v_eval_summary`

### Config Model
15+ sub-configs parsed from `config.toml` via Pydantic: `PathsConfig`, `WalkConfig`, `ArchivesConfig`, `IdentifyConfig`, `TriageConfig`, `OllamaConfig`, `FolderMetaConfig`, `ExtractConfig`, `ChunkConfig`, `RetrievalConfig`, `GenerationRuntimeConfig`, `APIConfig`, `UIConfig`, plus per-model configs for `embedding`, `summarization`, `contextual_retrieval`, `generation`, `reranker`, `transcription`, and `language_id`.

### Test Infrastructure
- 167 tests across 15 files. 86% overall coverage.
- No static binary fixtures — all test files generated programmatically (PyMuPDF for PDFs, zipfile for archives, zlib-constructed PNG bytes).
- External dependencies mocked: Ollama (returns `FolderInference`), Siegfried (returns JSON), patool (creates dummy files), ffprobe (returns duration JSON).
- Shared fixtures in `conftest.py`: `test_corpus`, `test_db`, `test_config`, `mock_plog`, `mock_ollama`, `mock_siegfried`, `mock_patool`.
- Run: `pytest tests/ --cov=pipeline --cov=cli --cov-report=term-missing -v`

### Pending Work

- All 7 pre-flight phases and RAG pipeline phases (8–14) are implemented. Remaining work focuses on expanding test coverage for `rag/`, `api/`, `eval/`, and `ui/` modules.

### Future Improvements

- **Per-task LLM endpoint routing** — Currently `[ollama].host` is a single global value shared by all phases. Different phases use different models that may live on different servers (e.g. remote MLX for context generation, local Ollama for embeddings, another endpoint for summarization). The config should support per-model or per-phase host definitions, e.g. `[endpoints.context]`, `[endpoints.embedding]`, `[endpoints.summarization]`, each with its own `host`, `model`, and auth settings. This would eliminate the manual config-switching workaround needed when switching between context generation (remote OpenAI-compat) and embedding (local Ollama native).
- **Part 2 test coverage** — no tests for rag/, api/, eval/, ui/ modules yet.
- **Chinese FTS5 tokenizer** — current Porter stemmer only handles English.
- **Reranker** — configured but not wired up.
- **Evaluation benchmarks** — eval framework exists but no questions loaded.
- **Whisper extractor** — stub exists, not implemented.

### Post-Pre-Flight Phases (8–14)

The TSD covers phases 0–7 only. Phases 8–14 are **out of scope** and "will be specified separately" in subsequent documents. All phases 8–14 are implemented and operational.
