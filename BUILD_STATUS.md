# RAG Pre-Flight Pipeline — Build Status

**Date:** 2026-05-26
**Project:** Local RAG Pre-Flight Pipeline (`rag-preflight`)
**Status:** All 7 pre-flight phases complete. Part 2 (phases 8–14) fully operational. Context augmentation complete (8,159/8,159).

---

## Current Status

### Part 1: Pre-Flight (phases 0–7) — COMPLETE

All 7 phases completed successfully on the sample corpus.

| Phase | Status | Files Processed | Failed |
|-------|--------|----------------|--------|
| 0. Archives | done | 2 | 0 |
| 1. Walk | done | 2177 | 0 |
| 2. Identify | done | 2177 | 0 |
| 3. Triage | done | 2152 | 23 |
| 4. Dedup | done | 232 groups | 0 |
| 5. Folder Meta | done | 71 | 0 |
| 6. Report | done | 2177 | 0 |
| 7. Exclude | ready | — | — |

Preflight completed at: 2026-05-02T03:56:16Z

### Part 2: RAG Pipeline (phases 8–14) — OPERATIONAL

| Phase | Status | Items Processed |
|-------|--------|----------------|
| 8. Extract | done | 325 extractions |
| 8b. Transcribe | implemented | Not used (opt-in) |
| 9. Summarize | done | 45 summaries |
| 10. Chunk | done | 8,159 chunks |
| 10.5. Context | done | 8,159 / 8,159 |
| 11. Embed | done | 8,159 chunks + 45 summaries + 71 folders |
| 12. Retrieval | done | Dense + BM25 + RRF fusion |
| 12. Generation | done | gpt-oss:20b with citations |
| 13. Eval | implemented | Not tested live |
| 14. API + UI | operational | API on :8001, UI on :7860 |

See `PART_2_STATUS.md` for full Part 2 details.

---

## Corpus Results

### Overview

| Metric | Value |
|--------|-------|
| Total files discovered | 2,177 |
| Unique files (after dedup) | 384 |
| Duplicate copies removed | 1,793 |
| Total folders | 71 |
| Total corpus size | 7.4 GB |
| Languages detected | English (63), Chinese (5) |

### File Categories (unique files)

| Category | Count | Description |
|----------|-------|-------------|
| cad | 112 | Windows PE executables (CAD-related) |
| data | 88 | PAK, QM, CSV, JSON, Draw.io, configs |
| image | 60 | JPEG, HEIC, PNG |
| video | 36 | MP4, QuickTime (1.41 hours total) |
| document | 29 | DOCX, DOC, TXT, HTML, Markdown, logs |
| pdf-text | 18 | Text-bearing PDFs |
| spreadsheet | 11 | XLSX |
| presentation | 8 | PPTX, PPSX |
| audio | 10 | MP3 (0.42 hours total) |
| executable | 6 | Windows PE, APK |
| archive | 5 | ZIP, RAR |
| code | 1 | Python source |

### Language Detection

| Language | Files | Percentage |
|----------|-------|------------|
| English | 63 | 92.6% |
| Chinese | 5 | 7.4% |

### OCR Workload

- **Scanned PDFs:** 0 files
- **Total pages to OCR:** 0

### Transcription Workload

- **Video files:** 36 files (1.41 hours)
- **Audio files:** 10 files (0.42 hours)

---

## All Issues Encountered and Resolved (Part 1)

### 1. sqlite-utils API Mismatch
**Problem:** Used `db.execute()` for SELECT queries (returns tuples), but code expected dicts.
**Fix:** Changed SELECT to `db.query()` (returns dicts), UPDATE/DELETE to raw `db.execute()`, lookups to `db["table"].rows_where()`.

### 2. Siegfried JSON Output Structure
**Problem:** TSD spec said `identification` field, but actual `sf -json` output uses `matches`.
**Fix:** Changed field mapping: `puid`→`id`, `name`→`format`.

### 3. SQLite Thread Safety in Phase 3
**Problem:** ThreadPoolExecutor triage workers shared the main thread's DB connection.
**Fix:** Each triage worker opens its own DB connection, closes in `finally` block.

### 4. Config.toml Overwritten by `init`
**Problem:** `init` wrote fresh config.toml every time, losing custom settings like `siegfried_path`.
**Fix:** Modified `init` to preserve existing config.toml using `tomllib` + `tomli_w`.

### 5. `status` Command Shows "unknown" for corpus_root
**Problem:** `init` didn't populate the `config` table in the DB.
**Fix:** Added `db["config"].insert()` for `corpus_root`, `cache_root`, `pipeline_version`.

### 6. Report Folder Count Incorrect
**Problem:** Report used `COUNT(DISTINCT folder_id)` from file table, not total folders.
**Fix:** Query `folder` table directly: `SELECT COUNT(*) AS total_folders FROM folder`.

### 7. Siegfried Not in Homebrew
**Problem:** `brew install siegfried` not available as Homebrew formula.
**Fix:** Downloaded binary from GitHub releases, placed at project root as `sf`.

### 8. `run-all` Command Click Context Bug
**Problem:** `ctx.invoke()` doesn't properly create subcommand contexts for nested invocation.
**Fix:** Changed to `ctx.forward()` which creates a fresh context for each subcommand.

### 9. Siegfried Binary Not Found on PATH
**Problem:** `shutil.which("sf")` fails when binary is in project root but not on PATH.
**Fix:** Added fallback lookup in project root directory.

### 10. Folder `INSERT OR REPLACE` Caused FK Constraint Failures
**Problem:** `_ensure_folder` used `replace=True`, deleting and re-inserting folder rows with `ON DELETE CASCADE` effects on child file rows.
**Fix:** Removed `replace=True` — if folder exists, return its id; only insert if not found.

### 11. Triage DB Updates Not Persisting
**Problem:** Per-thread DB connections used `execute()` which doesn't auto-commit for UPDATE statements. Changes were lost when connections closed.
**Fix:** Added explicit `tdb.conn.commit()` after each category's UPDATE statement.

### 12. Language Detection Never Ran for Non-PDF Files
**Problem:** `_detect_language` function existed but was never called in the triage worker's else branch.
**Fix:** Wired up `_detect_language` for all text-bearing categories (document, spreadsheet, data, etc.).

### 13. Phase0 + Phase1 FK Conflict (Archive Files)
**Problem:** Phase0 creates archive file rows with `hash_status='skipped'`. Walk phase tries `INSERT OR REPLACE` the same paths, triggering FK constraint failure due to self-referencing `extracted_from_archive_id` with `ON DELETE CASCADE`.
**Fix:** Walk phase detects existing files (`hash_status in ("done", "skipped")`) and adds them to the hash list without re-inserting. Uses `UPDATE` for files needing rehash.

### 14. Phase0 Extracted Folder Parent IDs Incorrect
**Problem:** Phase0 created extracted folders with `parent_id=None` and used `replace=True` on folder inserts.
**Fix:** Created `_ensure_folder()` helper that properly resolves parent_id. Extracted folder creation now looks up the parent folder id before inserting.

### 15. 795 Unknown Format Files (No PRONOM Match)
**Problem:** Most were extracted archive contents (.pak, .qm, etc.) with no PRONOM signature. `v_unknown_formats` view flagged them.
**Fix:** Added 12 UNKNOWN-* extension fallback entries to `seeds/format_policy.csv` and `EXT_FALLBACK` dict in `phase2_identify.py`. Each maps to a proper category and extract_strategy. Re-seeded DB and re-ran identify — zero unknown formats remain.

### 16. RAR Archive Extraction on ARM Mac
**Problem:** `rar` binary returned exit status -9 on ARM Mac during a prior run (older version).
**Fix:** Homebrew `rar` v7.20 (native ARM binary) extracts all RAR archives successfully. `unar` v1.10.8 also installed as a secondary fallback — patoolib 4.0.4 supports both.

---

## All Issues Encountered and Resolved (Part 2)

### 17. gemma4 Empty Responses via `generate()`
**Problem:** `client.generate()` returns empty content for gemma4:latest.
**Fix:** Switched to `client.chat()` with system prompt.

### 18. gemma4 Silent Fail with Low `num_predict`
**Problem:** `num_predict: 400` causes empty responses without error.
**Fix:** Removed `num_predict`, use `num_ctx: 16384` instead.

### 19. `Database` object has no `.path` attribute
**Problem:** `sqlite_utils.Database` doesn't expose the DB path.
**Fix:** Use `Path.cwd() / "corpus.db"` directly.

### 20. `embedding_ref.model` NOT NULL constraint
**Problem:** v2 migration created the column as NOT NULL, new embed code doesn't populate it.
**Fix:** Rebuilt table with nullable model, updated v2 migration DDL.

### 21. Chroma Metadata Type Error
**Problem:** Chroma rejects `None` values in metadata dicts.
**Fix:** Explicit type casting (str/int) in all embed functions.

### 22. Query Embedding Dimension Mismatch
**Problem:** Chroma's default embedder is 384-dim, chunks are 4096-dim.
**Fix:** Embed query with Ollama, pass `query_embeddings` instead of `query_texts`.

### 23. BM25 Returns Zero Results
**Problem:** FTS5 external content table not auto-populated.
**Fix:** Explicit INSERT into `chunk_fts` during chunking + FTS rebuild on retrieval.

### 24. Chroma ID Parsing
**Problem:** Chroma IDs stored as `c_N` strings, code tried `int("c_8")`.
**Fix:** Parse chunk_id from metadata field.

### 25. CLI `generate_answer()` Parameter Name
**Problem:** CLI passed `prompts_path=` but function expects `prompt_template=`.
**Fix:** Pass full template path.

### 26. SQLite Thread Safety in Gradio UI
**Problem:** DB connection shared across threads.
**Fix:** Thread-local storage with `_get_db()`.

### 27. Gradio 6.0 API Incompatibilities
**Problem:** `Chatbot(type="messages")` removed, `Blocks(theme=...)` moved.
**Fix:** Removed `type`, moved `theme` to `launch()`.

### 28. Context Augmentation Speed
**Problem:** gemma4 at ~1.6 chunks/min = ~85 hours.
**Fix:** Switched to MLX (`mlx-community/Qwen3-8B-4bit`) — completes at ~19 chunks/min (~7 hours).

---

## Project File Structure

```
.
├── cli.py                          # 30 click commands (14 pre-flight + 16 Part 2)
├── corpus.db                       # SQLite database (current run)
├── config.toml                     # Configuration (full post-preflight schema)
├── pyproject.toml                  # Dependencies
├── sf                              # Siegfried binary (v1.11.4)
├── README.md
├── BUILD_STATUS.md                 # This file
├── CLAUDE.md                       # AI assistant context
├── TEST_REPORT.md                  # Test suite report (167 tests, 86% coverage)
├── PART_2_STATUS.md                # Part 2 build status (phases 8-14)
├── seeds/
│   └── format_policy.csv           # 87 PRONOM policy entries
├── pipeline/
│   ├── __init__.py
│   ├── config.py                   # Pydantic config models (15 sub-configs)
│   ├── db.py                       # Schema (21 tables, 14 indexes, 9 views), migrations
│   ├── helpers.py                  # SHA-256, archive detection, error classification
│   ├── logging.py                  # JSON structured logging + rich console
│   ├── phase0_archives.py          # Archive expansion via patool
│   ├── phase1_walk.py              # File/folder walk + parallel SHA-256
│   ├── phase2_identify.py          # Siegfried PRONOM format ID + extension fallback
│   ├── phase3_triage.py            # PDF/media triage + language detection (threaded)
│   ├── phase4_dedup.py             # SQL-only exact deduplication
│   ├── phase5_folder_meta.py       # Ollama/gemma4 folder inference (bottom-up)
│   ├── phase6_report.py            # Aggregate report generation
│   └── phase7_exclude.py           # Exclusion marking (single + batch CSV)
├── rag/                            # Part 2: extraction through generation
│   ├── __init__.py
│   ├── phase8_extract.py           # Extraction dispatcher
│   ├── phase8b_transcribe.py       # Audio/video transcription (opt-in)
│   ├── phase9_summarize.py         # Document summarization
│   ├── phase10_chunk.py            # Sentence-window chunking
│   ├── phase10_5_context.py        # Contextual retrieval augmentation
│   ├── phase11_embed.py            # Embedding + Chroma management
│   ├── retrieval.py                # Hybrid retrieval (dense + BM25 + RRF)
│   ├── generation.py               # Citation-strict answer generation
│   ├── filters.py                  # Folder filter resolution
│   ├── citations.py                # Citation parsing utilities
│   └── extractors/                 # Text extraction backends
│       ├── __init__.py
│       ├── base.py
│       ├── docling_extractor.py
│       ├── tika_extractor.py
│       ├── ocr_extractor.py
│       ├── whisper_extractor.py    # Stub (opt-in)
│       ├── textutil_extractor.py
│       ├── filename_only_extractor.py
│       └── metadata_only_extractor.py
├── api/                            # Part 2: HTTP API
│   ├── __init__.py
│   ├── main.py                     # FastAPI app (10 endpoints)
│   └── schemas.py                  # Pydantic request/response models
├── eval/                           # Part 2: Evaluation
│   └── runner.py                   # add_question, run_eval, eval_report
├── ui/                             # Part 2: User interface
│   └── gradio_app.py               # Gradio chat interface
├── tests/                          # Part 1 test suite (167 tests)
│   ├── conftest.py
│   ├── test_helpers.py
│   ├── test_config.py
│   ├── test_db.py
│   ├── test_phase0_archives.py
│   ├── test_phase1_walk.py
│   ├── test_phase2_identify.py
│   ├── test_phase3_triage.py
│   ├── test_phase4_dedup.py
│   ├── test_phase5_folder_meta.py
│   ├── test_phase6_report.py
│   ├── test_phase7_exclude.py
│   ├── test_cli.py
│   └── test_e2e_pipeline.py
├── SourceData/                     # Sample corpus (7.4 GB, 2177 files)
│   └── preflight_report.md
├── chroma/                         # Chroma vector store (persistent collections)
└── .rag-cache/                     # Derived artifacts (extractions, models)
```

---

## Next Steps

### Immediate
1. ~~Complete context augmentation~~ — **DONE** (8,159/8,159).
2. **Write Part 2 tests** — no test coverage for rag/, api/, eval/, ui/ modules.

### Quality Improvements
3. **Chinese FTS5 tokenizer** — current Porter stemmer only handles English.
4. **Reranker implementation** — configured but not wired up.
5. **Evaluation benchmarks** — load questions, run eval, measure accuracy.
6. **Whisper extractor** — implement for audio/video transcription (92 files).

### Operational
7. **Multi-model embed switching** — embed-switch-to command exists, not tested with alternatives.
8. **Embed garbage collection** — embed-gc command exists, not tested.
