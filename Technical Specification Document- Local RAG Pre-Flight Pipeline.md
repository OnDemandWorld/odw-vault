# Technical Specification Document: Local RAG Pre-Flight Pipeline

**Document Version:** 1.0
**Status:** Approved for Implementation
**Target Executor:** Autonomous Coding Agent
**Scope:** Pre-flight inventory, identification, triage, and semantic enrichment phases of a local RAG knowledge base. Extraction, embedding, and retrieval phases are out of scope for this TSD and will be specified separately.

---

## 1. Project Overview & Goals

### 1.1 System Purpose

The system is a local, fully offline pre-flight pipeline that analyzes a hierarchical folder of mixed-format documents and produces a structured, queryable manifest of the corpus. The manifest serves as the foundation for downstream Retrieval-Augmented Generation (RAG) phases (extraction, chunking, embedding, retrieval) and as a standalone tool for human review of corpus composition.

The system operates against confidential corpora and must run entirely on-premises with no outbound network calls during inference. All format identification, triage, and language model inference must execute against locally hosted resources.

### 1.2 Core Functional Requirements

The system must recursively expand archive files in place, treating their contents as part of the corpus while preserving provenance. It must produce a content-addressed inventory of every file using SHA-256 hashing. It must identify file formats using PRONOM signatures rather than file extensions. It must triage files by category to estimate downstream extraction cost (OCR pages, transcription hours, document character volume). It must detect exact duplicates via hash equivalence. It must infer semantic labels for folders using a locally hosted language model, given that the source corpus is humanly organized and folder names carry meaning. It must produce a human-readable pre-flight report and expose the underlying database for interactive review.

### 1.3 Non-Functional Requirements

Every phase must be idempotent and resumable. Re-execution against an unchanged corpus must perform no work beyond verifying state. Originals must never be modified; all derived artifacts must be written under a single cache directory. All long-running operations must record status and write failures to a dedicated table for inspection. The system must operate without internet access once dependencies are installed and models are downloaded. The SQLite database must be the authoritative project artifact: deletion of the cache must permit reconstruction; deletion of the source corpus must not invalidate the database for query purposes.

### 1.4 Out of Scope

Text extraction, chunking, embedding generation, vector store population, retrieval, generation, user interface, multi-user access control, and authentication are explicitly out of scope for this TSD. The schema defined herein contains forward-compatible tables for these phases; their implementation is specified in subsequent documents.

---

## 2. Technical Stack Selection

### 2.1 Primary Language

**Python 3.11 or later.** Justification: the dominant ecosystem for document processing, format identification wrappers, and local LLM orchestration is Python. Required libraries (`sqlite-utils`, `pymupdf`, `patool`, `ollama`, `fasttext`) are Python-native or have first-class Python bindings. Python 3.11 is mandated as the minimum due to performance improvements in `pathlib` and the maturity of `tomllib` for configuration parsing.

### 2.2 Frameworks and Libraries

The system is implemented as a command-line application; no web framework is required at this stage. The following libraries are mandated:

**`sqlite-utils` (>=3.36)** as the database access layer. Justification: removes boilerplate for schema creation, bulk inserts, JSON column handling, and migrations; pairs natively with the chosen database; authored and maintained alongside Datasette which is the recommended exploration tool.

**`click` (>=8.1)** for the CLI surface. Justification: standard, declarative, supports nested subcommands required for per-phase invocation.

**`pydantic` (>=2.5)** for configuration parsing and structured-output validation from the LLM. Justification: required for reliable parsing of JSON returned by the folder-inference LLM step.

**`ollama` Python client (>=0.3)** for local LLM inference. Justification: official client, stable API, supports structured outputs.

**`pymupdf` (>=1.24, distributed as `PyMuPDF`)** for PDF triage. Justification: fastest open-source PDF library, accurate text extraction sampling, low memory footprint.

**`patool` (>=2.3)** plus system `unar` for archive expansion. Justification: `patool` provides a unified Python interface over many archive formats; `unar` (installed via Homebrew) handles RAR and obscure archive types `patool` alone cannot.

**`python-magic` (>=0.4.27)** as a fallback identifier when Siegfried produces no match. Justification: complements PRONOM-based identification with libmagic heuristics.

**`fasttext` (>=0.9.2)** with the `lid.176.bin` model for language detection. Justification: fastest accurate language identification with broad language coverage including Traditional and Simplified Chinese.

**`rich` (>=13.7)** for terminal output and progress bars. Justification: required for usability of long-running phases.

**`tenacity` (>=8.2)** for retry logic on Ollama calls. Justification: local LLM servers occasionally stall; retry-with-backoff is required for reliability.

External binaries (not Python packages) that must be present on `PATH`:

`sf` (Siegfried, >=1.11), installed via `brew install siegfried`. `ffprobe` (from FFmpeg, >=6.0), installed via `brew install ffmpeg`. `unar` (>=1.10), installed via `brew install unar`. `ollama` (>=0.3), installed via `brew install ollama`.

### 2.3 Database

**SQLite 3.40 or later** (the version bundled with Python 3.11 on macOS suffices). The database file is `corpus.db` at the project root.

Justification: the dataset is single-writer and single-machine by design (confidential local corpus). SQLite handles corpora into the millions of files without performance degradation. It produces a single portable file, which is operationally valuable for backup, transfer, and auditability. It supports JSON columns natively for storing tool outputs without information loss. It supports FTS5 virtual tables for full-text search on extracted content in later phases. No separate database server reduces deployment surface and matches the air-gap-capable design.

PostgreSQL is explicitly rejected for this use case due to operational overhead disproportionate to the workload.

### 2.4 Configuration

A single `config.toml` file at the project root drives all phases. The schema is specified in §5.1.

### 2.5 Runtime Topology

The system runs as a sequence of CLI subcommands, each implementing one phase. Inter-phase communication is exclusively through the SQLite database. No long-running processes, no message queues, no network services. Ollama runs as a separate background process listening on `localhost:11434`.

---

## 3. Data Modeling & Schemas

All tables use `INTEGER PRIMARY KEY AUTOINCREMENT` for surrogate keys unless otherwise noted. All timestamps are ISO-8601 strings stored in UTC. Foreign keys are enforced (`PRAGMA foreign_keys = ON` is set on every connection). JSON columns are stored as `TEXT` and validated at the application layer.

### 3.1 Entity Relationship Summary

A `folder` has many `file` rows and zero-or-one parent `folder`. A `file` has zero-or-one source `archive_expansion` (if it was extracted from an archive) and zero-or-many `extraction` rows (forward-compatible). A `file` belongs to exactly one `folder`. Files sharing a SHA-256 share a `dup_group_id`. The `format_policy` table is referenced by `pronom_id` from `file` (logical foreign key, not enforced because Siegfried may produce IDs not yet in policy).

### 3.2 Table: `schema_version`

Tracks applied migrations.

| Column | Type | Constraints | Notes |
|---|---|---|---|
| `version` | INTEGER | PRIMARY KEY | Monotonically increasing |
| `applied_at` | TEXT | NOT NULL DEFAULT (datetime('now')) | UTC timestamp |
| `description` | TEXT | NULL | Human-readable migration name |

### 3.3 Table: `config`

Persisted runtime configuration and pipeline metadata.

| Column | Type | Constraints |
|---|---|---|
| `key` | TEXT | PRIMARY KEY |
| `value` | TEXT | NOT NULL |
| `updated_at` | TEXT | NOT NULL DEFAULT (datetime('now')) |

Required keys: `corpus_root` (absolute path), `cache_root` (absolute path), `preflight_completed_at` (ISO timestamp, set after Phase 6), `preflight_approved_by` (string, set after Phase 7), `pipeline_version` (string).

### 3.4 Table: `pipeline_run`

One row per phase invocation.

| Column | Type | Constraints |
|---|---|---|
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT |
| `phase` | TEXT | NOT NULL CHECK (phase IN ('archives','walk','identify','triage','dedup','folder_meta','report')) |
| `started_at` | TEXT | NOT NULL DEFAULT (datetime('now')) |
| `finished_at` | TEXT | NULL |
| `status` | TEXT | NOT NULL DEFAULT 'running' CHECK (status IN ('running','done','failed','aborted')) |
| `files_processed` | INTEGER | NULL |
| `files_failed` | INTEGER | NULL |
| `notes` | TEXT | NULL |
| `config_snapshot_json` | TEXT | NULL |

Index: `idx_pipeline_run_phase` on `(phase, started_at DESC)`.

### 3.5 Table: `folder`

| Column | Type | Constraints |
|---|---|---|
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT |
| `path` | TEXT | NOT NULL UNIQUE |
| `rel_path` | TEXT | NOT NULL |
| `parent_id` | INTEGER | NULL REFERENCES folder(id) ON DELETE CASCADE |
| `name` | TEXT | NOT NULL |
| `depth` | INTEGER | NOT NULL |
| `is_extracted_archive` | INTEGER | NOT NULL DEFAULT 0 CHECK (is_extracted_archive IN (0,1)) |
| `source_archive_file_id` | INTEGER | NULL REFERENCES file(id) |
| `excluded` | INTEGER | NOT NULL DEFAULT 0 CHECK (excluded IN (0,1)) |
| `exclusion_reason` | TEXT | NULL |
| `file_count` | INTEGER | NULL |
| `total_bytes` | INTEGER | NULL |
| `document_count` | INTEGER | NULL |
| `dominant_format` | TEXT | NULL |
| `inferred_category` | TEXT | NULL |
| `inferred_label` | TEXT | NULL |
| `inferred_tags_json` | TEXT | NULL |
| `inferred_summary` | TEXT | NULL |
| `inference_model` | TEXT | NULL |
| `inference_prompt_hash` | TEXT | NULL |
| `inferred_at` | TEXT | NULL |
| `created_at` | TEXT | NOT NULL DEFAULT (datetime('now')) |
| `updated_at` | TEXT | NOT NULL DEFAULT (datetime('now')) |

Indexes: `idx_folder_parent` on `parent_id`; `idx_folder_depth` on `depth`; `idx_folder_excluded` on `excluded`.

### 3.6 Table: `file`

| Column | Type | Constraints |
|---|---|---|
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT |
| `folder_id` | INTEGER | NOT NULL REFERENCES folder(id) ON DELETE CASCADE |
| `path` | TEXT | NOT NULL UNIQUE |
| `rel_path` | TEXT | NOT NULL |
| `name` | TEXT | NOT NULL |
| `extension` | TEXT | NULL |
| `size_bytes` | INTEGER | NOT NULL |
| `mtime` | TEXT | NOT NULL |
| `sha256` | TEXT | NULL |
| `extracted_from_archive_id` | INTEGER | NULL REFERENCES file(id) |
| `pronom_id` | TEXT | NULL |
| `mime_type` | TEXT | NULL |
| `format_name` | TEXT | NULL |
| `format_version` | TEXT | NULL |
| `siegfried_json` | TEXT | NULL |
| `id_warning` | TEXT | NULL |
| `category` | TEXT | NULL |
| `extract_strategy` | TEXT | NULL |
| `is_encrypted` | INTEGER | NULL CHECK (is_encrypted IN (0,1) OR is_encrypted IS NULL) |
| `is_corrupt` | INTEGER | NULL CHECK (is_corrupt IN (0,1) OR is_corrupt IS NULL) |
| `page_count` | INTEGER | NULL |
| `duration_seconds` | REAL | NULL |
| `has_text_layer` | INTEGER | NULL |
| `triage_json` | TEXT | NULL |
| `dup_group_id` | INTEGER | NULL |
| `is_dup_primary` | INTEGER | NOT NULL DEFAULT 1 CHECK (is_dup_primary IN (0,1)) |
| `excluded` | INTEGER | NOT NULL DEFAULT 0 CHECK (excluded IN (0,1)) |
| `exclusion_reason` | TEXT | NULL |
| `hash_status` | TEXT | NOT NULL DEFAULT 'pending' |
| `identify_status` | TEXT | NOT NULL DEFAULT 'pending' |
| `triage_status` | TEXT | NOT NULL DEFAULT 'pending' |
| `error_message` | TEXT | NULL |
| `created_at` | TEXT | NOT NULL DEFAULT (datetime('now')) |
| `updated_at` | TEXT | NOT NULL DEFAULT (datetime('now')) |

`category` permitted values: `document`, `spreadsheet`, `presentation`, `pdf-text`, `pdf-scanned`, `image`, `image-with-text`, `video`, `audio`, `archive`, `code`, `executable`, `data`, `email`, `ebook`, `cad`, `ros-bag`, `unknown`.

`extract_strategy` permitted values: `docling`, `tika`, `ocr`, `whisper`, `textutil`, `filename-only`, `metadata-only`, `skip`, `manual`, `unsupported`.

Status columns permitted values: `pending`, `running`, `done`, `failed`, `skipped`.

Indexes: `idx_file_folder` on `folder_id`; `idx_file_sha256` on `sha256`; `idx_file_category` on `category`; `idx_file_pronom` on `pronom_id`; `idx_file_dup_group` on `dup_group_id`; `idx_file_hash_status` on `hash_status`; `idx_file_identify_status` on `identify_status`; `idx_file_triage_status` on `triage_status`; `idx_file_excluded` on `excluded`.

### 3.7 Table: `archive_expansion`

| Column | Type | Constraints |
|---|---|---|
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT |
| `archive_file_id` | INTEGER | NOT NULL REFERENCES file(id) |
| `extracted_to_path` | TEXT | NOT NULL |
| `extracted_to_folder_id` | INTEGER | NULL REFERENCES folder(id) |
| `tool` | TEXT | NOT NULL |
| `succeeded` | INTEGER | NOT NULL CHECK (succeeded IN (0,1)) |
| `file_count` | INTEGER | NULL |
| `error_message` | TEXT | NULL |
| `extracted_at` | TEXT | NOT NULL DEFAULT (datetime('now')) |

Index: `idx_archive_file` on `archive_file_id`.

### 3.8 Table: `format_policy`

Seeded from `seeds/format_policy.csv`; user-editable.

| Column | Type | Constraints |
|---|---|---|
| `pronom_id` | TEXT | PRIMARY KEY |
| `format_name` | TEXT | NULL |
| `category` | TEXT | NOT NULL |
| `extract_strategy` | TEXT | NOT NULL |
| `notes` | TEXT | NULL |
| `updated_at` | TEXT | NOT NULL DEFAULT (datetime('now')) |

### 3.9 Table: `failure`

| Column | Type | Constraints |
|---|---|---|
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT |
| `file_id` | INTEGER | NULL REFERENCES file(id) ON DELETE CASCADE |
| `folder_id` | INTEGER | NULL REFERENCES folder(id) ON DELETE CASCADE |
| `phase` | TEXT | NOT NULL |
| `tool` | TEXT | NULL |
| `error_class` | TEXT | NULL |
| `error_message` | TEXT | NULL |
| `traceback` | TEXT | NULL |
| `occurred_at` | TEXT | NOT NULL DEFAULT (datetime('now')) |

`error_class` controlled vocabulary: `permission`, `timeout`, `encrypted`, `corrupt`, `parse_error`, `tool_missing`, `unsupported_format`, `oversized`, `network`, `unknown`.

Indexes: `idx_failure_file` on `file_id`; `idx_failure_phase` on `phase`; `idx_failure_class` on `error_class`.

### 3.10 Forward-Compatible Tables

The following tables are created by the migration but not populated during pre-flight. They exist so downstream phases do not require schema changes: `extraction`, `summary`, `chunk`, `embedding_ref`, `chunk_fts` (FTS5 virtual table). Their definitions match those provided in the project plan and are reproduced verbatim in the migration script (§5.2).

### 3.11 Views

```sql
CREATE VIEW v_format_histogram AS
SELECT format_name, pronom_id, COUNT(*) AS n, SUM(size_bytes) AS bytes,
       SUM(CASE WHEN extract_strategy='skip' THEN 0 ELSE 1 END) AS extractable
FROM file WHERE is_dup_primary=1 AND excluded=0
GROUP BY pronom_id, format_name ORDER BY n DESC;

CREATE VIEW v_category_summary AS
SELECT category, COUNT(*) AS n, SUM(size_bytes) AS bytes
FROM file WHERE is_dup_primary=1 AND excluded=0
GROUP BY category ORDER BY n DESC;

CREATE VIEW v_ocr_workload AS
SELECT COUNT(*) AS scanned_pdfs, COALESCE(SUM(page_count),0) AS total_pages
FROM file WHERE category='pdf-scanned' AND is_dup_primary=1 AND excluded=0;

CREATE VIEW v_transcription_workload AS
SELECT category, COUNT(*) AS n,
       ROUND(COALESCE(SUM(duration_seconds),0)/3600.0, 2) AS total_hours
FROM file WHERE category IN ('audio','video') AND is_dup_primary=1 AND excluded=0
GROUP BY category;

CREATE VIEW v_duplicate_summary AS
SELECT dup_group_id, COUNT(*) AS copies, MIN(path) AS example_path,
       MAX(size_bytes) AS size_bytes
FROM file WHERE dup_group_id IS NOT NULL
GROUP BY dup_group_id HAVING copies>1 ORDER BY copies DESC;

CREATE VIEW v_problem_files AS
SELECT id, path, category, error_message, identify_status, triage_status
FROM file
WHERE error_message IS NOT NULL OR is_corrupt=1 OR is_encrypted=1
   OR id_warning IS NOT NULL OR category='unknown';

CREATE VIEW v_unknown_formats AS
SELECT pronom_id, format_name, COUNT(*) AS n
FROM file WHERE pronom_id IS NOT NULL
  AND pronom_id NOT IN (SELECT pronom_id FROM format_policy)
GROUP BY pronom_id, format_name ORDER BY n DESC;
```

---

## 4. API Design & Endpoints

The system exposes a CLI rather than an HTTP API for the pre-flight stage. CLI commands are specified with the same rigor as HTTP endpoints. The CLI is invoked as `rag-preflight <command> [options]`.

### 4.1 Command: `init`

Initialize the database and configuration in a target directory.

**Invocation:** `rag-preflight init --root <path> [--cache <path>] [--force]`

**Inputs:**

| Argument | Type | Required | Description |
|---|---|---|---|
| `--root` | path | yes | Absolute path to corpus root |
| `--cache` | path | no | Cache directory; defaults to `<root>/.rag-cache` |
| `--force` | flag | no | Overwrite existing `corpus.db` |

**Behavior:** Creates `corpus.db`, applies all schema migrations, seeds `format_policy` from `seeds/format_policy.csv`, writes `config.toml` template, creates `<cache>/{extractions,archives,logs}` directories.

**Exit codes:** `0` on success; `1` if database exists and `--force` not given; `2` if `--root` does not exist or is not a directory; `3` on permission errors.

### 4.2 Command: `archives`

Phase 0. Recursively expand archive files.

**Invocation:** `rag-preflight archives [--max-depth N] [--dry-run]`

**Inputs:**

| Argument | Type | Required | Description |
|---|---|---|---|
| `--max-depth` | int | no | Maximum recursion depth for nested archives; default 5 |
| `--dry-run` | flag | no | List archives that would be expanded; do not extract |

**Behavior:** Identifies archive files by extension and magic bytes, expands each into a sibling folder named `<archive>.extracted/`, records each expansion in `archive_expansion`. Re-runs internally until no new archives are found or `max_depth` reached. Skips archives already recorded as successfully expanded (idempotency check by `archive_file_id` and source mtime).

**Exit codes:** `0` on success; `4` if any archive failed to expand (details in `failure` table).

### 4.3 Command: `walk`

Phase 1. Inventory all files and folders; compute SHA-256.

**Invocation:** `rag-preflight walk [--workers N] [--rehash]`

**Inputs:**

| Argument | Type | Required | Description |
|---|---|---|---|
| `--workers` | int | no | Parallel hash workers; default `os.cpu_count()` |
| `--rehash` | flag | no | Force re-hash of files whose mtime/size match DB (default: skip) |

**Behavior:** Walks `corpus_root`. Inserts/updates rows in `folder` and `file`. Skips hidden files, `.DS_Store`, anything under `cache_root`, files exceeding `max_file_size_bytes` from config (flagged in `failure` with `error_class='oversized'`). Computes SHA-256 in parallel. Sets `hash_status='done'` per file.

**Exit codes:** `0` on success; `5` on incomplete walk (details in `failure`).

### 4.4 Command: `identify`

Phase 2. Run Siegfried; populate format fields and assign category/strategy.

**Invocation:** `rag-preflight identify [--reidentify]`

**Behavior:** Invokes `sf -json -multi 32 <corpus_root>`, parses output, joins to `file` by absolute path, writes `pronom_id`, `mime_type`, `format_name`, `format_version`, `siegfried_json`, `id_warning`. For each file, looks up `pronom_id` in `format_policy`; if found, sets `category` and `extract_strategy`; if not, sets `category='unknown'` and `extract_strategy='manual'` and inserts a row in `failure` with `error_class='unsupported_format'`. Sets `identify_status='done'`.

**Exit codes:** `0` on success; `6` if Siegfried binary missing or returns non-zero.

### 4.5 Command: `triage`

Phase 3. Format-specific cheap inspection.

**Invocation:** `rag-preflight triage [--workers N] [--categories CAT1,CAT2,...]`

**Behavior:** Selects files where `triage_status='pending'` and `excluded=0` and `is_dup_primary=1`. Dispatches by `category`:

PDF: open with `pymupdf`, sample three pages, count chars, set `page_count`, `has_text_layer`, classify as `pdf-text` or `pdf-scanned` (overrides `category`). Detect encryption.

Office documents: detect encryption via `msoffcrypto-tool` if present, else mark `is_encrypted=NULL`.

Audio/video: invoke `ffprobe -v quiet -print_format json -show_format`, store `duration_seconds`.

Images: read dimensions via `pymupdf` (works for common image formats) or `Pillow` if installed; store in `triage_json`.

Other categories: no-op, set `triage_status='done'`.

Failures recorded to `failure`; `triage_status='failed'` set.

**Exit codes:** `0` on success.

### 4.6 Command: `dedup`

Phase 4. Group exact duplicates.

**Invocation:** `rag-preflight dedup`

**Behavior:** SQL-only operation. Assigns `dup_group_id` to all files sharing a SHA-256 (group id is the smallest `file.id` in the group). Marks all but one member of each group as `is_dup_primary=0`. Selection of canonical: shortest `rel_path`, tiebreaker oldest `mtime`. Implemented as a single transaction.

**Exit codes:** `0` on success.

### 4.7 Command: `folder-meta`

Phase 5. LLM inference over folders.

**Invocation:** `rag-preflight folder-meta [--model NAME] [--reinfer] [--max-folders N]`

**Inputs:**

| Argument | Type | Required | Description |
|---|---|---|---|
| `--model` | string | no | Ollama model tag; default from config |
| `--reinfer` | flag | no | Re-infer folders even if `inference_prompt_hash` unchanged |
| `--max-folders` | int | no | Cap number of folders processed in this run (for testing) |

**Behavior:** Bottom-up traversal of `folder` where `excluded=0`. For each folder, builds prompt from folder name, parent inferred labels, sample of up to 30 child filenames, file-type histogram. Computes `inference_prompt_hash` (SHA-256 of canonicalized prompt). If hash matches existing row and `--reinfer` not set, skips. Otherwise calls Ollama with structured output schema (see §5.4). Validates response against Pydantic model. Writes `inferred_category`, `inferred_label`, `inferred_tags_json`, `inferred_summary`, `inference_model`, `inference_prompt_hash`, `inferred_at`. Retries up to 3 times on failure.

**Exit codes:** `0` on success; `7` if Ollama unreachable.

### 4.8 Command: `report`

Phase 6. Generate aggregates and markdown report.

**Invocation:** `rag-preflight report [--output PATH]`

**Behavior:** Computes folder-level aggregates (`file_count`, `total_bytes`, `document_count`, `dominant_format`) by querying `file` and updating `folder`. Generates `preflight_report.md` containing all view outputs plus a folder-tree summary with inferred labels. Sets `config.preflight_completed_at` to current timestamp.

**Exit codes:** `0` on success.

### 4.9 Command: `exclude`

Phase 7. Mark folders or files as excluded.

**Invocation:** `rag-preflight exclude --target {file,folder} --id N --reason TEXT`

Or batch mode: `rag-preflight exclude --from-file path/to/exclusions.csv`

CSV schema: `target,id,reason`.

**Exit codes:** `0` on success; `8` if target id does not exist.

### 4.10 Command: `approve`

Final sign-off. Sets `config.preflight_approved_by` and `config.preflight_completed_at`.

**Invocation:** `rag-preflight approve --by NAME`

### 4.11 Command: `status`

Diagnostic. Prints per-phase counts and last run timestamps.

**Invocation:** `rag-preflight status`

**Output (stdout, JSON):**

```json
{
  "corpus_root": "/path/to/corpus",
  "phases": {
    "archives": {"last_run": "2026-05-02T10:00:00Z", "status": "done", "expanded": 142},
    "walk": {"last_run": "2026-05-02T10:15:00Z", "status": "done", "files": 45231, "folders": 1893},
    "identify": {"last_run": "2026-05-02T10:30:00Z", "status": "done", "identified": 45102, "unknown": 129},
    "triage": {"last_run": "2026-05-02T11:00:00Z", "status": "done", "triaged": 45102, "failed": 17},
    "dedup": {"last_run": "2026-05-02T11:01:00Z", "status": "done", "groups": 3421, "duplicates": 9812},
    "folder_meta": {"last_run": "2026-05-02T11:45:00Z", "status": "done", "inferred": 1893},
    "report": {"last_run": "2026-05-02T11:46:00Z", "status": "done"}
  },
  "approved": false
}
```

### 4.12 Command: `serve`

Launches Datasette against `corpus.db` for interactive exploration.

**Invocation:** `rag-preflight serve [--port 8001]`

**Behavior:** Shells out to `datasette serve corpus.db --port <port> --setting truncate_cells_html 200`.

---

## 5. Implementation Details & Code Snippets

### 5.1 Configuration Schema

`config.toml`:

```toml
[paths]
corpus_root = "/Users/op/projects/corpus"
cache_root = "/Users/op/projects/corpus/.rag-cache"

[walk]
max_file_size_bytes = 5_368_709_120  # 5 GiB
skip_patterns = [".DS_Store", "Thumbs.db", "__MACOSX", "*.tmp"]
hash_workers = 0  # 0 = os.cpu_count()

[archives]
max_depth = 5
expand_extensions = [".zip", ".rar", ".7z", ".tar", ".tar.gz", ".tgz", ".tar.bz2"]
exclude_extensions = [".pages", ".numbers", ".key", ".docx", ".xlsx", ".pptx", ".epub", ".jar"]

[identify]
siegfried_path = "sf"
siegfried_workers = 32

[triage]
pdf_sample_pages = 3
pdf_text_threshold_chars_per_page = 50

[ollama]
host = "http://localhost:11434"
model = "qwen2.5:7b"
timeout_seconds = 120
max_retries = 3

[folder_meta]
max_filenames_in_prompt = 30
min_files_to_infer = 1
```

### 5.2 Migration Script Structure

```python
# pipeline/db.py
from pathlib import Path
import sqlite_utils

MIGRATIONS = [
    (1, "initial schema", """
        CREATE TABLE schema_version (...);
        CREATE TABLE config (...);
        CREATE TABLE pipeline_run (...);
        CREATE TABLE folder (...);
        CREATE TABLE file (...);
        CREATE TABLE archive_expansion (...);
        CREATE TABLE format_policy (...);
        CREATE TABLE failure (...);
        CREATE TABLE extraction (...);
        CREATE TABLE summary (...);
        CREATE TABLE chunk (...);
        CREATE TABLE embedding_ref (...);
        CREATE VIRTUAL TABLE chunk_fts USING fts5(
            text, content='chunk', content_rowid='id',
            tokenize='porter unicode61'
        );
        -- all indexes
        -- all views
    """),
]

def open_db(path: Path) -> sqlite_utils.Database:
    db = sqlite_utils.Database(path)
    db.conn.execute("PRAGMA foreign_keys = ON")
    db.conn.execute("PRAGMA journal_mode = WAL")
    db.conn.execute("PRAGMA synchronous = NORMAL")
    return db

def migrate(db: sqlite_utils.Database) -> None:
    db.executescript("CREATE TABLE IF NOT EXISTS schema_version "
                     "(version INTEGER PRIMARY KEY, applied_at TEXT, description TEXT)")
    current = next(iter(db.execute("SELECT COALESCE(MAX(version),0) FROM schema_version")))[0]
    for version, description, sql in MIGRATIONS:
        if version > current:
            with db.conn:
                db.executescript(sql)
                db.conn.execute(
                    "INSERT INTO schema_version(version, description) VALUES (?,?)",
                    (version, description)
                )
```

### 5.3 Archive Expansion (Phase 0) — Critical Logic

```python
# pipeline/phase0_archives.py
import patoolib
from pathlib import Path

ARCHIVE_MAGIC = {
    b"PK\x03\x04": "zip",
    b"Rar!\x1a\x07": "rar",
    b"7z\xbc\xaf\x27\x1c": "7z",
    b"\x1f\x8b": "gzip",
}

def is_archive(path: Path, config) -> bool:
    if path.suffix.lower() in config.archives.exclude_extensions:
        return False
    if path.suffix.lower() in config.archives.expand_extensions:
        return True
    try:
        with path.open("rb") as f:
            head = f.read(8)
        return any(head.startswith(m) for m in ARCHIVE_MAGIC)
    except OSError:
        return False

def expand_recursive(db, root: Path, config) -> int:
    expanded_total = 0
    for depth in range(config.archives.max_depth):
        archives = [p for p in root.rglob("*") if is_archive(p, config)]
        # filter out already-expanded archives by checking archive_expansion
        already = set(r["archive_file_path"] for r in db.execute(
            "SELECT f.path AS archive_file_path FROM archive_expansion ae "
            "JOIN file f ON f.id = ae.archive_file_id WHERE ae.succeeded = 1"
        ))
        archives = [p for p in archives if str(p) not in already]
        if not archives:
            break
        for arc in archives:
            target = arc.parent / f"{arc.name}.extracted"
            target.mkdir(exist_ok=True)
            try:
                patoolib.extract_archive(str(arc), outdir=str(target), verbosity=-1)
                file_count = sum(1 for _ in target.rglob("*") if _.is_file())
                # insert/upsert file row for arc, then archive_expansion row
                record_expansion(db, arc, target, "patool", True, file_count, None)
                expanded_total += 1
            except Exception as e:
                record_expansion(db, arc, target, "patool", False, 0, str(e))
                record_failure(db, file_path=arc, phase="archives",
                               error_class=classify_archive_error(e),
                               error_message=str(e))
    return expanded_total
```

### 5.4 Folder Inference (Phase 5) — Critical Logic

```python
# pipeline/phase5_folder_meta.py
import hashlib, json
from pydantic import BaseModel, Field
from typing import Literal
import ollama
from tenacity import retry, stop_after_attempt, wait_exponential

class FolderInference(BaseModel):
    category: Literal[
        "client-project", "internal-rnd", "vendor-docs",
        "admin-finance", "templates", "archive-historical",
        "personal", "unclear"
    ]
    label: str = Field(max_length=80)
    tags: list[str] = Field(max_length=8)
    summary: str = Field(max_length=400)

PROMPT_TEMPLATE = """You analyze folder structures in a company knowledge base.

Folder path: {rel_path}
Parent labels: {parent_labels}
Number of files: {file_count}
File-type histogram: {format_histogram}
Sample filenames (up to 30):
{filenames}

Respond in JSON matching this schema:
{{"category": one of [client-project, internal-rnd, vendor-docs, admin-finance,
                       templates, archive-historical, personal, unclear],
  "label": "short human-readable label, max 80 chars",
  "tags": ["up to 8 short tags"],
  "summary": "1-3 sentences describing what this folder contains, max 400 chars"}}
"""

def build_prompt(folder_row, db, config) -> tuple[str, str]:
    parents = list(db.execute(
        "WITH RECURSIVE chain(id, parent_id, label) AS ("
        " SELECT id, parent_id, inferred_label FROM folder WHERE id = ?"
        " UNION ALL"
        " SELECT f.id, f.parent_id, f.inferred_label FROM folder f JOIN chain c ON f.id = c.parent_id"
        ") SELECT label FROM chain WHERE label IS NOT NULL", [folder_row["id"]]
    ))
    parent_labels = " > ".join(p["label"] for p in reversed(parents)) or "(root)"
    files = list(db.execute(
        "SELECT name, category FROM file WHERE folder_id = ? AND excluded = 0 LIMIT ?",
        [folder_row["id"], config.folder_meta.max_filenames_in_prompt]
    ))
    histogram = {}
    for f in db.execute(
        "SELECT category, COUNT(*) c FROM file WHERE folder_id=? AND excluded=0 GROUP BY category",
        [folder_row["id"]]
    ):
        histogram[f["category"] or "unknown"] = f["c"]
    prompt = PROMPT_TEMPLATE.format(
        rel_path=folder_row["rel_path"],
        parent_labels=parent_labels,
        file_count=folder_row["file_count"] or 0,
        format_histogram=json.dumps(histogram),
        filenames="\n".join(f"- {f['name']}" for f in files),
    )
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    return prompt, prompt_hash

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def infer_folder(prompt: str, model: str, host: str) -> FolderInference:
    client = ollama.Client(host=host)
    response = client.generate(
        model=model, prompt=prompt, format="json",
        options={"temperature": 0.2, "num_ctx": 4096},
    )
    data = json.loads(response["response"])
    return FolderInference(**data)

def run_phase5(db, config) -> None:
    # bottom-up: order by depth DESC
    folders = list(db.execute(
        "SELECT * FROM folder WHERE excluded=0 ORDER BY depth DESC, id ASC"
    ))
    for folder in folders:
        prompt, prompt_hash = build_prompt(folder, db, config)
        if folder["inference_prompt_hash"] == prompt_hash:
            continue
        try:
            result = infer_folder(prompt, config.ollama.model, config.ollama.host)
            db["folder"].update(folder["id"], {
                "inferred_category": result.category,
                "inferred_label": result.label,
                "inferred_tags_json": json.dumps(result.tags),
                "inferred_summary": result.summary,
                "inference_model": config.ollama.model,
                "inference_prompt_hash": prompt_hash,
                "inferred_at": now_iso(),
            })
        except Exception as e:
            record_failure(db, folder_id=folder["id"], phase="folder_meta",
                           tool="ollama", error_class="parse_error",
                           error_message=str(e))
```

### 5.5 Hashing in Parallel (Phase 1)

```python
# pipeline/phase1_walk.py
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed

def sha256_file(path: str) -> tuple[str, str | None, str | None]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return path, h.hexdigest(), None
    except OSError as e:
        return path, None, str(e)

def hash_pending(db, workers: int) -> None:
    pending = [r["path"] for r in db.execute(
        "SELECT path FROM file WHERE hash_status='pending' AND excluded=0"
    )]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed(ex.submit(sha256_file, p) for p in pending):
            path, digest, err = fut.result()
            if err:
                db.execute("UPDATE file SET hash_status='failed', error_message=? "
                           "WHERE path=?", [err, path])
                record_failure(db, file_path=path, phase="walk",
                               error_class="permission", error_message=err)
            else:
                db.execute("UPDATE file SET sha256=?, hash_status='done' WHERE path=?",
                           [digest, path])
```

### 5.6 PDF Triage

```python
# pipeline/phase3_triage.py
import fitz  # pymupdf

def triage_pdf(path: str, config) -> dict:
    out = {"is_encrypted": 0, "is_corrupt": 0, "page_count": None,
           "has_text_layer": None, "category_override": None,
           "triage_json": None}
    try:
        doc = fitz.open(path)
    except Exception as e:
        out["is_corrupt"] = 1
        out["triage_json"] = json.dumps({"error": str(e)})
        return out
    if doc.is_encrypted and not doc.authenticate(""):
        out["is_encrypted"] = 1
        doc.close()
        return out
    out["page_count"] = doc.page_count
    n = doc.page_count
    sample = sorted({0, n // 2, max(0, n - 1)})
    chars_total = 0
    for i in sample:
        chars_total += len(doc.load_page(i).get_text("text"))
    avg = chars_total / max(1, len(sample))
    threshold = config.triage.pdf_text_threshold_chars_per_page
    out["has_text_layer"] = 1 if avg >= threshold else 0
    out["category_override"] = "pdf-text" if avg >= threshold else "pdf-scanned"
    out["triage_json"] = json.dumps({
        "sampled_pages": sample, "avg_chars_per_sampled_page": avg
    })
    doc.close()
    return out
```

### 5.7 Project Layout

```
rag-preflight/
├── corpus.db
├── config.toml
├── preflight_report.md
├── .rag-cache/
│   ├── extractions/
│   ├── archives/
│   └── logs/
├── seeds/
│   └── format_policy.csv
├── pipeline/
│   ├── __init__.py
│   ├── db.py
│   ├── config.py
│   ├── logging.py
│   ├── phase0_archives.py
│   ├── phase1_walk.py
│   ├── phase2_identify.py
│   ├── phase3_triage.py
│   ├── phase4_dedup.py
│   ├── phase5_folder_meta.py
│   ├── phase6_report.py
│   ├── phase7_exclude.py
│   └── helpers.py
├── tests/
│   ├── fixtures/
│   ├── test_phase0.py
│   ├── test_phase1.py
│   └── ...
├── cli.py
├── pyproject.toml
└── README.md
```

### 5.8 Format Policy Seed (excerpt)

`seeds/format_policy.csv`:

```csv
pronom_id,format_name,category,extract_strategy,notes
fmt/95,PDF/A,pdf-text,docling,Triage may downgrade to pdf-scanned
fmt/18,Acrobat PDF 1.4,pdf-text,docling,
fmt/19,Acrobat PDF 1.5,pdf-text,docling,
fmt/20,Acrobat PDF 1.6,pdf-text,docling,
fmt/276,Acrobat PDF 1.7,pdf-text,docling,
fmt/412,Microsoft Word DOCX,document,docling,
fmt/40,Microsoft Word 97-2003,document,textutil,Use macOS textutil fallback
fmt/214,Microsoft Excel OOXML,spreadsheet,tika,
fmt/215,Microsoft PowerPoint OOXML,presentation,docling,
fmt/101,XML 1.0,data,tika,
fmt/817,JSON,data,tika,
fmt/96,HTML 4.01,document,tika,
fmt/471,Markdown,document,tika,
x-fmt/111,Plain Text,document,tika,
fmt/353,TIFF,image,ocr,Likely scanned
fmt/41,JPEG,image,metadata-only,Skip OCR by default
fmt/12,PNG,image,metadata-only,
fmt/199,MPEG-4 Media,video,whisper,
fmt/131,MP3,audio,whisper,
fmt/141,WAV,audio,whisper,
fmt/189,ZIP,archive,skip,Expanded in Phase 0
fmt/484,RAR,archive,skip,Expanded in Phase 0
fmt/899,STEP CAD,cad,filename-only,Engineering geometry; index filename only
fmt/1147,ROS bag,ros-bag,filename-only,
```

The agent must extend this seed file with additional PRONOM IDs encountered during the first run; the policy table is a living artifact.

### 5.9 Logging

All phases write structured JSON logs to `<cache_root>/logs/<phase>-<timestamp>.jsonl`. Each log line: `{"ts": iso8601, "phase": str, "level": "info|warn|error", "event": str, "data": {...}}`. The CLI also displays human-readable progress via `rich.progress`.

### 5.10 Testing Requirements

The agent must produce unit tests for each phase using `pytest`. Test fixtures live under `tests/fixtures/` and include: a small synthetic corpus with one of each major file type; a nested archive containing a nested archive; a PDF with a text layer; a scanned PDF (image-only); a duplicate file pair; an encrypted PDF; an empty folder. Each phase must achieve at least 80% line coverage. End-to-end test: `init` → `archives` → `walk` → `identify` → `triage` → `dedup` → `folder-meta` (mocked Ollama) → `report` against the fixture corpus must complete with exit code 0 and produce a non-empty report.

---

## 6. Dependencies and References

### 6.1 Python Package Dependencies

Specified in `pyproject.toml`:

```toml
[project]
name = "rag-preflight"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "sqlite-utils>=3.36",
    "click>=8.1",
    "pydantic>=2.5",
    "ollama>=0.3",
    "PyMuPDF>=1.24",
    "patool>=2.3",
    "python-magic>=0.4.27",
    "fasttext>=0.9.2",
    "rich>=13.7",
    "tenacity>=8.2",
    "tomli>=2.0; python_version < '3.11'",
    "msoffcrypto-tool>=5.4",
]

[project.optional-dependencies]
dev = ["pytest>=7.4", "pytest-cov>=4.1", "datasette>=0.64", "ruff>=0.4"]

[project.scripts]
rag-preflight = "cli:main"
```

### 6.2 External Binary Dependencies

| Tool | Minimum Version | Install Command | Purpose |
|---|---|---|---|
| Siegfried | 1.11 | `brew install siegfried` | PRONOM-based format identification |
| FFmpeg (ffprobe) | 6.0 | `brew install ffmpeg` | Media duration |
| unar | 1.10 | `brew install unar` | RAR and obscure archive extraction |
| Ollama | 0.3 | `brew install ollama` | Local LLM server |

### 6.3 Models

The Ollama model specified in config must be pulled before running Phase 5: `ollama pull qwen2.5:7b`. The fasttext language identification model must be downloaded once: `lid.176.bin` from `https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin`, placed at `<cache_root>/models/lid.176.bin`. In air-gapped deployments these artifacts must be transferred manually.

### 6.4 Reference Implementations

The agent must consult the following open-source repositories for reference, but must not vendor their code:

`https://github.com/tw4l/brunnhilde` — reference for the Siegfried-to-SQLite pattern. Read `brunnhilde.py` for SQL query idioms used in reports.

`https://github.com/richardlehane/siegfried` — reference for Siegfried CLI output structure and JSON schema.

`https://github.com/simonw/sqlite-utils` — primary database library; consult documentation for `Database.upsert_all`, `Table.create_index`, and JSON column patterns.

`https://github.com/simonw/datasette` — exploration tool; no integration required beyond shelling out in the `serve` command.

`https://github.com/pymupdf/PyMuPDF` — consult `fitz.open` and `Page.get_text` documentation for Phase 3.

`https://github.com/ollama/ollama-python` — consult `Client.generate` with `format="json"` for structured output in Phase 5.

`https://github.com/wummel/patool` — consult supported formats list and `extract_archive` API for Phase 0.

### 6.5 Standards References

PRONOM file format registry: `https://www.nationalarchives.gov.uk/PRONOM/`. The `pronom_id` values used in `format_policy` must conform to this registry's identifier scheme (`fmt/N` and `x-fmt/N`).

### 6.6 Compliance Constraints

The system must not make outbound HTTP requests during any phase except for explicit one-time model downloads. The `ollama` client must be configured to point at `localhost:11434` only. Any network egress detected during runtime is a defect.

The system must not modify, move, rename, or delete any file under `corpus_root` except for the creation of `*.extracted` sibling folders during Phase 0. Any other write to `corpus_root` is a defect.

The `failure` table is the single source of truth for processing exceptions. Silent failure (catching an exception without recording it) is a defect.

---

**End of Technical Specification Document v1.0.**