# Developer Guide — RAG Pre-Flight Pipeline

This guide is for developers and AI agents extending this codebase. It covers architecture decisions, extension patterns, common pitfalls, and step-by-step instructions for adding new phases, categories, formats, and tests.

---

## 1. System Architecture

### 1.1 Design Philosophy

The pipeline follows a **sequential phase model** where each phase reads from and writes to a single SQLite database (`corpus.db`). Phases are:

- **Idempotent**: Running the same phase twice produces the same result. No phase should duplicate data or corrupt state on re-run.
- **Resumable**: If a phase fails partway through, re-running it should pick up where it left off (using `*_status` columns and cache hashes).
- **Self-contained**: Each phase has its own module (`phaseN_*.py`) with a single entry function `run_phaseN(db, config, plog, ...)`.
- **Failure-transparent**: Every error is recorded in the `failure` table. Silent failures are defects.

### 1.2 Data Flow

```
corpus_root/                    ┌────────────────────────────────────┐
├── file1.txt ─────────────────►│ phase1: walk + hash                │
├── file2.pdf ─────────────────►│   → file table (path, sha256)      │
├── archive.zip ──► extracted/ ─┤                                    │
└── subdir/file3.docx ─────────►│ phase2: identify (siegfried)       │
                                │   → pronom_id, category, strategy  │
                                │                                    │
                                │ phase3: triage (pdf/media/image)   │
                                │   → page_count, duration, language │
                                │                                    │
                                │ phase4: dedup (pure SQL)           │
                                │   → dup_group_id, is_dup_primary   │
                                │                                    │
                                │ phase5: folder-meta (ollama)       │
                                │   → inferred_category, label       │
                                │                                    │
                                │ phase6: report (aggregates)        │
                                │   → preflight_report.md            │
                                │                                    │
                                │ phase7: exclude + approve          │
                                │   → excluded=1, sign-off           │
                                └────────────────────────────────────┘
                                              │
                                              ▼
                                        corpus.db
                                        (15 tables, 7 views)
```

### 1.3 Key Architectural Decisions

| Decision | Rationale |
|----------|-----------|
| **SQLite, not PostgreSQL** | Single-file, no server, portable, sufficient for corpus-scale data |
| **sqlite-utils, not raw sqlite3** | Dict-returning queries, convenient insert/update/lookup APIs, built-in migration support |
| **Sequential phases, not DAG** | The pipeline is linear by design — each phase depends on the previous phase's output |
| **External binaries, not Python libs** | Siegfried is a compiled C binary (faster, better signatures). ffprobe handles media. patool wraps multiple extractors. |
| **Ollama local LLM, not cloud API** | Air-gap requirement. No outbound HTTP during runtime. |
| **Config as TOML + Pydantic** | Type-safe, validated at load time. Default values for all settings. |
| **Phase-specific status columns** | `hash_status`, `identify_status`, `triage_status` allow independent re-running of phases |

---

## 2. Database Deep Dive

### 2.1 Schema Overview

The schema is defined in `pipeline/db.py` as a single migration (version 1). All tables use `IF NOT EXISTS` so the migration is idempotent.

**Core entity tables:**

| Table | Primary Key | Key Columns | Purpose |
|-------|-------------|-------------|---------|
| `folder` | `id` (auto) | `path` (unique), `parent_id` (self-ref), `depth`, `inferred_category` | Directory tree with semantic metadata |
| `file` | `id` (auto) | `path` (unique), `folder_id` (FK), `sha256`, `category`, `extract_strategy` | File inventory with format/triage/dedup data |

**Provenance tables:**

| Table | Purpose |
|-------|---------|
| `archive_expansion` | Records which archive was extracted, where, by which tool, and how many files resulted |
| `pipeline_run` | Per-phase execution history (start time, status, files processed/failed) |
| `failure` | Error log with classification (phase, tool, error_class, message) |

**Configuration tables:**

| Table | Purpose |
|-------|---------|
| `config` | Key-value store for runtime settings (corpus_root, cache_root, pipeline_version, preflight_approved_by) |
| `format_policy` | PRONOM ID → category + extract_strategy mapping. Seeded from `seeds/format_policy.csv`. |

**Forward-compatible tables (empty during pre-flight):**

| Table | Purpose |
|-------|---------|
| `extraction` | Stores extracted text from Phase 8+ |
| `summary` | Stores document summaries from Phase 9+ |
| `chunk` | Stores document chunks from Phase 10+ |
| `embedding_ref` | Stores embedding vectors from Phase 11+ |
| `chunk_fts` | FTS5 virtual table for text search on chunks |

### 2.2 Important Relationships

```
folder (parent_id) ──► folder (id)          [self-referencing, ON DELETE CASCADE]
file (folder_id) ────► folder (id)          [ON DELETE CASCADE]
file (extracted_from_archive_id) ──► file (id)  [self-referencing, ON DELETE CASCADE]
archive_expansion (archive_file_id) ──► file (id)
failure (file_id) ──► file (id)             [ON DELETE CASCADE]
extraction (file_id) ──► file (id)          [ON DELETE CASCADE]
summary (file_id) ──► file (id)             [ON DELETE CASCADE]
chunk (file_id) ──► file (id)               [ON DELETE CASCADE]
embedding_ref (chunk_id) ──► chunk (id)     [ON DELETE CASCADE]
```

**Critical warning**: `INSERT OR REPLACE` on `file` or `folder` with self-referencing FK constraints will cascade-delete child rows. Always use:
- Check if row exists → `UPDATE` if yes, `INSERT` if no.
- Or use `insert(..., ignore=True)` to skip existing rows.

### 2.3 Views

Views are computed at query time and always reflect current data:

| View | WHERE clause | Use case |
|------|-------------|----------|
| `v_format_histogram` | `is_dup_primary=1 AND excluded=0` | Top formats by count |
| `v_category_summary` | `is_dup_primary=1 AND excluded=0` | Files per category |
| `v_ocr_workload` | `category='pdf-scanned' AND is_dup_primary=1 AND excluded=0` | OCR page count estimate |
| `v_transcription_workload` | `category IN ('audio','video') AND is_dup_primary=1 AND excluded=0` | Audio/video duration |
| `v_duplicate_summary` | `dup_group_id IS NOT NULL HAVING copies>1` | Duplicate group analysis |
| `v_problem_files` | `error_message IS NOT NULL OR is_corrupt=1 OR is_encrypted=1 OR id_warning IS NOT NULL OR category='unknown'` | Problem file identification |
| `v_unknown_formats` | `pronom_id NOT IN (SELECT pronom_id FROM format_policy)` | Formats needing policy entries |

### 2.4 Adding a New Table (Phase 8+)

To add a new table for a future phase:

1. Add a new migration tuple to `MIGRATIONS` in `db.py`:
```python
MIGRATIONS = [
    (1, "initial schema", "..."),
    (2, "extraction results table", """
        CREATE TABLE IF NOT EXISTS extraction_v2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER NOT NULL REFERENCES file(id) ON DELETE CASCADE,
            tool_version TEXT,
            ...
        );
    """),
]
```

2. The `migrate()` function automatically applies new migrations based on `schema_version`.

3. Update the `Config` model if the phase needs new settings.

---

## 3. Extending the Pipeline

### 3.1 Adding a New Phase

To add Phase 8 (text extraction) as an example:

**Step 1: Create the module** — `pipeline/phase8_extract.py`

```python
"""Phase 8: Text extraction.

Extracts text from files based on their extract_strategy.
Writes results to the extraction table.
"""

from __future__ import annotations

from pipeline.config import Config
from pipeline.logging import PhaseLogger


def run_phase8(
    db,
    config: Config,
    plog: PhaseLogger,
    categories: list[str] | None = None,
) -> dict:
    """Run text extraction on eligible files."""
    # 1. Find files that need extraction
    where = "extract_strategy IS NOT NULL AND extract_strategy NOT IN ('skip', 'unsupported')"
    if categories:
        cat_list = ",".join(f"'{c}'" for c in categories)
        where += f" AND category IN ({cat_list})"

    files = list(db["file"].rows_where(where))

    if not files:
        plog.info("No files need extraction.")
        return {"files_processed": 0, "files_failed": 0}

    # 2. Process each file
    processed = 0
    failed = 0

    for file_row in files:
        try:
            # Extract text based on strategy
            strategy = file_row["extract_strategy"]
            text = _extract_by_strategy(file_row["path"], strategy)

            # Write to extraction table
            db["extraction"].insert({
                "file_id": file_row["id"],
                "tool": strategy,
                "text_extracted": text,
                "char_count": len(text),
                "succeeded": 1,
            })
            processed += 1
        except Exception as e:
            from pipeline.helpers import record_failure
            record_failure(db, file_id=file_row["id"], phase="extract",
                          tool=strategy, error_class="extraction_error",
                          error_message=str(e))
            failed += 1

    plog.info(f"Extraction complete: {processed} processed, {failed} failed")
    return {"files_processed": processed, "files_failed": failed}


def _extract_by_strategy(filepath: str, strategy: str) -> str:
    """Route to the appropriate extractor."""
    if strategy == "docling":
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(filepath)
        return result.document.export_to_markdown()
    elif strategy == "tika":
        from tika import parser
        return parser.from_file(filepath)["content"]
    elif strategy == "filename-only":
        from pathlib import Path
        return Path(filepath).name
    elif strategy == "metadata-only":
        # Extract file metadata only
        import os
        stat = os.stat(filepath)
        return f"Size: {stat.st_size} bytes, Modified: {stat.st_mtime}"
    else:
        raise ValueError(f"Unknown extract strategy: {strategy}")
```

**Step 2: Add the CLI command** — in `cli.py`

```python
@main.command()
@click.option("--categories", default=None, help="Comma-separated categories to extract")
def extract(categories: str | None) -> None:
    """Phase 8: Text extraction."""
    from pipeline.phase8_extract import run_phase8

    cat_list = [c.strip() for c in categories.split(",")] if categories else None

    def wrapper(db, config, plog, **kw):
        return run_phase8(db, config, plog, categories=cat_list)

    _run_phase(click.get_current_context(), "extract", wrapper)
```

**Step 3: Update `run-all`** — add `"extract"` to the loop in the `run_all` command.

**Step 4: Add tests** — `tests/test_phase8_extract.py`

**Step 5: Update `pipeline_run.phase` CHECK constraint** — if the new phase name isn't in the existing constraint, add a migration or modify the schema.

### 3.2 Adding a New Format Policy Entry

When Siegfried identifies a new format that isn't in the policy:

1. **Identify the format** — check `v_unknown_formats` view for unhandled PRONOM IDs.

2. **Add to `seeds/format_policy.csv`**:
```csv
pronom_id,format_name,category,extract_strategy,notes
fmt/999,New Format Type,document,tika,Added 2026-05-02
```

3. **Add extension fallback** (if Siegfried can't identify it by content):
```python
# pipeline/phase2_identify.py
EXT_FALLBACK = {
    ...
    ".newext": ("UNKNOWN-newext", "data", "filename-only"),
}
```

4. **Re-seed the database**:
```python
# In cli.py init or manually:
db["format_policy"].insert_all(reader, replace=True)
```

5. **Re-run identify** with `--reidentify` flag.

### 3.3 Adding a New Category

Categories are free-form text in the `file.category` column. To add a new category:

1. **Define the category** — decide what format(s) map to it.
2. **Update `format_policy.csv`** with the new category name.
3. **Update any category-aware logic** — e.g., `v_ocr_workload` filters on `category='pdf-scanned'`.
4. **Update triage logic** if the new category needs special handling in `phase3_triage.py`.

### 3.4 Adding a New Extract Strategy

Strategies are free-form text in `file.extract_strategy`. To add one:

1. **Define the strategy** — decide which formats use it.
2. **Update `format_policy.csv`**.
3. **Implement the extractor** in the Phase 8+ extraction module.
4. **Handle it in `_extract_by_strategy()`** (or equivalent routing function).

---

## 4. sqlite-utils API Reference

This section covers the most commonly used patterns and pitfalls.

### 4.1 Reading Data

```python
# Returns list of dicts — preferred for SELECT
rows = list(db.query("SELECT * FROM file WHERE category = ?", ["pdf-text"]))

# Single row by primary key
row = db["file"].get(42)  # Raises NotFoundError if not found!

# Single row by condition
row = next(db["file"].rows_where("path = ?", ["/abs/path"]), None)  # Returns None if not found

# Iterate with conditions
for row in db["file"].rows_where("excluded = 0 AND hash_status = 'done'"):
    process(row)
```

### 4.2 Writing Data

```python
# Insert new row
db["file"].insert({
    "folder_id": 1,
    "path": "/abs/path/file.txt",
    "name": "file.txt",
    "size_bytes": 100,
    "mtime": "2026-01-01T00:00:00Z",
})

# Insert, skip if PK exists
db["file"].insert({...}, ignore=True)

# Update existing row
db["file"].update(42, {"category": "document", "pronom_id": "x-fmt/111"})

# Raw SQL with explicit commit
db.execute("UPDATE file SET category = ? WHERE id = ?", ["document", 42])
db.conn.commit()  # REQUIRED — sqlite-utils does NOT auto-commit for execute()

# Transaction (auto-commits on success, rolls back on error)
with db.conn:
    db.execute("UPDATE ...", [...])
    db.execute("INSERT ...", [...])
```

### 4.3 Thread-Safe DB Access

```python
# WRONG — shared connection across threads
def worker(file_id, db):
    db.execute("UPDATE file SET ...", [...])  # May cause "database is locked"

# CORRECT — each thread opens its own connection
def worker(file_id, db_path):
    tdb = sqlite_utils.Database(str(db_path))
    tdb.conn.execute("PRAGMA foreign_keys = ON")
    try:
        tdb.execute("UPDATE file SET ...", [...])
        tdb.conn.commit()  # REQUIRED
    finally:
        tdb.conn.close()
```

### 4.4 Common Pitfalls

| Pattern | Problem | Solution |
|---------|---------|----------|
| `db["table"].get(id)` returning None | Raises `NotFoundError` instead | Wrap in `try/except NotFoundError` |
| `db.execute("SELECT ...")` returning dicts | Returns tuples, not dicts | Use `db.query("SELECT ...")` |
| `db.execute("UPDATE ...")` persisting | No auto-commit | Call `db.conn.commit()` or use `with db.conn:` |
| `insert(..., replace=True)` on FK tables | Cascade-deletes child rows | Use `UPDATE` for existing, `INSERT` for new |
| `db["table"].lookup("col", val)` | API requires dict: `lookup({"col": val})` | Use `next(db["table"].rows_where(...), None)` |

---

## 5. Testing Guide

### 5.1 Test Structure

```
tests/
├── conftest.py              # Shared fixtures (auto-discovered by pytest)
├── test_helpers.py          # Unit tests for helper functions
├── test_config.py           # Unit tests for Pydantic models
├── test_db.py               # Unit tests for schema/migrations
├── test_phase0_archives.py  # Phase-specific tests
├── test_phase1_walk.py
├── test_phase2_identify.py
├── test_phase3_triage.py
├── test_phase4_dedup.py
├── test_phase5_folder_meta.py
├── test_phase6_report.py
├── test_phase7_exclude.py
├── test_cli.py              # Click command tests
└── test_e2e_pipeline.py     # Integration tests
```

### 5.2 Shared Fixtures

All fixtures are defined in `conftest.py` and auto-injected by name:

```python
def test_something(test_db, test_corpus, mock_plog):
    # test_db — fresh SQLite DB with full schema + seeded format_policy
    # test_corpus — (root_path, {name: Path}) dict of synthetic test files
    # mock_plog — no-op PhaseLogger (MagicMock)
```

### 5.3 Creating New Tests

**For a phase module** — create `tests/test_phaseN_new.py`:

```python
"""Tests for pipeline/phaseN_new.py."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from pipeline.config import Config, PathsConfig
from pipeline.phaseN_new import run_phaseN


class TestPhaseN:
    def _make_config(self, root):
        return Config(paths=PathsConfig(
            corpus_root=str(root), cache_root=str(root / ".rag-cache")))

    def test_basic(self, test_db, test_corpus, mock_plog):
        root, paths = test_corpus
        cfg = self._make_config(root)
        result = run_phaseN(test_db, cfg, mock_plog)
        assert result["files_processed"] >= 0
```

**Mocking an external service**:

```python
# Mock Ollama
from pipeline.phase5_folder_meta import FolderInference

inference = FolderInference(
    category="client-project",
    label="Test",
    tags=["test"],
    summary="A test folder.",
)
with patch("pipeline.phase5_folder_meta._call_ollama", return_value=inference):
    result = run_phase5(test_db, cfg, mock_plog)

# Mock Siegfried
sf_response = json.dumps({
    "siegfried": "v1.11.4",
    "files": [{"filename": "/path/file.txt", "filesize": 100,
               "modified": "2026-01-01", "matches": [...]}]
})
with patch("subprocess.run") as mock_run:
    mock_run.return_value = MagicMock(returncode=0, stdout=sf_response, stderr="")
    result = run_phase2(test_db, cfg, mock_plog)
```

### 5.4 Running Tests

```bash
# All tests
PYTHONPATH=. pytest tests/ -v

# Coverage report
PYTHONPATH=. pytest tests/ --cov=pipeline --cov=cli --cov-report=term-missing -v

# Specific test file
PYTHONPATH=. pytest tests/test_phase3_triage.py -v

# Specific test
PYTHONPATH=. pytest tests/test_phase3_triage.py::TestTriagePdf::test_text_pdf -v

# With coverage for a specific module
PYTHONPATH=. pytest tests/ --cov=pipeline/phase3_triage --cov-report=term-missing
```

### 5.5 Coverage Expectations

| Module | Expected Coverage | Notes |
|--------|-------------------|-------|
| helpers.py | 100% | Pure functions, easy to test |
| config.py | 100% | Pydantic models, validation |
| db.py | 90%+ | Schema creation, migration logic |
| phase0-7 | 80%+ | Each phase's main paths + error paths |
| cli.py | 60-70% | CLI dispatch is harder to isolate |
| logging.py | 80%+ | Structured logging, console output |

---

## 6. Configuration System

### 6.1 How Config Loading Works

1. `cli.py init` writes `config.toml` to the project root.
2. Each CLI command calls `get_config(ctx)` which loads `config.toml` via `load_config()`.
3. `load_config()` parses TOML → dict → validates through Pydantic `Config` model.
4. Pydantic fills in defaults for any missing sections/values.
5. The resulting `Config` object is passed to every phase function.

### 6.2 Adding a New Config Setting

**Step 1: Define the model** — in `pipeline/config.py`:

```python
class ExtractConfig(BaseModel):
    workers: int = 4
    timeout_seconds: int = 300
    max_chars_per_file: int = 1_000_000
```

**Step 2: Add to root Config**:

```python
class Config(BaseModel):
    paths: PathsConfig
    walk: WalkConfig = Field(default_factory=WalkConfig)
    # ... existing ...
    extract: ExtractConfig = Field(default_factory=ExtractConfig)
```

**Step 3: Update DEFAULT_CONFIG_TOML**:

```python
DEFAULT_CONFIG_TOML = """\
...
[extract]
workers = 4
timeout_seconds = 300
max_chars_per_file = 1000000
"""
```

**Step 4: Access in the phase**:

```python
def run_phase8(db, config: Config, plog: PhaseLogger):
    workers = config.extract.workers
    timeout = config.extract.timeout_seconds
```

### 6.3 Config Preservation

The `init` command preserves existing `config.toml` when re-run without `--force`:

```python
if config_path.exists() and not force:
    with open(config_path, "rb") as f:
        existing = tomllib.load(f)
    existing["paths"] = {"corpus_root": str(root), "cache_root": str(cache)}
    config_path.write_text(tomli_w.dumps(existing), encoding="utf-8")
```

This means custom settings (e.g., `siegfried_workers = 64`) survive `init --root /new/corpus`.

---

## 7. Troubleshooting

### 7.1 Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `database is locked` | Multiple threads sharing a DB connection | Each thread needs its own `sqlite_utils.Database()` instance |
| `NotFoundError` from `db[table].get(id)` | Row doesn't exist | Use `next(db[table].rows_where(...), None)` or catch the exception |
| `FK constraint failed` on INSERT | `replace=True` deletes parent row, cascading to children | Use `UPDATE` for existing rows |
| `Siegfried not found` | `sf` binary not on PATH and not in project root | Set `siegfried_path` in config.toml to full path |
| `Phase X processed 0 files` | No rows match the phase's WHERE clause | Run previous phases first, or check status columns |
| `triage_json` is NULL after triage | Per-thread connection didn't commit | Add `tdb.conn.commit()` after UPDATE in worker |
| `pickle.PicklingError` in ProcessPoolExecutor | Mocked object passed to subprocess | Use real file operations or mock at the function level, not with MagicMock across process boundaries |

### 7.2 Debugging Tips

```bash
# Check phase status
PYTHONPATH=. python cli.py status

# Query the DB directly
sqlite3 corpus.db "SELECT phase, status, files_processed FROM pipeline_run ORDER BY started_at DESC LIMIT 10;"

# Check for failures
sqlite3 corpus.db "SELECT phase, error_class, error_message FROM failure ORDER BY occurred_at DESC LIMIT 20;"

# Check unknown formats
sqlite3 corpus.db "SELECT * FROM v_unknown_formats;"

# Check problem files
sqlite3 corpus.db "SELECT path, category, error_message FROM v_problem_files LIMIT 20;"

# View the full schema
sqlite3 corpus.db ".schema"

# Check WAL file size (should be small after commits)
ls -la corpus.db*
```

### 7.3 Resetting and Re-running

```bash
# Re-run a single phase (idempotent)
PYTHONPATH=. python cli.py identify --reidentify

# Force re-hash all files
PYTHONPATH=. python cli.py walk --rehash

# Completely reset (destructive!)
rm corpus.db
rm config.toml
rm -rf .rag-cache/
PYTHONPATH=. python cli.py init --root /path/to/corpus
PYTHONPATH=. python cli.py run-all
```

---

## 8. Future Phase Design Checklist

When designing Phase 8+, use this checklist:

- [ ] **Module file**: `pipeline/phaseN_name.py` with `run_phaseN(db, config, plog, ...) -> dict`
- [ ] **Return value**: `{"files_processed": int, "files_failed": int}` (standard across all phases)
- [ ] **Idempotency**: Re-running doesn't duplicate data. Use status columns or cache hashes.
- [ ] **Failure recording**: All errors go to `record_failure(db, file_id=..., phase=..., ...)`.
- [ ] **Logging**: Use `plog.info()`, `plog.warning()`, `plog.debug()` for progress.
- [ ] **Progress bar**: Use `rich.progress.Progress` for long-running operations.
- [ ] **CLI command**: Add to `cli.py` using the `_run_phase()` wrapper pattern.
- [ ] **Tests**: Create `tests/test_phaseN_name.py` with 8+ tests covering happy path + error paths.
- [ ] **Config**: Add settings to `pipeline/config.py` if the phase is configurable.
- [ ] **Schema**: Add migration to `db.py` if new tables/columns are needed.
- [ ] **run-all**: Add command name to the loop in `cli.py:run_all()`.
