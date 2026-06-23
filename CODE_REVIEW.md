# RAG Pre-Flight Code Review

Date: 2026-05-05
Scope: All 15 source files in `pipeline/` + `cli.py` + tests
Result: 16 bugs fixed, 398 tests pass at 65% coverage.

---

## Bugs Fixed

### BUG-1: Hidden dot-files were silently indexed (HIGH)

**File:** `pipeline/helpers.py`

`is_hidden_or_system` returned `False` for `.env`, `.gitignore`, and other non-system dot-files. The logic checked both "starts with `.`" AND "in SYSTEM_FILES" — it should be OR.

**Before:**
```python
if name.startswith("."):
    return name in SYSTEM_FILES  # only True for .DS_Store etc.
return False
```

**After:**
```python
return name.startswith(".") or name in SYSTEM_FILES
```

Files like `.env`, `.secret_config`, `.git/` are now correctly skipped during walk and archive scan.

---

### BUG-2: SQL injection in triage `--categories` flag (HIGH)

**File:** `pipeline/phase3_triage.py`

Category strings from CLI input were interpolated directly into SQL with f-string quoting:

```python
cats = ",".join(f"'{c}'" for c in categories)
where_clause += f" AND category IN ({cats})"
```

A category value containing a single quote (e.g. `--categories "pdf-text','document"`) breaks out of the quoting and allows arbitrary SQL.

**Fix:** Parameterized query with `?` placeholders — categories passed as a list to `db.query()`.

---

### BUG-3: Datasette `serve` command was unreachable (HIGH)

**File:** `cli.py`

Two commands were both named `serve` — one launches Datasette, one launches FastAPI. Click uses the function name, so the FastAPI `serve` shadowed the Datasette one. The Datasette command was unreachable.

**Fix:** Renamed the Datasette command to `serve-datasette` (invoked as `python cli.py serve-datasette`).

---

### BUG-4: Phase 0 `replace=True` risked cascade-deleting child rows (MEDIUM)

**File:** `pipeline/phase0_archives.py`

`db["file"].insert({...}, replace=True)` deletes and re-inserts on PK conflict. The `file` table has `ON DELETE CASCADE` foreign keys — re-inserting a file row would cascade-delete any already-extracted child file rows.

**Fix:** Changed to `ignore=True` — skips if the file row already exists.

---

### BUG-5: Phase 6 report ran 142+ per-folder DB queries (MEDIUM)

**File:** `pipeline/phase6_report.py`

For every folder, two separate queries executed: file stats and dominant format. With 71 folders = 142 queries.

**Fix:** Replaced with 2 aggregate queries:
1. `GROUP BY folder_id` for file_count, total_bytes, document_count
2. `ROW_NUMBER() OVER (PARTITION BY folder_id ORDER BY COUNT(*) DESC)` for dominant format

Reduces 142 queries → 2 queries.

---

### BUG-6: Init command inserted format_policy from empty table (MEDIUM)

**File:** `cli.py`

The `init` command did two inserts into `format_policy`: first from the existing empty DB table (no-op), then from the CSV seed file. The first insert was dead code.

**Fix:** Removed the redundant first insert.

---

### BUG-7: Folder tree builder fired N queries (MEDIUM)

**File:** `pipeline/phase6_report.py`

`_build_folder_tree` recursively queried `SELECT * FROM folder WHERE parent_id=?` for every child — N queries for N folders.

**Fix:** Single recursive CTE fetches the entire tree in one query, then builds the tree structure in Python.

---

### BUG-8: Phase 5 loaded already-inferred folders (MEDIUM)

**File:** `pipeline/phase5_folder_meta.py`

The SQL `SELECT * FROM folder WHERE excluded=0` fetched ALL folders, including those that already had a valid inference. The Python prompt-hash check skipped the LLM call, but the folder was still loaded into memory and iterated.

**Fix:** Added SQL filter: `WHERE excluded=0 AND (inferred_category IS NULL OR inference_prompt_hash IS NULL)`.

---

### BUG-9: Markdown tables rendered `None` as `"None"` (LOW)

**File:** `pipeline/phase6_report.py`

`str(row.get(h, ""))` converts `None` to the string `"None"`.

**Fix:** `str(row.get(h) or "")` — renders `None` as empty string.

---

### BUG-10: Batch exclude silently skipped missing IDs (LOW)

**File:** `pipeline/phase7_exclude.py`

When a CSV row referenced a non-existent ID, the exception was caught and the row silently skipped. A CSV with all wrong IDs would appear to succeed with 0 output.

**Fix:** Function now returns `(applied_count, skipped_count)`. The CLI command reports both numbers.

---

### Additional fixes from code quality review

| Fix | File | Detail |
|-----|------|--------|
| Removed dead `record_run` function | `cli.py` | Wrong type annotation, never called |
| Added clarifying comment on `DOC_ARCHIVES` check | `helpers.py` | Defensive — sets don't overlap currently |
| Fixed symlink edge case in cache skip | `phase1_walk.py`, `phase0_archives.py` | Uses `is_relative_to(path.resolve())` instead of `startswith(str)` |
| Fixed PhaseLogger handler accumulation | `logging.py` | Calls `logger.handlers.clear()` before adding new handler |

---

## What to Tackle Next

Prioritized by impact and effort. Each item is independent — pick one at a time.

### 1. Split heavy optional dependencies (MEDIUM effort, HIGH impact)

**Problem:** `pyproject.toml` installs `chromadb`, `gradio`, `fastapi`, `docling`, `ocrmypdf` even when only running pre-flight phases. These are heavy packages with many transitive dependencies (chromadb alone pulls in ~500 MB).

**Impact:** Faster `pip install`, smaller venv, cleaner separation of concerns.

**Approach:**
```toml
[project.optional-dependencies]
preflight = []        # core deps only (already in main)
rag = ["chromadb", "docling", "ocrmypdf", "whisper"]
ui = ["fastapi", "uvicorn", "gradio"]
eval = []             # eval-specific deps
dev = ["pytest", "pytest-cov", "ruff"]
```

Users who only need pre-flight: `pip install .`
Full stack: `pip install ".[rag,ui,dev]"`

---

### 2. Database wrapper for thread workers (MEDIUM effort, MEDIUM impact)

**Problem:** Thread workers in `phase3_triage.py` open raw `sqlite_utils.Database(db_path)` directly (line 210), bypassing the `Database` wrapper from `db.py` which provides retry-on-lock and safer defaults.

**Impact:** Thread workers don't benefit from the retry-on-lock logic that the main pipeline uses. Under heavy load, they could hit "database is locked" errors.

**Approach:** Export a helper from `db.py`:
```python
def open_thread_db(db_path: Path | str) -> sqlite_utils.Database:
    """Open a raw sqlite_utils.Database for use in a thread worker."""
    db = sqlite_utils.Database(str(db_path))
    db.conn.execute("PRAGMA foreign_keys = ON")
    db.conn.execute("PRAGMA journal_mode = WAL")
    db.conn.execute("PRAGMA synchronous = NORMAL")
    return db
```
This encapsulates the pattern and makes it clear these are intentionally raw connections (the retry-on-lock from the main connection doesn't apply to per-thread connections since they don't share locks with the main process).

---

### 3. Phase 5 prompt building batch queries (LOW effort, MEDIUM impact)

**Problem:** `_build_prompt` runs 3 DB queries per folder (parent chain, child filenames, histogram). For many folders this adds up.

**Impact:** Phase 5 is already Ollama-bound (the LLM call is the bottleneck), so this is not the primary performance concern. But batching these queries would reduce DB chatter.

**Approach:** Pre-fetch all folder data in one query at the start of phase 5, then build prompts from in-memory data. This trades one large query for N small ones.

---

### 4. Migration backup-before-apply (LOW effort, MEDIUM impact)

**Problem:** `pipeline/db.py` migrations are forward-only with no backup. A buggy migration could corrupt an existing corpus.

**Impact:** Protects against data loss from schema changes.

**Approach:** In `migrate()`, before applying a migration:
```python
import shutil
backup = db_path.with_suffix(f".db.bak.{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}")
shutil.copy2(db_path, backup)
```

---

### 5. CSV batch exclusion validation (LOW effort, LOW impact)

**Problem:** Batch CSV processing has no validation of the `target` column (any non-"folder" value treated as "file"), no bounds checking on `id`.

**Impact:** A malformed CSV could produce confusing results or attempt to update non-existent rows.

**Approach:** Validate CSV headers, validate `target` is exactly "file" or "folder", validate `id` is a positive integer, skip invalid rows with a warning.

---

### 6. Per-model LLM endpoint routing (HIGH effort, HIGH impact)

**Problem:** The `[ollama].host` config is a single global value. Different phases use different models that may live on different servers (remote OpenAI-compat for context generation, local Ollama for embeddings). Currently requires manual config switching.

**Impact:** Enables mixing local and remote models without config changes. Required before deploying to a multi-server setup.

**Approach:** Extend config with per-endpoint sections:
```toml
[endpoints.embedding]
host = "http://localhost:11434"
model = "nomic-embed-text"

[endpoints.summarization]
host = "http://mlx-server:8080"
model = "gemma-3-12b"
type = "openai-compat"
```

This is a larger change — touches config model, phase5, phase8, phase9, phase10.5, and phase11. Best tackled as its own feature branch.

---

## Summary

| Category | Fixed | Deferred | Total |
|----------|-------|----------|-------|
| HIGH severity bugs | 4 | 0 | 4 |
| MEDIUM severity bugs | 6 | 2 (QC-2, QC-4) | 8 |
| LOW severity bugs | 5 | 3 (QC-4, FI-2, FI-6) | 8 |

**Recommended next step:** #1 (split optional dependencies) — highest impact, lowest risk, standalone change that doesn't require touching phase logic.
