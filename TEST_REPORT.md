# Test Suite Report — RAG Pre-Flight Pipeline

## Overview

The test suite covers all 7 pre-flight phases plus CLI commands and end-to-end integration, meeting the TSD §5.10 requirement of 80%+ line coverage.

- **167 tests**, all passing
- **86% overall coverage** (target: 80%)
- **15 test files** across `tests/`
- **pytest + pytest-cov** framework
- No static binary fixtures — all test files generated programmatically at runtime

## Test Files

| File | Tests | Purpose |
|------|-------|---------|
| `conftest.py` | fixtures | Shared fixtures: corpus factory, DB, config, mocks |
| `test_helpers.py` | 20 | `sha256_file`, `is_hidden_or_system`, `is_archive_extension`, `record_failure`, etc. |
| `test_config.py` | 15 | Pydantic model validation, `load_config()`, `DEFAULT_CONFIG_TOML` |
| `test_db.py` | 12 | `open_db`, `migrate`, schema (tables/views/indexes), idempotency |
| `test_phase0_archives.py` | 10 | Archive discovery, patool expansion, dry-run, idempotency |
| `test_phase1_walk.py` | 11 | Folder/file row creation, SHA-256 hashing, hidden/cache skipping |
| `test_phase2_identify.py` | 13 | Siegfried format ID, extension fallback, error handling |
| `test_phase3_triage.py` | 16 | PDF text/scanned/encrypted triage, image dimensions, language detection |
| `test_phase4_dedup.py` | 8 | SQL deduplication, dup groups, primary assignment, idempotency |
| `test_phase5_folder_meta.py` | 15 | FolderInference model, prompt building, Ollama mocking, bottom-up traversal |
| `test_phase6_report.py` | 10 | Markdown report generation, format tables, folder taxonomy |
| `test_phase7_exclude.py` | 8 | Single/batch exclusion, nonexistent target handling |
| `test_cli.py` | 14 | Click commands: init, status, exclude, approve, serve, run-all |
| `test_e2e_pipeline.py` | 6 | Walk+identify integration, report generation, idempotency |

## Architecture

### Fixture Strategy

**Programmatic corpus generation** — no binary files checked into the repo. A `create_test_corpus(tmp_path)` factory creates:
- Text files via `Path.write_text()`
- PDFs via PyMuPDF: text-bearing, scanned (image-only), encrypted, minimal blank
- PNG via raw zlib-constructed bytes (1×1 pixel)
- ZIP archives via `zipfile.ZipFile`, including nested archives
- Hidden files (`.DS_Store`), cache dirs (`.rag-cache/`)

### Shared Fixtures

- **`test_corpus`** — function-scoped `(corpus_root, path_dict)` tuple
- **`test_db`** — fresh SQLite DB with full schema + seeded `format_policy` from `seeds/format_policy.csv`
- **`test_config`** — `Config` pointing to test corpus
- **`mock_plog`** — no-op `PhaseLogger` (MagicMock)
- **`mock_ollama`** — patches `_call_ollama` returning valid `FolderInference`
- **`mock_siegfried`** — patches `subprocess.run` with Siegfried JSON

### Mocking External Dependencies

| Dependency | Mock Target | Strategy |
|------------|------------|----------|
| **Ollama** | `pipeline.phase5_folder_meta._call_ollama` | Return `FolderInference(category=..., label=..., tags=..., summary=...)` |
| **Siegfried** | `subprocess.run` | Return pre-built JSON matching `_mock_sf()` factory |
| **patool** | `patoolib.extract_archive` | Side effect that creates dummy files in `outdir` |
| **PhaseLogger** | `MagicMock` | No-op info/warning/error/debug |
| **lingua** | `HAS_LINGUA` flag | `@pytest.mark.skipif` or `patch("pipeline.phase3_triage.HAS_LINGUA", False)` |

## Coverage Summary

| Module | Coverage | Notes |
|--------|----------|-------|
| `pipeline/config.py` | 100% | All Pydantic models + `load_config()` |
| `pipeline/helpers.py` | 100% | SHA-256, hidden/system detection, archive classification |
| `pipeline/phase4_dedup.py` | 100% | Pure SQL, all paths exercised |
| `pipeline/phase7_exclude.py` | 100% | Single + batch exclusion |
| `pipeline/phase5_folder_meta.py` | 96% | Happy path + model validation |
| `pipeline/phase0_archives.py` | 98% | Archive expansion, provenance |
| `pipeline/phase1_walk.py` | 96% | Walk, hash, skip logic |
| `pipeline/db.py` | 93% | Schema migration, views |
| `pipeline/logging.py` | 87% | Structured JSON logging |
| `pipeline/phase6_report.py` | 84% | Report sections, table formatting |
| `pipeline/phase2_identify.py` | 89% | Siegfried parsing, extension fallback |
| `pipeline/phase3_triage.py` | 80% | PDF triage, language detection |
| `cli.py` | 67% | CLI command dispatch |
| **TOTAL** | **86%** | Exceeds 80% TSD target |

## Known Coverage Gaps

### cli.py (67%)
- **`run_all`** — sequential phase execution via `ctx.forward()` not exercised end-to-end (relies on real DB + config.toml setup)
- **`serve`** — Datasette subprocess launch; hard to test without installing Datasette in the test venv
- **`init` with existing config** — TOML update path uses `tomllib`/`tomli_w`, tested only for happy path
- **Error branches** — malformed config.toml, corrupted DB paths

### pipeline/phase3_triage.py (80%)
- **Lingua language detection** — depends on `lingua` package availability; Chinese (Traditional/Simplified) detection not covered
- **`_triage_media` (ffprobe)** — tested only for corrupt media and missing ffprobe; no valid duration test because no real video/audio fixtures
- **Thread pool paths** — `_triage_worker` runs in threads; race conditions and per-thread DB connection behavior not tested under load

### pipeline/phase6_report.py (84%)
- **`_build_language_summary`** — no test with actual `detected_language` in triage_json (requires lingua)
- **`_build_folder_tree`** — no test with deeply nested folders + inferred labels
- **Optional sections** — OCR workload, transcription workload, problem files, unknown formats views not exercised

### pipeline/phase2_identify.py (89%)
- **Siegfried per-file error handling** — when sf returns partial JSON with some files failing, the error-isolation path is not fully covered
- **`reidentify=True`** — force re-identification of already-identified files

## Bugs Found During Test Development

These were real defects or API mismatches discovered while writing tests:

1. **PyMuPDF zero-page save error** — `_make_empty_pdf` called `doc.save()` on a zero-page document. Newer PyMuPDF raises `ValueError: cannot save with zero pages`. Fixed by creating a 1-page blank PDF instead.

2. **`sqlite_utils.Database.lookup` API mismatch** — several tests used `db["table"].lookup("column", value)` but `lookup()` requires a dict (`lookup({"column": value})`). Fixed by using `next(db["table"].rows_where("column = ?", [value]))` instead.

3. **`sqlite_utils.Database.get` raises `NotFoundError`** — `phase7_exclude.py` used `db[table].get(id)` expecting a `None` return for missing rows. `sqlite_utils` raises `NotFoundError` instead. Fixed by catching the exception in the pipeline code.

4. **`PhaseLogger` type mismatch** — `PhaseLogger.__init__` expects `Path`, not `str`. E2E tests passed `str(root / ".rag-cache")` and got `TypeError`.

5. **Phase2 requires pre-existing DB rows** — `run_phase2` only processes files already in the `file` table with `hash_status='done'`. Tests that only created files on disk (without DB rows) silently processed 0 files.

6. **Phase5 requires folder rows** — `run_phase5` queries the `folder` table. Empty DB = 0 folders = 0 inferences. Tests without inserted folder rows were no-ops.

7. **Phase3 DB locking** — e2e tests opened separate `sqlite_utils.Database` connections and got `database is locked`. Solution: reuse a single connection or ensure WAL mode is active.

8. **`test_db` lacked `.path` attribute** — `sqlite_utils.Database` does not expose its path directly. Tests that needed the path (for spawning subprocess workers) failed with `AttributeError`. Fixed by attaching `db.path = db_path` in the fixture.

9. **List comprehension syntax error** — `test_phase0_archives.py` had `[a for a in ".rag-cache" in str(a) for a in archives]` (undefined `a` before use). Correct form: `[a for a in archives if ".rag-cache" in str(a)]`.

10. **`cli.approve` parameter mismatch** — `@click.option("--by", "approver", ...)` set the parameter to `approver` but the function signature used `by`. Click raised `TypeError: unexpected keyword argument 'approver'`.

## Future Improvements

### 1. Property-Based Testing
Use `hypothesis` for generating arbitrary inputs to helpers like `sha256_file`, `_apply_extension_fallback`, and `_fmt_table`. This would catch edge cases with unusual file paths, Unicode filenames, and malformed data.

### 2. Integration Tests with Real External Services
Currently all external dependencies are mocked. Consider adding a `@pytest.mark.slow` or `@pytest.mark.integration` test class that runs with:
- **Real Siegfried** (`sf` binary in PATH)
- **Real Ollama** (`gemma4:latest` running locally)
- **Real patool** (with actual archives to extract)
- **Real ffprobe** (with actual media files)

These would validate that the mock contracts match real behavior.

### 3. Concurrency/Thread Safety Tests
Phase 3 triage uses a thread pool with per-thread DB connections. Tests should:
- Run triage with 50+ files and `workers=8`
- Verify no `database is locked` errors
- Confirm all `triage_status` transitions to `done` or `failed`

### 4. Snapshot/Regression Tests
Generate a baseline `corpus.db` from the real `SourceData/` corpus and use it as a reference. Future test runs could compare key metrics (file counts, category distributions, dedup groups) against the snapshot to detect regressions.

### 5. CLI End-to-End Tests
The `cli.py` module is at 67% coverage. A proper e2e CLI test would:
- Use `CliRunner` with a real `init --root`
- Execute `run-all` with all mocks applied
- Verify the generated `preflight_report.md` content
- Test `status` command JSON output structure

### 6. Database Migration Tests
Currently `migrate()` is only tested for idempotency on a fresh DB. If a future schema migration is added (e.g., `schema_version = 2`), tests should verify:
- Migration from v1 → v2 preserves existing data
- Forward-only migration (no downgrade)
- Partial migration recovery

### 7. Test Matrix for Python Versions
Currently tested only on Python 3.11 on ARM macOS. A CI matrix should cover:
- Python 3.11, 3.12
- ARM64 and x86_64 macOS
- Linux (Ubuntu) — important for production deployment

### 8. Mutation Testing
Use `mutmut` or `cosmic-ray` to verify that tests actually catch bugs. Mutate source code (change `>` to `>=`, flip `True`/`False`) and verify tests fail. This measures test *quality*, not just coverage.

### 9. Performance Benchmarks
Add timing assertions for known slow operations:
- `run_phase1` with 2000+ files should complete in < 60s
- `run_phase2` with mocked sf should complete in < 10s
- `run_phase6` report generation should complete in < 5s

### 10. Fixture Factory Pattern
Consider replacing the single `create_test_corpus` factory with a more composable builder:

```python
builder = CorpusBuilder(tmp_path)
builder.text_file("doc.txt", content="Hello" * 100)
builder.pdf("scan.pdf", type="scanned")
builder.duplicate("doc.txt", "doc_copy.txt")
root = builder.build()
```

This makes individual test setup more readable and avoids creating unnecessary files (encrypted PDFs, nested ZIPs) in tests that don't need them.
