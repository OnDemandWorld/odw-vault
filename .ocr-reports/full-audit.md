# ODW.ai Vault — Full-Repository Code Audit

- **Date:** 2026-11-24
- **Branch / commit:** `dev` @ `e1846a0` (up to date with `origin/dev`; `main` @ `09d9d4d`)
- **Mode:** Open Code Review — delegation mode, host-driven analysis (no external OCR CLI, no separate API config)
- **Scope:** entire current source tree, not a diff — all 124 tracked Python files (61 production + 63 test) plus 8 infra/prompt/ignore files
- **Companion data:** [`full-audit.json`](./full-audit.json) — finding IDs `#1`–`#99` below map 1:1 to the `id` field there
- **Source modifications:** none (read-only pass; only `.ocr-reports/` was created)

## Summary

| Severity | Count | IDs |
|---|---|---|
| **High** | 9 | #1–#9 |
| **Medium** | 58 | #10–#67 |
| **Low** | 32 | #68–#99 |
| **Total** | **99** | |

**Category breakdown:** bug 54 · security 10 · error-handling 11 · concurrency 6 · performance 6 · test 11 · maintainability 1

### Coverage

| Area | Files | Reviewed | Skipped |
|---|---|---|---|
| `api/` | 6 | 6 | 0 |
| `rag/` (incl. `extractors/`) | 24 | 24 | 0 |
| `pipeline/` | 15 | 15 | 0 |
| `cli.py` | 1 | 1 | 0 |
| `ui/` | 13 | 13 | 0 |
| `eval/` | 2 | 2 | 0 |
| `tests/` | 63 | 63 | 0 |
| Infra (`Dockerfile`, `pyproject.toml`, `ci.yml`, `config.example.toml`) | 4 | 4 | 0 |
| Prompts + ignore files (`prompts/*.txt`, `rag/prompts/*.txt`, `.gitignore`, `.dockerignore`) | 4 | 4 | 0 |
| **Total** | **132** | **132** | **0** — coverage 100% |

Excluded from scope per guardrails: `.git/`, `.venv/`, `__pycache__/`, build outputs, lockfiles, the `sf` binary, `seeds/*.csv`, `data/`, `resource_img/` media, and documentation (`*.md`).

### Method & verification

Six parallel review batches (API, RAG core, RAG phases/extractors, pipeline/CLI/infra, UI/eval, test quality) read every assigned file in full. Library-API claims were checked against the **installed** versions in `.venv` (pywhispercpp 1.5.0, ocrmypdf 17.7.0, chromadb 1.5.9, gradio 6.19.0, tika 3.1.0, pydantic 2.13, fastapi 0.138). The host then personally re-verified **all 9 High findings and 18 Medium/Low findings** against the source (`"verification": "host"` in the JSON); remaining findings carry agent verification with cited evidence. Likely false positives were discarded silently (notable discards: RRF math is correct; FTS5 MATCH injection is prevented by token sanitization; Chroma per-call `with PersistentClient` is safe in chromadb 1.5.9; zip-slip via patool backends assessed and ruled out; upload path traversal is blocked by O_EXCL + resolve-containment checks).

**Editorial note:** pure test-quality findings (tautologies, false confidence) are capped at Medium because they do not break production at runtime — except **#9**, which has real side effects on the developer's/production database and stays High.

### Cross-cutting themes

1. **Exclusion is not enforced end-to-end** (#1, #2, #14, #15): `excluded` flags are written but never respected by retrieval, and Chroma vectors outlive their SQLite eligibility. This is the single most important privacy fix.
2. **SQLite/Chroma dual-store divergence** (#3, #4, #14, #15, #16, #19, #24): the two stores drift on delete/rechunk/re-embed/crash paths; several paths destroy data without rollback.
3. **Dead config knobs** (#21, #25, #28, #47, #58, #77): `require_citations`, `context_prompt_hash`, `prompt_version`, `--reinfer`, `skip_patterns`, `pdf_sample_pages`, `min_files_to_infer`, `max_tokens` are accepted but never honored.
4. **Event-loop blocking in the async API** (#32, #33, #34): synchronous SQLite commits and OTLP HTTP on the loop freeze all concurrent SSE streams.
5. **UI trust boundary** (#5, #6, #39, #40, #88): unescaped interpolation into `innerHTML`/`href` plus unauthenticated settings endpoints.
6. **Container is non-functional as shipped** (#8, #56, #57).

---

## Round 1 Remediation Progress (2026-11-24)

**Branch:** `fix/ocr-full-audit-round-1` (off `dev` @ `e1846a0`) — for Jack's review; nothing merged to `main`.

| Disposition | Count | IDs |
|---|---|---|
| **Fixed** | 66 | all 9 High (#1–#9); 56 Medium; 1 Low (#79, coupled with #36) |
| **Skipped (sign-off needed)** | 2 | #38 (CORS default vs documented integration contract), #43 (feedback protocol redesign) |
| **Open (Low, not in this round's scope)** | 31 | #68–#99 except #79 |

### Baseline (before any edits)
- `python -m pytest tests/ -q --no-cov` → **737 passed, 3 skipped, 0 failed**
- `python -m ruff check tests/test_api_auth.py` (CI strict gate) → **pass**
- `python -m ruff check pipeline/ rag/ api/ eval/ ui/ cli.py tests/` → **194 pre-existing errors** (informational per CI)
- No type-checker configured; no Makefile.
- Side effect observed: the baseline run created a repo-root `corpus.db` (live confirmation of finding #9).

### Verification protocol used per pass
1. Targeted suites: `python -m pytest tests/test_ocr_round1_fixes.py <affected module suites> -q --no-cov`
2. Lint on changed files: `python -m ruff check <changed files>` (no new errors vs baseline)
3. Full suite: `python -m pytest tests/ -q --no-cov -p no:cacheprovider`
4. Final state: **838 passed, 0 failed** (737 baseline + 98 new/rewritten regression tests + 3 formerly-skipped watcher tests activated by installing watchdog).

New regression coverage lives in `tests/test_ocr_round1_fixes.py` (node-executed XSS payloads, real-endpoint API tests, FK/rollback data-loss tests, dedup/exclusion/GC invariants) plus rewrites of the false-confidence suites (#59–#67).

### Skipped items — decisions needed from Jack
- **#38** — CORS `*` default is documented as intentional (integration contract / Loop canvas). Changing the default origin allowlist needs product sign-off. Mitigation today: `VAULT_CORS_ORIGINS` + `VAULT_API_KEY`.
- **#43** — correct fix requires returning query_log ids to the client and keying feedback by (conversation, message), replacing process-global mappings in the legacy Gradio path. Coordinate with the #44 ordinal fix already landed.
- **#56 (partial)** — unar/p7zip installed in-image; Siegfried binary stays user-provided (mount at `/app/sf`); bundling a pinned release needs a URL/version decision. Docker build not executed (no daemon in review environment).

### Remaining open items
The 31 Low findings (#68–#99 except #79) were out of this round's requested scope (High → Medium). They remain documented in `full-audit.json` with root causes and suggested fixes for a follow-up round.

---

## HIGH — fix first (9)

- [x] **#1 — Excluded files still retrieved and cited** · `rag/retrieval.py:153-199` · bug
  The only query-time exclusion is `preflight_report.md`; `file.excluded=1` is ignored by BM25, dense, path-match, and summary paths. Confidential documents marked excluded remain answerable. **Fix:** enforce `excluded=0` in every retrieval query + filter/delete dense vectors. *(host-verified; pair with #2, #14)*
- [x] **#2 — Folder exclusion never cascades or binds** · `pipeline/phase7_exclude.py:25-37` · security
  `vault exclude --target folder` sets one row; descendants and their files stay `excluded=0` and are extracted, embedded, and served. **Fix:** recursive cascade to descendant folders/files (or folder-exclusion joins downstream). *(host-verified)*
- [x] **#3 — sync_all destroys index of transiently unreadable files** · `rag/indexer.py:393-410` · bug (data loss)
  Hash/stat/resolve OSError → path missing from `disk_files` → classified deleted → `remove_file()` destroys chunks + vectors. **Fix:** track scan errors; confirm absence with `exists()` before removal. *(host-verified)*
- [x] **#4 — DELETE /files/{id}: disk unlink before FK-failing DB delete, no rollback** · `api/main.py:1742-1838` · bug (data loss)
  Original unlinked + vectors deleted, then `DELETE FROM file` hits non-cascade FKs (`archive_expansion`, `folder.source_archive_file_id`, `extracted_from_archive_id`) → 500 with a poisoned open transaction later committed by an unrelated request. **Fix:** transaction + FK cleanup first, disk unlink last. *(host-verified)*
- [x] **#5 — XSS in markdown link renderer** · `ui/gradio_app.py:2407-2408` · security
  `_md()` escapes only `& < >`; raw URL goes into `href="$2"` — quote breakout and `javascript:` schemes accepted. Model output influenced by corpus prompt-injection executes JS in every viewer's browser. **Fix:** escape quotes, allowlist schemes. *(host-verified)*
- [x] **#6 — Stored XSS via conversation id** · `ui/gradio_app.py:2524-2526` · security
  Client-supplied `conv_id` persisted verbatim and interpolated unescaped into `data-conv-id` via `innerHTML`; unauthenticated endpoint → payload hits every UI user. **Fix:** escape the id or enforce UUIDs server-side. *(host-verified)*
- [x] **#7 — Transcription uses nonexistent pywhispercpp APIs** · `rag/phase8b_transcribe.py:49-70` · bug
  `fname_audio=`, `print_system_info`, `new_full_params`, `process`, `.segments` do not exist in installed pywhispercpp 1.5.0 → 100% of transcriptions fail. **Fix:** rewrite against `Model(...).transcribe(path)`; pin `models_dir` to local cache (offline guarantee). *(host-verified)*
- [x] **#8 — Docker entry point cannot find config.toml** · `Dockerfile:44-55` (+ `cli.py:18`, `pyproject.toml:58`) · bug
  `py-modules=["cli"]` installs cli.py into site-packages; `PROJECT_ROOT` resolves there, never `/app`. Container exits 2 on boot; seeds also unreachable. **Fix:** env-anchored root (`VAULT_HOME`) or run `python /app/cli.py`. *(host-verified)*
- [x] **#9 — Test leaks unjoined eval daemon thread into the real DB** · `tests/test_review_round6_fixes.py:93-102` · bug
  `_load_config` unmocked, thread not joined; late execution opens/migrates the real repo `corpus.db` and can fire real Ollama/Chroma traffic; `_eval_runs` never reset → flaky 409s. **Fix:** mock config + runner, join thread, autouse reset fixture. *(host-verified; pair with #82)*

## MEDIUM (58)

### Retrieval & generation quality
- [x] **#10** — Reranker embed-scores ignore the query (near-constant ~0.5 ranking) · `rag/reranker.py:115-139`
- [x] **#11** — Path-match hits flood top BM25 RRF ranks with cover/TOC chunks · `rag/retrieval.py:194-199` *(host-verified)*
- [x] **#12** — Workspace filter broadens to whole workspace when companion filter matches nothing · `rag/filters.py:66-67` *(host-verified)*
- [x] **#13** — Context duplicated in prompt for dense hits (stored doc already contains it) · `rag/retrieval.py:840-855` *(host-verified)*
- [x] **#21** — `require_citations` is a dead flag; uncited answers pass as success · `rag/generation.py:121-135` *(host-verified)*

### Index lifecycle / store consistency
- [x] **#14** — Re-embed never deletes vectors of now-ineligible chunks · `rag/phase11_embed.py:200-237`
- [x] **#15** — Rechunk orphans Chroma vectors (cascade deletes refs only) · `rag/phase10_chunk.py:117-136`
- [x] **#16** — Watcher's long-lived Chroma client defeats per-query reload (stale reads) · `rag/retrieval.py:45-63`
- [x] **#17** — Watcher delete lookup fails on relative/symlinked corpus roots · `rag/watcher.py:285-295`
- [x] **#18** — Watcher stop() closes Chroma client while worker still flushing · `rag/watcher.py:142-155`
- [x] **#19** — Indexer SELECT omits `c.start_page`; page metadata never written · `rag/indexer.py:797-810` *(host-verified)*
- [x] **#20** — Uploads/sync always `is_dup_primary=1`; identical content double-indexed · `rag/indexer.py:252-272`
- [x] **#24** — Manual chunk_fts inserts duplicate migration-4 triggers (double postings; crash leaves orphans) · `rag/phase10_chunk.py:205-215` + `pipeline/db.py:505-517`

### Extraction / chunking / context phases
- [x] **#22** — Phase 8 selects `whisper` files but has no extractor → permanent failure rows · `rag/phase8_extract.py:158-163`
- [x] **#23** — Artifact write failure swallowed: `succeeded=1` with NULL text, no failure row · `rag/phase8_extract.py:56-79`
- [x] **#25** — `context_prompt_hash` cache dead: prompt changes never invalidate · `rag/phase10_5_context.py:244-254`
- [x] **#26** — Short-context clear/regenerate loop never converges (wasted LLM calls every run) · `rag/phase10_5_context.py:372-382`
- [x] **#27** — Transcribe not resumable; `--workers` ignored · `rag/phase8b_transcribe.py:141-156`
- [x] **#28** — Summary cache ignores `prompt_version` and re-extraction freshness · `rag/phase9_summarize.py:63-71`
- [x] **#29** — Docling converter (full model load) rebuilt per file · `rag/extractors/docling_extractor.py:12-15`
- [x] **#30** — Chunk phase lacks per-file error isolation; one bad file aborts all, no failure row · `rag/phase10_chunk.py:141-234`

### API layer
- [x] **#31** — 422 handler serializes raw pydantic `ctx` (live exceptions) → 500 instead of 422 · `api/main.py:350-368` *(host-verified)*
- [x] **#32** — Async upload does blocking SQLite/FS work on the event loop · `api/main.py:1384-1529`
- [x] **#33** — Stream refuse-path persists synchronously on the event loop · `api/main.py:842-861`
- [x] **#34** — OTLP export blocks inline in `span.end()` (up to 2s × 3 spans per query) · `api/spans.py:308-323`
- [x] **#35** — GET /folders O(n²) parent lookup · `api/main.py:1027-1042` *(host-verified)*
- [x] **#36** — `extraction_path` field carries full document text (~1MB) · `api/main.py:1056-1088` *(host-verified)*
- [x] **#37** — Audit date-only `end` bound = midnight; final day silently dropped · `api/audit.py:99-104` *(host-verified)*
- [ ] **#38** — Default CORS `*` + default-open auth → any website can read the local corpus via the operator's browser · `api/main.py:149-240`

### UI / eval
- [x] **#39** — Folder-name XSS re-injected via `opt.text` in `_buildFolderTree` · `ui/gradio_app.py:2625-2633`
- [x] **#40** — Unauthenticated provider/settings + conversation endpoints; non-loopback bind enables corpus exfiltration via attacker-controlled generation endpoint · `ui/gradio_app.py:1131-1232`
- [x] **#41** — Citations panel never cleared; stale sources under new answers · `ui/gradio_app.py:2274-2277`
- [x] **#42** — Saved-conversation replay drops assistant turns (context loss; Anthropic 400) · `ui/gradio_app.py:2571-2579`
- [ ] **#43** — Process-global history/log-id mapping corrupts feedback across sessions · `ui/gradio_app.py:532-575`
- [x] **#44** — Message feedback always updates the newest assistant message · `ui/gradio_app.py:804-813`
- [x] **#45** — Mid-stream conversation switch clobbers state (no stream token/abort) · `ui/gradio_app.py:2299-2324`
- [x] **#46** — recall@10 silently capped at top_k_chunks=8 · `eval/runner.py:115-126` *(host-verified)*

### Pre-flight pipeline / CLI / infra
- [x] **#47** — `--reinfer` is a no-op; folder labels never refresh · `pipeline/phase5_folder_meta.py:143-149` *(host-verified)*
- [x] **#48** — `init --force` leaves `-wal`/`-shm` → old DB resurrects into the "new" one · `cli.py:64-65` *(host-verified)*
- [x] **#49** — `init --force` resets whole config; non-force init drops `chroma_root` · `cli.py:69-84` *(host-verified)*
- [x] **#50** — Failed archives retried on every depth pass (5× failure rows); some failures unrecorded · `pipeline/phase0_archives.py:126-134`
- [x] **#51** — `is_corrupt` never persisted for media/images · `pipeline/phase3_triage.py:258-271`
- [x] **#52** — Language detection reads ZIP/binary containers as UTF-8 → garbage labels · `pipeline/phase3_triage.py:169-182`
- [x] **#53** — Batch exclude CSV: unvalidated `target` (capital-F "Folder" excludes a file); malformed rows abort batch · `pipeline/phase7_exclude.py:53-57`
- [x] **#54** — approve/status/exclude create empty corpus.db without init guard; approve fakes success · `cli.py:314-321`
- [x] **#55** — Siegfried rescans entire corpus (incl. multi-GB model cache) every identify run · `pipeline/phase2_identify.py:88-93`
- [x] **#56** — Docker image lacks unar + sf → phases 0/2 unusable in container · `Dockerfile:22-27`
- [x] **#57** — Container runs as root, 0.0.0.0 bind, secrets-bearing config mount · `Dockerfile:14-55`
- [x] **#58** — `walk.skip_patterns`, `pdf_sample_pages`, `min_files_to_infer` dead (Thumbs.db/*.tmp indexed) · `pipeline/phase1_walk.py:27-31`

### Test quality (false confidence)
- [x] **#59** — Conversation-retrieval test is a pure mock tautology · `tests/test_conversation.py:314-353`
- [x] **#60** — Reranker toggle tests re-implement the gate locally (wrong config shape) · `tests/test_reranker.py:183-276`
- [x] **#61** — Query-history suite tests hand-written SQL, never the /queries endpoint · `tests/test_query_history.py:61-228`
- [x] **#62** — Thread-affinity test asserts an always-true disjunction · `tests/test_review_round4_fixes.py:157-189`
- [x] **#63** — Phase-5 success/max_folders tests vacuous (no folder rows) · `tests/test_phase5_folder_meta.py:228-242`
- [x] **#64** — Health-cache test hits real Ollama/Chroma + fixed /tmp path; can't detect removed cache · `tests/test_error_shaping.py:26-39`
- [x] **#65** — `archives --dry-run` CLI assertion can never fail · `tests/test_cli.py:183-196`
- [x] **#66** — Walk idempotency/hash-failure tests assert only `>= 0` · `tests/test_phase1_walk.py:71-115`
- [x] **#67** — CLI reranker-default test greps source text instead of invoking the command · `tests/test_review_round4_fixes.py:364-367`

## LOW (32)

### RAG core & phases
- [ ] **#68** — Query-embed failure swallowed as empty result (outage looks like content refusal) · `rag/retrieval.py:132-136` *(host-verified)*
- [ ] **#69** — LIKE wildcards unescaped in `_find_files_by_path` (`%` token matches whole corpus) · `rag/retrieval.py:483-496` *(host-verified)*
- [ ] **#70** — Untrusted chunk text injected into prompt without spotlighting; fake `[N]` markers bind to real chunks · `rag/generation.py:143-149`
- [ ] **#71** — Indexer page mapping: final chunk `end_page = page_count + 1` · `rag/indexer.py:749-750`
- [ ] **#72** — `sync_file` returns success despite failed extraction/embedding stages · `rag/indexer.py:277-295`
- [ ] **#73** — `on_moved` bypasses `_should_skip` (indexes into .rag-cache/hidden dirs) · `rag/watcher.py:60-64`
- [ ] **#74** — Paragraph strategy emits unbounded chunks; embedding truncates tail (dense recall loss) · `rag/chunk_strategies.py:329-357`
- [ ] **#75** — Phase-10 page mapping: same `end_page` off-by-one as #71 · `rag/phase10_chunk.py:179-184`
- [ ] **#76** — `--folder` interpolated into SQL (apostrophe crashes; injection surface) · `rag/phase8b_transcribe.py:147-148`
- [ ] **#77** — `max_tokens` ignored (`num_predict` never set); claimed length validation absent · `rag/phase9_summarize.py:43-56`
- [ ] **#78** — `force_perform` is not an ocrmypdf kwarg (real flag: `force_ocr`) → PriorOcrFoundError failures · `rag/extractors/ocr_extractor.py:32-39`

### API
- [x] **#79** — /files/{id}/text serves oldest extraction row (no ORDER BY) · `api/main.py:1100-1108`
- [ ] **#80** — Refusal `done` event: `total_ms` may be null, `conversation_id` missing · `api/main.py:851-860`
- [ ] **#81** — /query/stream skips the generation-endpoint precheck /query performs · `api/main.py:665-688`
- [ ] **#82** — `_eval_runs` unbounded; wedged thread blocks all future eval runs (409 forever) · `api/main.py:1160-1205` *(host-verified)*
- [ ] **#83** — Eval status endpoint returns raw traceback fragments unauthenticated · `api/main.py:1181-1205`
- [ ] **#84** — Unparseable audit date bounds → silent empty 200 result · `api/audit.py:99-104` *(host-verified)*

### UI / benchmarks
- [ ] **#85** — `query_log.answer_model` records config default, not active provider · `ui/gradio_app.py:553-556`
- [ ] **#86** — Provider registry: write failures reported as success; corrupt file reseeds away keys; no locking · `ui/gradio_app.py:173-215`
- [ ] **#87** — Stop button aborts client only; backend generates and persists full answer · `ui/gradio_app.py:2188-2204`
- [ ] **#88** — Legacy citations HTML unescaped (snippet + rel_path XSS) · `ui/gradio_app_legacy.py:164-183`
- [ ] **#89** — Legacy feedback UPDATE…ORDER BY…LIMIT invalid in bundled SQLite; feedback never recorded · `ui/gradio_app_legacy.py:291-300`
- [ ] **#90** — Legacy generation ignores configured generation endpoint (cloud models fail) · `ui/gradio_app_legacy.py:134-135`
- [ ] **#91** — benchmark.py posts 3 args to 4-input /chat (Gradio 6 rejects); length counts parts · `ui/benchmark.py:220-224`
- [ ] **#92** — benchmark_gpt_oss.py unpacks 4-tuples into 3 vars → crash after all queries run · `ui/benchmark_gpt_oss.py:124` *(host-verified)*

### Pipeline / CLI
- [ ] **#93** — run_phase3 db_path fallback points at nonexistent cache/corpus.db · `pipeline/phase3_triage.py:210-211`
- [ ] **#94** — Walk "succeeds" with 0 files on missing root; relative config paths CWD-dependent · `pipeline/phase1_walk.py:98-108`
- [ ] **#95** — CLI phase commands never apply pending migrations · `cli.py:116-123`
- [ ] **#96** — Hash failures all classified 'permission'; oversized/symlink failures unlinked, stale rows stay valid · `pipeline/phase1_walk.py:224-233`
- [ ] **#97** — Identify coverage guard only logs; files stamped unknown/done, files_failed=0 · `pipeline/phase2_identify.py:126-132`

### Tests
- [ ] **#98** — e2e walk-idempotent never compares the two runs · `tests/test_e2e_pipeline.py:55-70`
- [ ] **#99** — ffprobe/lingua-gated assertions pass silently when deps absent; phase8b test hard-fails if whisper installed · `tests/test_phase3_triage.py:82-132`

---

## Suggested remediation order

1. **Privacy/exclusion cluster:** #1 + #2 + #14 + #15 (one coordinated change: enforce exclusion at query time, cascade folder exclusion, GC ineligible vectors).
2. **Data-loss paths:** #3, #4, #48 (confirm-before-delete; transactional deletes; WAL cleanup).
3. **UI security:** #5, #6, #39, #40 (+ #88 if legacy UI is kept).
4. **Broken features:** #7 (transcription), #8 + #56 + #57 (container), #22 (whisper dispatch), #47 (`--reinfer`).
5. **Retrieval quality:** #10, #11, #12, #13, #21, #46.
6. **API responsiveness:** #32, #33, #34, #31, #35.
7. **Test confidence:** #9 first (real side effects), then #59–#67, #98, #99.
8. Remaining Mediums and Lows by area.

**Verification protocol for future sessions:** each fix should cite its finding ID, add/repair a regression test that fails before the fix (especially where #59–#67 show existing tests give false confidence), and re-run `pytest tests/` — noting that #9 must be fixed before trusting suite runs on a machine with a live `corpus.db`.
