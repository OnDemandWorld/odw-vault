# Pre-Flight Plan: Local RAG for Project Knowledge Base

## Purpose

Produce a structured, queryable inventory of the source folder before any extraction, embedding, or RAG work begins. The pre-flight is complete when you can answer, with data: *what is in this corpus, what can be extracted, what it will cost to extract, what should be excluded, and how the corpus is organized semantically.*

The output is a SQLite database (`corpus.db`) plus a human-readable report. Both feed every later phase of the pipeline.

## Guiding principles

Originals are never modified. Everything derived lives under `.rag-cache/` mirroring the source tree. Every phase is idempotent and resumable: re-running on an unchanged file is a no-op. Every long operation has a status column and writes failures to a `failure` table rather than crashing. The SQLite database is the project's source of truth — delete the cache, keep the DB, and you can rebuild; delete the DB, keep the cache, and you can re-index.

## Tooling

Siegfried for format identification (`brew install siegfried`). sqlite-utils as the database library. Datasette for browsing the DB during review. Ollama with a small fast model (`qwen2.5:7b` or `llama3.1:8b`) for folder-level semantic inference. patool plus libarchive plus `unar` for archive expansion. pdfplumber or pymupdf for PDF triage. ffprobe (from ffmpeg) for media duration. fasttext `lid.176` or lingua-py for language detection.

No extraction yet. No embedding yet. No vector store yet. Pre-flight is purely about understanding the corpus.

## Phases

### Phase 0 — Archive expansion

Recursively walk the source tree. For every archive (`.zip`, `.rar`, `.7z`, `.tar`, `.tar.gz`, `.tar.bz2`, `.tgz`), extract it in place into a sibling folder named `<archive>.extracted/`. Record the relationship in the `archive_expansion` table: which archive produced which folder, what tool was used, success/failure, file count. Re-run until the walk finds no new archives (archives can contain archives).

Deliberately do *not* delete the archive itself. The archive is the primary record; the extracted folder is derived. If an archive is encrypted, log it to `failure` with `error_class='encrypted'` and move on.

Edge cases worth handling: macOS `.pages`, `.numbers`, `.key` files are technically zip archives but should be treated as documents, not expanded — keep an exclusion list of archive-like document formats. Files appearing inside `__MACOSX/` or named `.DS_Store` are skipped at walk time.

### Phase 1 — Walk and hash

Walk the (now-expanded) tree. For every folder, insert a row into `folder` with path, parent, depth. For every file, insert a row into `file` with path, size, mtime. Compute SHA-256 incrementally and store it. Hashing is the slowest part of phase 1 — parallelize it across cores. On a typical project corpus, expect roughly 200–500 MB/sec on Apple Silicon SSDs, so even a few hundred GB completes in well under an hour.

Skip rules: hidden files, system files, anything inside `.rag-cache/`, files larger than a configurable threshold (default 5 GB — flag for manual review rather than auto-process).

### Phase 2 — Format identification

Run Siegfried over the entire tree once: `sf -json -multi 32 <root> > sf_results.json`. Parse the output and update each `file` row with `pronom_id`, `mime_type`, `format_name`, `format_version`, `siegfried_json`, `id_warning`. This is faster than calling Siegfried per-file.

Then assign `category` and `extract_strategy` for each file by looking up `pronom_id` in the `format_policy` table. If the PRONOM ID is unknown to the policy table, set `category='unknown'`, `extract_strategy='manual'` and log it. The first time you run pre-flight on a new domain, expect to add 20–50 rows to `format_policy` for formats specific to that corpus. For your robotics work this will likely include CAD formats, ROS-specific files, and various proprietary vendor formats.

### Phase 3 — Triage

Format-specific quick checks that don't actually extract content but tell you *what extraction will cost*. Run only on files where `extract_strategy` warrants it.

For PDFs: open with pymupdf, sample first/middle/last page, count extracted characters. If under a threshold (say 50 chars/page average), classify as `pdf-scanned`; otherwise `pdf-text`. Record `page_count`. Flag encrypted PDFs.

For Office files: detect encryption, count pages/slides where cheap.

For audio/video: `ffprobe` for duration, store `duration_seconds`. This drives Whisper transcription cost estimation.

For images: store dimensions. Don't try to detect "contains text" yet — too expensive at pre-flight; defer to extraction phase if you decide to OCR images.

For CAD, ROS bags, code, executables: just record that they exist and their size; no triage beyond that.

### Phase 4 — Exact deduplication

Group files by SHA-256. For each group with more than one member, assign a `dup_group_id` and mark all but one (the canonical copy — pick the one with the shortest path or oldest mtime) with `is_dup_primary=0`. Downstream phases query with `WHERE is_dup_primary=1`.

This is free because you already have hashes from phase 1. Expect 10–40% duplication in real project corpora — clients re-send the same datasheet, engineers copy reference docs across projects, archives contain copies of files that also exist outside the archive.

Near-duplicate detection (same document, different versions) is deferred to a post-extraction phase since it requires text.

### Phase 5 — Folder semantic inference

For each folder containing documents (skip leaf folders that are pure archive expansions or system noise), build a small prompt: folder path, parent folder labels already inferred, sample of up to 30 child filenames, file-type histogram. Send to Ollama with a structured-output prompt asking for `{category, label, tags[], summary}`. Cache by hash of the prompt — don't re-run unless inputs change.

Suggested categories tuned for your context: `client-project`, `internal-rnd`, `vendor-docs`, `admin-finance`, `templates`, `archive-historical`, `personal`, `unclear`. The model should also extract probable client name, robot platform mentions, and date range when visible from filenames.

Run this bottom-up so children inform parents. The root folder gets a final pass that summarizes the whole corpus structure.

This is the only LLM use in pre-flight, and it's bounded: number of folders, not number of files. Even on a corpus with 100,000 files there will be at most a few thousand folders, which `qwen2.5:7b` on Apple Silicon handles in under an hour.

### Phase 6 — Aggregate and report

Compute folder-level aggregates: file count, total bytes, document count, dominant format. Update `folder` rows.

Generate a markdown report with: total files and bytes, format histogram (top 30 plus "other"), category breakdown, extraction-strategy distribution, OCR workload (count and total pages of `pdf-scanned`), transcription workload (hours of audio/video), duplicate ratio, language distribution sample, list of unknown formats needing policy entries, list of files flagged as encrypted/corrupt/oversized, top 20 largest folders, and the inferred folder taxonomy.

Open the same database in Datasette for interactive exploration: `datasette serve corpus.db`.

This is the **decision point**. Review the report before committing to extraction. Frequently you'll find something that changes strategy: 80% of PDFs are scanned and OCR will take three days; this whole subtree is duplicates; this folder is personal files and shouldn't be indexed; this client's data should be excluded for NDA reasons.

### Phase 7 — Mark exclusions and approve

Manual step. Walk through the report with stakeholders. Mark folders or files as `excluded=1` where appropriate (NDA, personal, irrelevant, too-noisy). The schema supports this; downstream phases respect it.

When the report is signed off, pre-flight is done. Tag the database (`config` table: `preflight_completed_at`, `preflight_approved_by`) and proceed to extraction.

## Expected outputs at the end of pre-flight

A populated `corpus.db` with every file inventoried, identified, triaged, and deduplicated, and every folder semantically labeled. A markdown report (`preflight_report.md`) suitable for sharing. A list of formats added to `format_policy.csv` during this run. A list of files/folders excluded with reasons. Concrete cost estimates for the next phase: how many GB to OCR, how many hours to transcribe, how many documents to embed, total expected text volume.

## Time estimate for a real corpus

For a corpus of around 500 GB and 100,000 files on a modern Mac: archive expansion 30–90 min depending on archive count, walk and hash 30–60 min, Siegfried identification 10–20 min, triage 30–60 min (PDF triage dominates), dedup near-instant, folder inference 30–90 min depending on folder count and model size, report generation seconds. Realistic wall-clock: half a working day, mostly unattended. The human review afterward is the bottleneck.

---

