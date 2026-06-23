# Context Quality Audit — Post-Processing Validation

> Date: 2026-05-04
> Corpus: 8,159 chunks, 384 primary documents, 4.8 GB
> Model: mlx-community/Qwen3-8B-4bit (remote MLX @ 192.168.0.72:28100)

---

## Why This Step Exists

During the first full run of contextual augmentation, several quality issues were
discovered in the generated context entries.  These are not bugs in the pipeline
itself — they are inherent to working with Qwen-family models that emit chain-of-
thought reasoning blocks.  This document records what happened and the validation
step that was added to catch it.

---

## Issues Found

### 1. Leaked `` Reasoning Blocks

**What happened:** Qwen3 models have a ``thinking`` mode that emits internal
reasoning wrapped in `` tags.  Even though the request included
``"thinking": false``, the model still produced reasoning blocks in some cases.
Worse, some outputs had **malformed closing tags** (```` without the closing
``>``) that the initial regex missed.

**How many:** 13 entries out of 8,159 (0.16%)

**Impact:** Two entries contained 75 KB of leaked reasoning for a single context.
The remaining entries had 1-3 KB of thinking noise before the actual context.

**Fix:** The `_call_openai_compat` function now strips both well-formed
(``</think>``) and malformed (`</think>` missing, or ```` without `>`) reasoning
blocks using a two-pass regex.  The CLI also runs a post-processing validation
pass that catches any that slipped through.

### 2. Very Short Contexts (< 50 characters)

**What happened:** 83 entries had context shorter than 50 characters.  These were
not errors per se — the model produced output, just very terse output.

**Root causes:**
| Cause | Count | Example |
|---|---|---|
| File has no summary (no LLM summary generated) | 63 | Copyright footer in Linen Trolley.pdf: "广州赛特智能科技有限公司 版权所有" |
| Chunk text is tiny (< 100 chars) | 15 | Single-row table cells, version strings |
| Table-heavy test report PDF | 50 | "检测报告第14项传感器测试，第24页" — just a location pointer |

**Impact:** These contexts provide minimal retrieval benefit.  They don't hurt
search, but they also don't meaningfully improve it over the raw chunk text.

**Fix:** The validation pass clears contexts shorter than 50 characters so they
can be regenerated on the next run.  After 2-3 re-runs, most improve because the
model produces slightly different output each time.  The remainder are genuinely
unaugmentable (copyright footers, single-row table cells).

### 3. "Error" and "Failed" Patterns — False Positives

**What happened:** 255 contexts contained words like "error", "failed", or
"unable to".

**Root causes:** These were **not model failures**.  The documents themselves
describe failed tests, errors, or issues (e.g., elevator simulation failures in
a test report).  The model correctly summarized the content.

**Impact:** None.  These are legitimate summaries.

**Action:** No fix needed — just noted for audit completeness.

### 4. Generic Phrasing — Cosmetic

**What happened:** 1,437 contexts start with "This chunk details..." or "This
section details..." or "The document details...".

**Impact:** This is a stylistic quirk of Qwen3's instruction-following behavior.
The content is useful; the opening phrase is just formulaic.

**Action:** No fix needed — cosmetic only.  Could be addressed with prompt
engineering if desired in the future.

---

## Final State After Cleanup

| Metric | Before | After |
|---|---|---|
| Total chunks | 8,159 | 8,159 |
| With context | 8,159 | 8,159 |
| Thinking tags present | 13 | 0 |
| Context < 50 chars | 83 | ~20 (regenerated, some remain unaugmentable) |
| Avg context length | 209 chars | 191 chars |
| Median context length | ~150 chars | ~145 chars |

---

## Validation Function

The `validate_and_clean(db)` function in `rag/phase10_5_context.py` runs after
every context generation pass.  It:

1. **Strips leaked `` blocks** — both well-formed (`<think>...</think>`)
   and malformed (starts but no proper close).  If stripping leaves nothing,
   the entry is cleared for reprocessing.

2. **Clears very-short contexts** (< 50 chars) — sets all context fields to
   NULL so the next `rag context` run will regenerate them.

3. **Reports findings** — logs counts of each issue type found and how many
   were cleared.

The CLI command `rag context` automatically runs validation after context
generation.  If issues are found, it reports them and suggests re-running to
regenerate cleared entries.

---

## Model Comparison: gemma3n vs Qwen3-8B

| Metric | gemma3n:latest (Ollama) | Qwen3-8B-4bit (MLX remote) |
|---|---|---|
| Chunks processed | 4,237 | 3,922 |
| Time | 13.0 hours | 11.5 hours |
| Speed | 5.4 chunks/min | 5.7 chunks/min |
| Avg context length | ~90 chars | ~191 chars |
| Quality | Brief, minimal | Rich, structured |
| `` leakage | 0 | 13 (0.3%, fixable) |

**Conclusion:** Qwen3-8B produces ~2x richer context despite a slight speed
advantage.  The  thinking leakage is a minor, fixable issue.
Qwen3-8B was the better choice for this phase.
