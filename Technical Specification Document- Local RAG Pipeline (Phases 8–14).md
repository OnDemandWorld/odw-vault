# Technical Specification Document: Local RAG Pipeline (Phases 8–14)

**Document Version:** 2.0
**Status:** Approved for Implementation — Supersedes v1.0
**Predecessor:** Pre-Flight TSD v1.0 (BUILD_STATUS confirms 7/7 phases complete)
**Input Substrate:** `corpus.db` SQLite database with populated pre-flight tables and forward-compatible `extraction`, `summary`, `chunk`, `embedding_ref`, `chunk_fts` tables.
**Scope:** Text extraction (Phase 8), summarization (Phase 9), chunking (Phase 10), contextual retrieval augmentation (Phase 10.5), embedding (Phase 11), retrieval and generation (Phase 12), query API (Phase 13), and user interface (Phase 14).

**Changes from v1.0:** Model identities corrected against current Ollama library (verified 2026-05-02). Gemma 4 family adopted as primary generation stack. Configuration architecture elevated to first-class principle: every model identity is config-driven and swappable without code changes. Contextual Retrieval (Anthropic, September 2024) added as Phase 10.5. `max_context_tokens` raised in line with model capabilities. Reranker explicitly defaulted off pending baseline evaluation.

---

## 1. Project Overview & Goals

### 1.1 System Purpose

The system extends the validated pre-flight pipeline into a complete, fully local Retrieval-Augmented Generation system serving the ODW.ai Vault project knowledge base. It transforms the inventoried corpus into a queryable knowledge layer that supports natural-language questions in English and Traditional Chinese, returns answers grounded in source documents with explicit citations, and enforces folder-scoped retrieval to maintain client-data segregation.

### 1.2 Core Functional Requirements

The system must extract text from documents using format-appropriate extractors selected by the `extract_strategy` column populated in pre-flight. It must transcribe audio and video using local whisper.cpp with word-level timestamps. It must produce document-level summaries using a configurable local LLM. It must chunk extracted text using a sentence-window strategy with persistent character offsets for citation. It must generate per-chunk contextual augmentations (Anthropic-style Contextual Retrieval) that are concatenated to chunks at embedding time but never shown to the answering LLM in place of the original text. It must generate embeddings at three levels (chunks, document summaries, folder summaries) and store them in a vector store namespaced by embedding model, permitting multiple embedding model versions to coexist for A/B comparison. It must perform hybrid retrieval combining dense vector search, BM25 sparse search, and structured metadata filtering, with optional reranking. It must generate citation-strict answers via a configurable local LLM. It must expose the system through a FastAPI HTTP service and a Gradio chat UI.

### 1.3 Non-Functional Requirements

The system must operate entirely offline once dependencies and models are installed. **Every model identity at every phase must be configurable via `config.toml` and overridable per-invocation via CLI flag, with no model name hard-coded in source.** Re-running any phase against unchanged inputs must perform no work. The vector store must be model-namespaced so that swapping embedding models does not invalidate prior work and permits side-by-side evaluation. Every artifact must record which model produced it, when, and against which configuration hash. Failures must be recorded to the existing `failure` table; silent exception suppression is a defect. The reranker is explicitly optional. The system must function correctly with `models.reranker.enabled = false`, falling back to RRF-fused dense+BM25 results without reranking; reranker should be added only after baseline evaluation establishes its value.

### 1.4 Operating Constraints

The system runs on a single Apple Silicon Mac. Ollama serves all LLM and embedding inference on `localhost:11434`. The corpus is small (≈75 working documents from a 2,177-file inventory) so total build time is expected in tens of minutes for embedding and short hours for transcription. Bilingual retrieval (English + Traditional Chinese) must be a first-class requirement.

### 1.5 Configuration-Driven Architecture (Load-Bearing Principle)

This principle is elevated from a convenience to a structural requirement. Three consequences flow from it:

Embedding vectors are not permanent — when the embedding model changes, vectors become invalid (different vector space). Re-embedding must therefore be cheap and obvious, not a frightening rebuild. This is achieved by separating model-independent artifacts (extracted text, chunks, summaries — all in SQLite) from model-dependent artifacts (vectors — in Chroma, namespaced per model).

Multiple model versions must coexist. The system must support running model A in production while embedding the corpus with model B in the background, then atomically switching after evaluation. Chroma collections are therefore namespaced by `{kind}__{embedding.collection_suffix}`.

Every artifact records which model produced it. The `model_run`, `embedding_ref.embedding_model`, `summary.model`, and `extraction.tool`+`tool_version` columns enforce this. The retrieval path verifies the active configuration matches the vector store metadata before serving a query.

### 1.6 Success Criteria

A hand-curated evaluation set of at least 30 questions covering single-document factual lookups, multi-document synthesis, project-scoped queries, bilingual queries, and adversarial negatives ("about a client we have not served") achieves at least 80% retrieval recall (correct source document among the top-5 retrieved chunks) and at least 70% answer correctness on human grading. The folder-scope filter passes adversarial tests with zero leakage. Swapping the configured embedding model and running `rag embed` produces a parallel set of vector collections that can be evaluated side-by-side without affecting the active production set.

### 1.7 Out of Scope

Multi-user authentication and authorization beyond a folder-filter convenience, role-based access control, network-exposed deployment, mobile interfaces, fine-tuning of any model, RAPTOR-style hierarchical clustering of chunks, and ingestion of structured databases are explicitly out of scope for this TSD.

---

## 2. Technical Stack Selection

### 2.1 Primary Language

**Python 3.11+,** consistent with the pre-flight pipeline. No language change permitted.

### 2.2 Frameworks and Libraries

**LlamaIndex (>=0.11)** as the orchestration framework for ingestion, chunking, retrieval, and prompt assembly. Provides `IngestionPipeline` with `DocstoreStrategy.UPSERTS` (content-hash incremental indexing aligned with the SHA-256 hashes already in `file.sha256`), native Ollama integration, native Chroma integration, and `SentenceWindowNodeParser` with character offsets needed for citation. Components are individually replaceable; framework used as a toolkit.

**Chroma (>=0.5)** as the vector store. Persistent single-directory store with zero operational overhead, sufficient for the working-set scale, supports per-collection metadata filtering required for folder-scoped retrieval, supports multiple coexisting collections required for the model-namespacing principle.

**Docling (>=2.0)** as the primary document extractor for PDF, DOCX, PPTX, XLSX, HTML.

**Apache Tika (>=2.9, via `tika-python` client)** as a fallback extractor for legacy `.doc`, `.rtf`, `.eml`, `.msg`, `.epub`, and the 23 triage-failed files identified in BUILD_STATUS. Tika runs as a local server on `localhost:9998`.

**pywhispercpp (>=1.2)** as the binding to whisper.cpp for audio and video transcription.

**FastAPI (>=0.110)** with **Uvicorn (>=0.27)** for the HTTP API.

**Gradio (>=4.40)** for the chat interface.

**rank-bm25 (>=0.2.2)** for BM25 sparse retrieval to complement dense vectors via reciprocal rank fusion.

**`ollama` Python client (>=0.3),** **`pydantic` (>=2.5),** **`tenacity` (>=8.2),** **`rich` (>=13.7),** **`click` (>=8.1),** and **`sqlite-utils` (>=3.36)** are reused from pre-flight without version changes.

External binaries (must be on `PATH` or in project root, consistent with existing pre-flight conventions):

| Tool | Minimum Version | Purpose |
|---|---|---|
| Ollama | 0.3 | LLM and embedding server |
| Apache Tika Server | 2.9 | Fallback document extraction |
| ocrmypdf | 15.4 | OCR pre-processing |
| Tesseract | 5.3 (with `eng`, `chi_tra`, `chi_sim` packs) | OCR engine used by ocrmypdf |
| FFmpeg | 6.0 | Reused from pre-flight; media decoding |

### 2.3 Models — Recommended Configuration (Verified Against Ollama Library 2026-05-02)

All model identities below are **defaults shipped in `config.toml`** and are **fully overridable**. The system must not contain any model name as a literal anywhere except in the seed config file and in documentation.

| Role | Default Model (Ollama tag) | Disk | Context | Rationale |
|---|---|---|---|---|
| Generation (primary) | `gemma4:26b` | ~16 GB | 256K | MoE 25.2B total / 3.8B active. Near-frontier quality (MMLU Pro 82.6, BigBench Extra Hard 64.8, MRCR v2 long-context 44.1). Runs at 3.8B-dense speed on Apple Silicon. Multimodal. |
| Generation (alternate) | `qwen3:30b-a3b-instruct-2507-q4_K_M` | 19 GB | 256K | Strong Chinese, well-tested. For A/B comparison against Gemma 4. |
| Generation (max quality) | `gemma4:31b` | ~20 GB | 256K | Dense flagship. MMLU Pro 85.2, AIME 2026 89.2. Use when RAM permits and quality matters more than speed. |
| Generation (fallback) | `gemma4:e4b` | ~5 GB | 128K | RAM-constrained fallback. 4.5B effective params, multimodal, audio-aware. |
| Summarization | `gemma4:e4b` | ~5 GB | 128K | Bounded task, speed matters. 128K context covers most documents in a single pass. |
| Contextual augmentation | `gemma4:e2b` | ~2 GB | 128K | Bounded augmentation task; smallest fast model is ideal. ~2,000 calls per corpus build. |
| Embedding (primary) | `qwen3-embedding:8b` | 4.7 GB | 40K | MTEB Multilingual #1 at time of writing (70.58). Strong Chinese. Matryoshka representation learning; dimension truncation supported. |
| Embedding (alternate small) | `qwen3-embedding:4b` | 2.5 GB | 40K | Same family at half size. Use under RAM pressure. |
| Embedding (alternate fast) | `nomic-embed-text-v2-moe:latest` | ~475 MB | 8K | MoE multilingual baseline for ablation. |
| Embedding (alternate Google) | `embeddinggemma:latest` | ~600 MB | 2K | Google's multilingual on-device embedder for ablation. |
| Reranker (optional) | `dengcao/Qwen3-Reranker-0.6B` | ~600 MB | — | Community upload; not in official Ollama library. Disabled by default. |
| Transcription | Whisper Large-v3 via whisper.cpp | ~3 GB | — | Apple Silicon Core ML acceleration; word-level timestamps for media chunk citations. |
| Language identification | fasttext `lid.176.bin` | 130 MB | — | Already used in pre-flight. |

Notes on Gemma 4 specifically: the model card specifies recommended sampling parameters of `temperature=1.0, top_p=0.95, top_k=64`, which are unusual for citation-strict RAG. The shipping configuration sets these defaults but raises them as candidates for empirical tuning during evaluation. Gemma 4 also supports configurable thinking mode via the `<|think|>` token in the system prompt; thinking is **off by default** for RAG generation (it slows generation and clutters citation output) but available as a per-request toggle in the API.

### 2.4 Database

**SQLite** (`corpus.db`, the existing pre-flight database, extended via additive migration to schema version 2) plus **Chroma** (`chroma/` directory at project root, multiple collections namespaced per embedding model).

Justification: SQLite remains authoritative for content-versioned artifacts (extracted text references, chunks, summaries, contextual augmentations, query logs, evaluation records) because these are model-independent and benefit from relational queries and FTS5. Chroma holds only embedding-versioned artifacts (vectors), namespaced by model. This separation makes embedding-model swaps a fast, low-risk operation.

Extracted text bodies remain on disk under `.rag-cache/extractions/<sha-prefix>/<sha>.md`. Transcripts under `.rag-cache/transcripts/<sha-prefix>/<sha>.json`. Chunk contextual augmentations cached under `.rag-cache/contexts/<chunk-hash>.txt`.

---

## 3. Data Modeling & Schemas

The pre-flight schema is extended via additive migration `schema_version=2`.

### 3.1 Migration `schema_version=2`

```sql
ALTER TABLE embedding_ref ADD COLUMN config_hash TEXT;
ALTER TABLE embedding_ref ADD COLUMN is_current INTEGER NOT NULL DEFAULT 1
    CHECK (is_current IN (0,1));

CREATE INDEX idx_embedding_ref_current ON embedding_ref(embedding_model, is_current);

ALTER TABLE chunk ADD COLUMN context_text TEXT;
ALTER TABLE chunk ADD COLUMN context_model TEXT;
ALTER TABLE chunk ADD COLUMN context_prompt_hash TEXT;
ALTER TABLE chunk ADD COLUMN context_generated_at TEXT;

-- model_run, query_log, eval_question, eval_run, folder_embedding_ref,
-- summary_embedding_ref, and views as defined below
```

### 3.2 Table: `model_run` (new)

A permanent ledger of every model invocation by phase.

| Column | Type | Constraints |
|---|---|---|
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT |
| `role` | TEXT | NOT NULL CHECK (role IN ('extraction','transcription','summarization','contextual_augmentation','embedding','reranker','generation','language_id')) |
| `model_name` | TEXT | NOT NULL |
| `model_version` | TEXT | NULL |
| `tool` | TEXT | NULL |
| `config_hash` | TEXT | NOT NULL |
| `phase` | TEXT | NOT NULL |
| `started_at` | TEXT | NOT NULL DEFAULT (datetime('now')) |
| `finished_at` | TEXT | NULL |
| `status` | TEXT | NOT NULL DEFAULT 'running' CHECK (status IN ('running','done','failed','aborted')) |
| `items_processed` | INTEGER | NULL |
| `items_failed` | INTEGER | NULL |
| `notes` | TEXT | NULL |

Indexes: `idx_model_run_role` on `(role, started_at DESC)`; `idx_model_run_phase` on `(phase, started_at DESC)`.

### 3.3 Table: `extraction` (existing, populated in Phase 8)

One row per `(file_id, tool)`. `file_sha256` denormalized from `file.sha256` at insertion; mismatch invalidates the extraction. `text_path` is an absolute path under `.rag-cache/extractions/`. `tool_version` records the resolved version. `metadata_json` stores tool-specific metadata: page count, table count, document title/author/dates for documents; for transcripts, model name, detected language, and timestamp granularity flag.

### 3.4 Table: `summary` (existing, populated in Phase 9)

One row per `(file_id, model, prompt_hash)`. Multiple summaries per file are permitted to support A/B comparison across summarization models. Retrieval uses the row matching the currently configured summarization model.

### 3.5 Table: `chunk` (existing, extended)

One row per chunk, plus four new columns from migration v2:

`context_text` — Anthropic-style contextual augmentation generated in Phase 10.5. Used at embedding time only; never shown to the answering LLM in place of `text`.
`context_model` — Ollama tag of the model that generated `context_text`.
`context_prompt_hash` — SHA-256 of the augmentation prompt; permits idempotent re-runs.
`context_generated_at` — ISO timestamp.

Other population rules unchanged from v1: `text` is the chunk text as fed to the embedding model when no context augmentation is enabled, or the raw chunk text when augmentation is enabled (in which case the embedded text is `context_text + "\n\n" + text`). `char_start` and `char_end` are byte offsets into the source extraction's text. `page_start` and `page_end` are populated from extraction metadata when available. For media transcripts, `time_start_seconds` and `time_end_seconds` flow through LlamaIndex chunk metadata.

### 3.6 Table: `embedding_ref` (existing, extended)

One row per `(chunk_id, embedding_model)`. `vector_store='chroma'`. `collection` follows the naming convention `{kind}__{suffix}`. `external_id` is the Chroma document id (UUID). `embedding_model` is the full Ollama tag including version. `dim` is the actual vector dimensionality returned by the model on first call. `config_hash` is a SHA-256 of the embedding-relevant configuration block at insertion time. `is_current=1` if this is the actively-served model for this chunk, `0` if superseded.

### 3.7 Tables: `folder_embedding_ref` and `summary_embedding_ref` (new)

```sql
CREATE TABLE folder_embedding_ref (
    folder_id       INTEGER NOT NULL REFERENCES folder(id) ON DELETE CASCADE,
    vector_store    TEXT NOT NULL,
    collection      TEXT NOT NULL,
    external_id     TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    dim             INTEGER NOT NULL,
    config_hash     TEXT NOT NULL,
    is_current      INTEGER NOT NULL DEFAULT 1 CHECK (is_current IN (0,1)),
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (folder_id, embedding_model)
);

CREATE TABLE summary_embedding_ref (
    summary_id      INTEGER NOT NULL REFERENCES summary(id) ON DELETE CASCADE,
    vector_store    TEXT NOT NULL,
    collection      TEXT NOT NULL,
    external_id     TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    dim             INTEGER NOT NULL,
    config_hash     TEXT NOT NULL,
    is_current      INTEGER NOT NULL DEFAULT 1 CHECK (is_current IN (0,1)),
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (summary_id, embedding_model)
);
```

### 3.8 Table: `query_log` (new)

| Column | Type | Constraints |
|---|---|---|
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT |
| `asked_at` | TEXT | NOT NULL DEFAULT (datetime('now')) |
| `user` | TEXT | NULL |
| `query_text` | TEXT | NOT NULL |
| `query_lang` | TEXT | NULL |
| `folder_filter_json` | TEXT | NULL |
| `retrieved_chunks_json` | TEXT | NOT NULL |
| `answer_text` | TEXT | NOT NULL |
| `answer_model` | TEXT | NOT NULL |
| `embedding_model` | TEXT | NOT NULL |
| `reranker_model` | TEXT | NULL |
| `latency_ms` | INTEGER | NOT NULL |
| `retrieval_ms` | INTEGER | NULL |
| `generation_ms` | INTEGER | NULL |
| `feedback` | TEXT | NULL CHECK (feedback IN ('up','down') OR feedback IS NULL) |
| `feedback_note` | TEXT | NULL |
| `feedback_at` | TEXT | NULL |

Indexes: `idx_query_log_asked` on `asked_at DESC`; partial `idx_query_log_feedback` on `feedback` where not null.

### 3.9 Tables: `eval_question` and `eval_run` (new)

```sql
CREATE TABLE eval_question (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    question                TEXT NOT NULL,
    expected_file_ids_json  TEXT NOT NULL,
    expected_answer         TEXT,
    category                TEXT,
    lang                    TEXT,
    created_at              TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE eval_run (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id             INTEGER NOT NULL REFERENCES eval_question(id) ON DELETE CASCADE,
    run_at                  TEXT NOT NULL DEFAULT (datetime('now')),
    embedding_model         TEXT NOT NULL,
    generation_model        TEXT NOT NULL,
    reranker_model          TEXT,
    contextual_augmentation INTEGER NOT NULL DEFAULT 0 CHECK (contextual_augmentation IN (0,1)),
    retrieval_recall_at_5   INTEGER NOT NULL CHECK (retrieval_recall_at_5 IN (0,1)),
    retrieval_recall_at_10  INTEGER NOT NULL CHECK (retrieval_recall_at_10 IN (0,1)),
    human_grade             INTEGER CHECK (human_grade IS NULL OR human_grade BETWEEN 0 AND 5),
    automated_grade         REAL,
    answer_text             TEXT,
    notes                   TEXT
);
CREATE INDEX idx_eval_run_models ON eval_run(embedding_model, generation_model, run_at DESC);
```

`eval_question.category` controlled vocabulary: `lookup`, `synthesis`, `scoped`, `negative`, `bilingual`.

### 3.10 Chroma Collection Naming Convention

Three collection kinds, namespaced per embedding model:

```
chunks__<suffix>
summaries__<suffix>
folders__<suffix>
```

`<suffix>` is taken from `config.models.embedding.collection_suffix`. Multiple suffixes coexist. The active suffix is read from `config.toml` at every retrieval call. Switching suffixes is a single config edit and requires no code change.

Each collection's metadata records on creation: `{"embedding_model": "<full ollama tag>", "dim": <int>, "created_at": "<iso8601>", "chunker_version": "<str>", "contextual_augmentation": <bool>, "context_model": "<str|null>", "source_db_path": "<absolute path>"}`. The retrieval layer asserts these match the active configuration before serving any query.

### 3.11 Views

```sql
CREATE VIEW v_extraction_status AS
SELECT f.category, COUNT(*) AS total,
       SUM(CASE WHEN e.id IS NOT NULL THEN 1 ELSE 0 END) AS extracted,
       SUM(CASE WHEN f.extract_status='failed' THEN 1 ELSE 0 END) AS failed,
       SUM(CASE WHEN f.extract_status='pending' THEN 1 ELSE 0 END) AS pending
FROM file f
LEFT JOIN extraction e ON e.file_id = f.id
WHERE f.is_dup_primary=1 AND f.excluded=0
GROUP BY f.category;

CREATE VIEW v_embedding_coverage AS
SELECT er.embedding_model, er.collection,
       COUNT(DISTINCT er.chunk_id) AS chunks_embedded,
       COUNT(DISTINCT c.file_id) AS files_covered
FROM embedding_ref er JOIN chunk c ON c.id = er.chunk_id
WHERE er.is_current = 1
GROUP BY er.embedding_model, er.collection;

CREATE VIEW v_context_coverage AS
SELECT context_model, COUNT(*) AS chunks_augmented,
       COUNT(DISTINCT file_id) AS files_covered
FROM chunk
WHERE context_text IS NOT NULL
GROUP BY context_model;

CREATE VIEW v_query_volume AS
SELECT DATE(asked_at) AS day, COUNT(*) AS queries,
       AVG(latency_ms) AS avg_latency_ms,
       SUM(CASE WHEN feedback='up' THEN 1 ELSE 0 END) AS up,
       SUM(CASE WHEN feedback='down' THEN 1 ELSE 0 END) AS down
FROM query_log
GROUP BY DATE(asked_at) ORDER BY day DESC;

CREATE VIEW v_eval_summary AS
SELECT embedding_model, generation_model, contextual_augmentation,
       COUNT(*) AS questions,
       AVG(retrieval_recall_at_5) AS recall_at_5,
       AVG(retrieval_recall_at_10) AS recall_at_10,
       AVG(human_grade) AS avg_human_grade
FROM eval_run
GROUP BY embedding_model, generation_model, contextual_augmentation;
```

---

## 4. Configuration Schema (Single Source of Truth)

The `config.toml` file is the sole source of model identities and operational parameters. The system reads it at startup of every CLI invocation and at every API request. No model name appears as a string literal in source code outside the seed config and tests.

```toml
# ============================================================
# config.toml — RAG Pipeline (post pre-flight)
# Every model identity below is a default. Override per-invocation
# via CLI flags (e.g. `rag embed --model NAME`) or by editing this
# file. The pipeline never hard-codes a model name.
# ============================================================

[paths]
corpus_root = "./data/my-corpus"
cache_root  = "./data/.rag-cache"
chroma_root = "./chroma"

[ollama]
host             = "http://localhost:11434"
timeout_seconds  = 300
max_retries      = 3
keep_alive       = "10m"

# ============================================================
# Models — change any name and re-run the affected phase.
# ============================================================

[models.embedding]
name              = "qwen3-embedding:8b"
collection_suffix = "qwen3emb8b"
batch_size        = 32
normalize         = true
# Optional Matryoshka truncation; null = use native dimension.
truncate_dim      = 0   # 0 means no truncation

[models.embedding.alternatives]
# Reference list for `rag embed --switch-to <key>` and `rag eval`.
# Not loaded at runtime; documents the supported alternatives.
qwen3_4b      = { name = "qwen3-embedding:4b",            suffix = "qwen3emb4b" }
nomic_v2      = { name = "nomic-embed-text-v2-moe:latest", suffix = "nomicv2"   }
embedding_gemma = { name = "embeddinggemma:latest",        suffix = "gemmaemb"  }

[models.summarization]
name           = "gemma4:e4b"
temperature    = 0.3
max_tokens     = 400
prompt_version = "v1"

[models.contextual_retrieval]
# Phase 10.5: Anthropic-style chunk-level context augmentation.
# Disable to skip Phase 10.5 entirely; the embedding phase will use raw chunk text.
enabled            = true
name               = "gemma4:e2b"
temperature        = 0.1
max_context_tokens = 16384
prompt_version     = "v1"

[models.generation]
name               = "gemma4:26b"
fallback_name      = "gemma4:e4b"
alternate_name     = "qwen3:30b-a3b-instruct-2507-q4_K_M"
# Sampling — Gemma 4 card recommends 1.0/0.95/64; lower temperature for citation-strict RAG.
temperature        = 0.5
top_p              = 0.95
top_k              = 64
max_context_tokens = 16384
prompt_version     = "v1"
thinking           = false       # Gemma 4 thinking mode; off by default for RAG

[models.reranker]
# Optional. Start disabled. Add only after baseline eval shows recall headroom.
# Only community uploads exist on Ollama for Qwen3 reranker.
enabled  = false
name     = "dengcao/Qwen3-Reranker-0.6B"
top_n_in  = 50
top_n_out = 8

[models.transcription]
backend          = "whisper.cpp"
model            = "large-v3"
language         = "auto"
threads          = 8
word_timestamps  = true
# Per-folder opt-in glob patterns. Empty list = transcribe nothing.
opt_in_globs = [
    "Project/*/Media/*",
    "Project/*/*Site Visit*/*",
    "Project/*/*Demonstration*/*",
]

[models.language_id]
backend     = "fasttext"
model_path  = ".rag-cache/models/lid.176.bin"

# ============================================================
# Pipeline parameters (model-independent)
# ============================================================

[extract]
docling_workers              = 4
tika_url                     = "http://localhost:9998"
size_threshold_for_summary   = 500
tika_brute_force_fallback    = true   # retry triage-failed files via Tika

[chunk]
chunker         = "sentence-window"
window_size     = 5
target_tokens   = 512
chunker_version = "1"

[retrieval]
top_k_chunks       = 8
top_k_documents    = 20
top_k_folders      = 5
dense_candidates   = 50
bm25_candidates    = 50
hierarchical       = true
rrf_k              = 60

[generation_runtime]
require_citations         = true
refuse_on_empty_context   = true

[api]
host = "127.0.0.1"
port = 8765

[ui]
host = "127.0.0.1"
port = 7860
```

The `config_hash` referenced throughout this TSD is computed as `sha256(canonical_json(config.models.<role> ∪ config.chunk ∪ config.extract))` for the relevant role. Changes to unrelated config sections do not invalidate prior work. The hash is recorded in every `model_run`, `embedding_ref`, `folder_embedding_ref`, and `summary_embedding_ref` row.

---

## 5. CLI and API Design

The pre-flight CLI binary is renamed from `rag-preflight` to `rag` for the unified system. All existing pre-flight commands remain available under their original names. New commands follow.

### 5.1 CLI Commands (Phases 8–14)

All commands accept `--config <path>` and obey existing exit-code conventions.

#### `rag extract`

Phase 8. Extracts text per `extract_strategy`.

`rag extract [--workers N] [--strategy STRATEGY] [--limit N] [--reextract]`

Selects rows from `file` where `is_dup_primary=1 AND excluded=0 AND extract_status='pending'` (or all rows if `--reextract`). Dispatches by `extract_strategy`. Writes markdown to `.rag-cache/extractions/<sha-prefix>/<sha>.md`. Inserts `extraction` rows. Updates `file.extract_status`. Records failures in `failure` with `phase='extract'`. Records start/end in `model_run`.

The 23 triage-failed files identified in BUILD_STATUS are routed through Tika as a brute-force fallback when `extract.tika_brute_force_fallback=true`. Successful Tika fallbacks update `file.extract_strategy='tika'`. Tika failure marks the file `extract_status='failed'` with `error_class='unsupported_format'`.

Exit codes: `0` on success; `9` if extraction failure rate exceeds `5%` (configurable).

#### `rag transcribe`

Phase 8 sub-command for audio/video.

`rag transcribe [--workers 1] [--folder PATH_PREFIX] [--limit N] [--model NAME]`

Selects audio/video files where `extract_status='pending'`. Defaults to per-folder opt-in via `models.transcription.opt_in_globs`. Calls whisper.cpp via `pywhispercpp`. Stores transcript JSON (with word timestamps) under `.rag-cache/transcripts/`. Stores markdown rendering under `.rag-cache/extractions/`. Inserts `extraction` row with `tool='whisper.cpp'`.

#### `rag summarize`

Phase 9.

`rag summarize [--model NAME] [--limit N] [--resummarize]`

Selects extractions with `text_char_count >= config.extract.size_threshold_for_summary` and no current `summary` row for `(file_id, configured_model, prompt_hash)`. Generates summaries via Ollama. Validates response with Pydantic. Inserts into `summary`. Records `model_run`.

#### `rag chunk`

Phase 10.

`rag chunk [--chunker NAME] [--window N] [--rechunk]`

Selects extractions not yet chunked under the active `chunker_version`. Runs the configured chunker. Inserts into `chunk`. FTS5 index updates via existing triggers. `--rechunk` deletes prior chunks for those files (cascading via `embedding_ref` foreign key) before inserting new ones.

#### `rag context`

Phase 10.5. Generates Anthropic-style contextual augmentations.

`rag context [--model NAME] [--limit N] [--regenerate]`

Skipped entirely if `models.contextual_retrieval.enabled = false`. For every chunk where `context_text IS NULL` (or all if `--regenerate`), constructs a prompt containing the full extracted document plus the specific chunk, asks the configured small LLM to produce a 50–100 token context blurb, writes it to `chunk.context_text` along with `context_model` and `context_prompt_hash`. Idempotent on prompt hash. Records `model_run` with `role='contextual_augmentation'`.

For long documents that exceed the model's context window, the prompt uses the document's existing summary (from `summary` table) plus the surrounding paragraph window of the chunk, rather than the full document. This degradation is logged.

#### `rag embed`

Phase 11.

```
rag embed [--model NAME] [--collections KIND1,KIND2,...] [--reembed]
rag embed --switch-to NAME           # re-embed and update active config
rag embed --gc                       # delete superseded collections
rag embed --list                     # show all collections and current status
```

Default: for each chunk lacking an `embedding_ref` row for the configured embedding model, embeds and inserts into `chunks__<suffix>` Chroma collection. Records `embedding_ref` row with `is_current=1`. The embedded text is `context_text + "\n\n" + text` if `context_text` exists, else `text` alone. Equivalently for `summary_embedding_ref` and `folder_embedding_ref`.

`--switch-to` performs the embed against the new model into new collections, then in a single transaction sets `is_current=0` on prior model's `*_embedding_ref` rows and `is_current=1` for the new model, and rewrites `config.models.embedding.{name,collection_suffix}` in `config.toml` (preserving comments via `tomli_w`). Old collections remain on disk; rollback is one config edit.

`--gc` deletes Chroma collections that have no `is_current=1` rows in any `*_embedding_ref` table; prompts before deletion.

#### `rag query`

Phase 12 terminal entry point.

`rag query "<question>" [--folder PATH_PREFIX] [--top-k N] [--no-rerank] [--no-augment] [--json]`

#### `rag serve`

Phase 13. Starts the FastAPI server.

`rag serve [--host HOST] [--port PORT] [--reload]`

#### `rag ui`

Phase 14. Starts the Gradio UI.

`rag ui [--host HOST] [--port PORT]`

#### `rag eval`

Evaluation harness.

```
rag eval add --question "..." --expects FILE_IDS --category CAT --lang LANG
rag eval run [--embedding NAME] [--generation NAME] [--reranker NAME] [--no-augment]
rag eval report [--last N] [--compare]
```

`run` overrides take precedence over `config.toml` for the duration of the run; the active configuration is restored afterward. Results recorded in `eval_run` with full model identity. `report --compare` shows a side-by-side table of all `(embedding_model, generation_model, contextual_augmentation)` combinations evaluated.

#### `rag models`

```
rag models list                         # active models per role + collection state
rag models pull                         # ollama pull every model named in config (active set)
rag models pull --all                   # also pull all alternatives
rag models bench --role embedding       # micro-benchmark the configured model
rag models check                        # verify every configured model is present in Ollama
```

### 5.2 HTTP API (FastAPI, mounted at `http://127.0.0.1:8765`)

#### `POST /query`

Request:
```json
{
  "query": "What robots did we deploy at Gleneagles Hospital?",
  "folder_filter": {"path_prefix": "Project/Gleneagles Hospital"},
  "top_k_chunks": 8,
  "use_reranker": true,
  "use_augmentation": true,
  "thinking": false,
  "stream": false,
  "user": "alice"
}
```

Validation: `query` required, non-empty, max 2000 chars. `folder_filter` must contain exactly one of `path_prefix`, `folder_id`, `inferred_category` if present. `top_k_chunks` 1–20. `use_reranker`, `use_augmentation`, `thinking` default from config.

Response 200:
```json
{
  "answer": "Based on the deployment records, two robot models were used at Gleneagles Hospital: the Pudu Flashbot for delivery [1][2] and a Temi unit for reception [3].",
  "citations": [
    {
      "marker": "[1]",
      "file_id": 412,
      "path": "Project/Gleneagles Hospital/Flashbot Mapping/deployment_log_2024-03-12.pdf",
      "page": 3,
      "chunk_id": 18432,
      "snippet": "Pudu Flashbot units deployed on level 3 corridor..."
    }
  ],
  "retrieved_chunks": [
    {"chunk_id": 18432, "file_id": 412, "score": 0.78, "source": "rerank"}
  ],
  "metrics": {"retrieval_ms": 240, "generation_ms": 2880, "total_ms": 3120},
  "models": {
    "embedding": "qwen3-embedding:8b",
    "generation": "gemma4:26b",
    "reranker": null,
    "contextual_augmentation": "gemma4:e2b"
  },
  "query_log_id": 871
}
```

Response 400 on validation; 422 if `folder_filter` references unknown path/category; 503 if Ollama unreachable after retries or if the required Chroma collection does not exist for the active embedding model.

#### `POST /query/stream`

Same request body; returns `text/event-stream`. Events: `retrieval` (once, after retrieval), `token` (per token), `citations` (once, after generation), `done` (once, with metrics), `error` (on failure).

#### `POST /feedback`

Request: `{"query_log_id": 871, "feedback": "up", "note": "correct"}`. Response 200: `{"ok": true}`. 404 if log id not found.

#### `GET /folders`

Returns folder tree with inferred labels for the UI's filter picker.

#### `GET /files/{file_id}`

Returns file metadata, summary, extraction path, page count, parent folder.

#### `GET /files/{file_id}/text`

Returns extracted markdown as `text/plain`. 404 if no extraction.

#### `GET /models`

Active model configuration plus Chroma collection state. Same content as `rag models list` in JSON form.

#### `GET /health`

Returns Ollama reachability, presence of required Chroma collection for active embedding model, database accessibility, fasttext model availability. Used by UI status badge.

#### `POST /eval/run`

Triggers `rag eval run` server-side. Response 200: `{"questions": 30, "recall_at_5": 0.83, "recall_at_10": 0.93, "run_id": 17}`.

### 5.3 Retrieval Algorithm Specification

The retrieval pipeline is deterministic:

**Step 1 — Configuration assertion.** Verify Chroma collection `chunks__<active_suffix>` exists; reject 503 otherwise. Verify the collection's metadata `embedding_model` matches active config; reject 503 on mismatch. If `models.contextual_retrieval.enabled` differs between collection metadata and active config, log a warning but proceed (it affects what was embedded, not what can be retrieved).

**Step 2 — Query language detection.** Run fasttext on the query.

**Step 3 — Folder filter resolution.** If `folder_filter` provided, resolve to set `allowed_file_ids`. Empty result returns 422. Absent filter means `allowed_file_ids = None`.

**Step 4 — Hierarchical narrowing** (if `retrieval.hierarchical=true`). Query `folders__<suffix>` (top-`top_k_folders`) and `summaries__<suffix>` (top-`top_k_documents`). Derive `candidate_file_ids`. Intersect with `allowed_file_ids`.

**Step 5 — Dense chunk retrieval.** Query `chunks__<suffix>` with `where={"file_id": {"$in": list(candidate_file_ids)}}`, top-`dense_candidates`.

**Step 6 — BM25 chunk retrieval.** SQL query on `chunk_fts` with same `file_id` restriction, top-`bm25_candidates`. Tokenization for BM25: `unicode61 remove_diacritics 2` (configured at FTS5 creation in pre-flight) plus a query-side bigram fallback for CJK queries.

**Step 7 — Reciprocal Rank Fusion** (k=`retrieval.rrf_k`).

**Step 8 — Reranking** (if `use_reranker` and `models.reranker.enabled`). Pass top-`top_n_in` to reranker; take top-`top_n_out`.

**Step 9 — Context assembly.** Order final chunks by `(file_id, ordinal)`. Build numbered context with file path and page reference. **Use `chunk.text` (the original chunk), never `chunk.context_text`**, regardless of whether augmentation was used at embedding time.

**Step 10 — Generation.** Call configured generation model with system prompt (§6.1) and assembled context. If `models.generation.thinking=true` (or per-request override), prepend `<|think|>` to the system prompt for Gemma 4. Stream if requested.

**Step 11 — Citation parsing.** Extract `[N]` markers; resolve to chunk metadata; build citation array.

**Step 12 — Logging.** Insert `query_log` row with all metadata.

---

## 6. Implementation Details & Code Snippets

### 6.1 Generation System Prompt (Pinned, Versioned)

`prompts/generation_v1.txt`. `prompt_version` recorded in `query_log`.

```
You are the ODW.ai Vault internal knowledge assistant. You help staff find
information about company projects, products, deployments, and operations
by answering questions using ONLY the provided context excerpts.

You answer in the same language as the user's question (English or
Traditional Chinese). Match the user's terminology and tone.

RULES:
1. Every factual claim MUST be supported by a citation marker [N] where N
   refers to a numbered context excerpt below. Use markers inline.
2. If the context does not contain enough information to answer, say so
   explicitly. Do not guess. Do not use external knowledge about products,
   clients, robots, sites, or contracts beyond what the context says.
3. When synthesizing across multiple sources, cite each.
4. Preserve technical terminology, model numbers, robot platform names,
   client names, and site names exactly as they appear in the context.
5. Never invent file names, page numbers, or citation markers that are
   not in the provided context.
6. If asked about a client or project not present in the context, state
   that you have no information about it; do not speculate.

CONTEXT EXCERPTS:
{numbered_chunks}

USER QUESTION: {query}

ANSWER:
```

### 6.2 Contextual Retrieval Prompt (Phase 10.5)

`prompts/contextual_retrieval_v1.txt`.

```
<document>
{document_text}
</document>

Here is the chunk we want to situate within the whole document:

<chunk>
{chunk_text}
</chunk>

Please give a short, succinct context (50-100 tokens) to situate this
chunk within the overall document, for the purposes of improving search
retrieval of the chunk. Answer ONLY with the succinct context and nothing
else. Do not include preamble, do not repeat the chunk content, do not
add commentary. Output language: match the document's language.
```

### 6.3 Configuration Loader

```python
# pipeline/config.py
import hashlib, json, tomllib
from pathlib import Path
from pydantic import BaseModel, Field

class EmbeddingConfig(BaseModel):
    name: str
    collection_suffix: str
    batch_size: int = 32
    normalize: bool = True
    truncate_dim: int = 0

class GenerationConfig(BaseModel):
    name: str
    fallback_name: str | None = None
    alternate_name: str | None = None
    temperature: float = 0.5
    top_p: float = 0.95
    top_k: int = 64
    max_context_tokens: int = 16384
    prompt_version: str = "v1"
    thinking: bool = False

class ContextualRetrievalConfig(BaseModel):
    enabled: bool = True
    name: str
    temperature: float = 0.1
    max_context_tokens: int = 16384
    prompt_version: str = "v1"

# ... (similar models for summarization, reranker, transcription, language_id)

class ModelsConfig(BaseModel):
    embedding: EmbeddingConfig
    summarization: "SummarizationConfig"
    contextual_retrieval: ContextualRetrievalConfig
    generation: GenerationConfig
    reranker: "RerankerConfig"
    transcription: "TranscriptionConfig"
    language_id: "LanguageIdConfig"

class AppConfig(BaseModel):
    paths: "PathsConfig"
    ollama: "OllamaConfig"
    models: ModelsConfig
    extract: "ExtractConfig"
    chunk: "ChunkConfig"
    retrieval: "RetrievalConfig"
    generation_runtime: "GenerationRuntimeConfig"
    api: "ApiConfig"
    ui: "UiConfig"

def load_config(path: Path) -> AppConfig:
    with path.open("rb") as f:
        data = tomllib.load(f)
    return AppConfig.model_validate(data)

def config_hash(role_block: dict, chunk_block: dict, extract_block: dict) -> str:
    payload = json.dumps(
        {"role": role_block, "chunk": chunk_block, "extract": extract_block},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

### 6.4 Embedding Phase — Critical Logic

```python
# pipeline/phase11_embed.py
import uuid
import chromadb
import ollama
from tenacity import retry, stop_after_attempt, wait_exponential

def collection_name(kind: str, suffix: str) -> str:
    return f"{kind}__{suffix}"

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=20))
def embed_batch(texts: list[str], model: str, host: str) -> list[list[float]]:
    client = ollama.Client(host=host)
    resp = client.embed(model=model, input=texts)
    return resp["embeddings"]

def ensure_collection(chroma, kind, suffix, model, dim, cfg_hash, ctx_aug):
    name = collection_name(kind, suffix)
    try:
        col = chroma.get_collection(name)
        meta = col.metadata or {}
        if meta.get("embedding_model") != model:
            raise RuntimeError(
                f"Collection {name} was built with {meta.get('embedding_model')}, "
                f"refusing to mix with {model}"
            )
        return col
    except Exception:
        return chroma.create_collection(
            name=name,
            metadata={
                "embedding_model": model, "dim": dim,
                "config_hash": cfg_hash, "created_at": now_iso(),
                "contextual_augmentation": int(ctx_aug),
            },
        )

def embed_chunks(db, chroma, cfg) -> int:
    cfg_hash = compute_embedding_config_hash(cfg)
    model = cfg.models.embedding.name
    suffix = cfg.models.embedding.collection_suffix
    use_aug = cfg.models.contextual_retrieval.enabled
    run_id = start_model_run(db, role="embedding", model_name=model,
                             config_hash=cfg_hash, phase="embed")

    pending = list(db.query("""
        SELECT c.id AS chunk_id, c.text, c.context_text,
               c.file_id, f.folder_id, f.rel_path,
               fo.inferred_category, fo.inferred_label
        FROM chunk c
        JOIN file f ON f.id = c.file_id
        JOIN folder fo ON fo.id = f.folder_id
        WHERE c.id NOT IN (
            SELECT chunk_id FROM embedding_ref
            WHERE embedding_model = ? AND is_current = 1
        )
        AND f.is_dup_primary = 1 AND f.excluded = 0
    """, [model]))

    if not pending:
        finish_model_run(db, run_id, status="done", items_processed=0)
        return 0

    def text_for_embedding(row):
        if use_aug and row["context_text"]:
            return f"{row['context_text']}\n\n{row['text']}"
        return row["text"]

    sample = embed_batch([text_for_embedding(pending[0])],
                         model, cfg.ollama.host)
    dim = len(sample[0])
    col = ensure_collection(chroma, "chunks", suffix, model, dim, cfg_hash, use_aug)

    bs = cfg.models.embedding.batch_size
    processed = 0
    for i in range(0, len(pending), bs):
        batch = pending[i:i+bs]
        texts = [text_for_embedding(r) for r in batch]
        vectors = embed_batch(texts, model, cfg.ollama.host)
        ids = [str(uuid.uuid4()) for _ in batch]
        metadatas = [{
            "chunk_id": r["chunk_id"],
            "file_id": r["file_id"],
            "folder_id": r["folder_id"],
            "rel_path": r["rel_path"],
            "inferred_category": r["inferred_category"] or "",
            "inferred_label": r["inferred_label"] or "",
        } for r in batch]
        col.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
        rows = [{
            "chunk_id": r["chunk_id"], "vector_store": "chroma",
            "collection": col.name, "external_id": ids[j],
            "embedding_model": model, "dim": dim,
            "config_hash": cfg_hash, "is_current": 1,
        } for j, r in enumerate(batch)]
        db["embedding_ref"].insert_all(rows, replace=False)
        processed += len(batch)

    finish_model_run(db, run_id, status="done", items_processed=processed)
    return processed
```

### 6.5 Switch-To Re-Embed (Atomic Model Swap)

```python
def switch_to(new_model_name: str, new_suffix: str | None,
              db, chroma, cfg, config_path: Path) -> None:
    if new_suffix is None:
        new_suffix = suffix_from_model(new_model_name)
    tmp_cfg = cfg.copy_with_embedding(name=new_model_name,
                                      collection_suffix=new_suffix)
    embed_chunks(db, chroma, tmp_cfg)
    embed_summaries(db, chroma, tmp_cfg)
    embed_folders(db, chroma, tmp_cfg)
    with db.conn:
        for table in ("embedding_ref", "folder_embedding_ref",
                      "summary_embedding_ref"):
            db.execute(f"UPDATE {table} SET is_current=0 "
                       f"WHERE embedding_model != ?", [new_model_name])
            db.execute(f"UPDATE {table} SET is_current=1 "
                       f"WHERE embedding_model = ?", [new_model_name])
    rewrite_config_embedding(config_path, new_model_name, new_suffix)
```

### 6.6 Hybrid Retrieval — Critical Logic

```python
# rag/retrieval.py
from dataclasses import dataclass
import time

@dataclass
class Hit:
    chunk_id: int; file_id: int; folder_id: int
    rel_path: str; page_start: int | None; text: str
    dense_score: float | None = None; bm25_score: float | None = None
    rerank_score: float | None = None; fused_score: float | None = None

def reciprocal_rank_fuse(dense, bm25, k):
    by_id = {}
    for rank, h in enumerate(dense):
        h.fused_score = 1.0 / (k + rank + 1)
        by_id[h.chunk_id] = h
    for rank, h in enumerate(bm25):
        c = 1.0 / (k + rank + 1)
        if h.chunk_id in by_id:
            by_id[h.chunk_id].fused_score += c
            by_id[h.chunk_id].bm25_score = h.bm25_score
        else:
            h.fused_score = c
            by_id[h.chunk_id] = h
    return sorted(by_id.values(), key=lambda x: x.fused_score, reverse=True)

def retrieve(query, db, chroma, cfg,
             folder_filter=None, top_k_chunks=None,
             use_reranker=None, use_augmentation=None):
    t0 = time.monotonic()
    top_k_chunks = top_k_chunks or cfg.retrieval.top_k_chunks
    use_reranker = (cfg.models.reranker.enabled
                    if use_reranker is None else use_reranker)
    suffix = cfg.models.embedding.collection_suffix

    assert_collection_matches_config(chroma, suffix, cfg)

    allowed_file_ids = (resolve_folder_filter(db, folder_filter)
                        if folder_filter else None)

    if cfg.retrieval.hierarchical:
        candidate_file_ids = hierarchical_narrow(
            chroma, suffix, query, db,
            cfg.retrieval.top_k_folders, cfg.retrieval.top_k_documents,
        )
        if allowed_file_ids is not None:
            candidate_file_ids &= set(allowed_file_ids)
    else:
        candidate_file_ids = set(allowed_file_ids) if allowed_file_ids else None

    dense = dense_search(chroma, suffix, query,
                         candidate_file_ids, cfg.retrieval.dense_candidates)
    bm25  = bm25_search(db, query,
                         candidate_file_ids, cfg.retrieval.bm25_candidates)
    fused = reciprocal_rank_fuse(dense, bm25, cfg.retrieval.rrf_k)

    if use_reranker and cfg.models.reranker.enabled and len(fused) > top_k_chunks:
        top_in = fused[: cfg.models.reranker.top_n_in]
        reranked = call_reranker(query, top_in, cfg)
        final = reranked[: cfg.models.reranker.top_n_out]
    else:
        final = fused[:top_k_chunks]

    return final, {
        "retrieval_ms": int((time.monotonic() - t0) * 1000),
        "candidate_file_count": len(candidate_file_ids) if candidate_file_ids else 0,
        "dense_count": len(dense), "bm25_count": len(bm25),
        "reranked": use_reranker and cfg.models.reranker.enabled,
    }
```

### 6.7 Citation-Strict Generation (Gemma 4 + Qwen3 compatible)

```python
# rag/generation.py
import re, time
import ollama
from tenacity import retry, stop_after_attempt

CITATION_RE = re.compile(r"\[(\d+)\]")

def assemble_context(hits, max_chars):
    parts = []; used = 0
    for i, h in enumerate(hits, start=1):
        page_ref = f", page {h.page_start}" if h.page_start else ""
        block = f"[{i}] (source: {h.rel_path}{page_ref})\n{h.text}\n"
        if used + len(block) > max_chars:
            break
        parts.append(block); used += len(block)
    return "\n".join(parts)

@retry(stop=stop_after_attempt(2))
def generate_answer(query, hits, cfg, prompt_template):
    t0 = time.monotonic()
    if cfg.generation_runtime.refuse_on_empty_context and not hits:
        return {
            "answer": "I don't have any information in the knowledge base "
                      "that answers this question.",
            "citations": [], "generation_ms": 0,
            "model": cfg.models.generation.name,
        }

    context = assemble_context(
        hits, cfg.models.generation.max_context_tokens * 3,  # ~3 chars/token rough
    )
    system_prompt = prompt_template.format(numbered_chunks=context, query=query)
    if cfg.models.generation.thinking:
        system_prompt = "<|think|>\n" + system_prompt

    client = ollama.Client(host=cfg.ollama.host)
    response = client.generate(
        model=cfg.models.generation.name,
        prompt=system_prompt,
        options={
            "temperature": cfg.models.generation.temperature,
            "top_p": cfg.models.generation.top_p,
            "top_k": cfg.models.generation.top_k,
            "num_ctx": cfg.models.generation.max_context_tokens,
        },
    )
    answer = response["response"].strip()
    used = {int(m) for m in CITATION_RE.findall(answer)}
    citations = [{
        "marker": f"[{i+1}]", "file_id": hits[i].file_id,
        "rel_path": hits[i].rel_path, "page": hits[i].page_start,
        "chunk_id": hits[i].chunk_id, "snippet": hits[i].text[:240],
    } for i in range(len(hits)) if (i+1) in used]
    return {
        "answer": answer, "citations": citations,
        "generation_ms": int((time.monotonic() - t0) * 1000),
        "model": cfg.models.generation.name,
    }
```

### 6.8 Project Layout (Additions)

```
rag-pipeline/
├── corpus.db                           # from pre-flight
├── chroma/                             # vector store
├── config.toml                         # SINGLE SOURCE OF TRUTH for models
├── prompts/
│   ├── generation_v1.txt
│   ├── summarization_v1.txt
│   ├── contextual_retrieval_v1.txt
│   └── folder_meta_v1.txt              # from pre-flight
├── .rag-cache/
│   ├── extractions/
│   ├── transcripts/
│   ├── contexts/
│   ├── models/                         # fasttext lid.176.bin
│   └── logs/
├── pipeline/                           # extends pre-flight
│   ├── (phase0–phase7 unchanged)
│   ├── config.py
│   ├── phase8_extract.py
│   ├── phase8b_transcribe.py
│   ├── phase9_summarize.py
│   ├── phase10_chunk.py
│   ├── phase10_5_context.py            # Anthropic contextual augmentation
│   ├── phase11_embed.py
│   └── extractors/
│       ├── docling_extractor.py
│       ├── tika_extractor.py
│       ├── textutil_extractor.py
│       ├── ocr_extractor.py
│       ├── whisper_extractor.py
│       ├── filename_only_extractor.py
│       └── metadata_only_extractor.py
├── rag/
│   ├── retrieval.py
│   ├── generation.py
│   ├── reranker.py
│   ├── filters.py
│   └── citations.py
├── api/
│   ├── main.py
│   └── schemas.py
├── ui/
│   └── gradio_app.py
├── eval/
│   ├── questions.yaml
│   ├── runner.py
│   └── metrics.py
├── seeds/
│   └── format_policy.csv               # from pre-flight
├── cli.py                              # extended
├── pyproject.toml
└── tests/
    ├── (existing pre-flight tests)
    ├── test_phase8_extract.py
    ├── test_phase9_summarize.py
    ├── test_phase10_chunk.py
    ├── test_phase10_5_context.py
    ├── test_phase11_embed.py
    ├── test_retrieval.py
    ├── test_generation.py
    └── test_api.py
```

### 6.9 Logging and Observability

All new phases emit JSONL logs to `.rag-cache/logs/<phase>-<ts>.jsonl`. Every API request is logged to `.rag-cache/logs/api-<date>.jsonl` with `{ts, method, path, latency_ms, status, query_log_id, models}`. Datasette `serve` from pre-flight remains available for inspection.

### 6.10 Testing Requirements

Coverage target 80% for all new modules. Required tests:

For each extractor: a fixture file in `tests/fixtures/`; assert non-empty extracted text and correct metadata.

For chunking: synthetic 10-page document with deterministic content; assert `(char_start, char_end)` round-trip back to source byte-for-byte.

For contextual retrieval: assert that re-running with the same `context_prompt_hash` is a no-op; assert that the embedding text combines `context_text + "\n\n" + text` only when `enabled=true`.

For embedding: mocked Ollama embed endpoint; assert correct collection name computed from config; assert `embedding_ref` rows have correct `config_hash`; assert idempotency (zero new rows on identical re-run); assert that `--switch-to` flips `is_current` atomically.

For retrieval: synthetic Chroma collection with known vectors; assert RRF fusion produces documented ranking; assert folder filter eliminates non-matching files; assert collection-mismatch triggers 503.

For generation: mocked Ollama generate endpoint; assert citations parsed match marker positions; assert refusal on empty context; assert thinking-mode token correctly prepended.

For the API: FastAPI `TestClient` covering each endpoint with valid and invalid payloads.

End-to-end: pre-flight fixture corpus → extract → chunk → context → embed (mocked Ollama) → query (mocked Ollama) returns structured response with non-empty citations.

### 6.11 Migration and Backwards Compatibility

The migration adding `model_run`, `query_log`, `eval_question`, `eval_run`, `folder_embedding_ref`, `summary_embedding_ref`, the four new `chunk` columns, and the two new `embedding_ref` columns is `schema_version=2`. The migration script in `pipeline/db.py` applies it only when the current version is `1`. Existing pre-flight databases accept `rag` commands after running `rag init --migrate`.

---

## 7. Dependencies and References

### 7.1 New Python Dependencies

`pyproject.toml` additions:

```toml
[project.optional-dependencies]
rag = [
    "llama-index>=0.11",
    "llama-index-vector-stores-chroma>=0.2",
    "llama-index-embeddings-ollama>=0.3",
    "llama-index-llms-ollama>=0.3",
    "llama-index-readers-file>=0.1",
    "chromadb>=0.5",
    "docling>=2.0",
    "tika>=2.6",
    "pywhispercpp>=1.2",
    "rank-bm25>=0.2.2",
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "gradio>=4.40",
    "ocrmypdf>=15.4",
    "sse-starlette>=2.0",
    "tomli-w>=1.0",
    "fasttext>=0.9.2",
]
```

Reused from pre-flight without version change: `sqlite-utils`, `click`, `pydantic`, `ollama`, `PyMuPDF`, `patool`, `python-magic`, `rich`, `tenacity`, `msoffcrypto-tool`.

### 7.2 New External Binaries

| Tool | Install | Purpose |
|---|---|---|
| Apache Tika Server | `brew install tika` | Fallback extractor + brute-force retry for triage failures |
| ocrmypdf | `brew install ocrmypdf` | OCR pre-processing layer |
| Tesseract w/ language packs | `brew install tesseract tesseract-lang` | OCR engine |

Siegfried, FFmpeg, `unar`, `rar` reused from pre-flight.

### 7.3 Models to Pull (Active Set)

```bash
# Embedding (primary)
ollama pull qwen3-embedding:8b                                 # 4.7 GB

# Generation stack (Gemma 4)
ollama pull gemma4:e2b                                          # ~2 GB  — contextual augmentation
ollama pull gemma4:e4b                                          # ~5 GB  — summarization, fallback gen
ollama pull gemma4:26b                                          # ~16 GB — primary generation

# Active set total: ~28 GB
```

Optional (for A/B comparison):

```bash
ollama pull qwen3-embedding:4b                                 # 2.5 GB
ollama pull nomic-embed-text-v2-moe:latest                     # ~475 MB
ollama pull embeddinggemma:latest                              # ~600 MB
ollama pull qwen3:30b-a3b-instruct-2507-q4_K_M                 # 19 GB — alternate generation
ollama pull gemma4:31b                                         # ~20 GB — max-quality generation
ollama pull dengcao/Qwen3-Reranker-0.6B                        # ~600 MB — community upload
```

Whisper models are auto-downloaded by `pywhispercpp` to `~/.cache/whisper-cpp/` on first use. The fasttext language identification model `lid.176.bin` is fetched once during `rag init --migrate` to `.rag-cache/models/lid.176.bin`. In air-gapped environments, all artifact classes are transferred manually; `rag models pull --skip-download` verifies presence without fetching.

### 7.4 Reference Implementations

The autonomous agent must consult, but not vendor, the following repositories:

`https://github.com/run-llama/llama_index` — `IngestionPipeline`, `DocstoreStrategy.UPSERTS`, `SentenceWindowNodeParser`, `DocumentSummaryIndex` (for inspiration on summary-routed retrieval).

`https://github.com/chroma-core/chroma` — `PersistentClient`, collection metadata patterns, `where` filter syntax.

`https://github.com/anthropics/anthropic-cookbook/tree/main/skills/contextual-embeddings` — canonical Contextual Retrieval implementation, reference for the Phase 10.5 prompt and cost-control patterns.

`https://github.com/DS4SD/docling` — `DocumentConverter` API and Markdown export.

`https://github.com/chrismattmann/tika-python` — Tika server client; the agent must launch the server explicitly (`tika.server.start_server`).

`https://github.com/abdeladim-s/pywhispercpp` — whisper.cpp bindings; enable Core ML on Apple Silicon via documented parameters.

`https://huggingface.co/Qwen/Qwen3-Embedding-8B` — instruction-aware embedding usage for Qwen3 family; query-side prompt prefix conventions.

`https://huggingface.co/google/gemma-3-27b-it` and Gemma 4 model cards on Ollama — sampling parameter and thinking-mode conventions.

`https://github.com/ollama/ollama-python` — `Client.embed`, `Client.generate`, streaming generation.

`https://github.com/sysid/sse-starlette` — SSE response helper for the streaming endpoint.

### 7.5 Compliance Constraints (Reaffirmed)

The system makes no outbound HTTP calls during runtime except to `localhost`. Tika and Ollama bind to localhost. FastAPI binds to `127.0.0.1` by default; binding to `0.0.0.0` requires explicit `--host` and is logged as a security event.

The system never modifies files under `corpus_root`. Extraction outputs go to `.rag-cache/extractions/`; transcripts to `.rag-cache/transcripts/`; chunk contexts to `.rag-cache/contexts/`; vectors to `chroma/`; logs to `.rag-cache/logs/`.

Silent exception suppression remains a defect.

### 7.6 Operational Defaults Specific to the ODW.ai Vault Corpus

Based on the BUILD_STATUS report:

Transcription is opt-in by folder via `models.transcription.opt_in_globs`, defaulting to project-media folders only. The 36 video files dominated by 1.41 hours of QuickTime in non-deployment contexts are excluded by default.

The 23 triage-failed files are routed through Tika as a brute-force fallback in `rag extract`. Successful Tika extractions update `extract_strategy='tika'` for traceability. Tika failures mark the file `failed` with `error_class='unsupported_format'`.

CAD and executable files (`category` in `cad`, `executable`) default to `extract_strategy='filename-only'`. The filename-only extractor produces a one-line "document" containing the filename and parent folder labels, preserving filename signal (model numbers, client names, version strings) for retrieval without attempting binary content extraction.

Retrieval defaults `top_k_chunks=8`, `top_k_documents=20`, `top_k_folders=5`, reranker disabled, contextual augmentation enabled. These match the small working-set size and the Gemma 4 family's strong long-context handling.

Generation `max_context_tokens=16384` raised from v1's 8192, in line with Gemma 4 and Qwen3-30B-A3B both supporting 256K. Further raising (32768+) is recommended once baseline quality is established and RAM headroom is confirmed.

### 7.7 Model-Swap Workflow (Operational Recipe)

To try a new embedding model that has just become available:

1. `ollama pull <new-model>`
2. Edit `config.toml`: change `models.embedding.name` and `models.embedding.collection_suffix`. (Or run `rag embed --switch-to <new-model>` which performs both atomically.)
3. `rag embed` — embeds against new model into new collections; old collections remain.
4. `rag eval run` — evaluates the new model against the eval suite.
5. `rag eval report --compare` — side-by-side numerical comparison.
6. If the new model is better: keep the config; old collections can be removed with `rag embed --gc`.
7. If the new model is worse: revert the config edit; old collections were never invalidated, no rebuild required.

The same workflow applies to generation and summarization models, with the difference that generation-model swaps require no re-embedding — just edit the config and run `rag eval run`. This is the practical payoff of the configuration-driven architecture.

---

**End of Technical Specification Document v2.0.**