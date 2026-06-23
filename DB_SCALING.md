# Database Layer — Scaling Strategy

Date: 2026-05-05
Current stack: SQLite (metadata) + ChromaDB (vectors) + FTS5 (text search)

---

## Current Architecture

The system uses three storage layers, each with a distinct role:

| Component | Role | Current choice |
|-----------|------|----------------|
| Metadata store | File inventory, categories, dedup, format, provenance, pipeline runs | SQLite 3 |
| Vector store | Embeddings for retrieval | ChromaDB (`PersistentClient`) |
| Full-text search | Text search over extracted content | SQLite FTS5 (`chunk_fts`) |

This separation — relational metadata + specialized vector DB — is the industry-standard pattern. LangChain, LlamaIndex, and Haystack all assume this split.

---

## Where SQLite Is the Right Choice Today

- **Single corpus, 2177 files, batch pipeline.** No concurrent writers, no multi-tenant access.
- **Zero ops.** No server to manage, no connection pool, no migration tool beyond a Python script.
- **Easy backups.** `cp corpus.db corpus.db.bak` — atomic with WAL mode.
- **ACID guarantees.** Foreign keys, transactions, constraints all work.
- **Air-gap capable.** No network dependency.

The file table at 10M rows is well within SQLite's capability. The practical bottleneck will be FTS5 index size and query latency on unindexed text columns before hitting any SQLite limit.

---

## Where SQLite Hits Limits

### Single database file
One `corpus.db` = one corpus. Multiple projects means multiple files. This works fine for hundreds of projects but breaks down when you need:
- **Cross-project queries** — search across all clients or corpora at once
- **Centralized access control** — row-level permissions (who can read which corpus)
- **Connection sharing** across processes on different machines

### Write concurrency
SQLite allows unlimited readers but only **one writer at a time**. WAL mode improves reader/writer concurrency but writes are still serialized. This is fine for the current sequential batch pipeline. It becomes a bottleneck if:
- Multiple ingestion pipelines write to the same DB concurrently
- A query API has high write-throughput (e.g. user feedback logging concurrent with extraction)
- Real-time indexing requires simultaneous reads and writes from different services

### Distributed deployment
SQLite is local-only. Multi-machine setups require a network-accessible database or a replication layer (LiteFS).

---

## Scaling Triggers and Migration Targets

### Trigger 1: Multiple concurrent writers
**Signal:** "database is locked" errors despite retry logic, or write queueing slows pipeline phases.

**Move to:** PostgreSQL

**Why:** True MVCC — readers never block writers, writers never block readers. Connection pooling, connection sharing across processes and machines.

**Migration path:** Straightforward. The current schema uses standard SQL — no SQLite-specific features beyond `AUTOINCREMENT` (maps to `SERIAL`) and `CHECK` constraints (identical). FTS5 maps to PostgreSQL's built-in full-text search (`tsvector` / `tsquery`).

### Trigger 2: Cross-project queries or multi-tenant access control
**Signal:** Need to search across corpora, or need per-user/per-project read permissions.

**Move to:** PostgreSQL with row-level security (RLS)

**Why:** Single database, row-level `SELECT` permissions. `ALTER TABLE file ENABLE ROW LEVEL SECURITY` + policies per tenant. SQLite has no equivalent.

**Migration path:** Add a `project_id` column to `file`, `folder`, and all related tables. Create a `project` table. Set up RLS policies. The application layer passes the current project context to the DB session.

### Trigger 3: Full-text search becomes a bottleneck
**Signal:** FTS5 queries over extracted text take >1s, or the FTS index size approaches the SQLite file size limit for acceptable performance.

**Move to:** Elasticsearch or OpenSearch

**Why:** Distributed search, relevance tuning (BM25 weights, field boosting), multi-language analyzers, faceted search. FTS5 is good but not tunable.

**Migration path:** Keep SQLite for metadata. Ship extracted text to Elasticsearch via a bulk indexer. Change search queries to hit ES instead of FTS5. Join results back to SQLite by `file_id`.

### Trigger 4: Vector store outgrows single machine
**Signal:** ChromaDB `PersistentClient` hits memory limits, or HNSW index build time becomes prohibitive.

**Move to:** Qdrant, Weaviate, or pgvector

**Why:** Distributed vectors, horizontal scaling, native HNSW at scale, multi-tenant collections.

**Migration path:** This is the harder switch — embeddings are tied to the vector store's index format. Export existing embeddings, re-import to the new store. The `embedding_ref` table already abstracts the collection reference, which helps.

---

## Recommended Sequence

| Phase | Metadata | Vector store | Text search | When |
|-------|----------|-------------|-------------|------|
| **Now** | SQLite | ChromaDB (PersistentClient) | SQLite FTS5 | Single project, batch pipeline |
| **2nd project** | SQLite (one DB per project) | ChromaDB (PersistentClient) | SQLite FTS5 | Thin API layer routes by project ID |
| **Concurrent writes** | PostgreSQL | ChromaDB (PersistentClient or server) | SQLite FTS5 or pg full-text | Multiple ingestion pipelines, query API |
| **Cross-project search** | PostgreSQL + RLS | ChromaDB server or Qdrant | Elasticsearch | Multi-tenant, cross-corpus queries |
| **Scale vectors** | PostgreSQL | Qdrant / Weaviate | Elasticsearch | Distributed vectors, 100K+ chunks |

---

## Preparation Steps (Do Now, Regardless of Migration)

These changes make the eventual migration easier and are valuable regardless:

1. **No SQLite-specific SQL in application code.** Use parameterized queries, standard SQL syntax. Avoid `PRAGMA` in application logic (keep it in `db.py` connection setup).

2. **Abstract the DB connection.** The current `Database` wrapper in `db.py` is good — if migrating, this is the single file that changes.

3. **Keep FTS5 usage isolated.** The `chunk_fts` virtual table is only queried from retrieval code. If migrating to Elasticsearch, only that module changes.

4. **The `embedding_ref` table is the right abstraction.** It decouples the application from the vector store's internal IDs. Keep this pattern.

5. **Avoid `sqlite_utils`-specific APIs in phase code.** The `db["table"].insert()`, `db["table"].update()` pattern is sqlite_utils-specific. For migration, ensure critical writes use `db.execute()` with standard SQL so the translation layer is smaller.
