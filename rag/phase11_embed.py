"""Phase 11: Embedding generation with Chroma vector store.

Generates embeddings for chunks, summaries, and folders using Ollama,
stores vectors in Chroma persistent collections, and tracks references
in the embedding_ref tables.
"""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb
import ollama
from rich.console import Console
from rich.table import Table
from tenacity import retry, stop_after_attempt, wait_exponential

from pipeline.config import AppConfig, embedding_config_hash
from pipeline.db import finish_model_run, heartbeat_model_run, start_model_run
from pipeline.helpers import now_iso, record_failure

logger = logging.getLogger(__name__)
console = Console()

# Chroma collection prefixes
CHUNKS_PREFIX = "chunks__"
SUMMARIES_PREFIX = "summaries__"
FOLDERS_PREFIX = "folders__"

# Batch size for $in filter chunking (Chroma limitation)
IN_FILTER_BATCH = 1000

# Max chars per chunk text for embedding (Ollama context limit).
# Truncate oversized chunks; the embedding model only needs semantic content.
_MAX_CHUNK_CHARS = 8192


# ---------------------------------------------------------------------------
# Ollama embed with retry
# ---------------------------------------------------------------------------


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=30))
def _ollama_embed(
    host: str, model: str, texts: list[str], truncate_dim: int = 0,
    timeout: int = 300,
) -> list[list[float]]:
    """Call Ollama embed API and return list of vectors."""
    client = ollama.Client(host=host, timeout=timeout)
    kwargs: dict = {"model": model, "input": texts}
    if truncate_dim > 0:
        kwargs["options"] = {"truncate_dim": truncate_dim}
    resp = client.embed(**kwargs)
    vectors = resp.get("embeddings", [])
    if not vectors:
        raise ValueError("Ollama returned empty embeddings")
    return vectors


# ---------------------------------------------------------------------------
# Chroma collection helpers
# ---------------------------------------------------------------------------


def _ensure_collection(
    client: chromadb.PersistentClient,
    name: str,
    *,
    embedding_model: str,
    dim: int,
    config_hash: str,
    source_db_path: str,
    contextual_augmentation: bool = False,
) -> chromadb.Collection:
    """Get or create a Chroma collection with required metadata."""
    existing = [c.name for c in client.list_collections()]
    if name in existing:
        coll = client.get_collection(name)
        # Verify metadata consistency
        meta = coll.metadata or {}
        stored_model = meta.get("embedding_model")
        stored_dim = meta.get("dim")
        if stored_model and stored_model != embedding_model:
            raise ValueError(
                f"Collection '{name}' was created with model '{stored_model}', "
                f"cannot embed with '{embedding_model}'"
            )
        if stored_dim and stored_dim != dim:
            raise ValueError(f"Collection '{name}' has dim={stored_dim}, expected {dim}")
        return coll
    else:
        metadata = {
            "embedding_model": embedding_model,
            "dim": dim,
            "config_hash": config_hash,
            "created_at": now_iso(),
            "contextual_augmentation": contextual_augmentation,
            "source_db_path": source_db_path,
        }
        return client.create_collection(name=name, metadata=metadata)


def _get_dim_from_ollama(host: str, model: str) -> int:
    """Probe Ollama for the embedding model's dimension."""
    vectors = _ollama_embed(host, model, ["probe"])
    return len(vectors[0])


# ---------------------------------------------------------------------------
# Chunk embedding
# ---------------------------------------------------------------------------


def _embed_chunks(
    db,
    client: chromadb.PersistentClient,
    host: str,
    model: str,
    suffix: str,
    batch_size: int,
    truncate_dim: int,
    config_hash: str,
    reembed: bool,
    run_id: int | None = None,
    timeout: int = 300,
) -> int:
    """Embed chunks and return count of new embeddings."""
    collection_name = f"{CHUNKS_PREFIX}{suffix}"

    # Build SQL for chunks without embeddings for this model
    if reembed:
        sql = """
            SELECT c.id AS chunk_id, c.text, c.context_text,
                   c.file_id, c.metadata_json, f.folder_id, f.rel_path,
                   f.category AS file_category
            FROM chunk c
            JOIN file f ON f.id = c.file_id
            WHERE f.is_dup_primary = 1 AND f.excluded = 0
            ORDER BY c.id
        """
        rows = list(db.query(sql))
    else:
        sql = """
            SELECT c.id AS chunk_id, c.text, c.context_text,
                   c.file_id, c.metadata_json, f.folder_id, f.rel_path,
                   f.category AS file_category
            FROM chunk c
            JOIN file f ON f.id = c.file_id
            WHERE f.is_dup_primary = 1 AND f.excluded = 0
              AND c.id NOT IN (
                  SELECT chunk_id FROM embedding_ref WHERE embedding_model = ?
              )
            ORDER BY c.id
        """
        rows = list(db.query(sql, [model]))

    if not rows:
        logger.info("No chunks to embed.")
        return 0

    # Get folder-level metadata for efficient joins
    file_ids = [r["file_id"] for r in rows]
    folder_info = {}
    for fid_batch in range(0, len(file_ids), IN_FILTER_BATCH):
        batch_ids = file_ids[fid_batch : fid_batch + IN_FILTER_BATCH]
        placeholders = ",".join("?" for _ in batch_ids)
        file_rows = db.query(
            f"SELECT id, folder_id, rel_path, category, pronom_id FROM file WHERE id IN ({placeholders})",
            batch_ids,
        )
        for fr in file_rows:
            folder_info[fr["id"]] = dict(fr)

    folder_ids = list(set(fr["folder_id"] for fr in folder_info.values()))
    folder_meta = {}
    for fid_batch in range(0, len(folder_ids), IN_FILTER_BATCH):
        batch_ids = folder_ids[fid_batch : fid_batch + IN_FILTER_BATCH]
        placeholders = ",".join("?" for _ in batch_ids)
        folder_rows = db.query(
            f"SELECT id, rel_path, inferred_category, inferred_label FROM folder WHERE id IN ({placeholders})",
            batch_ids,
        )
        for fmr in folder_rows:
            folder_meta[fmr["id"]] = dict(fmr)

    # Discover dimension
    dim = _get_dim_from_ollama(host, model)

    coll = _ensure_collection(
        client,
        collection_name,
        embedding_model=model,
        dim=dim,
        config_hash=config_hash,
        source_db_path=str(Path.cwd() / "corpus.db"),
    )

    # When re-embedding, remove old refs for this model so new rows won't duplicate
    if reembed:
        db.execute("DELETE FROM embedding_ref WHERE embedding_model = ?", [model])

    embedded = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        texts = []
        for row in batch:
            ctx = row.get("context_text")
            raw = f"{ctx}\n\n{row['text']}" if ctx else row["text"]
            texts.append(raw[:_MAX_CHUNK_CHARS])

        vectors = _ollama_embed(host, model, texts, truncate_dim, timeout)

        ids = [f"c_{r['chunk_id']}" for r in batch]
        metadatas = []
        for r in batch:
            fi = folder_info.get(r["file_id"], {})
            fldr = folder_meta.get(fi.get("folder_id", 0), {})
            meta = {}
            meta["chunk_id"] = int(r["chunk_id"])
            meta["file_id"] = int(r["file_id"])
            meta["folder_id"] = int(fi.get("folder_id", 0) or 0)
            meta["rel_path"] = str(r.get("rel_path", ""))
            meta["inferred_category"] = str(fldr.get("inferred_category", "") or "")
            meta["inferred_label"] = str(fldr.get("inferred_label", "") or "")
            metadatas.append(meta)

        # Upsert into Chroma (add for new, update for existing)
        if reembed:
            coll.upsert(ids=ids, embeddings=vectors, metadatas=metadatas, documents=texts)
        else:
            coll.add(ids=ids, embeddings=vectors, metadatas=metadatas, documents=texts)

        # Record embedding_ref rows
        for r in batch:
            db["embedding_ref"].insert(
                {
                    "chunk_id": r["chunk_id"],
                    "vector_store": "chroma",
                    "collection": collection_name,
                    "external_id": f"c_{r['chunk_id']}",
                    "embedding_model": model,
                    "dim": dim,
                    "config_hash": config_hash,
                    "is_current": 1,
                    "created_at": now_iso(),
                }
            )

        embedded += len(batch)
        if run_id and embedded % 100 == 0:
            heartbeat_model_run(db, run_id, embedded, 0)
        logger.debug("Embedded chunks %d-%d", i, i + len(batch))

    return embedded


# ---------------------------------------------------------------------------
# Summary embedding
# ---------------------------------------------------------------------------


def _embed_summaries(
    db,
    client: chromadb.PersistentClient,
    host: str,
    model: str,
    suffix: str,
    batch_size: int,
    truncate_dim: int,
    config_hash: str,
    reembed: bool,
    run_id: int | None = None,
    timeout: int = 300,
) -> int:
    """Embed document summaries and return count."""
    collection_name = f"{SUMMARIES_PREFIX}{suffix}"

    if reembed:
        sql = """
            SELECT s.id AS summary_id, s.summary_text, s.file_id, s.model
            FROM summary s
            JOIN file f ON f.id = s.file_id
            WHERE f.is_dup_primary = 1 AND f.excluded = 0
            ORDER BY s.id
        """
        rows = list(db.query(sql))
    else:
        sql = """
            SELECT s.id AS summary_id, s.summary_text, s.file_id, s.model
            FROM summary s
            JOIN file f ON f.id = s.file_id
            WHERE f.is_dup_primary = 1 AND f.excluded = 0
              AND s.id NOT IN (
                  SELECT summary_id FROM summary_embedding_ref WHERE embedding_model = ?
              )
            ORDER BY s.id
        """
        rows = list(db.query(sql, [model]))

    if not rows:
        logger.info("No summaries to embed.")
        return 0

    dim = _get_dim_from_ollama(host, model)

    coll = _ensure_collection(
        client,
        collection_name,
        embedding_model=model,
        dim=dim,
        config_hash=config_hash,
        source_db_path=str(Path.cwd() / "corpus.db"),
    )

    embedded = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        texts = [r["summary_text"] for r in batch]
        vectors = _ollama_embed(host, model, texts, truncate_dim, timeout)

        ids = [f"s_{r['summary_id']}" for r in batch]
        metadatas = [
            {
                "summary_id": int(r["summary_id"]),
                "file_id": int(r["file_id"]),
                "model": str(r["model"] or ""),
            }
            for r in batch
        ]

        if reembed:
            coll.upsert(ids=ids, embeddings=vectors, metadatas=metadatas, documents=texts)
        else:
            coll.add(ids=ids, embeddings=vectors, metadatas=metadatas, documents=texts)

        for r in batch:
            db["summary_embedding_ref"].insert(
                {
                    "summary_id": r["summary_id"],
                    "vector_store": "chroma",
                    "collection": collection_name,
                    "external_id": f"s_{r['summary_id']}",
                    "embedding_model": model,
                    "dim": dim,
                    "config_hash": config_hash,
                    "is_current": 1,
                    "created_at": now_iso(),
                }
            )

        embedded += len(batch)
        if run_id and embedded % 100 == 0:
            heartbeat_model_run(db, run_id, embedded, 0)
        logger.debug("Embedded summaries %d-%d", i, i + len(batch))

    return embedded


# ---------------------------------------------------------------------------
# Folder embedding
# ---------------------------------------------------------------------------


def _embed_folders(
    db,
    client: chromadb.PersistentClient,
    host: str,
    model: str,
    suffix: str,
    batch_size: int,
    truncate_dim: int,
    config_hash: str,
    reembed: bool,
    run_id: int | None = None,
    timeout: int = 300,
) -> int:
    """Embed folder metadata and return count."""
    collection_name = f"{FOLDERS_PREFIX}{suffix}"

    if reembed:
        sql = """
            SELECT id, rel_path, inferred_category, inferred_label, inferred_summary
            FROM folder
            WHERE excluded = 0
            ORDER BY id
        """
        rows = list(db.query(sql))
    else:
        sql = """
            SELECT f.id, f.rel_path, f.inferred_category, f.inferred_label, f.inferred_summary
            FROM folder f
            WHERE f.excluded = 0
              AND f.id NOT IN (
                  SELECT folder_id FROM folder_embedding_ref WHERE embedding_model = ?
              )
            ORDER BY f.id
        """
        rows = list(db.query(sql, [model]))

    if not rows:
        logger.info("No folders to embed.")
        return 0

    dim = _get_dim_from_ollama(host, model)

    coll = _ensure_collection(
        client,
        collection_name,
        embedding_model=model,
        dim=dim,
        config_hash=config_hash,
        source_db_path=str(Path.cwd() / "corpus.db"),
    )

    embedded = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        texts = []
        for r in batch:
            parts = [
                r["rel_path"],
                r["inferred_category"] or "",
                r["inferred_label"] or "",
                r["inferred_summary"] or "",
            ]
            texts.append(" ".join(p for p in parts if p))

        vectors = _ollama_embed(host, model, texts, truncate_dim, timeout)

        ids = [f"f_{r['id']}" for r in batch]
        metadatas = [
            {
                "folder_id": int(r["id"]),
                "rel_path": str(r["rel_path"] or ""),
                "inferred_category": str(r["inferred_category"] or ""),
                "inferred_label": str(r["inferred_label"] or ""),
            }
            for r in batch
        ]

        if reembed:
            coll.upsert(ids=ids, embeddings=vectors, metadatas=metadatas, documents=texts)
        else:
            coll.add(ids=ids, embeddings=vectors, metadatas=metadatas, documents=texts)

        for r in batch:
            db["folder_embedding_ref"].insert(
                {
                    "folder_id": r["id"],
                    "vector_store": "chroma",
                    "collection": collection_name,
                    "external_id": f"f_{r['id']}",
                    "embedding_model": model,
                    "dim": dim,
                    "config_hash": config_hash,
                    "is_current": 1,
                    "created_at": now_iso(),
                }
            )

        embedded += len(batch)
        if run_id and embedded % 100 == 0:
            heartbeat_model_run(db, run_id, embedded, 0)
        logger.debug("Embedded folders %d-%d", i, i + len(batch))

    return embedded


# ---------------------------------------------------------------------------
# Public: run_embed
# ---------------------------------------------------------------------------


def run_embed(
    db,
    cfg: AppConfig,
    model: str | None = None,
    collections: list[str] | None = None,
    reembed: bool = False,
) -> int:
    """Embed chunks, summaries, and folders. Returns number of new embeddings.

    Args:
        db: sqlite_utils Database instance.
        cfg: AppConfig with embedding and Ollama settings.
        model: Override model name. Uses config default if None.
        collections: Which collections to embed. None = all of ['chunks', 'summaries', 'folders'].
        reembed: If True, re-embed even already-embedded items.
    """
    emb = cfg.models.embedding
    model_name = model or emb.name
    suffix = emb.collection_suffix
    batch_size = emb.batch_size
    truncate_dim = emb.truncate_dim
    c_hash = embedding_config_hash(emb)

    chroma_path = str(Path(cfg.paths.chroma_root).expanduser().resolve())
    client = chromadb.PersistentClient(path=chroma_path)

    if collections is None:
        collections = ["chunks", "summaries", "folders"]

    run_id = start_model_run(
        db,
        role="embedding",
        model_name=model_name,
        config_hash=c_hash,
        phase="embed",
    )

    total = 0
    failed = 0

    if "chunks" in collections:
        try:
            n = _embed_chunks(
                db,
                client,
                cfg.ollama.host,
                model_name,
                suffix,
                batch_size,
                truncate_dim,
                c_hash,
                reembed,
                run_id=run_id,
                timeout=cfg.ollama.timeout_seconds,
            )
            total += n
        except Exception as e:
            record_failure(
                db,
                phase="embed",
                tool=model_name,
                error_class=type(e).__name__,
                error_message=str(e),
            )
            failed += 1
            logger.error("Chunk embedding failed: %s", e)

    if "summaries" in collections:
        try:
            n = _embed_summaries(
                db,
                client,
                cfg.ollama.host,
                model_name,
                suffix,
                batch_size,
                truncate_dim,
                c_hash,
                reembed,
                run_id=run_id,
                timeout=cfg.ollama.timeout_seconds,
            )
            total += n
        except Exception as e:
            record_failure(
                db,
                phase="embed",
                tool=model_name,
                error_class=type(e).__name__,
                error_message=str(e),
            )
            failed += 1
            logger.error("Summary embedding failed: %s", e)

    if "folders" in collections:
        try:
            n = _embed_folders(
                db,
                client,
                cfg.ollama.host,
                model_name,
                suffix,
                batch_size,
                truncate_dim,
                c_hash,
                reembed,
                run_id=run_id,
                timeout=cfg.ollama.timeout_seconds,
            )
            total += n
        except Exception as e:
            record_failure(
                db,
                phase="embed",
                tool=model_name,
                error_class=type(e).__name__,
                error_message=str(e),
            )
            failed += 1
            logger.error("Folder embedding failed: %s", e)

    finish_model_run(
        db,
        run_id,
        status="done",
        items_processed=total,
        items_failed=failed,
    )

    logger.info("Embedding complete: %d total embeddings, %d failures", total, failed)
    return total


# ---------------------------------------------------------------------------
# Public: run_embed_switch_to
# ---------------------------------------------------------------------------


def run_embed_switch_to(
    db,
    cfg: AppConfig,
    config_path: Path,
    model_name: str | None = None,
    suffix: str | None = None,
) -> None:
    """Embed against a new model, flip is_current atomically, rewrite config.toml.

    Args:
        db: sqlite_utils Database instance.
        cfg: Current AppConfig.
        config_path: Path to config.toml to rewrite.
        model_name: New embedding model name.
        suffix: New collection suffix.
    """
    emb = cfg.models.embedding
    new_model = model_name or emb.name
    new_suffix = suffix or emb.collection_suffix
    new_batch_size = emb.batch_size
    new_truncate_dim = emb.truncate_dim
    _ = embedding_config_hash(
        type(emb)(
            name=new_model,
            collection_suffix=new_suffix,
            batch_size=new_batch_size,
            normalize=emb.normalize,
            truncate_dim=new_truncate_dim,
        )
    )

    console.print(f"[bold]Switching to model '{new_model}' (suffix: '{new_suffix}')[/]")

    # Build a temporary config override for the new model
    class _TempEmbeddingConfig:
        def __init__(self):
            self.name = new_model
            self.collection_suffix = new_suffix
            self.batch_size = new_batch_size
            self.normalize = emb.normalize
            self.truncate_dim = new_truncate_dim

        def model_dump(self):
            return {
                "name": self.name,
                "collection_suffix": self.collection_suffix,
                "batch_size": self.batch_size,
                "normalize": self.normalize,
                "truncate_dim": self.truncate_dim,
            }

    temp_cfg = _TempEmbeddingConfig()
    temp_c_hash = embedding_config_hash(temp_cfg)

    # Embed into new collections
    chroma_path = str(Path(cfg.paths.chroma_root).expanduser().resolve())
    client = chromadb.PersistentClient(path=chroma_path)

    console.print("Embedding chunks with new model...")
    chunk_count = _embed_chunks(
        db,
        client,
        cfg.ollama.host,
        new_model,
        new_suffix,
        new_batch_size,
        new_truncate_dim,
        temp_c_hash,
        reembed=False,
    )
    console.print(f"  Embedded {chunk_count} chunks")

    console.print("Embedding summaries with new model...")
    summary_count = _embed_summaries(
        db,
        client,
        cfg.ollama.host,
        new_model,
        new_suffix,
        new_batch_size,
        new_truncate_dim,
        temp_c_hash,
        reembed=False,
    )
    console.print(f"  Embedded {summary_count} summaries")

    console.print("Embedding folders with new model...")
    folder_count = _embed_folders(
        db,
        client,
        cfg.ollama.host,
        new_model,
        new_suffix,
        new_batch_size,
        new_truncate_dim,
        temp_c_hash,
        reembed=False,
    )
    console.print(f"  Embedded {folder_count} folders")

    # Flip is_current atomically
    console.print("Switching current embeddings...")
    with db.conn:
        db.conn.execute(
            "UPDATE embedding_ref SET is_current = 0 WHERE embedding_model != ?", [new_model]
        )
        db.conn.execute(
            "UPDATE embedding_ref SET is_current = 1 WHERE embedding_model = ?", [new_model]
        )
        db.conn.execute(
            "UPDATE folder_embedding_ref SET is_current = 0 WHERE embedding_model != ?", [new_model]
        )
        db.conn.execute(
            "UPDATE folder_embedding_ref SET is_current = 1 WHERE embedding_model = ?", [new_model]
        )
        db.conn.execute(
            "UPDATE summary_embedding_ref SET is_current = 0 WHERE embedding_model != ?",
            [new_model],
        )
        db.conn.execute(
            "UPDATE summary_embedding_ref SET is_current = 1 WHERE embedding_model = ?", [new_model]
        )
        db.conn.commit()

    # Rewrite config.toml
    _rewrite_config_embedding(config_path, new_model, new_suffix)

    console.print(
        f"[green]Switch complete. "
        f"Chunks: {chunk_count}, Summaries: {summary_count}, Folders: {folder_count}[/]"
    )


def _rewrite_config_embedding(config_path: Path, model: str, suffix: str) -> None:
    """Rewrite config.toml with new embedding model and suffix, preserving other content."""
    try:
        import tomli_w
    except ImportError:
        raise ImportError(
            "tomli-w is required to rewrite config.toml. Install with: pip install tomli-w"
        ) from None

    # Read existing config
    import tomllib

    with open(config_path, "rb") as f:
        raw = tomllib.load(f)

    # Update embedding model
    if "models" not in raw:
        raw["models"] = {}
    if "embedding" not in raw["models"]:
        raw["models"]["embedding"] = {}

    raw["models"]["embedding"]["name"] = model
    raw["models"]["embedding"]["collection_suffix"] = suffix

    # Write back (tomli-w doesn't preserve comments, but preserves structure)
    with open(config_path, "w") as f:
        tomli_w.dump(raw, f)


# ---------------------------------------------------------------------------
# Public: run_embed_gc
# ---------------------------------------------------------------------------


def run_embed_gc(db, cfg: AppConfig) -> None:
    """Delete Chroma collections with no is_current=1 rows."""
    chroma_path = str(Path(cfg.paths.chroma_root).expanduser().resolve())
    client = chromadb.PersistentClient(path=chroma_path)

    # Find all collections referenced in embedding_ref tables
    all_collections = set()

    for table in ["embedding_ref", "folder_embedding_ref", "summary_embedding_ref"]:
        rows = list(db.query(f"SELECT DISTINCT collection FROM {table}"))
        for r in rows:
            all_collections.add(r["collection"])

    # Find collections that have is_current=1 rows
    current_collections = set()
    for table in ["embedding_ref", "folder_embedding_ref", "summary_embedding_ref"]:
        rows = list(db.query(f"SELECT DISTINCT collection FROM {table} WHERE is_current = 1"))
        for r in rows:
            current_collections.add(r["collection"])

    # Collections to garbage collect
    gc_candidates = all_collections - current_collections

    if not gc_candidates:
        console.print("[green]No collections to garbage collect.[/]")
        return

    # Show candidates
    table = Table(title="Candidate collections for deletion")
    table.add_column("Collection")
    for name in sorted(gc_candidates):
        table.add_row(name)
    console.print(table)

    # Prompt before deletion
    confirm = console.input("\nDelete these collections? [bold red](y/N)[/]: ").strip().lower()

    if confirm not in ("y", "yes"):
        console.print("[yellow]Aborted.[/]")
        return

    # Delete from Chroma
    deleted = 0
    for name in sorted(gc_candidates):
        try:
            client.delete_collection(name)
            deleted += 1
            logger.info("Deleted Chroma collection '%s'", name)
        except Exception as e:
            console.print(f"[red]Failed to delete '{name}': {e}[/]")

    # Clean up DB references
    with db.conn:
        for table in ["embedding_ref", "folder_embedding_ref", "summary_embedding_ref"]:
            db.conn.execute(
                f"DELETE FROM {table} WHERE collection IN ({','.join('?' for _ in gc_candidates)})",
                list(gc_candidates),
            )
        db.conn.commit()

    console.print(f"[green]Deleted {deleted} collection(s).[/]")


# ---------------------------------------------------------------------------
# Public: run_embed_list
# ---------------------------------------------------------------------------


def run_embed_list(db, cfg: AppConfig) -> list[dict]:
    """Show all Chroma collections and their current status."""
    chroma_path = str(Path(cfg.paths.chroma_root).expanduser().resolve())
    client = chromadb.PersistentClient(path=chroma_path)

    # Query all embedding_ref tables
    results = []

    # Chunk embeddings
    rows = list(
        db.query("""
        SELECT er.collection, er.embedding_model, er.dim, er.is_current,
               er.vector_store, COUNT(*) AS chunk_count
        FROM embedding_ref er
        GROUP BY er.collection, er.embedding_model, er.dim, er.is_current, er.vector_store
    """)
    )
    for r in rows:
        results.append(
            {
                "collection": r["collection"],
                "model": r["embedding_model"],
                "dim": r["dim"],
                "is_current": bool(r["is_current"]),
                "vector_store": r["vector_store"],
                "chunk_count": r["chunk_count"],
                "summary_count": 0,
                "folder_count": 0,
            }
        )

    # Summary embeddings
    rows = list(
        db.query("""
        SELECT ser.collection, ser.embedding_model, ser.dim, ser.is_current,
               ser.vector_store, COUNT(*) AS summary_count
        FROM summary_embedding_ref ser
        GROUP BY ser.collection, ser.embedding_model, ser.dim, ser.is_current, ser.vector_store
    """)
    )
    for r in rows:
        # Merge with existing result if same collection+model
        key = (r["collection"], r["embedding_model"])
        existing = next((x for x in results if (x["collection"], x["model"]) == key), None)
        if existing:
            existing["summary_count"] = r["summary_count"]
        else:
            results.append(
                {
                    "collection": r["collection"],
                    "model": r["embedding_model"],
                    "dim": r["dim"],
                    "is_current": bool(r["is_current"]),
                    "vector_store": r["vector_store"],
                    "chunk_count": 0,
                    "summary_count": r["summary_count"],
                    "folder_count": 0,
                }
            )

    # Folder embeddings
    rows = list(
        db.query("""
        SELECT fer.collection, fer.embedding_model, fer.dim, fer.is_current,
               fer.vector_store, COUNT(*) AS folder_count
        FROM folder_embedding_ref fer
        GROUP BY fer.collection, fer.embedding_model, fer.dim, fer.is_current, fer.vector_store
    """)
    )
    for r in rows:
        key = (r["collection"], r["embedding_model"])
        existing = next((x for x in results if (x["collection"], x["model"]) == key), None)
        if existing:
            existing["folder_count"] = r["folder_count"]
        else:
            results.append(
                {
                    "collection": r["collection"],
                    "model": r["embedding_model"],
                    "dim": r["dim"],
                    "is_current": bool(r["is_current"]),
                    "vector_store": r["vector_store"],
                    "chunk_count": 0,
                    "summary_count": 0,
                    "folder_count": r["folder_count"],
                }
            )

    # Also list collections from Chroma that may not be in DB
    chroma_collections = {c.name for c in client.list_collections()}
    db_collections = {r["collection"] for r in results}
    for name in chroma_collections - db_collections:
        results.append(
            {
                "collection": name,
                "model": "(unknown)",
                "dim": 0,
                "is_current": False,
                "vector_store": "chroma",
                "chunk_count": 0,
                "summary_count": 0,
                "folder_count": 0,
            }
        )

    # Display table
    table = Table(title="Embedding collections")
    table.add_column("Collection")
    table.add_column("Model")
    table.add_column("Dim")
    table.add_column("Current")
    table.add_column("Chunks")
    table.add_column("Summaries")
    table.add_column("Folders")

    for r in sorted(results, key=lambda x: x["collection"]):
        table.add_row(
            r["collection"],
            r["model"],
            str(r["dim"]),
            "yes" if r["is_current"] else "no",
            str(r["chunk_count"]),
            str(r["summary_count"]),
            str(r["folder_count"]),
        )

    console.print(table)
    return results
