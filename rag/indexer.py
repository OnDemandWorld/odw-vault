"""Incremental indexing engine for the Vault RAG pipeline.

Provides single-file and bulk sync operations that detect changes via
SHA-256 hashing and run modified files through the extraction -> summarization
-> chunking -> embedding pipeline.  Idempotent: running sync_file() twice on
the same unchanged file produces no duplicate records.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from pipeline.config import AppConfig, embedding_config_hash
from pipeline.helpers import now_iso, record_failure

logger = logging.getLogger(__name__)

# Sentence boundary regex — mirrors rag.phase10_chunk
_SENTENCE_RE = re.compile(r"(?<=[.!?。！？\n])\s+")  # noqa: RUF001

# Extension -> (pronom_id, mime_type, category, extract_strategy)
_EXT_MAP: dict[str, tuple[str, str, str, str]] = {
    ".txt": ("x-fmt/111", "text/plain", "document", "textutil"),
    ".md": ("x-fmt/111", "text/plain", "document", "textutil"),
    ".csv": ("x-fmt/18", "text/csv", "data", "textutil"),
    ".json": ("fmt/817", "application/json", "data", "textutil"),
    ".pdf": ("fmt/16", "application/pdf", "document", "docling"),
    ".png": ("fmt/13", "image/png", "image", "metadata-only"),
    ".jpg": ("fmt/43", "image/jpeg", "image", "metadata-only"),
    ".jpeg": ("fmt/43", "image/jpeg", "image", "metadata-only"),
    ".docx": (
        "fmt/412",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "document",
        "docling",
    ),
    ".html": ("fmt/96", "text/html", "document", "textutil"),
    ".xml": ("fmt/101", "application/xml", "data", "textutil"),
}

# Max extracted text stored in DB (1 MB)
_MAX_EXTRACT_TEXT_DB = 1_000_000

# Max chars per chunk text sent to the embedding model
_MAX_CHUNK_CHARS = 8192


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 of a file in 8 KB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _last_rowid(db) -> int:
    """Return last inserted rowid (works around sqlite_utils wrapper quirk)."""
    return db.conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def _ensure_folder(db, folder_path: Path, corpus_root: Path) -> int:
    """Ensure a folder record exists.  Returns folder_id."""
    rel = str(folder_path.relative_to(corpus_root)) if folder_path != corpus_root else "."
    row = next(iter(db.query("SELECT id FROM folder WHERE path = ?", [str(folder_path)])), None)
    if row:
        return row["id"]

    parent_id = None
    if folder_path != corpus_root and folder_path.parent != folder_path:
        parent_id = _ensure_folder(db, folder_path.parent, corpus_root)

    db["folder"].insert(
        {
            "path": str(folder_path),
            "rel_path": rel,
            "name": folder_path.name or ".",
            "depth": 0 if rel == "." else rel.count("/"),
            "parent_id": parent_id,
            "excluded": 0,
        }
    )
    folder_id = _last_rowid(db)
    db.conn.commit()
    return folder_id


def _identify_format(file_path: Path) -> tuple[str, str, str, str]:
    """Return (pronom_id, mime_type, category, extract_strategy) by extension."""
    ext = file_path.suffix.lower()
    if ext in _EXT_MAP:
        return _EXT_MAP[ext]
    return ("UNKNOWN", "application/octet-stream", "unknown", "filename-only")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences (mirrors phase10_chunk)."""
    raw = _SENTENCE_RE.split(text)
    return [s for s in raw if s.strip()]


def _token_estimate(text: str) -> int:
    """Rough token count estimate."""
    return max(1, len(text) // 4)


def _cleanup_file_derivatives(db, file_id: int) -> None:
    """Delete all derived data for a file (chunks, embeddings, extraction, summary).

    Does NOT delete the file row itself.
    """
    # Chroma deletion is handled by the caller before invoking this.

    # Chunk FTS
    db.execute(
        "DELETE FROM chunk_fts WHERE rowid IN "
        "(SELECT id FROM chunk WHERE file_id = ?)",
        [file_id],
    )
    # Embedding refs for chunks
    db.execute(
        "DELETE FROM embedding_ref WHERE chunk_id IN "
        "(SELECT id FROM chunk WHERE file_id = ?)",
        [file_id],
    )
    # Chunks
    db.execute("DELETE FROM chunk WHERE file_id = ?", [file_id])
    # Extraction
    db.execute("DELETE FROM extraction WHERE file_id = ?", [file_id])
    # Summary + its embedding refs
    db.execute(
        "DELETE FROM summary_embedding_ref WHERE summary_id IN "
        "(SELECT id FROM summary WHERE file_id = ?)",
        [file_id],
    )
    db.execute("DELETE FROM summary WHERE file_id = ?", [file_id])
    # Failures
    db.execute("DELETE FROM failure WHERE file_id = ?", [file_id])
    db.conn.commit()


# ---------------------------------------------------------------------------
# IncrementalIndexer
# ---------------------------------------------------------------------------


class IncrementalIndexer:
    """Handles incremental index updates for file changes."""

    def __init__(self, db, cfg: AppConfig, chroma_client=None):
        self.db = db
        self.cfg = cfg
        self.chroma_client = chroma_client

    # ------------------------------------------------------------------
    # sync_file
    # ------------------------------------------------------------------

    def sync_file(self, file_path: Path) -> dict:
        """Process a single new or modified file through the pipeline.

        Returns a dict with *status*, *file_id*, and optional *error*.

        Steps:
          1. Compute SHA-256 hash
          2. Check if file already exists with same hash (skip if unchanged)
          3. If new or changed: insert/update file record
          4. Run format identification (extension-based)
          5. Run extraction (using appropriate strategy)
          6. Run summarization (if text >= threshold)
          7. Run chunking
          8. Run embedding
          9. Return result
        """
        file_path = Path(file_path).resolve()
        corpus_root = self.cfg.corpus_root_path

        # --- 1. Hash ---
        try:
            new_hash = _sha256_file(file_path)
        except OSError as exc:
            logger.error("Cannot read %s: %s", file_path, exc)
            return {"status": "error", "error": str(exc)}

        # --- 2. Check existing ---
        existing = next(
            iter(self.db.query("SELECT id, sha256 FROM file WHERE path = ?", [str(file_path)])),
            None,
        )

        if existing and existing["sha256"] == new_hash:
            logger.debug("File unchanged, skipping: %s", file_path)
            return {"status": "skipped_unchanged", "file_id": existing["id"]}

        # --- 3. Insert / update file record ---
        folder_id = _ensure_folder(self.db, file_path.parent, corpus_root)
        pronom_id, mime_type, category, extract_strategy = _identify_format(file_path)
        stat = file_path.stat()

        if existing:
            file_id = existing["id"]
            # Clean up old derived data (Chroma + SQLite) before reprocessing
            self._remove_chroma_vectors(file_id)
            _cleanup_file_derivatives(self.db, file_id)
            mtime_iso = datetime.fromtimestamp(stat.st_mtime, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
            self.db.execute(
                """UPDATE file SET sha256=?, size_bytes=?, extension=?,
                       pronom_id=?, mime_type=?, category=?, extract_strategy=?,
                       mtime=?, hash_status='done', identify_status='done'
                    WHERE id=?""",
                [
                    new_hash, stat.st_size, file_path.suffix.lower(),
                    pronom_id, mime_type, category, extract_strategy,
                    mtime_iso,
                    file_id,
                ],
            )
            self.db.conn.commit()
            action = "updated"
        else:
            mtime_iso = datetime.fromtimestamp(stat.st_mtime, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
            self.db["file"].insert(
                {
                    "folder_id": folder_id,
                    "path": str(file_path),
                    "rel_path": str(file_path.relative_to(corpus_root)),
                    "name": file_path.name,
                    "extension": file_path.suffix.lower(),
                    "size_bytes": stat.st_size,
                    "mtime": mtime_iso,
                    "sha256": new_hash,
                    "pronom_id": pronom_id,
                    "mime_type": mime_type,
                    "category": category,
                    "extract_strategy": extract_strategy,
                    "is_dup_primary": 1,
                    "excluded": 0,
                    "hash_status": "done",
                    "identify_status": "done",
                    "triage_status": "done",
                }
            )
            self.db.conn.commit()
            file_id = _last_rowid(self.db)
            action = "created"

        # --- 4-8. Pipeline stages (each wrapped in try/except) ---
        try:
            self._run_extraction(file_id, file_path, extract_strategy)
            self._run_summarization(file_id)
            self._run_chunking(file_id)
            self._run_embedding(file_id)
        except Exception as exc:
            logger.error("Pipeline error for %s: %s", file_path, exc)
            record_failure(
                self.db,
                file_id=file_id,
                phase="indexer",
                error_class=type(exc).__name__,
                error_message=str(exc),
            )
            return {"status": "error", "file_id": file_id, "error": str(exc)}

        logger.info("Synced file %s (id=%d, action=%s)", file_path, file_id, action)
        return {"status": "success", "file_id": file_id, "action": action}

    # ------------------------------------------------------------------
    # remove_file
    # ------------------------------------------------------------------

    def remove_file(self, file_id: int) -> dict:
        """Remove a file and all its derived data from DB and Chroma."""
        file_row = next(
            iter(self.db.query("SELECT id, path FROM file WHERE id = ?", [file_id])), None
        )
        if not file_row:
            return {"status": "not_found", "file_id": file_id}

        # 1. Delete from Chroma
        self._remove_chroma_vectors(file_id)

        # 2. Delete from SQLite (cascading)
        _cleanup_file_derivatives(self.db, file_id)
        self.db.execute("DELETE FROM file WHERE id = ?", [file_id])
        self.db.conn.commit()

        logger.info("Removed file_id=%d", file_id)
        return {"status": "removed", "file_id": file_id}

    # ------------------------------------------------------------------
    # sync_all
    # ------------------------------------------------------------------

    def sync_all(self) -> dict:
        """Full sync: detect all changes and process them.

        1. Walk corpus_root, compute hashes for all files
        2. Compare with DB: find new, modified, deleted files
        3. Process new/modified files
        4. Remove deleted files
        5. Return summary
        """
        corpus_root = self.cfg.corpus_root_path
        if not corpus_root.exists():
            return {"error": f"Corpus root does not exist: {corpus_root}"}

        # --- 1. Walk & hash ---
        disk_files: dict[str, str] = {}  # absolute path -> sha256
        for dirpath, dirnames, filenames in os.walk(str(corpus_root)):
            dp = Path(dirpath)
            # Skip hidden / cache directories
            dirnames[:] = [
                d for d in dirnames
                if not d.startswith(".") and d != "__MACOSX"
            ]
            for fname in filenames:
                if fname.startswith(".") or fname in {".DS_Store", "Thumbs.db"}:
                    continue
                fp = dp / fname
                try:
                    disk_files[str(fp)] = _sha256_file(fp)
                except OSError:
                    logger.warning("Cannot read %s, skipping", fp)

        # --- 2. Compare with DB ---
        db_files: dict[str, str] = {}
        for row in self.db.query("SELECT path, sha256 FROM file"):
            db_files[row["path"]] = row["sha256"]

        new_paths = [p for p in disk_files if p not in db_files]
        modified_paths = [
            p for p in disk_files
            if p in db_files and db_files[p] != disk_files[p]
        ]
        deleted_paths = [p for p in db_files if p not in disk_files]

        logger.info(
            "Sync detection: %d new, %d modified, %d deleted",
            len(new_paths), len(modified_paths), len(deleted_paths),
        )

        # --- 3. Process new / modified ---
        processed = 0
        skipped = 0
        failed = 0

        for p in new_paths + modified_paths:
            result = self.sync_file(Path(p))
            if result["status"] == "success":
                processed += 1
            elif result["status"] == "skipped_unchanged":
                skipped += 1
            else:
                failed += 1

        # --- 4. Remove deleted ---
        removed = 0
        for p in deleted_paths:
            row = next(
                iter(self.db.query("SELECT id FROM file WHERE path = ?", [p])), None
            )
            if row:
                res = self.remove_file(row["id"])
                if res["status"] == "removed":
                    removed += 1

        summary = {
            "new": len(new_paths),
            "modified": len(modified_paths),
            "deleted": len(deleted_paths),
            "processed": processed,
            "skipped": skipped,
            "removed": removed,
            "failed": failed,
        }
        logger.info("Sync complete: %s", summary)
        return summary

    # ------------------------------------------------------------------
    # Private: Chroma helpers
    # ------------------------------------------------------------------

    def _remove_chroma_vectors(self, file_id: int) -> None:
        """Remove all Chroma vectors associated with a file's chunks."""
        if not self.chroma_client:
            return

        chunk_ids = [
            r["id"]
            for r in self.db.query("SELECT id FROM chunk WHERE file_id = ?", [file_id])
        ]
        if not chunk_ids:
            return

        suffix = self.cfg.models.embedding.collection_suffix
        for prefix in ("chunks__", "summaries__"):
            coll_name = f"{prefix}{suffix}"
            try:
                coll = self.chroma_client.get_collection(coll_name)
                chroma_ids = [f"{'c' if prefix == 'chunks__' else 's'}_{cid}" for cid in chunk_ids]
                coll.delete(ids=chroma_ids)
            except Exception:
                logger.debug("Chroma delete skipped for %s", coll_name)

    # ------------------------------------------------------------------
    # Private: pipeline stages
    # ------------------------------------------------------------------

    def _run_extraction(self, file_id: int, file_path: Path, strategy: str) -> None:
        """Phase 8 — text extraction."""
        from rag.phase8_extract import EXTRACTOR_MAP

        if strategy in ("skip", "manual", "unsupported"):
            logger.debug("Skipping extraction for strategy=%s", strategy)
            return

        extractor_fn = EXTRACTOR_MAP.get(strategy)
        if not extractor_fn:
            logger.warning("No extractor for strategy '%s', using filename-only", strategy)
            from rag.extractors.filename_only_extractor import extract_filename_only
            extractor_fn = extract_filename_only

        try:
            result = extractor_fn(str(file_path))
            text = result.get("text", "") or ""
            tool = result.get("tool", strategy)
            page_count = result.get("page_count")

            stored_text = text[:_MAX_EXTRACT_TEXT_DB]
            self.db["extraction"].insert(
                {
                    "file_id": file_id,
                    "tool": tool,
                    "text_extracted": stored_text if stored_text else None,
                    "char_count": len(text),
                    "page_count": page_count,
                    "succeeded": 1,
                }
            )
            self.db.conn.commit()
            logger.debug("Extracted %d chars from file_id=%d", len(text), file_id)

        except Exception as exc:
            self.db["extraction"].insert(
                {
                    "file_id": file_id,
                    "tool": strategy,
                    "text_extracted": None,
                    "char_count": 0,
                    "succeeded": 0,
                }
            )
            self.db.conn.commit()
            record_failure(
                self.db, file_id=file_id, phase="extract",
                tool=strategy, error_class=type(exc).__name__,
                error_message=str(exc),
            )
            logger.warning("Extraction failed for file_id=%d: %s", file_id, exc)

    def _run_summarization(self, file_id: int) -> None:
        """Phase 9 — document summarization (if text meets threshold)."""
        threshold = self.cfg.extract.size_threshold_for_summary

        ext_row = next(
            iter(self.db.query(
                "SELECT text_extracted, char_count FROM extraction "
                "WHERE file_id = ? AND succeeded = 1",
                [file_id],
            )),
            None,
        )
        if not ext_row or ext_row["char_count"] < threshold:
            return

        model = self.cfg.models.summarization.name
        existing = next(
            iter(self.db.query(
                "SELECT 1 FROM summary WHERE file_id = ? AND model = ?",
                [file_id, model],
            )),
            None,
        )
        if existing:
            return

        try:
            import ollama
            from tenacity import retry, stop_after_attempt, wait_exponential

            text = ext_row["text_extracted"][:8000]
            prompt = (
                "Summarize the following document excerpt in 3-5 concise paragraphs.\n"
                "Focus on key facts, figures, decisions, and outcomes.\n"
                "Use the same language as the document.\n\n"
                f"Document:\n{text}\n\nSummary:\n"
            )

            @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=30))
            def _call(prompt_text: str) -> str:
                client = ollama.Client(host=self.cfg.ollama.host)
                resp = client.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful summarization assistant."},
                        {"role": "user", "content": prompt_text},
                    ],
                    options={"temperature": self.cfg.models.summarization.temperature, "num_ctx": 16384},
                )
                return resp.get("message", {}).get("content", "").strip()

            summary_text = _call(prompt)
            if summary_text:
                self.db["summary"].insert(
                    {
                        "file_id": file_id,
                        "model": model,
                        "summary_text": summary_text,
                        "generated_at": now_iso(),
                    }
                )
                self.db.conn.commit()
                logger.debug("Summarized file_id=%d (%d chars)", file_id, len(summary_text))

        except Exception as exc:
            record_failure(
                self.db, file_id=file_id, phase="summarize",
                tool=model, error_class=type(exc).__name__,
                error_message=str(exc),
            )
            logger.warning("Summarization failed for file_id=%d: %s", file_id, exc)

    def _run_chunking(self, file_id: int) -> None:
        """Phase 10 — sentence-window chunking."""
        existing = next(
            iter(self.db.query("SELECT 1 FROM chunk WHERE file_id = ? LIMIT 1", [file_id])),
            None,
        )
        if existing:
            return

        ext_row = next(
            iter(self.db.query(
                "SELECT id, text_extracted, page_count FROM extraction "
                "WHERE file_id = ? AND succeeded = 1",
                [file_id],
            )),
            None,
        )
        if not ext_row or not ext_row["text_extracted"]:
            return

        text = ext_row["text_extracted"]
        page_count = ext_row["page_count"]
        window_size = self.cfg.chunk.window_size
        sentences = _split_sentences(text)
        if not sentences:
            return

        # Pre-compute character offsets
        offsets: list[tuple[int, int]] = []
        pos = 0
        for s in sentences:
            start = text.find(s, pos)
            if start < 0:
                start = pos
            end = start + len(s)
            offsets.append((start, end))
            pos = end

        chunks_to_insert = []
        for i in range(len(sentences)):
            lo = max(0, i - window_size)
            hi = min(len(sentences) - 1, i + window_size)
            window_text = " ".join(sentences[lo : hi + 1])
            char_start = offsets[lo][0]
            char_end = offsets[hi][1]

            meta = {
                "extraction_id": ext_row["id"],
                "char_start": char_start,
                "char_end": char_end,
            }

            start_page = None
            end_page = None
            if page_count and page_count > 0:
                total = max(len(text), 1)
                start_page = max(1, int((char_start / total) * page_count) + 1)
                end_page = max(1, int((char_end / total) * page_count) + 1)

            chunks_to_insert.append(
                {
                    "file_id": file_id,
                    "chunk_index": i,
                    "text": window_text,
                    "token_count": _token_estimate(window_text),
                    "start_page": start_page,
                    "end_page": end_page,
                    "metadata_json": json.dumps(meta, ensure_ascii=False),
                    "created_at": now_iso(),
                }
            )

        if chunks_to_insert:
            self.db["chunk"].insert_all(chunks_to_insert)
            self.db.conn.commit()

            # Populate FTS5
            inserted = list(
                self.db.query(
                    "SELECT id, text FROM chunk WHERE file_id = ? ORDER BY chunk_index",
                    [file_id],
                )
            )
            for ch in inserted:
                self.db.execute(
                    "INSERT INTO chunk_fts(rowid, text) VALUES (?, ?)",
                    [ch["id"], ch["text"]],
                )
            self.db.conn.commit()
            self.db.execute('INSERT INTO chunk_fts(chunk_fts) VALUES("rebuild")')
            self.db.conn.commit()

            logger.debug("Created %d chunks for file_id=%d", len(chunks_to_insert), file_id)

    def _run_embedding(self, file_id: int) -> None:
        """Phase 11 — embed chunks into Chroma."""
        model = self.cfg.models.embedding.name
        suffix = self.cfg.models.embedding.collection_suffix
        collection_name = f"chunks__{suffix}"

        chunks = list(
            self.db.query(
                """SELECT c.id, c.text, c.context_text, c.file_id,
                          f.folder_id, f.rel_path, f.category
                   FROM chunk c
                   JOIN file f ON f.id = c.file_id
                   WHERE c.file_id = ?
                     AND c.id NOT IN (
                         SELECT chunk_id FROM embedding_ref WHERE embedding_model = ?
                     )
                   ORDER BY c.id""",
                [file_id, model],
            )
        )
        if not chunks:
            return

        if not self.chroma_client:
            logger.warning("No Chroma client — skipping embedding for file_id=%d", file_id)
            return

        try:
            import chromadb

            from rag.phase11_embed import _ensure_collection, _get_dim_from_ollama, _ollama_embed

            dim = _get_dim_from_ollama(self.cfg.ollama.host, model)
            c_hash = embedding_config_hash(self.cfg.models.embedding)

            coll = _ensure_collection(
                self.chroma_client,
                collection_name,
                embedding_model=model,
                dim=dim,
                config_hash=c_hash,
                source_db_path=str(Path.cwd() / "corpus.db"),
            )

            batch_size = self.cfg.models.embedding.batch_size
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                texts = []
                for ch in batch:
                    ctx = ch.get("context_text")
                    raw = f"{ctx}\n\n{ch['text']}" if ctx else ch["text"]
                    texts.append(raw[:_MAX_CHUNK_CHARS])

                vectors = _ollama_embed(self.cfg.ollama.host, model, texts, self.cfg.models.embedding.truncate_dim)

                ids = [f"c_{ch['id']}" for ch in batch]
                metadatas = [
                    {
                        "chunk_id": int(ch["id"]),
                        "file_id": int(ch["file_id"]),
                        "folder_id": int(ch.get("folder_id") or 0),
                        "rel_path": str(ch.get("rel_path", "")),
                    }
                    for ch in batch
                ]
                coll.add(ids=ids, embeddings=vectors, metadatas=metadatas, documents=texts)

                for ch in batch:
                    self.db["embedding_ref"].insert(
                        {
                            "chunk_id": ch["id"],
                            "vector_store": "chroma",
                            "collection": collection_name,
                            "external_id": f"c_{ch['id']}",
                            "embedding_model": model,
                            "dim": dim,
                            "config_hash": c_hash,
                            "is_current": 1,
                            "created_at": now_iso(),
                        }
                    )
                self.db.conn.commit()

            logger.debug("Embedded %d chunks for file_id=%d", len(chunks), file_id)

        except Exception as exc:
            record_failure(
                self.db, file_id=file_id, phase="embed",
                tool=model, error_class=type(exc).__name__,
                error_message=str(exc),
            )
            logger.warning("Embedding failed for file_id=%d: %s", file_id, exc)
