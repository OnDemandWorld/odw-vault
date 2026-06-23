"""Phase 10: Sentence-window chunking.

Splits extracted text into overlapping sentence-window chunks. Each chunk
contains a focal sentence plus ``window_size`` surrounding sentences on each
side. Char offsets are tracked as byte positions into the source extraction
text. Chunks are inserted into the ``chunk`` table; the FTS5 trigger on
``chunk_fts`` handles index updates automatically.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

# Regex that matches sentence boundaries. Covers English (.!?) and common
# Chinese sentence terminators. The split keeps the delimiter so
# offsets remain accurate.
_SENTENCE_RE = re.compile(r"(?<=[.!?。！？\n])\s+")  # noqa: RUF001


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using a regex-based boundary detector."""
    raw = _SENTENCE_RE.split(text)
    # Filter out empty strings that can appear at boundaries
    return [s for s in raw if s.strip()]


def _token_estimate(text: str) -> int:
    """Rough token count: len(text) // 4."""
    return max(1, len(text) // 4)


def run_chunk(
    db,
    cfg,
    chunker: str | None = None,
    window: int | None = None,
    rechunk: bool = False,
) -> tuple[int, int]:
    """Sentence-window chunking.

    Returns (total_chunks_created, files_processed) counts.
    """
    chunk_cfg = cfg.chunk
    selected_chunker = chunker or chunk_cfg.chunker
    window_size = window if window is not None else chunk_cfg.window_size

    if selected_chunker != "sentence-window":
        logger.warning(
            "Unknown chunker '%s', falling back to sentence-window",
            selected_chunker,
        )

    # Select files with successful extractions that haven't been chunked yet
    if rechunk:
        rows = list(
            db.query(
                """
                SELECT e.id AS extraction_id,
                       e.file_id,
                       e.text_extracted,
                       e.page_count AS extraction_page_count,
                       e.char_count,
                       e.tool
                FROM extraction e
                JOIN file f ON f.id = e.file_id
                WHERE e.succeeded = 1
                  AND e.text_extracted IS NOT NULL
                  AND e.text_extracted != ''
                  AND f.is_dup_primary = 1
                  AND f.excluded = 0
                ORDER BY e.file_id
                """,
            )
        )
    else:
        rows = list(
            db.query(
                """
                SELECT e.id AS extraction_id,
                       e.file_id,
                       e.text_extracted,
                       e.page_count AS extraction_page_count,
                       e.char_count,
                       e.tool
                FROM extraction e
                JOIN file f ON f.id = e.file_id
                WHERE e.succeeded = 1
                  AND e.text_extracted IS NOT NULL
                  AND e.text_extracted != ''
                  AND f.is_dup_primary = 1
                  AND f.excluded = 0
                  AND e.file_id NOT IN (
                      SELECT DISTINCT file_id FROM chunk
                  )
                ORDER BY e.file_id
                """,
            )
        )

    if not rows:
        logger.info("No extractions eligible for chunking.")
        return (0, 0)

    # If rechunk, delete existing chunks for these files
    if rechunk:
        file_ids = [r["file_id"] for r in rows]
        placeholders = ",".join("?" for _ in file_ids)
        db.execute(
            f"DELETE FROM chunk WHERE file_id IN ({placeholders})",
            file_ids,
        )
        db.conn.commit()
        logger.info("Deleted existing chunks for %d files (rechunk mode)", len(file_ids))

    total_chunks = 0
    files_processed = 0

    for row in rows:
        extraction_id = row["extraction_id"]
        file_id = row["file_id"]
        text = row["text_extracted"]
        page_count = row["extraction_page_count"]

        sentences = _split_sentences(text)
        if not sentences:
            continue

        # Pre-compute character offsets for each sentence
        offsets: list[tuple[int, int]] = []
        pos = 0
        for s in sentences:
            start = text.find(s, pos)
            if start < 0:
                start = pos  # fallback
            end = start + len(s)
            offsets.append((start, end))
            pos = end

        chunks_to_insert = []
        for i, _sentence in enumerate(sentences):
            # Window: sentences[i-window_size .. i+window_size], clamped
            lo = max(0, i - window_size)
            hi = min(len(sentences) - 1, i + window_size)

            window_sentences = sentences[lo : hi + 1]
            chunk_text = " ".join(window_sentences)

            # Byte offsets into the extraction text
            char_start = offsets[lo][0]
            char_end = offsets[hi][1]

            # Page info from extraction metadata
            meta: dict[str, object] = {
                "extraction_id": extraction_id,
                "extraction_tool": row["tool"],
                "char_start": char_start,
                "char_end": char_end,
            }

            start_page = None
            end_page = None
            if page_count is not None and page_count > 0:
                # Approximate page mapping based on position in text
                page_start = int((char_start / max(len(text), 1)) * page_count) + 1
                page_end = int((char_end / max(len(text), 1)) * page_count) + 1
                start_page = max(1, page_start)
                end_page = max(1, page_end)

            chunks_to_insert.append(
                {
                    "file_id": file_id,
                    "chunk_index": i,
                    "text": chunk_text,
                    "token_count": _token_estimate(chunk_text),
                    "start_page": start_page,
                    "end_page": end_page,
                    "metadata_json": json.dumps(meta, ensure_ascii=False),
                    "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            )

        if not chunks_to_insert:
            continue

        db["chunk"].insert_all(chunks_to_insert)
        db.conn.commit()

        # Populate FTS5 table (external content table needs explicit inserts).
        # The rowid must match chunk.id for content='chunk' to work.
        inserted_chunks = list(
            db.query(
                "SELECT id, text FROM chunk WHERE file_id = ? ORDER BY chunk_index",
                [file_id],
            )
        )
        for ch in inserted_chunks:
            db.execute("INSERT INTO chunk_fts(rowid, text) VALUES (?, ?)", [ch["id"], ch["text"]])
        db.conn.commit()

        total_chunks += len(chunks_to_insert)
        files_processed += 1
        logger.debug(
            "file_id=%d: %d chunks from %d sentences (%d chars)",
            file_id,
            len(chunks_to_insert),
            len(sentences),
            row["char_count"],
        )

    logger.info(
        "Chunking complete: %d chunks across %d files",
        total_chunks,
        files_processed,
    )

    # Rebuild FTS index to ensure consistency
    db.execute('INSERT INTO chunk_fts(chunk_fts) VALUES("rebuild")')
    db.conn.commit()

    return (total_chunks, files_processed)
