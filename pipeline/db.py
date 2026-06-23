"""Database schema, migrations, and connection helpers."""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

import sqlite_utils

# ---------------------------------------------------------------------------
# Retry helper for SQLite lock errors
# ---------------------------------------------------------------------------


def _retry_on_locked(fn, max_retries=5, backoff=0.1):
    """Retry *fn* on 'database is locked' errors with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return fn()
        except sqlite3.OperationalError as e:
            if "locked" not in str(e).lower() or attempt == max_retries - 1:
                raise
            time.sleep(backoff * (attempt + 1))


# ---------------------------------------------------------------------------
# Database wrapper: safer defaults than raw sqlite_utils
# ---------------------------------------------------------------------------


class Database:
    """Thin wrapper around sqlite_utils.Database with safer defaults.

    - db.query() returns list() by default (generators exhaust silently)
    - db.transaction() provides retry-on-lock and rollback-on-error
    - Fully compatible with existing db["table"], db.conn usage
    """

    def __init__(self, path: Path | str):
        self._db = sqlite_utils.Database(str(path))
        self._db.conn.execute("PRAGMA foreign_keys = ON")
        self._db.conn.execute("PRAGMA journal_mode = WAL")
        self._db.conn.execute("PRAGMA synchronous = NORMAL")

    def __getitem__(self, table: str):
        return self._db[table]

    @property
    def conn(self):
        return self._db.conn

    def query(self, sql: str, params=None, *, lazy: bool = False):
        """Execute a SELECT query. Returns list by default, or generator if lazy=True."""
        rows = self._db.query(sql, params)
        return rows if lazy else list(rows)

    def execute(self, sql: str, params=None):
        """Execute a write statement with retry on lock."""
        return _retry_on_locked(lambda: self._db.execute(sql, params))

    def executescript(self, sql: str):
        """Execute a SQL script (multiple statements)."""
        return self._db.executescript(sql)

    @contextmanager
    def transaction(self):
        """Context manager for atomic writes with retry on lock."""
        try:
            yield self._db
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise


# ---------------------------------------------------------------------------
# Helper: safely add a column (SQLite lacks ALTER TABLE … ADD COLUMN IF NOT EXISTS)
# ---------------------------------------------------------------------------


def _add_column_if_missing(db, table: str, column: str, definition: str) -> None:
    """Add *column* to *table* only if it does not already exist."""
    cols = [row[1] for row in db.execute(f"PRAGMA table_info({table})").fetchall()]
    if column not in cols:
        db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


# All migrations, applied in order. Each tuple: (version, description, sql).
MIGRATIONS = [
    (
        1,
        "initial schema",
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL DEFAULT (datetime('now')),
            description TEXT
        );

        CREATE TABLE IF NOT EXISTS config (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS pipeline_run (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phase TEXT NOT NULL CHECK (phase IN ('archives','walk','identify','triage','dedup','folder_meta','report')),
            started_at TEXT NOT NULL DEFAULT (datetime('now')),
            finished_at TEXT,
            status TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running','done','failed','aborted')),
            files_processed INTEGER,
            files_failed INTEGER,
            notes TEXT,
            config_snapshot_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_pipeline_run_phase ON pipeline_run(phase, started_at DESC);

        CREATE TABLE IF NOT EXISTS folder (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE,
            rel_path TEXT NOT NULL,
            parent_id INTEGER REFERENCES folder(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            depth INTEGER NOT NULL,
            is_extracted_archive INTEGER NOT NULL DEFAULT 0 CHECK (is_extracted_archive IN (0,1)),
            source_archive_file_id INTEGER REFERENCES file(id),
            excluded INTEGER NOT NULL DEFAULT 0 CHECK (excluded IN (0,1)),
            exclusion_reason TEXT,
            file_count INTEGER,
            total_bytes INTEGER,
            document_count INTEGER,
            dominant_format TEXT,
            inferred_category TEXT,
            inferred_label TEXT,
            inferred_tags_json TEXT,
            inferred_summary TEXT,
            inference_model TEXT,
            inference_prompt_hash TEXT,
            inferred_at TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_folder_parent ON folder(parent_id);
        CREATE INDEX IF NOT EXISTS idx_folder_depth ON folder(depth);
        CREATE INDEX IF NOT EXISTS idx_folder_excluded ON folder(excluded);

        CREATE TABLE IF NOT EXISTS file (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            folder_id INTEGER NOT NULL REFERENCES folder(id) ON DELETE CASCADE,
            path TEXT NOT NULL UNIQUE,
            rel_path TEXT NOT NULL,
            name TEXT NOT NULL,
            extension TEXT,
            size_bytes INTEGER NOT NULL,
            mtime TEXT NOT NULL,
            sha256 TEXT,
            extracted_from_archive_id INTEGER REFERENCES file(id),
            pronom_id TEXT,
            mime_type TEXT,
            format_name TEXT,
            format_version TEXT,
            siegfried_json TEXT,
            id_warning TEXT,
            category TEXT,
            extract_strategy TEXT,
            is_encrypted INTEGER CHECK (is_encrypted IN (0,1) OR is_encrypted IS NULL),
            is_corrupt INTEGER CHECK (is_corrupt IN (0,1) OR is_corrupt IS NULL),
            page_count INTEGER,
            duration_seconds REAL,
            has_text_layer INTEGER,
            triage_json TEXT,
            dup_group_id INTEGER,
            is_dup_primary INTEGER NOT NULL DEFAULT 1 CHECK (is_dup_primary IN (0,1)),
            excluded INTEGER NOT NULL DEFAULT 0 CHECK (excluded IN (0,1)),
            exclusion_reason TEXT,
            hash_status TEXT NOT NULL DEFAULT 'pending',
            identify_status TEXT NOT NULL DEFAULT 'pending',
            triage_status TEXT NOT NULL DEFAULT 'pending',
            error_message TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_file_folder ON file(folder_id);
        CREATE INDEX IF NOT EXISTS idx_file_sha256 ON file(sha256);
        CREATE INDEX IF NOT EXISTS idx_file_category ON file(category);
        CREATE INDEX IF NOT EXISTS idx_file_pronom ON file(pronom_id);
        CREATE INDEX IF NOT EXISTS idx_file_dup_group ON file(dup_group_id);
        CREATE INDEX IF NOT EXISTS idx_file_hash_status ON file(hash_status);
        CREATE INDEX IF NOT EXISTS idx_file_identify_status ON file(identify_status);
        CREATE INDEX IF NOT EXISTS idx_file_triage_status ON file(triage_status);
        CREATE INDEX IF NOT EXISTS idx_file_excluded ON file(excluded);

        CREATE TABLE IF NOT EXISTS archive_expansion (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            archive_file_id INTEGER NOT NULL REFERENCES file(id),
            extracted_to_path TEXT NOT NULL,
            extracted_to_folder_id INTEGER REFERENCES folder(id),
            tool TEXT NOT NULL,
            succeeded INTEGER NOT NULL CHECK (succeeded IN (0,1)),
            file_count INTEGER,
            error_message TEXT,
            extracted_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_archive_file ON archive_expansion(archive_file_id);

        CREATE TABLE IF NOT EXISTS format_policy (
            pronom_id TEXT PRIMARY KEY,
            format_name TEXT,
            category TEXT NOT NULL,
            extract_strategy TEXT NOT NULL,
            notes TEXT,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS failure (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER REFERENCES file(id) ON DELETE CASCADE,
            folder_id INTEGER REFERENCES folder(id) ON DELETE CASCADE,
            phase TEXT NOT NULL,
            tool TEXT,
            error_class TEXT,
            error_message TEXT,
            traceback TEXT,
            occurred_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_failure_file ON failure(file_id);
        CREATE INDEX IF NOT EXISTS idx_failure_phase ON failure(phase);
        CREATE INDEX IF NOT EXISTS idx_failure_class ON failure(error_class);

        -- Forward-compatible tables (empty during pre-flight)
        CREATE TABLE IF NOT EXISTS extraction (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER NOT NULL REFERENCES file(id) ON DELETE CASCADE,
            tool TEXT NOT NULL,
            text_extracted TEXT,
            char_count INTEGER,
            page_count INTEGER,
            succeeded INTEGER NOT NULL DEFAULT 0 CHECK (succeeded IN (0,1)),
            error_message TEXT,
            extracted_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER NOT NULL REFERENCES file(id) ON DELETE CASCADE,
            model TEXT NOT NULL,
            summary_text TEXT,
            generated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS chunk (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER NOT NULL REFERENCES file(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            token_count INTEGER,
            start_page INTEGER,
            end_page INTEGER,
            metadata_json TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS embedding_ref (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id INTEGER NOT NULL REFERENCES chunk(id) ON DELETE CASCADE,
            model TEXT,
            embedding_blob BLOB,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
            text, content='chunk', content_rowid='id',
            tokenize='porter unicode61'
        );

        -- Views
        CREATE VIEW IF NOT EXISTS v_format_histogram AS
        SELECT format_name, pronom_id, COUNT(*) AS n, SUM(size_bytes) AS bytes,
               SUM(CASE WHEN extract_strategy='skip' THEN 0 ELSE 1 END) AS extractable
        FROM file WHERE is_dup_primary=1 AND excluded=0
        GROUP BY pronom_id, format_name ORDER BY n DESC;

        CREATE VIEW IF NOT EXISTS v_category_summary AS
        SELECT category, COUNT(*) AS n, SUM(size_bytes) AS bytes
        FROM file WHERE is_dup_primary=1 AND excluded=0
        GROUP BY category ORDER BY n DESC;

        CREATE VIEW IF NOT EXISTS v_ocr_workload AS
        SELECT COUNT(*) AS scanned_pdfs, COALESCE(SUM(page_count),0) AS total_pages
        FROM file WHERE category='pdf-scanned' AND is_dup_primary=1 AND excluded=0;

        CREATE VIEW IF NOT EXISTS v_transcription_workload AS
        SELECT category, COUNT(*) AS n,
               ROUND(COALESCE(SUM(duration_seconds),0)/3600.0, 2) AS total_hours
        FROM file WHERE category IN ('audio','video') AND is_dup_primary=1 AND excluded=0
        GROUP BY category;

        CREATE VIEW IF NOT EXISTS v_duplicate_summary AS
        SELECT dup_group_id, COUNT(*) AS copies, MIN(path) AS example_path,
               MAX(size_bytes) AS size_bytes
        FROM file WHERE dup_group_id IS NOT NULL
        GROUP BY dup_group_id HAVING copies>1 ORDER BY copies DESC;

        CREATE VIEW IF NOT EXISTS v_problem_files AS
        SELECT id, path, category, error_message, identify_status, triage_status
        FROM file
        WHERE error_message IS NOT NULL OR is_corrupt=1 OR is_encrypted=1
           OR id_warning IS NOT NULL OR category='unknown';

        CREATE VIEW IF NOT EXISTS v_unknown_formats AS
        SELECT pronom_id, format_name, COUNT(*) AS n
        FROM file WHERE pronom_id IS NOT NULL
          AND pronom_id NOT IN (SELECT pronom_id FROM format_policy)
        GROUP BY pronom_id, format_name ORDER BY n DESC;
        """,
    ),
    (
        2,
        "post-preflight tables and views",
        """
        -- New tables
        CREATE TABLE IF NOT EXISTS model_run (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            role            TEXT NOT NULL CHECK (role IN ('extraction','transcription','summarization','contextual_augmentation','embedding','reranker','generation','language_id')),
            model_name      TEXT NOT NULL,
            model_version   TEXT,
            tool            TEXT,
            config_hash     TEXT NOT NULL,
            phase           TEXT NOT NULL,
            started_at      TEXT NOT NULL DEFAULT (datetime('now')),
            finished_at     TEXT,
            status          TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running','done','failed','aborted')),
            items_processed INTEGER,
            items_failed    INTEGER,
            notes           TEXT
        );

        CREATE TABLE IF NOT EXISTS query_log (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            asked_at            TEXT NOT NULL DEFAULT (datetime('now')),
            user                TEXT,
            query_text          TEXT NOT NULL,
            query_lang          TEXT,
            folder_filter_json  TEXT,
            retrieved_chunks_json TEXT NOT NULL,
            answer_text         TEXT NOT NULL,
            answer_model        TEXT NOT NULL,
            embedding_model     TEXT NOT NULL,
            reranker_model      TEXT,
            latency_ms          INTEGER NOT NULL,
            retrieval_ms        INTEGER,
            generation_ms       INTEGER,
            feedback            TEXT NULL CHECK (feedback IN ('up','down') OR feedback IS NULL),
            feedback_note       TEXT,
            feedback_at         TEXT
        );

        CREATE TABLE IF NOT EXISTS eval_question (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            question                TEXT NOT NULL,
            expected_file_ids_json  TEXT NOT NULL,
            expected_answer         TEXT,
            category                TEXT,
            lang                    TEXT,
            created_at              TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS eval_run (
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

        CREATE TABLE IF NOT EXISTS folder_embedding_ref (
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

        CREATE TABLE IF NOT EXISTS summary_embedding_ref (
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

        -- New indexes
        CREATE INDEX IF NOT EXISTS idx_embedding_ref_current ON embedding_ref(embedding_model, is_current);
        CREATE INDEX IF NOT EXISTS idx_model_run_role ON model_run(role, started_at DESC);
        CREATE INDEX IF NOT EXISTS idx_model_run_phase ON model_run(phase, started_at DESC);
        CREATE INDEX IF NOT EXISTS idx_query_log_asked ON query_log(asked_at DESC);
        CREATE INDEX IF NOT EXISTS idx_query_log_feedback ON query_log(feedback) WHERE feedback IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_eval_run_models ON eval_run(embedding_model, generation_model, run_at DESC);

        -- New views (DROP + CREATE for idempotency)
        DROP VIEW IF EXISTS v_extraction_status;
        CREATE VIEW v_extraction_status AS
        SELECT f.category, COUNT(*) AS total,
               SUM(CASE WHEN e.id IS NOT NULL THEN 1 ELSE 0 END) AS extracted,
               SUM(CASE WHEN f.extract_status='failed' THEN 1 ELSE 0 END) AS failed,
               SUM(CASE WHEN f.extract_status='pending' THEN 1 ELSE 0 END) AS pending
        FROM file f
        LEFT JOIN extraction e ON e.file_id = f.id
        WHERE f.is_dup_primary=1 AND f.excluded=0
        GROUP BY f.category;

        DROP VIEW IF EXISTS v_embedding_coverage;
        CREATE VIEW v_embedding_coverage AS
        SELECT er.embedding_model, er.collection,
               COUNT(DISTINCT er.chunk_id) AS chunks_embedded,
               COUNT(DISTINCT c.file_id) AS files_covered
        FROM embedding_ref er JOIN chunk c ON c.id = er.chunk_id
        WHERE er.is_current = 1
        GROUP BY er.embedding_model, er.collection;

        DROP VIEW IF EXISTS v_context_coverage;
        CREATE VIEW v_context_coverage AS
        SELECT context_model, COUNT(*) AS chunks_augmented,
               COUNT(DISTINCT file_id) AS files_covered
        FROM chunk
        WHERE context_text IS NOT NULL
        GROUP BY context_model;

        DROP VIEW IF EXISTS v_query_volume;
        CREATE VIEW v_query_volume AS
        SELECT DATE(asked_at) AS day, COUNT(*) AS queries,
               AVG(latency_ms) AS avg_latency_ms,
               SUM(CASE WHEN feedback='up' THEN 1 ELSE 0 END) AS up,
               SUM(CASE WHEN feedback='down' THEN 1 ELSE 0 END) AS down
        FROM query_log
        GROUP BY DATE(asked_at) ORDER BY day DESC;

        DROP VIEW IF EXISTS v_eval_summary;
        CREATE VIEW v_eval_summary AS
        SELECT embedding_model, generation_model, contextual_augmentation,
               COUNT(*) AS questions,
               AVG(retrieval_recall_at_5) AS recall_at_5,
               AVG(retrieval_recall_at_10) AS recall_at_10,
               AVG(human_grade) AS avg_human_grade
        FROM eval_run
        GROUP BY embedding_model, generation_model, contextual_augmentation;
        """,
    ),
]


def open_db(db_path: Path) -> Database:
    """Open a SQLite database with recommended pragmas."""
    return Database(db_path)


def migrate(db: Database) -> None:
    """Apply all pending migrations."""
    db.executescript(
        "CREATE TABLE IF NOT EXISTS schema_version "
        "(version INTEGER PRIMARY KEY, applied_at TEXT, description TEXT)"
    )
    current = next(iter(db.execute("SELECT COALESCE(MAX(version),0) FROM schema_version")))[0]
    for version, description, sql in MIGRATIONS:
        if version > current:
            with db.conn:
                # v2 ALTER TABLE statements: apply via helper for idempotency
                # since SQLite lacks ADD COLUMN IF NOT EXISTS.
                if version == 2:
                    _add_column_if_missing(db, "embedding_ref", "embedding_model", "TEXT")
                    _add_column_if_missing(db, "embedding_ref", "config_hash", "TEXT")
                    _add_column_if_missing(
                        db,
                        "embedding_ref",
                        "is_current",
                        "INTEGER NOT NULL DEFAULT 1 CHECK (is_current IN (0,1))",
                    )
                    _add_column_if_missing(db, "embedding_ref", "collection", 'TEXT DEFAULT ""')
                    _add_column_if_missing(db, "embedding_ref", "vector_store", 'TEXT DEFAULT ""')
                    _add_column_if_missing(db, "embedding_ref", "external_id", 'TEXT DEFAULT ""')
                    _add_column_if_missing(db, "embedding_ref", "dim", "INTEGER DEFAULT 0")
                    _add_column_if_missing(db, "chunk", "context_text", "TEXT")
                    _add_column_if_missing(db, "chunk", "context_model", "TEXT")
                    _add_column_if_missing(db, "chunk", "context_prompt_hash", "TEXT")
                    _add_column_if_missing(db, "chunk", "context_generated_at", "TEXT")
                    db.conn.commit()
                db.executescript(sql)
                db.conn.execute(
                    "INSERT INTO schema_version(version, description) VALUES (?,?)",
                    (version, description),
                )


def is_db_initialized(db_path: Path) -> bool:
    """Check if the database already has schema applied."""
    if not db_path.exists():
        return False
    db = open_db(db_path)
    try:
        result = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
        ).fetchone()
        return result is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Model run tracking helpers
# ---------------------------------------------------------------------------


def start_model_run(
    db: Database,
    role: str,
    model_name: str,
    config_hash: str,
    phase: str,
    model_version: str | None = None,
    tool: str | None = None,
) -> int:
    """Insert a new model_run row and return its id."""
    cursor = db.conn.execute(
        """INSERT INTO model_run (role, model_name, model_version, tool, config_hash, phase, started_at)
           VALUES (?, ?, ?, ?, ?, ?, datetime('now'))""",
        (role, model_name, model_version, tool, config_hash, phase),
    )
    db.conn.commit()
    return cursor.lastrowid


def finish_model_run(
    db: Database,
    run_id: int,
    status: str,
    items_processed: int = 0,
    items_failed: int = 0,
    notes: str | None = None,
) -> None:
    """Update a model_run row with completion details."""
    db.conn.execute(
        "UPDATE model_run SET finished_at=datetime('now'), status=?, items_processed=?, items_failed=?, notes=? WHERE id=?",
        (status, items_processed, items_failed, notes, run_id),
    )
    db.conn.commit()


def heartbeat_model_run(
    db: Database,
    run_id: int,
    items_processed: int = 0,
    items_failed: int = 0,
) -> None:
    """Update a running model_run's progress counters."""
    db.conn.execute(
        "UPDATE model_run SET items_processed=?, items_failed=? WHERE id=?",
        (items_processed, items_failed, run_id),
    )
    db.conn.commit()


def cleanup_stale_runs(
    db: Database,
    max_age_hours: int = 24,
) -> int:
    """Mark model_runs stuck as 'running' for >N hours as 'aborted'.

    Returns the number of runs cleaned up.
    """
    rows = db.query(
        "SELECT id FROM model_run WHERE status='running' "
        "AND started_at < datetime('now', ?)",
        [f"-{max_age_hours} hours"],
    )
    for row in rows:
        db["model_run"].update(row["id"], {
            "status": "aborted",
            "notes": "abandoned — process killed or crashed",
        })
    db.conn.commit()
    return len(rows)
