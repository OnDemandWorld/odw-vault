"""File-system watcher for automatic RAG index updates.

Monitors the configured corpus_root directory for file changes
(create, modify, delete, move) and delegates processing to the
existing IncrementalIndexer.  Uses watchdog (FSEvents on macOS,
inotify on Linux) for efficient event-driven monitoring.

Usage:
    watcher = CorpusWatcher(cfg, db, chroma_client)
    watcher.start()
    # ... later ...
    watcher.stop()
"""

from __future__ import annotations

import fnmatch
import logging
import threading
import time
from pathlib import Path

from pipeline.config import AppConfig
from pipeline.helpers import ARCHIVE_OS_PATTERNS, SYSTEM_FILES
from rag.indexer import IncrementalIndexer

try:
    from watchdog.events import FileSystemEvent, FileSystemEventHandler
    from watchdog.observers import Observer
except ImportError:
    raise ImportError(
        "watchdog is required for file watching. "
        "Install with: pip install odw-vault[watch]"
    ) from None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event handler
# ---------------------------------------------------------------------------


class _SyncHandler(FileSystemEventHandler):
    """Watchdog event handler that filters and enqueues file events."""

    def __init__(self, watcher: CorpusWatcher) -> None:
        super().__init__()
        self._watcher = watcher

    def on_created(self, event: FileSystemEvent) -> None:
        self._handle(event, "sync")

    def on_modified(self, event: FileSystemEvent) -> None:
        self._handle(event, "sync")

    def on_deleted(self, event: FileSystemEvent) -> None:
        self._handle(event, "delete")

    def on_moved(self, event: FileSystemEvent) -> None:
        # Treat move as delete(src) + sync(dest)
        if not event.is_directory:
            self._watcher._enqueue(event.src_path, "delete")
            self._watcher._enqueue(event.dest_path, "sync")

    def _handle(self, event: FileSystemEvent, action: str) -> None:
        if event.is_directory:
            return
        path = str(event.src_path)
        if self._watcher._should_skip(path):
            return
        self._watcher._enqueue(path, action)


# ---------------------------------------------------------------------------
# CorpusWatcher
# ---------------------------------------------------------------------------


class CorpusWatcher:
    """Watches corpus_root and auto-syncs the RAG index on file changes.

    Events are debounced (coalesced per path) before being dispatched
    to IncrementalIndexer.sync_file() or remove_file().
    """

    def __init__(self, cfg: AppConfig, db, chroma_client=None) -> None:
        self._cfg = cfg
        self._db = db
        self._indexer = IncrementalIndexer(db, cfg, chroma_client)
        self._pending: dict[str, tuple[str, float]] = {}  # path -> (action, timestamp)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._observer: Observer | None = None
        self._worker: threading.Thread | None = None
        self._stats: dict = {"processed": 0, "failed": 0, "last_event_at": None}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start watching corpus_root for file changes."""
        if self._observer is not None:
            logger.warning("Watcher already running")
            return

        watch_path = str(self._cfg.corpus_root_path)
        recursive = self._cfg.watcher.recursive

        self._observer = Observer()
        handler = _SyncHandler(self)
        self._observer.schedule(handler, watch_path, recursive=recursive)
        self._observer.daemon = True
        self._observer.start()

        self._worker = threading.Thread(target=self._flush_loop, daemon=True)
        self._worker.start()

        logger.info("CorpusWatcher started on %s (recursive=%s)", watch_path, recursive)

    def stop(self) -> None:
        """Stop watching and drain pending events."""
        self._stop_event.set()
        if self._worker is not None:
            self._worker.join(timeout=10)
            self._worker = None
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
        logger.info("CorpusWatcher stopped")

    def status(self) -> dict:
        """Return watcher status dict."""
        with self._lock:
            pending_count = len(self._pending)
        return {
            "watching": self._observer is not None and self._observer.is_alive(),
            "pending": pending_count,
            "processed": self._stats["processed"],
            "failed": self._stats["failed"],
            "last_event_at": self._stats["last_event_at"],
        }

    @property
    def indexer(self) -> IncrementalIndexer:
        """Expose the underlying indexer (for startup_sync)."""
        return self._indexer

    # Context manager for tests
    def __enter__(self) -> CorpusWatcher:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Internal: event filtering
    # ------------------------------------------------------------------

    def _should_skip(self, path: str) -> bool:
        """Return True if this path should be ignored."""
        p = Path(path)
        name = p.name

        # Skip system files
        if name in SYSTEM_FILES:
            return True

        # Skip hidden files/dirs (any component starts with '.')
        for part in p.parts:
            if part.startswith(".") and part not in (".", ".."):
                return True

        # Skip archive OS patterns
        for part in p.parts:
            if part in ARCHIVE_OS_PATTERNS:
                return True

        # Skip configured ignore patterns
        for pattern in self._cfg.watcher.ignore_patterns:
            if fnmatch.fnmatch(name, pattern):
                return True
            # Also check if any path component matches (e.g. ".rag-cache")
            if pattern in p.parts:
                return True

        return False

    # ------------------------------------------------------------------
    # Internal: queue management
    # ------------------------------------------------------------------

    def _enqueue(self, path: str, action: str) -> None:
        """Add or update a path in the pending queue."""
        with self._lock:
            self._pending[path] = (action, time.monotonic())
            self._stats["last_event_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        logger.debug("Enqueued %s: %s", path, action)

    # ------------------------------------------------------------------
    # Internal: worker loop
    # ------------------------------------------------------------------

    def _flush_loop(self) -> None:
        """Background worker that flushes debounced events to the indexer."""
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=1.0)
            self._flush_ready()

        # Final drain on shutdown
        self._flush_ready(force=True)

    def _flush_ready(self, force: bool = False) -> None:
        """Process all pending events that have passed the debounce window."""
        now = time.monotonic()
        debounce = 0.0 if force else self._cfg.watcher.debounce_seconds
        ready: list[tuple[str, str]] = []

        with self._lock:
            for path, (action, ts) in list(self._pending.items()):
                if now - ts >= debounce:
                    ready.append((path, action))
                    del self._pending[path]

        for path, action in ready:
            try:
                if action == "delete":
                    self._handle_delete(path)
                else:
                    self._indexer.sync_file(Path(path))
                self._stats["processed"] += 1
                logger.info("Watcher synced: %s (%s)", path, action)
            except Exception as exc:
                self._stats["failed"] += 1
                logger.error("Watcher sync failed for %s: %s", path, exc)

    def _handle_delete(self, path: str) -> None:
        """Look up file_id by path and remove from index."""
        row = next(
            iter(self._db.query("SELECT id FROM file WHERE path = ?", [path])),
            None,
        )
        if row:
            self._indexer.remove_file(row["id"])
            logger.info("Watcher removed file_id=%d (%s)", row["id"], path)
        else:
            logger.debug("Delete event for unindexed path: %s", path)
