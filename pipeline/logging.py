"""Structured JSON logging for pipeline phases."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console

console = Console(stderr=True)


class JSONFileHandler(logging.Handler):
    """Writes structured JSON log lines to a .jsonl file in the cache logs directory."""

    def __init__(self, log_path: Path) -> None:
        super().__init__()
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, record: logging.LogRecord) -> None:
        entry = {
            "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "phase": getattr(record, "phase", "unknown"),
            "level": record.levelname.lower(),
            "event": record.getMessage(),
            "data": getattr(record, "extra_data", {}),
        }
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass  # Don't crash if log dir is unwritable


class PhaseLogger:
    """Convenience wrapper that writes both to JSON log file and rich console."""

    def __init__(self, phase: str, cache_root: Path) -> None:
        self.phase = phase
        self.log_path = (
            cache_root / "logs" / f"{phase}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}.jsonl"
        )
        self.file_handler = JSONFileHandler(self.log_path)
        self.logger = logging.getLogger(f"rag_preflight.{phase}")
        self.logger.setLevel(logging.DEBUG)
        # Clear any handlers from a prior run of the same phase to avoid
        # duplicate log writes.
        self.logger.handlers.clear()
        self.logger.addHandler(self.file_handler)
        self.logger.propagate = False

    def _log(self, level: int, msg: str, **kwargs: object) -> None:
        record = self.logger.makeRecord(self.logger.name, level, "", 0, msg, (), None)
        record.phase = self.phase
        record.extra_data = kwargs
        self.logger.handle(record)

    def info(self, msg: str, **kwargs: object) -> None:
        self._log(logging.INFO, msg, **kwargs)
        console.print(f"[dim][{self.phase}] {msg}[/dim]")

    def warning(self, msg: str, **kwargs: object) -> None:
        self._log(logging.WARNING, msg, **kwargs)
        console.print(f"[yellow][{self.phase}] WARNING: {msg}[/yellow]")

    def error(self, msg: str, **kwargs: object) -> None:
        self._log(logging.ERROR, msg, **kwargs)
        console.print(f"[red][{self.phase}] ERROR: {msg}[/red]")

    def debug(self, msg: str, **kwargs: object) -> None:
        self._log(logging.DEBUG, msg, **kwargs)
