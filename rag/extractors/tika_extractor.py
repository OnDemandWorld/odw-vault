"""Apache Tika-based extractor for .doc, .rtf, .eml, .msg, .epub.

Offline-first: the tika-python library's default behaviour downloads a ~100MB
server jar from search.maven.org (no timeout) and races when several workers
start at once. We disable that machinery (TikaClientOnly) and only ever start
a server from a jar that already exists locally.
"""

from __future__ import annotations

import os
import threading
import time
import zipfile
from urllib.parse import urlparse

_START_LOCK = threading.Lock()
_READY_ENDPOINTS: set[str] = set()
_STARTUP_DEADLINE_SECONDS = 30.0


def _probe(tika_url: str, timeout: float = 3.0) -> bool:
    import requests

    try:
        resp = requests.get(
            tika_url.rstrip("/") + "/tika",
            timeout=timeout,
            proxies={"http": None, "https": None},
        )
        return resp.status_code < 500
    except Exception:
        return False


def _local_jar() -> str | None:
    env_jar = os.environ.get("TIKA_SERVER_JAR")
    if env_jar and os.path.isfile(env_jar) and zipfile.is_zipfile(env_jar):
        return env_jar
    from tika import tika as tika_core

    candidate = os.path.join(tika_core.TikaJarPath, "tika-server.jar")
    if os.path.isfile(candidate) and zipfile.is_zipfile(candidate):
        return candidate
    return None


def _ensure_server(tika_url: str, brute_force: bool) -> str | None:
    """Return None when the server is ready, else an error message."""
    if tika_url in _READY_ENDPOINTS or _probe(tika_url):
        _READY_ENDPOINTS.add(tika_url)
        return None
    if not brute_force:
        return f"Tika server unavailable at {tika_url}"

    with _START_LOCK:
        if tika_url in _READY_ENDPOINTS or _probe(tika_url):
            _READY_ENDPOINTS.add(tika_url)
            return None
        jar = _local_jar()
        if jar is None:
            return (
                f"Tika server unavailable at {tika_url} and no local server jar found. "
                "Start Tika manually or point TIKA_SERVER_JAR at a server jar "
                "(offline mode: the jar is never downloaded automatically)."
            )
        from tika import tika as tika_core

        parsed = urlparse(tika_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 9998
        try:
            tika_core.startServer(jar, tika_core.TikaJava, tika_core.TikaJavaArgs, host, port)
        except Exception as e:
            return f"Tika server failed to start: {e}"
        deadline = time.monotonic() + _STARTUP_DEADLINE_SECONDS
        while time.monotonic() < deadline:
            if _probe(tika_url, timeout=2.0):
                _READY_ENDPOINTS.add(tika_url)
                return None
            time.sleep(1.0)
        return f"Tika server did not become ready at {tika_url} within {_STARTUP_DEADLINE_SECONDS:.0f}s"


def extract_tika(
    filepath: str,
    tika_url: str = "http://localhost:9998",
    brute_force: bool = True,
) -> tuple[str | None, dict, bool, str | None]:
    """Extract text and metadata via Apache Tika server.

    Uses an existing server at tika_url; if none is running and brute_force is
    True, starts one from a local jar (never downloads). Returns
    (text, metadata, succeeded, error_message).
    """
    try:
        from tika import parser as tika_parser
        from tika import tika as tika_core

        # Skip the library's auto-download/auto-start path entirely.
        tika_core.TikaClientOnly = True

        err = _ensure_server(tika_url, brute_force)
        if err:
            return None, {}, False, err

        try:
            parsed = tika_parser.from_file(filepath, serverEndpoint=tika_url)
        except Exception:
            # Server may have died after being marked ready (e.g. watcher
            # outlives it); force a re-probe on the next call.
            _READY_ENDPOINTS.discard(tika_url)
            raise

        if not parsed:
            return None, {}, False, "Tika returned empty response"

        text = parsed.get("content") or ""
        metadata = parsed.get("metadata", {})

        # Normalize metadata dict (tika returns it as a dict with string keys)
        if isinstance(metadata, dict):
            clean_meta = {}
            for k, v in metadata.items():
                clean_meta[str(k).lower()] = str(v) if v is not None else None
            metadata = clean_meta

        if not text.strip():
            if brute_force:
                return None, {}, False, "Tika returned empty text (file may be binary/non-text)"
            return None, {}, False, "Tika returned empty text"

        return text, metadata, True, None

    except ImportError:
        return None, {}, False, "tika not installed"
    except Exception as e:
        return None, {}, False, str(e)
