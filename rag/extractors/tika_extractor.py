"""Apache Tika-based extractor for .doc, .rtf, .eml, .msg, .epub."""

from __future__ import annotations


def extract_tika(
    filepath: str,
    tika_url: str = "http://localhost:9998",
    brute_force: bool = True,
) -> tuple[str | None, dict, bool, str | None]:
    """Extract text and metadata via Apache Tika server.

    Attempts to connect to an existing Tika server at tika_url.
    If that fails and brute_force is True, tries starting one via tika.server.startServer().

    Returns (text, metadata, succeeded, error_message).
    """
    try:
        from tika import parser as tika_parser

        # Try the primary URL first
        try:
            parsed = tika_parser.from_file(filepath, serverEndpoint=tika_url)
        except Exception:
            # Fallback: try starting Tika server locally
            try:
                from tika import server as tika_server

                tika_server.startServer()
                parsed = tika_parser.from_file(filepath, serverEndpoint=tika_url)
            except Exception as start_err:
                if brute_force:
                    # Last resort: try without explicit endpoint (uses default)
                    parsed = tika_parser.from_file(filepath)
                else:
                    return None, {}, False, f"Tika server unavailable: {start_err}"

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
