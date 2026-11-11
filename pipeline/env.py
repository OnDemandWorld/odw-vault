"""Proxy environment normalization — import before any httpx-based library.

`ollama` instantiates a global httpx Client at import time, so a malformed
NO_PROXY entry crashes `import ollama` itself. Importing this module first
normalizes the environment before that happens.
"""

from __future__ import annotations

import ipaddress
import os


def sanitize_proxy_env() -> None:
    """Make NO_PROXY/no_proxy parseable by httpx and bypass proxies for loopback.

    httpx 0.28.x does not recognize bracketed IPv6 entries ("[::1]") in
    NO_PROXY and builds the invalid URL pattern "all://*[::1]", raising
    InvalidURL when any client is constructed. Unbracket them — curl
    semantics are preserved. Also ensure localhost/127.0.0.1 bypass the
    proxy so internal calls (Ollama, sibling services) never route through
    an external proxy and 502.
    """
    for key in ("NO_PROXY", "no_proxy"):
        parts = [p.strip() for p in os.environ.get(key, "").split(",") if p.strip()]
        cleaned: list[str] = []
        for part in parts:
            if part.startswith("["):
                # Bracketed IPv6, with or without the closing bracket —
                # httpx cannot parse either form; keep only valid addresses.
                inner = part[1:-1] if part.endswith("]") else part[1:]
                try:
                    ipaddress.IPv6Address(inner)
                except ValueError:
                    continue
                part = inner
            cleaned.append(part)
        for host in ("localhost", "127.0.0.1"):
            if host not in cleaned:
                cleaned.append(host)
        os.environ[key] = ",".join(cleaned)


sanitize_proxy_env()
