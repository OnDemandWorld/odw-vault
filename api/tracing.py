"""Distributed tracing support for the Vault HTTP service (V1.5 F-3).

Strictly additive and best-effort: this module never changes a business
response body or status code. It only

* reads an inbound ``X-Trace-Id`` request header (generating a UUID4 when it
  is absent),
* echoes the id back as a response header so callers can correlate, and
* binds the id into the logging context for the request lifetime so existing
  endpoint logs carry a ``trace_id`` field without any per-call changes.

Vault uses the standard-library ``logging`` module (not structlog), so the
trace id is held in a :class:`contextvars.ContextVar` and injected into every
:class:`logging.LogRecord` via a :class:`logging.Filter` — the stdlib
equivalent of ``structlog.contextvars.bind_contextvars(trace_id=...)``.
"""

from __future__ import annotations

import contextvars
import logging
import uuid

# Header used to propagate the trace id across services (e.g. Desk/Loop/Recap
# -> Vault). Callers may set it; when absent a fresh UUID4 is generated.
TRACE_HEADER = "X-Trace-Id"

# ContextVar holding the trace id for the lifetime of the current request.
# ``None`` outside a traced request (background threads, CLI, startup hooks).
trace_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "vault_trace_id", default=None
)


def new_trace_id() -> str:
    """Generate a fresh trace id (UUID4)."""
    return str(uuid.uuid4())


def get_trace_id() -> str | None:
    """Return the trace id bound to the current context, if any."""
    return trace_id_var.get()


class TraceIdFilter(logging.Filter):
    """Attach the current ``trace_id`` to every log record (best-effort).

    Records emitted while a request is in flight gain a ``trace_id`` attribute
    holding the bound id; records emitted outside a traced context get
    ``trace_id=None`` so a ``%(trace_id)s`` formatter never raises.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = trace_id_var.get()
        return True


def install_trace_id_filter(*loggers: logging.Logger) -> None:
    """Idempotently attach :class:`TraceIdFilter` to the given loggers.

    Attaching to the originating logger (e.g. the ``api.main`` logger) means
    every record it emits carries ``trace_id`` *before* propagation, so any
    downstream handler or formatter — including pytest's ``caplog`` — observes
    it. When called with no arguments the root logger is used.
    """
    targets = loggers or (logging.getLogger(),)
    for log in targets:
        if not any(isinstance(existing, TraceIdFilter) for existing in log.filters):
            log.addFilter(TraceIdFilter())
