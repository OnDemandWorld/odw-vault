"""Lightweight distributed-tracing spans for the Vault HTTP service (V1.6 F-2).

Strictly additive and best-effort: this module never changes a business
response body or status code. It builds on the V1.5 ``trace_id`` mechanism
(:mod:`api.tracing`) and adds a minimal span model on top:

* :class:`Span` — ``name / trace_id / span_id / parent_span_id / start_ms /
  duration_ms / attrs / status``.
* :func:`start_span` / :meth:`Span.end` — a :class:`contextvars.ContextVar`
  span stack gives automatic parent/child nesting; child spans reuse the
  current ``trace_id`` (starting a fresh one only when no trace is bound).
* Sampling — ``TRACE_SAMPLE_RATE`` (env, default ``1.0``). Unsampled root
  spans are no-ops and their children inherit the no-op decision, so nothing
  is recorded or exported for an unsampled trace.
* Exporters — :class:`ConsoleSpanExporter` (default, logs the span dict via
  the module logger) and :class:`OtlpHttpSpanExporter` (best-effort HTTP POST
  to ``OTLP_ENDPOINT`` with a short timeout, silent degradation on failure).
  ``TRACE_EXPORTER=console|otlp|none`` selects one (default ``console``).

The :func:`span` context manager is the intended call-site API::

    with span("vault.query") as sp:
        sp.set_attr("user", req.user)
        ...

It records ``status="error"`` when the wrapped block raises, then re-raises so
the caller's exception (e.g. an ``HTTPException``) propagates unchanged.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from api.tracing import get_trace_id, new_trace_id

logger = logging.getLogger(__name__)

# ContextVar holding the tuple of currently-active spans (a stack). The last
# element is the innermost span and becomes the parent of any new span. Empty
# outside a traced block. Held in a ContextVar so each request context (FastAPI
# runs handlers in their own context copy) gets an isolated stack.
_span_stack_var: contextvars.ContextVar[tuple[Span, ...]] = contextvars.ContextVar(
    "vault_span_stack", default=()
)


def _new_span_id() -> str:
    """Generate a fresh span id (16 hex chars, distinct from the trace id)."""
    return uuid.uuid4().hex[:16]


# ---------------------------------------------------------------------------
# Configuration (env-based, read per-call so it can be toggled / monkeypatched)
# ---------------------------------------------------------------------------


def _sample_rate() -> float:
    """Resolve ``TRACE_SAMPLE_RATE`` (env, default ``1.0``), clamped to [0, 1]."""
    raw = os.environ.get("TRACE_SAMPLE_RATE", "").strip()
    if not raw:
        return 1.0
    try:
        rate = float(raw)
    except ValueError:
        logger.warning("Invalid TRACE_SAMPLE_RATE %r; defaulting to 1.0", raw)
        return 1.0
    return max(0.0, min(1.0, rate))


def _exporter_name() -> str:
    """Resolve ``TRACE_EXPORTER`` (env, default ``console``)."""
    return os.environ.get("TRACE_EXPORTER", "console").strip().lower() or "console"


def _otlp_endpoint() -> str:
    """Resolve ``OTLP_ENDPOINT`` (env). Empty string disables OTLP export."""
    return os.environ.get("OTLP_ENDPOINT", "").strip()


def _should_sample(rate: float) -> bool:
    """Make a root sampling decision for the given rate."""
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    return random.random() < rate


# ---------------------------------------------------------------------------
# Span model
# ---------------------------------------------------------------------------


@dataclass
class Span:
    """A single unit of traced work.

    The public fields mirror the design contract; ``sampled`` / ``_ended`` /
    ``_start_mono`` are internal bookkeeping and are excluded from
    :meth:`to_dict`.
    """

    name: str
    trace_id: str
    span_id: str
    parent_span_id: str | None
    start_ms: float
    duration_ms: float | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    sampled: bool = True
    _ended: bool = field(default=False, repr=False)
    _start_mono: float = field(default=0.0, repr=False)

    @property
    def is_noop(self) -> bool:
        """True when this span is unsampled and therefore records nothing."""
        return not self.sampled

    def set_attr(self, key: str, value: Any) -> None:
        """Attach an attribute (best-effort; no-op spans still accept writes)."""
        self.attrs[key] = value

    def end(self, status: str = "ok") -> None:
        """Finish the span, pop it from the stack, and export it (if sampled).

        Idempotent. Best-effort: any export failure is swallowed so ending a
        span can never break the surrounding business logic.
        """
        if self._ended:
            return
        self._ended = True
        self.status = status
        self.duration_ms = round((time.monotonic() - self._start_mono) * 1000, 3)

        # Pop this span off the stack (forgiving of out-of-order ends).
        stack = _span_stack_var.get()
        if stack and stack[-1] is self:
            _span_stack_var.set(stack[:-1])

        if not self.sampled:
            return
        with contextlib.suppress(Exception):
            export_span(self)

    def to_dict(self) -> dict[str, Any]:
        """Return the span as a plain dict (the documented 8-field contract)."""
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "start_ms": self.start_ms,
            "duration_ms": self.duration_ms,
            "attrs": dict(self.attrs),
            "status": self.status,
        }


def _noop_span(name: str) -> Span:
    """Build an unsampled span that records/exports nothing (used on failure)."""
    return Span(
        name=name,
        trace_id=get_trace_id() or "",
        span_id=_new_span_id(),
        parent_span_id=None,
        start_ms=round(time.time() * 1000, 3),
        sampled=False,
        _start_mono=time.monotonic(),
    )


def start_span(name: str, attrs: dict[str, Any] | None = None) -> Span:
    """Start a span, push it on the stack, and return it.

    Children auto-parent to the current innermost span and inherit both its
    ``trace_id`` and its sampling decision. A root span reuses the bound V1.5
    ``trace_id`` (starting a fresh one when none is bound) and rolls the
    sampling decision from ``TRACE_SAMPLE_RATE``.

    Best-effort: any unexpected error yields a no-op span so tracing can never
    break a request.
    """
    try:
        stack = _span_stack_var.get()
        parent = stack[-1] if stack else None
        if parent is not None:
            trace_id = parent.trace_id
            sampled = parent.sampled
            parent_span_id = parent.span_id
        else:
            trace_id = get_trace_id() or new_trace_id()
            sampled = _should_sample(_sample_rate())
            parent_span_id = None

        span_obj = Span(
            name=name,
            trace_id=trace_id,
            span_id=_new_span_id(),
            parent_span_id=parent_span_id,
            start_ms=round(time.time() * 1000, 3),
            attrs=dict(attrs or {}),
            sampled=sampled,
            _start_mono=time.monotonic(),
        )
        _span_stack_var.set((*stack, span_obj))
        return span_obj
    except Exception:  # pragma: no cover - defensive, tracing must never fail
        logger.debug("start_span failed for %r", name, exc_info=True)
        return _noop_span(name)


def get_current_span() -> Span | None:
    """Return the innermost active span, or ``None`` outside a traced block."""
    stack = _span_stack_var.get()
    return stack[-1] if stack else None


@contextlib.contextmanager
def span(name: str, attrs: dict[str, Any] | None = None):
    """Context manager wrapping a block in a span.

    Records ``status="error"`` and re-raises if the block raises (so the
    caller's exception/status code is unchanged); otherwise ``status="ok"``.
    """
    sp = start_span(name, attrs)
    try:
        yield sp
    except Exception:
        sp.end(status="error")
        raise
    else:
        sp.end(status="ok")


# ---------------------------------------------------------------------------
# Exporters (VS2)
# ---------------------------------------------------------------------------


class SpanExporter:
    """Base exporter interface."""

    def export(self, span_obj: Span) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class ConsoleSpanExporter(SpanExporter):
    """Default exporter: logs the span dict via the module logger."""

    def export(self, span_obj: Span) -> None:
        logger.info("span %s", span_obj.to_dict())


class OtlpHttpSpanExporter(SpanExporter):
    """Best-effort OTLP/HTTP exporter.

    POSTs a minimal OTLP-shaped JSON payload to ``OTLP_ENDPOINT`` with a short
    timeout. Any failure (no endpoint, network error, non-2xx, timeout) is
    swallowed and logged at debug level — export never blocks or breaks a
    request. Uses only the standard library (``urllib``) so no new dependency
    is introduced.
    """

    def __init__(self, endpoint: str | None = None, timeout: float = 2.0) -> None:
        self.endpoint = endpoint if endpoint is not None else _otlp_endpoint()
        self.timeout = timeout

    def _payload(self, span_obj: Span) -> bytes:
        start_ns = int(span_obj.start_ms * 1_000_000)
        end_ns = int((span_obj.start_ms + (span_obj.duration_ms or 0.0)) * 1_000_000)
        body = {
            "resourceSpans": [
                {
                    "scopeSpans": [
                        {
                            "spans": [
                                {
                                    "traceId": span_obj.trace_id,
                                    "spanId": span_obj.span_id,
                                    "parentSpanId": span_obj.parent_span_id or "",
                                    "name": span_obj.name,
                                    "startTimeUnixNano": start_ns,
                                    "endTimeUnixNano": end_ns,
                                    "attributes": [
                                        {"key": k, "value": {"stringValue": str(v)}}
                                        for k, v in span_obj.attrs.items()
                                    ],
                                    "status": {"code": 1 if span_obj.status == "ok" else 2},
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        return json.dumps(body).encode("utf-8")

    def export(self, span_obj: Span) -> None:
        if not self.endpoint:
            return
        try:
            import urllib.request

            req = urllib.request.Request(
                self.endpoint,
                data=self._payload(span_obj),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp.read()
        except Exception:  # best-effort, silent degradation
            logger.debug("OTLP span export to %s failed", self.endpoint, exc_info=True)


class _NullExporter(SpanExporter):
    """Exporter for ``TRACE_EXPORTER=none`` — discards everything."""

    def export(self, span_obj: Span) -> None:
        return None


# A test/override hook. When set, :func:`get_exporter` returns this instead of
# resolving ``TRACE_EXPORTER`` — lets tests capture spans deterministically.
_exporter_override: SpanExporter | None = None


def set_exporter(exporter: SpanExporter | None) -> None:
    """Install (or clear, with ``None``) a process-wide exporter override."""
    global _exporter_override
    _exporter_override = exporter


def get_exporter() -> SpanExporter:
    """Resolve the active exporter (override → ``TRACE_EXPORTER`` → console)."""
    if _exporter_override is not None:
        return _exporter_override
    name = _exporter_name()
    if name == "none":
        return _NullExporter()
    if name == "otlp":
        return OtlpHttpSpanExporter()
    return ConsoleSpanExporter()


def export_span(span_obj: Span) -> None:
    """Export a span via the active exporter (best-effort, never raises)."""
    try:
        get_exporter().export(span_obj)
    except Exception:  # best-effort, silent degradation
        logger.debug("span export failed for %r", span_obj.name, exc_info=True)
