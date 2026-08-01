"""Best-effort audit logging for ODW Vault (V1.3 F-Vault-1).

The audit trail records who/when/what for sensitive operations (query, file
management, pipeline sync, feedback) to support compliance forensics. Audit
writes are strictly *best-effort*: any failure (e.g. a missing or locked
``audit_log`` table) is swallowed and logged as a warning so the main request
flow is never blocked, and no existing response shape or status code changes.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Column order shared by every audit read path (GET /audit and GET /audit/export).
AUDIT_COLUMNS = "id, ts, actor, action, resource_type, resource_id, detail, status"


def record_audit(
    db,
    actor: str,
    action: str,
    resource_type: str,
    resource_id=None,
    detail: str | None = None,
    status: str = "ok",
) -> None:
    """Record a single audit event.

    BEST-EFFORT: catches every error, logs a warning, and never raises, so a
    failing audit write can never alter the caller's response.

    Args:
        db: A ``pipeline.db.Database`` (or compatible) connection.
        actor: Authenticated principal, or ``anonymous``/configured service name.
        action: Short verb, e.g. ``query``, ``file.upload``, ``file.delete``,
            ``pipeline.sync``, ``feedback``.
        resource_type: Kind of resource affected, e.g. ``query_log``, ``file``.
        resource_id: Optional identifier of the affected resource (stored as text).
        detail: Optional human-readable detail (e.g. the query text or filename).
        status: Outcome marker, defaults to ``ok``.
    """
    try:
        db.conn.execute(
            """INSERT INTO audit_log
               (actor, action, resource_type, resource_id, detail, status)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                actor,
                action,
                resource_type,
                None if resource_id is None else str(resource_id),
                detail,
                status,
            ),
        )
        db.conn.commit()
    except Exception as exc:  # best-effort: never propagate to the caller
        logger.warning("audit write failed (best-effort, ignoring): %s", exc)


def query_audit_records(
    db,
    *,
    start: str | None = None,
    end: str | None = None,
    actor: str | None = None,
    action: str | None = None,
    limit: int = 1000,
) -> list:
    """Read ``audit_log`` records, most recent first, with optional filters.

    This is the single shared audit read seam (V1.4 F-3'). It reuses and extends
    the V1.3 ``GET /audit`` read path — same columns (:data:`AUDIT_COLUMNS`) and
    same ``ORDER BY ts DESC, id DESC`` ordering — and adds time-range filtering
    (``start``/``end`` against ``ts``) plus ``actor`` filtering on top of the
    existing exact-match ``action`` filter.

    Time bounds accept any ISO-8601 string SQLite's ``datetime()`` understands
    (e.g. ``2026-01-01``, ``2026-01-01T00:00:00``); they are normalised via
    ``datetime(?)`` so they compare correctly against the stored
    ``YYYY-MM-DD HH:MM:SS`` timestamps. ``start`` is inclusive (``>=``) and
    ``end`` is inclusive (``<=``).

    Args:
        db: A ``pipeline.db.Database`` (or compatible) connection.
        start: Inclusive lower bound on ``ts`` (ISO-8601). ``None`` = unbounded.
        end: Inclusive upper bound on ``ts`` (ISO-8601). ``None`` = unbounded.
        actor: Exact-match actor filter. ``None`` = no filter.
        action: Exact-match action filter. ``None`` = no filter.
        limit: Maximum number of rows to return.

    Returns:
        A list of dict-like rows (one per audit record).
    """
    conditions: list[str] = []
    params: list = []
    if start:
        conditions.append("ts >= datetime(?)")
        params.append(start)
    if end:
        conditions.append("ts <= datetime(?)")
        params.append(end)
    if actor:
        conditions.append("actor = ?")
        params.append(actor)
    if action:
        conditions.append("action = ?")
        params.append(action)
    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

    return list(
        db.query(
            f"""SELECT {AUDIT_COLUMNS}
                FROM audit_log {where_clause}
                ORDER BY ts DESC, id DESC
                LIMIT ?""",
            [*params, limit],
        )
    )
