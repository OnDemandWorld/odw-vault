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
