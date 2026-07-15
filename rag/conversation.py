"""Conversation management for multi-turn dialog.

Handles conversation lifecycle (create, list, get messages) and
provides the sliding-window history for prompt injection.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_SIZE = 5  # number of recent turns (user+assistant pairs)


def create_conversation(db, user: str | None = None, title: str | None = None) -> str:
    """Create a new conversation and return its ID."""
    conv_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    db.execute(
        "INSERT INTO conversation (id, user, title, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        [conv_id, user, title, now, now],
    )
    db.conn.commit()
    logger.info("Created conversation %s", conv_id)
    return conv_id


def get_or_create_conversation(
    db, conversation_id: str | None, user: str | None = None
) -> str:
    """Return an existing conversation ID or create a new one."""
    if conversation_id:
        rows = list(db.query(
            "SELECT id FROM conversation WHERE id = ?", [conversation_id]
        ))
        if rows:
            return conversation_id
        logger.warning(
            "conversation_id %s not found, creating new", conversation_id
        )
    return create_conversation(db, user=user)


def add_message(
    db,
    conversation_id: str,
    role: str,
    content: str,
    query_log_id: int | None = None,
) -> int:
    """Add a message to a conversation and return its ID."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    cursor = db.execute(
        "INSERT INTO message (conversation_id, role, content, query_log_id, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        [conversation_id, role, content, query_log_id, now],
    )
    # Update conversation updated_at and title (if first user message)
    db.execute(
        "UPDATE conversation SET updated_at = ? WHERE id = ?",
        [now, conversation_id],
    )
    if role == "user":
        # Set title from first user message if not already set
        db.execute(
            "UPDATE conversation SET title = ? "
            "WHERE id = ? AND (title IS NULL OR title = '')",
            [_truncate_title(content), conversation_id],
        )
    db.conn.commit()
    return cursor.lastrowid


def get_history(
    db,
    conversation_id: str,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> list[dict]:
    """Get recent conversation history as a list of {role, content} dicts.

    Returns the most recent `window_size` turns (each turn = 1 user + 1 assistant
    message), ordered chronologically. This is used for prompt injection.
    """
    # Each turn is 2 messages (user + assistant), so fetch window_size * 2
    limit = window_size * 2
    rows = db.query(
        "SELECT role, content FROM message "
        "WHERE conversation_id = ? "
        "ORDER BY created_at DESC LIMIT ?",
        [conversation_id, limit],
    )

    # Reverse to chronological order
    rows = list(reversed(list(rows)))
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def list_conversations(db, user: str | None = None, limit: int = 50) -> list[dict]:
    """List conversations, most recently updated first."""
    if user:
        rows = db.query(
            "SELECT c.id, c.title, c.user, c.created_at, c.updated_at, "
            "COUNT(m.id) as message_count "
            "FROM conversation c LEFT JOIN message m ON m.conversation_id = c.id "
            "WHERE c.user = ? "
            "GROUP BY c.id "
            "ORDER BY c.updated_at DESC LIMIT ?",
            [user, limit],
        )
    else:
        rows = db.query(
            "SELECT c.id, c.title, c.user, c.created_at, c.updated_at, "
            "COUNT(m.id) as message_count "
            "FROM conversation c LEFT JOIN message m ON m.conversation_id = c.id "
            "GROUP BY c.id "
            "ORDER BY c.updated_at DESC LIMIT ?",
            [limit],
        )
    return [dict(r) for r in rows]


def get_conversation_messages(db, conversation_id: str) -> list[dict]:
    """Get all messages for a conversation in chronological order."""
    rows = db.query(
        "SELECT id, role, content, query_log_id, created_at "
        "FROM message WHERE conversation_id = ? "
        "ORDER BY created_at",
        [conversation_id],
    )
    return [dict(r) for r in rows]


def delete_conversation(db, conversation_id: str) -> bool:
    """Delete a conversation and all its messages. Returns True if deleted."""
    # Messages are cascade-deleted via FK
    db.execute("DELETE FROM conversation WHERE id = ?", [conversation_id])
    db.conn.commit()
    rows = list(db.query("SELECT changes() as c"))
    return rows[0]["c"] > 0


def _default_title(user: str | None) -> str:
    """Generate a default conversation title."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    if user:
        return f"Chat with {user} — {now}"
    return f"Conversation — {now}"


def _truncate_title(text: str, max_len: int = 60) -> str:
    """Truncate text for use as a conversation title."""
    text = text.strip().replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "…"
