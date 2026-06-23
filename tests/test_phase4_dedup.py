"""Tests for pipeline/phase4_dedup.py."""

from datetime import UTC, datetime

from pipeline.phase4_dedup import run_phase4


def _insert_file(db, path, sha256, rel_path="test", mtime=None, excluded=0, hash_status="done"):
    """Insert a file row for dedup testing."""
    if mtime is None:
        mtime = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    parent = rel_path.rsplit("/", 1)[0] if "/" in rel_path else ""
    folder_path = f"/{parent}" if parent else "/"
    db["folder"].insert(
        {
            "path": folder_path,
            "rel_path": parent,
            "parent_id": None,
            "name": parent or "root",
            "depth": 0,
        },
        ignore=True,
    )
    folder_row = next(db["folder"].rows_where("path = ?", [folder_path]), None)
    folder_id = folder_row["id"] if folder_row else 1
    db["file"].insert(
        {
            "folder_id": folder_id,
            "path": f"/{rel_path}",
            "rel_path": rel_path,
            "name": rel_path.split("/")[-1],
            "size_bytes": 100,
            "mtime": mtime,
            "sha256": sha256,
            "hash_status": hash_status,
            "identify_status": "pending",
            "triage_status": "pending",
            "excluded": excluded,
        }
    )


class TestDedup:
    def test_creates_dup_groups(self, test_db, mock_plog):
        _insert_file(test_db, "/a.txt", "abc123", "a.txt")
        _insert_file(test_db, "/b.txt", "abc123", "b.txt")
        from pipeline.config import Config, PathsConfig

        cfg = Config(paths=PathsConfig(corpus_root="/tmp", cache_root="/tmp/.rag-cache"))
        result = run_phase4(test_db, cfg, mock_plog)
        assert result["files_processed"] == 1
        # Both files should share a dup_group_id
        a = next(test_db["file"].rows_where("path = ?", ["/a.txt"]))
        b = next(test_db["file"].rows_where("path = ?", ["/b.txt"]))
        assert a["dup_group_id"] == b["dup_group_id"]

    def test_marks_non_primary(self, test_db, mock_plog):
        _insert_file(test_db, "/a.txt", "abc123", "a.txt", mtime="2026-01-01T00:00:00Z")
        _insert_file(test_db, "/b.txt", "abc123", "b.txt", mtime="2026-01-02T00:00:00Z")
        from pipeline.config import Config, PathsConfig

        cfg = Config(paths=PathsConfig(corpus_root="/tmp", cache_root="/tmp/.rag-cache"))
        run_phase4(test_db, cfg, mock_plog)
        # Shortest path = /a.txt should be primary
        a = next(test_db["file"].rows_where("path = ?", ["/a.txt"]))
        b = next(test_db["file"].rows_where("path = ?", ["/b.txt"]))
        assert a["is_dup_primary"] == 1
        assert b["is_dup_primary"] == 0

    def test_no_duplicates(self, test_db, mock_plog):
        _insert_file(test_db, "/a.txt", "hash1", "a.txt")
        _insert_file(test_db, "/b.txt", "hash2", "b.txt")
        from pipeline.config import Config, PathsConfig

        cfg = Config(paths=PathsConfig(corpus_root="/tmp", cache_root="/tmp/.rag-cache"))
        result = run_phase4(test_db, cfg, mock_plog)
        assert result["files_processed"] == 0

    def test_excluded_ignored(self, test_db, mock_plog):
        _insert_file(test_db, "/a.txt", "abc123", "a.txt", excluded=1)
        _insert_file(test_db, "/b.txt", "abc123", "b.txt")
        from pipeline.config import Config, PathsConfig

        cfg = Config(paths=PathsConfig(corpus_root="/tmp", cache_root="/tmp/.rag-cache"))
        result = run_phase4(test_db, cfg, mock_plog)
        # Only one non-excluded file with this hash, so no dup group
        assert result["files_processed"] == 0

    def test_pending_hash_ignored(self, test_db, mock_plog):
        _insert_file(test_db, "/a.txt", "abc123", "a.txt", hash_status="done")
        _insert_file(test_db, "/b.txt", "abc123", "b.txt", hash_status="pending")
        from pipeline.config import Config, PathsConfig

        cfg = Config(paths=PathsConfig(corpus_root="/tmp", cache_root="/tmp/.rag-cache"))
        result = run_phase4(test_db, cfg, mock_plog)
        # Only one file has hash_status='done', so no dup group
        assert result["files_processed"] == 0

    def test_three_copies(self, test_db, mock_plog):
        _insert_file(test_db, "/a.txt", "abc123", "a.txt", mtime="2026-01-01T00:00:00Z")
        _insert_file(test_db, "/b.txt", "abc123", "b.txt", mtime="2026-01-02T00:00:00Z")
        _insert_file(test_db, "/c.txt", "abc123", "c.txt", mtime="2026-01-03T00:00:00Z")
        from pipeline.config import Config, PathsConfig

        cfg = Config(paths=PathsConfig(corpus_root="/tmp", cache_root="/tmp/.rag-cache"))
        result = run_phase4(test_db, cfg, mock_plog)
        assert result["files_processed"] == 1
        # One primary, two non-primary
        primary_count = test_db.execute(
            "SELECT COUNT(*) FROM file WHERE sha256='abc123' AND is_dup_primary=1"
        ).fetchone()[0]
        non_primary_count = test_db.execute(
            "SELECT COUNT(*) FROM file WHERE sha256='abc123' AND is_dup_primary=0"
        ).fetchone()[0]
        assert primary_count == 1
        assert non_primary_count == 2

    def test_multiple_groups(self, test_db, mock_plog):
        _insert_file(test_db, "/a.txt", "hash_a", "a.txt", mtime="2026-01-01T00:00:00Z")
        _insert_file(test_db, "/b.txt", "hash_a", "b.txt", mtime="2026-01-02T00:00:00Z")
        _insert_file(test_db, "/c.txt", "hash_b", "c.txt", mtime="2026-01-01T00:00:00Z")
        _insert_file(test_db, "/d.txt", "hash_b", "d.txt", mtime="2026-01-02T00:00:00Z")
        from pipeline.config import Config, PathsConfig

        cfg = Config(paths=PathsConfig(corpus_root="/tmp", cache_root="/tmp/.rag-cache"))
        result = run_phase4(test_db, cfg, mock_plog)
        assert result["files_processed"] == 2

    def test_idempotent(self, test_db, mock_plog):
        _insert_file(test_db, "/a.txt", "abc123", "a.txt", mtime="2026-01-01T00:00:00Z")
        _insert_file(test_db, "/b.txt", "abc123", "b.txt", mtime="2026-01-02T00:00:00Z")
        from pipeline.config import Config, PathsConfig

        cfg = Config(paths=PathsConfig(corpus_root="/tmp", cache_root="/tmp/.rag-cache"))
        run_phase4(test_db, cfg, mock_plog)
        # Run again
        run_phase4(test_db, cfg, mock_plog)
        # Should still have exactly 1 primary
        primary_count = test_db.execute(
            "SELECT COUNT(*) FROM file WHERE sha256='abc123' AND is_dup_primary=1"
        ).fetchone()[0]
        assert primary_count == 1
