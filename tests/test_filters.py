"""Tests for rag/filters.py — resolve_folder_filter with in-memory DB."""


from pipeline.db import migrate, open_db
from rag.filters import resolve_folder_filter


def _seed_db(tmp_path):
    db_path = tmp_path / "test.db"
    db = open_db(db_path)
    migrate(db)
    # Insert folders
    db["folder"].insert_all([
        {"id": 1, "path": "Project/Alpha", "rel_path": "Project/Alpha", "name": "Alpha", "depth": 1, "parent_id": None, "excluded": 0},
        {"id": 2, "path": "Project/Beta", "rel_path": "Project/Beta", "name": "Beta", "depth": 1, "parent_id": None, "excluded": 0},
        {"id": 3, "path": "Archive/Old", "rel_path": "Archive/Old", "name": "Old", "depth": 1, "parent_id": None, "excluded": 0},
    ])
    # Insert files
    db["file"].insert_all([
        {"id": 1, "path": "Project/Alpha/a.txt", "name": "a.txt", "rel_path": "Project/Alpha/a.txt", "folder_id": 1,
         "size_bytes": 100, "mtime": "2026-01-01", "is_dup_primary": 1, "excluded": 0, "sha256": "abc",
         "hash_status": "done", "identify_status": "done", "triage_status": "pending"},
        {"id": 2, "path": "Project/Beta/b.txt", "name": "b.txt", "rel_path": "Project/Beta/b.txt", "folder_id": 2,
         "size_bytes": 100, "mtime": "2026-01-01", "is_dup_primary": 1, "excluded": 0, "sha256": "def",
         "hash_status": "done", "identify_status": "done", "triage_status": "pending"},
        {"id": 3, "path": "Archive/Old/c.txt", "name": "c.txt", "rel_path": "Archive/Old/c.txt", "folder_id": 3,
         "size_bytes": 100, "mtime": "2026-01-01", "is_dup_primary": 1, "excluded": 0, "sha256": "ghi",
         "hash_status": "done", "identify_status": "done", "triage_status": "pending"},
        {"id": 4, "path": "Archive/Old/d.txt", "name": "d.txt", "rel_path": "Archive/Old/d.txt", "folder_id": 3,
         "size_bytes": 100, "mtime": "2026-01-01", "is_dup_primary": 1, "excluded": 1, "sha256": "jkl",
         "hash_status": "done", "identify_status": "done", "triage_status": "pending"},  # excluded
    ])
    db.conn.commit()
    return db


class TestResolveFolderFilter:
    def test_none_filter_returns_none(self, tmp_path):
        db = _seed_db(tmp_path)
        assert resolve_folder_filter(db, {}) is None

    def test_empty_filter_returns_none(self, tmp_path):
        db = _seed_db(tmp_path)
        assert resolve_folder_filter(db, {}) is None

    def test_path_prefix_match(self, tmp_path):
        db = _seed_db(tmp_path)
        result = resolve_folder_filter(db, {"path_prefix": "Project/"})
        assert result is not None
        assert 1 in result  # Alpha
        assert 2 in result  # Beta
        assert 3 not in result  # Archive

    def test_path_prefix_partial_no_match(self, tmp_path):
        db = _seed_db(tmp_path)
        result = resolve_folder_filter(db, {"path_prefix": "Project/Alpha"})
        assert result is not None
        assert 1 in result
        assert 2 not in result

    def test_folder_id_match(self, tmp_path):
        db = _seed_db(tmp_path)
        result = resolve_folder_filter(db, {"folder_id": 2})
        assert result == {2}

    def test_excluded_files_ignored(self, tmp_path):
        db = _seed_db(tmp_path)
        result = resolve_folder_filter(db, {"path_prefix": "Archive/"})
        assert result is not None
        assert 3 in result
        assert 4 not in result  # excluded

    def test_no_matches_returns_none(self, tmp_path):
        db = _seed_db(tmp_path)
        result = resolve_folder_filter(db, {"path_prefix": "Nonexistent/"})
        assert result is None

    def test_folder_id_no_match(self, tmp_path):
        db = _seed_db(tmp_path)
        result = resolve_folder_filter(db, {"folder_id": 999})
        assert result is None
