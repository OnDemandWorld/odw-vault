"""Tests for V1.1 M4 multi-workspace (knowledge base) isolation.

Workspaces are a strictly-additive logical isolation layer over the shared
corpus: every ``file`` row carries a ``workspace`` label (default ``default``).
These tests cover:

- the idempotent migration (column + index + backfill),
- ``POST /files/upload`` workspace tagging,
- workspace filtering at the retrieval file-filtering seam
  (``rag.filters.resolve_folder_filter``) — the authoritative isolation check,
- ``POST /query`` isolation end-to-end (retrieval mocked at the same seam used
  by ``tests/test_api_query.py``, but routing through the *real*
  ``resolve_folder_filter`` so the isolation assertion is genuine),
- ``GET /workspaces`` and ``GET /files?workspace=`` listing/filtering.

No Ollama / Chroma / network access is required.
"""

from __future__ import annotations

import io
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import sqlite_utils
from fastapi.testclient import TestClient

from api.main import app
from pipeline.db import migrate, open_db
from rag.filters import resolve_folder_filter
from rag.retrieval import Hit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_db(tmp_path: Path) -> sqlite_utils.Database:
    """Create a migrated test DB with check_same_thread=False for TestClient."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    db = sqlite_utils.Database(conn)
    migrate(db)
    return db


def _make_cfg(tmp_path: Path) -> MagicMock:
    """Build a MagicMock config exposing every attribute the endpoints touch."""
    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    cfg = MagicMock()
    cfg.ollama.host = "http://localhost:11434"
    cfg.corpus_root_path = corpus
    cfg.chroma_root_path = str(tmp_path / "chroma")
    cfg.models.embedding.collection_suffix = "test"
    cfg.models.embedding.name = "test-embed"
    cfg.models.generation.name = "test-gen"
    cfg.models.generation.thinking = False
    cfg.models.reranker.enabled = False
    cfg.models.contextual_retrieval.enabled = False
    return cfg


def _seed_workspaces(db: sqlite_utils.Database) -> dict[str, int]:
    """Seed one folder and three files across two workspaces (A, A, B).

    Returns a mapping of rel_path -> file id.
    """
    db["folder"].insert({
        "path": "root", "rel_path": "root", "name": "root", "depth": 0, "excluded": 0,
    })
    db.conn.commit()
    folder_id = next(iter(db.query("SELECT id FROM folder LIMIT 1")))["id"]

    specs = [
        ("alpha_one.txt", "alpha"),
        ("alpha_two.txt", "alpha"),
        ("beta_one.txt", "beta"),
    ]
    for name, workspace in specs:
        db["file"].insert({
            "folder_id": folder_id,
            "path": f"/corpus/{name}",
            "rel_path": name,
            "name": name,
            "size_bytes": 100,
            "mtime": "2026-01-01T00:00:00",
            "sha256": f"sha-{name}",
            "hash_status": "done",
            "identify_status": "done",
            "triage_status": "pending",
            "is_dup_primary": 1,
            "excluded": 0,
            "workspace": workspace,
        })
    db.conn.commit()
    return {
        name: next(iter(db.query("SELECT id FROM file WHERE name = ?", [name])))["id"]
        for name, _ in specs
    }


def _gen_result() -> dict:
    return {
        "answer": "Answer grounded in the retrieved context [1].",
        "citations": [],
        "generation_ms": 5.0,
        "model": "test-gen",
        "refused": False,
    }


def _fake_retrieve_factory(db: sqlite_utils.Database):
    """Build a retrieve() stand-in that routes through the REAL
    resolve_folder_filter, so isolation assertions are genuine while still
    avoiding Ollama/Chroma. Returns one Hit per allowed file.
    """

    def _fake_retrieve(
        query,
        db=None,
        chroma_client=None,
        chroma_path=None,
        cfg=None,
        folder_filter=None,
        top_k_chunks=None,
        use_reranker=None,
        use_augmentation=None,
    ):
        allowed = resolve_folder_filter(db, folder_filter) if folder_filter else None
        rows = list(db.query(
            "SELECT id, folder_id, rel_path FROM file WHERE excluded = 0 ORDER BY id"
        ))
        if allowed is not None:
            rows = [r for r in rows if r["id"] in allowed]
        hits = [
            Hit(
                chunk_id=r["id"],
                file_id=r["id"],
                folder_id=r["folder_id"],
                rel_path=r["rel_path"],
                page_start=None,
                text=f"text of {r['rel_path']}",
                dense_score=0.9,
                bm25_score=0.8,
                fused_score=0.85,
            )
            for r in rows
        ]
        return hits, {"retrieval_ms": 1.0, "query_lang": "en"}

    return _fake_retrieve


# ---------------------------------------------------------------------------
# V1 — migration
# ---------------------------------------------------------------------------


class TestWorkspaceMigration:
    def test_file_has_workspace_column_and_index(self, tmp_path):
        db = open_db(tmp_path / "m.db")
        migrate(db)
        cols = [r[1] for r in db.execute("PRAGMA table_info(file)").fetchall()]
        assert "workspace" in cols
        idx = db.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_file_workspace'"
        ).fetchone()
        assert idx is not None

    def test_migration_is_idempotent(self, tmp_path):
        db = open_db(tmp_path / "m.db")
        migrate(db)
        migrate(db)  # second run must not raise
        cols = [r[1] for r in db.execute("PRAGMA table_info(file)").fetchall()]
        assert cols.count("workspace") == 1

    def test_existing_rows_backfill_to_default(self, tmp_path):
        db = open_db(tmp_path / "m.db")
        migrate(db)
        db["folder"].insert({"path": "a", "rel_path": "a", "name": "a", "depth": 0, "excluded": 0})
        db.conn.commit()
        fid = next(iter(db.execute("SELECT id FROM folder LIMIT 1")))[0]
        # Insert WITHOUT specifying workspace -> column default applies.
        db.execute(
            "INSERT INTO file (folder_id, path, rel_path, name, size_bytes, mtime) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (fid, "/x", "x.txt", "x.txt", 1, "2026-01-01"),
        )
        db.conn.commit()
        ws = next(iter(db.execute("SELECT workspace FROM file WHERE name='x.txt'")))[0]
        assert ws == "default"


# ---------------------------------------------------------------------------
# V2 — upload tagging
# ---------------------------------------------------------------------------


class TestUploadWorkspaceTag:
    def test_upload_with_workspace_tags_rows(self, tmp_path):
        db = _make_test_db(tmp_path)
        cfg = _make_cfg(tmp_path)
        with patch("api.main._load_config", return_value=cfg), \
             patch("api.main._get_db", return_value=db):
            client = TestClient(app)
            response = client.post(
                "/files/upload",
                files=[("files", ("a.txt", io.BytesIO(b"aaa"), "text/plain"))],
                data={"workspace": "alpha"},
            )
        assert response.status_code == 200
        data = response.json()
        assert data == {"uploaded": 1, "failed": []}  # response shape unchanged
        rows = list(db.query("SELECT workspace FROM file WHERE name='a.txt'"))
        assert rows[0]["workspace"] == "alpha"

    def test_upload_without_workspace_defaults(self, tmp_path):
        db = _make_test_db(tmp_path)
        cfg = _make_cfg(tmp_path)
        with patch("api.main._load_config", return_value=cfg), \
             patch("api.main._get_db", return_value=db):
            client = TestClient(app)
            response = client.post(
                "/files/upload",
                files=[("files", ("b.txt", io.BytesIO(b"bbb"), "text/plain"))],
            )
        assert response.status_code == 200
        rows = list(db.query("SELECT workspace FROM file WHERE name='b.txt'"))
        assert rows[0]["workspace"] == "default"


# ---------------------------------------------------------------------------
# V3 — isolation at the retrieval file-filtering seam
# ---------------------------------------------------------------------------


class TestWorkspaceFilterSeam:
    def test_workspace_filter_returns_only_that_workspace(self, tmp_path):
        db = _make_test_db(tmp_path)
        ids = _seed_workspaces(db)
        result = resolve_folder_filter(db, {"workspace": "alpha"})
        assert result == {ids["alpha_one.txt"], ids["alpha_two.txt"]}
        assert ids["beta_one.txt"] not in result

    def test_no_workspace_filter_scopes_whole_corpus(self, tmp_path):
        db = _make_test_db(tmp_path)
        _seed_workspaces(db)
        # V1.0 behavior: empty filter => None (no scoping).
        assert resolve_folder_filter(db, {}) is None

    def test_workspace_composes_with_path_prefix(self, tmp_path):
        # path_prefix matches folder.rel_path, so seed two folders: files in
        # "Project/Alpha" (workspace alpha) and "Project/Beta" (workspace beta).
        db = _make_test_db(tmp_path)
        db["folder"].insert_all([
            {"path": "Project/Alpha", "rel_path": "Project/Alpha", "name": "Alpha",
             "depth": 1, "excluded": 0},
            {"path": "Project/Beta", "rel_path": "Project/Beta", "name": "Beta",
             "depth": 1, "excluded": 0},
        ])
        db.conn.commit()
        alpha_folder = next(iter(db.query(
            "SELECT id FROM folder WHERE rel_path='Project/Alpha'")))["id"]
        beta_folder = next(iter(db.query(
            "SELECT id FROM folder WHERE rel_path='Project/Beta'")))["id"]

        def _add(name, folder_id, workspace):
            db["file"].insert({
                "folder_id": folder_id, "path": f"/c/{name}", "rel_path": name,
                "name": name, "size_bytes": 1, "mtime": "2026-01-01",
                "sha256": f"s-{name}", "is_dup_primary": 1, "excluded": 0,
                "workspace": workspace,
            })

        _add("a.txt", alpha_folder, "alpha")
        _add("b.txt", beta_folder, "beta")
        db.conn.commit()
        a_id = next(iter(db.query("SELECT id FROM file WHERE name='a.txt'")))["id"]

        # Intersection: workspace=alpha AND path_prefix=Project/ => only a.txt.
        result = resolve_folder_filter(db, {"workspace": "alpha", "path_prefix": "Project/"})
        assert result == {a_id}
        # Conflicting intersection: workspace=beta AND path_prefix=Project/Alpha => empty.
        assert resolve_folder_filter(
            db, {"workspace": "beta", "path_prefix": "Project/Alpha"}
        ) is None

    def test_workspace_with_no_files_returns_none(self, tmp_path):
        db = _make_test_db(tmp_path)
        _seed_workspaces(db)
        assert resolve_folder_filter(db, {"workspace": "ghost"}) is None


# ---------------------------------------------------------------------------
# V3 — isolation through POST /query (retrieval mocked, real filter seam)
# ---------------------------------------------------------------------------


class TestQueryWorkspaceIsolation:
    def _query(self, db, cfg, payload):
        with patch("api.main._load_config", return_value=cfg), \
             patch("api.main.ollama.Client") as mock_ollama, \
             patch("api.main.chromadb.PersistentClient") as mock_chroma, \
             patch("api.main.retrieve", side_effect=_fake_retrieve_factory(db)), \
             patch("api.main.generate_answer", return_value=_gen_result()), \
             patch("api.main._get_db", return_value=db):
            mock_ollama.return_value.list.return_value = {"models": []}
            mock_chroma.return_value.get_collection.return_value = MagicMock()
            client = TestClient(app)
            return client.post("/query", json=payload)

    def test_query_workspace_a_excludes_b(self, tmp_path):
        db = _make_test_db(tmp_path)
        _seed_workspaces(db)
        resp = self._query(db, _make_cfg(tmp_path),
                           {"query": "anything", "folder_filter": {"workspace": "alpha"}})
        assert resp.status_code == 200
        paths = {c["rel_path"] for c in resp.json()["retrieved_chunks"]}
        assert paths == {"alpha_one.txt", "alpha_two.txt"}
        assert "beta_one.txt" not in paths

    def test_query_workspace_b_only(self, tmp_path):
        db = _make_test_db(tmp_path)
        _seed_workspaces(db)
        resp = self._query(db, _make_cfg(tmp_path),
                           {"query": "anything", "folder_filter": {"workspace": "beta"}})
        assert resp.status_code == 200
        paths = {c["rel_path"] for c in resp.json()["retrieved_chunks"]}
        assert paths == {"beta_one.txt"}

    def test_query_without_workspace_returns_both_v10_compat(self, tmp_path):
        db = _make_test_db(tmp_path)
        _seed_workspaces(db)
        resp = self._query(db, _make_cfg(tmp_path), {"query": "anything"})
        assert resp.status_code == 200
        paths = {c["rel_path"] for c in resp.json()["retrieved_chunks"]}
        assert paths == {"alpha_one.txt", "alpha_two.txt", "beta_one.txt"}

    def test_query_unknown_workspace_returns_422(self, tmp_path):
        db = _make_test_db(tmp_path)
        _seed_workspaces(db)
        resp = self._query(db, _make_cfg(tmp_path),
                           {"query": "anything", "folder_filter": {"workspace": "ghost"}})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# V4 — /workspaces listing and /files?workspace= filtering
# ---------------------------------------------------------------------------


class TestWorkspaceListing:
    def test_get_workspaces_lists_both_with_counts(self, tmp_path):
        db = _make_test_db(tmp_path)
        _seed_workspaces(db)
        with patch("api.main._get_db", return_value=db):
            client = TestClient(app)
            resp = client.get("/workspaces")
        assert resp.status_code == 200
        data = resp.json()
        by_name = {w["workspace"]: w["file_count"] for w in data["workspaces"]}
        assert by_name == {"alpha": 2, "beta": 1}
        assert data["total"] == 2

    def test_files_filter_by_workspace(self, tmp_path):
        db = _make_test_db(tmp_path)
        _seed_workspaces(db)
        with patch("api.main._get_db", return_value=db):
            client = TestClient(app)
            all_resp = client.get("/files")
            alpha_resp = client.get("/files?workspace=alpha")
            beta_resp = client.get("/files?workspace=beta")
        # Unfiltered => whole corpus (V1.0 behavior).
        assert all_resp.json()["total"] == 3
        alpha_names = {i["name"] for i in alpha_resp.json()["items"]}
        assert alpha_names == {"alpha_one.txt", "alpha_two.txt"}
        beta_names = {i["name"] for i in beta_resp.json()["items"]}
        assert beta_names == {"beta_one.txt"}

    def test_folders_filter_by_workspace(self, tmp_path):
        db = _make_test_db(tmp_path)
        _seed_workspaces(db)
        # Add a folder that only holds a beta file's sibling-less folder.
        db["folder"].insert({
            "path": "beta_only", "rel_path": "beta_only", "name": "beta_only",
            "depth": 0, "excluded": 0,
        })
        db.conn.commit()
        with patch("api.main._get_db", return_value=db):
            client = TestClient(app)
            all_resp = client.get("/folders")
            alpha_resp = client.get("/folders?workspace=alpha")
        # Unfiltered lists both folders.
        assert len(all_resp.json()) == 2
        # Workspace scoping keeps only folders with files in that workspace.
        alpha_folders = {f["rel_path"] for f in alpha_resp.json()}
        assert alpha_folders == {"root"}
