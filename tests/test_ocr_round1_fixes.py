"""Regression tests for OCR full-audit round 1 fixes.

Covers findings #1 (exclusion enforced at query time), #2 (folder exclusion
cascade), #14 (re-embed GC of stale vectors), #15 (rechunk Chroma cleanup).
See .ocr-reports/full-audit.json for the finding details.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.db import migrate, open_db
from tests.conftest import seed_test_extractions, seed_test_files


def _make_cfg(tmp_path):
    from pipeline.config import (
        AppConfig,
        ChunkConfig,
        ContextualRetrievalConfig,
        EmbeddingConfig,
        ExtractConfig,
        GenerationConfig,
        GenerationRuntimeConfig,
        ModelsConfig,
        OllamaConfig,
        PathsConfig,
        RerankerConfig,
        RetrievalConfig,
        SummarizationConfig,
    )

    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    cache = corpus / ".rag-cache"
    cache.mkdir(exist_ok=True)
    chroma = tmp_path / "chroma"
    chroma.mkdir(exist_ok=True)

    return AppConfig(
        paths=PathsConfig(
            corpus_root=str(corpus),
            cache_root=str(cache),
            chroma_root=str(chroma),
        ),
        ollama=OllamaConfig(),
        models=ModelsConfig(
            embedding=EmbeddingConfig(name="test-embed", collection_suffix="test"),
            summarization=SummarizationConfig(name="test-summarize"),
            contextual_retrieval=ContextualRetrievalConfig(enabled=False, name="test-context"),
            generation=GenerationConfig(
                name="test-gen", fallback_name="test-fb", alternate_name="test-alt"
            ),
            reranker=RerankerConfig(enabled=False),
        ),
        generation_runtime=GenerationRuntimeConfig(refuse_on_empty_context=True),
        chunk=ChunkConfig(),
        retrieval=RetrievalConfig(),
        extract=ExtractConfig(),
    )


def _seed_db(tmp_path):
    db = open_db(tmp_path / "test.db")
    migrate(db)
    return db


def _insert_folder(db, path, rel_path, name, depth, parent_id=None, excluded=0):
    table = db["folder"]
    table.insert(
        {
            "path": path,
            "rel_path": rel_path,
            "parent_id": parent_id,
            "name": name,
            "depth": depth,
            "excluded": excluded,
        }
    )
    db.conn.commit()
    return table.last_rowid


class TestFolderExclusionCascade:
    """OCR #2: excluding a folder must cascade to descendants and their files."""

    def _tree(self, db):
        root = _insert_folder(db, "/c", ".", "root", 0)
        parent = _insert_folder(db, "/c/hr", "hr", "hr", 1, root)
        child = _insert_folder(db, "/c/hr/2024", "hr/2024", "2024", 2, parent)
        other = _insert_folder(db, "/c/eng", "eng", "eng", 1, root)
        f_parent = seed_test_files(db, files=[{"name": "a.txt", "folder_id": parent}])[0]
        f_child = seed_test_files(db, files=[{"name": "b.txt", "folder_id": child}])[0]
        f_other = seed_test_files(db, files=[{"name": "c.txt", "folder_id": other}])[0]
        return root, parent, child, other, f_parent, f_child, f_other

    def test_folder_exclusion_cascades(self, tmp_path):
        from pipeline.config import Config, PathsConfig
        from pipeline.phase7_exclude import run_phase7_exclude

        db = _seed_db(tmp_path)
        cfg = Config(paths=PathsConfig(corpus_root=str(tmp_path), cache_root=str(tmp_path)))
        root, parent, child, other, f_parent, f_child, f_other = self._tree(db)

        run_phase7_exclude(db, cfg, "folder", parent, "confidential")

        assert db["folder"].get(parent)["excluded"] == 1
        assert db["folder"].get(child)["excluded"] == 1
        assert db["folder"].get(root)["excluded"] == 0
        assert db["folder"].get(other)["excluded"] == 0
        assert db["file"].get(f_parent)["excluded"] == 1
        assert db["file"].get(f_child)["excluded"] == 1
        assert db["file"].get(f_other)["excluded"] == 0
        assert db["file"].get(f_child)["exclusion_reason"] == "confidential"

    def test_batch_csv_folder_exclusion_cascades(self, tmp_path):
        from pipeline.config import Config, PathsConfig
        from pipeline.phase7_exclude import run_phase7_exclude_batch

        db = _seed_db(tmp_path)
        cfg = Config(paths=PathsConfig(corpus_root=str(tmp_path), cache_root=str(tmp_path)))
        _root, parent, child, _other, f_parent, f_child, _f_other = self._tree(db)

        csv_path = tmp_path / "exclusions.csv"
        csv_path.write_text(f"target,id,reason\nfolder,{parent},secret\n", encoding="utf-8")
        applied, skipped = run_phase7_exclude_batch(db, cfg, csv_path)

        assert (applied, skipped) == (1, 0)
        assert db["folder"].get(child)["excluded"] == 1
        assert db["file"].get(f_parent)["excluded"] == 1
        assert db["file"].get(f_child)["excluded"] == 1


class TestRetrievalExclusion:
    """OCR #1: excluded files must not surface via dense or BM25 retrieval."""

    def _run_retrieve(self, db, cfg, tmp_path, dense_metadatas, dense_docs):
        from rag.retrieval import _retrieve_with_client

        mock_coll = MagicMock()
        mock_coll.metadata = {"embedding_model": "test-embed"}
        mock_coll.query.return_value = {
            "ids": [[f"c_{m['chunk_id']}" for m in dense_metadatas]],
            "metadatas": [dense_metadatas],
            "documents": [dense_docs],
            "distances": [[0.1] * len(dense_metadatas)],
        }
        client = MagicMock()

        def get_collection(name):
            if name.startswith("chunks__"):
                return mock_coll
            raise Exception("collection absent")

        client.get_collection.side_effect = get_collection

        with (
            patch("ollama.Client") as mock_ollama,
            patch("rag.retrieval._detect_language", return_value="en"),
        ):
            mock_ollama.return_value.embed.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
            hits, _metrics = _retrieve_with_client(
                "energy", db, client, str(tmp_path / "chroma"), cfg
            )
        return hits

    def _seed_two_files(self, db, excluded=0, folder_excluded=0):
        fid_ok = _insert_folder(db, "/c/ok", "ok", "ok", 0)
        fid_bad = _insert_folder(db, "/c/bad", "bad", "bad", 0, excluded=folder_excluded)
        f_ok = seed_test_files(db, files=[{"name": "ok.txt", "folder_id": fid_ok}])[0]
        f_bad = seed_test_files(
            db, files=[{"name": "bad.txt", "folder_id": fid_bad, "excluded": excluded}]
        )[0]
        for fid, fname in ((f_ok, "ok.txt"), (f_bad, "bad.txt")):
            db["chunk"].insert(
                {
                    "file_id": fid,
                    "text": "energy report findings for " + fname,
                    "chunk_index": 0,
                    "token_count": 6,
                    "metadata_json": "{}",
                }
            )
        db.conn.commit()
        chunks = {r["file_id"]: r["id"] for r in db.query("SELECT id, file_id FROM chunk")}
        return f_ok, f_bad, chunks

    def test_excluded_file_filtered_from_dense_and_bm25(self, tmp_path):
        db = _seed_db(tmp_path)
        cfg = _make_cfg(tmp_path)
        f_ok, f_bad, chunks = self._seed_two_files(db, excluded=1)

        metas = [
            {"chunk_id": chunks[f_ok], "file_id": f_ok, "folder_id": 1, "rel_path": "ok.txt"},
            {"chunk_id": chunks[f_bad], "file_id": f_bad, "folder_id": 2, "rel_path": "bad.txt"},
        ]
        hits = self._run_retrieve(
            db, cfg, tmp_path, metas, ["energy report ok", "energy report bad"]
        )

        assert hits, "non-excluded file must still be retrievable"
        assert all(h.file_id == f_ok for h in hits)

    def test_folder_excluded_file_filtered(self, tmp_path):
        db = _seed_db(tmp_path)
        cfg = _make_cfg(tmp_path)
        f_ok, f_bad, chunks = self._seed_two_files(db, excluded=0, folder_excluded=1)

        metas = [
            {"chunk_id": chunks[f_ok], "file_id": f_ok, "folder_id": 1, "rel_path": "ok.txt"},
            {"chunk_id": chunks[f_bad], "file_id": f_bad, "folder_id": 2, "rel_path": "bad.txt"},
        ]
        hits = self._run_retrieve(
            db, cfg, tmp_path, metas, ["energy report ok", "energy report bad"]
        )

        assert hits
        assert all(h.file_id == f_ok for h in hits)


class TestReembedStaleVectorGC:
    """OCR #14: re-embed must delete vectors of now-ineligible chunks."""

    def _seed_chunk(self, db, excluded=0):
        db["folder"].insert({"path": "t", "rel_path": "t", "name": "t", "depth": 0})
        db.conn.commit()
        fids = seed_test_files(
            db, files=[{"name": "t.txt", "folder_id": 1, "excluded": excluded}]
        )
        seed_test_extractions(db, fids, text="Sample text.")
        db["chunk"].insert(
            {
                "file_id": fids[0],
                "text": "a chunk about energy",
                "chunk_index": 0,
                "token_count": 4,
                "metadata_json": "{}",
            }
        )
        db.conn.commit()
        return fids[0]

    def test_reembed_deletes_stale_vectors(self, tmp_path):
        from rag.phase11_embed import run_embed

        db = _seed_db(tmp_path)
        cfg = _make_cfg(tmp_path)
        self._seed_chunk(db)

        mock_coll = MagicMock()
        mock_coll.name = "chunks__test"
        mock_coll.metadata = {"embedding_model": "test-embed", "dim": 3}
        # c_999 belongs to a chunk that no longer exists / is ineligible.
        mock_coll.get.return_value = {"ids": ["c_1", "c_999"]}
        mock_client = MagicMock()
        mock_client.list_collections.return_value = [mock_coll]
        mock_client.get_collection.return_value = mock_coll

        with (
            patch("rag.phase11_embed.chromadb.PersistentClient", return_value=mock_client),
            patch("rag.phase11_embed._ollama_embed", return_value=[[0.1, 0.2, 0.3]]),
        ):
            run_embed(db, cfg, collections=["chunks"], reembed=True)

        deleted = [c.kwargs.get("ids") for c in mock_coll.delete.call_args_list]
        assert ["c_999"] in deleted
        assert all("c_1" not in (ids or []) for ids in deleted)

    def test_reembed_no_eligible_chunks_clears_collection(self, tmp_path):
        from rag.phase11_embed import run_embed

        db = _seed_db(tmp_path)
        cfg = _make_cfg(tmp_path)
        self._seed_chunk(db, excluded=1)

        mock_coll = MagicMock()
        mock_coll.name = "chunks__test"
        mock_coll.metadata = {"embedding_model": "test-embed", "dim": 3}
        mock_coll.get.return_value = {"ids": ["c_1"]}
        mock_client = MagicMock()
        mock_client.list_collections.return_value = [mock_coll]
        mock_client.get_collection.return_value = mock_coll

        with (
            patch("rag.phase11_embed.chromadb.PersistentClient", return_value=mock_client),
            patch("rag.phase11_embed._ollama_embed", return_value=[[0.1, 0.2, 0.3]]),
        ):
            result = run_embed(db, cfg, collections=["chunks"], reembed=True)

        assert result == 0
        mock_coll.delete.assert_called_once_with(ids=["c_1"])


class TestRechunkVectorCleanup:
    """OCR #15: rechunk must delete the old chunks' Chroma vectors."""

    def test_rechunk_deletes_chroma_vectors(self, tmp_path):
        from rag.phase10_chunk import run_chunk

        db = _seed_db(tmp_path)
        cfg = _make_cfg(tmp_path)
        db["folder"].insert({"path": "t", "rel_path": "t", "name": "t", "depth": 0})
        db.conn.commit()
        fids = seed_test_files(db, files=[{"name": "t.txt", "folder_id": 1}])
        seed_test_extractions(db, fids, text="First sentence. Second sentence.")
        chunk_table = db["chunk"]
        chunk_table.insert(
            {
                "file_id": fids[0],
                "text": "old chunk",
                "chunk_index": 0,
                "token_count": 2,
                "metadata_json": "{}",
            }
        )
        db.conn.commit()
        old_chunk_id = chunk_table.last_rowid
        db["embedding_ref"].insert(
            {
                "chunk_id": old_chunk_id,
                "vector_store": "chroma",
                "collection": "chunks__test",
                "external_id": f"c_{old_chunk_id}",
                "embedding_model": "test-embed",
                "dim": 3,
                "config_hash": "h",
                "is_current": 1,
            }
        )
        db.conn.commit()

        mock_coll = MagicMock()
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_coll
        mock_pc = MagicMock()
        mock_pc.return_value.__enter__.return_value = mock_client

        with patch("chromadb.PersistentClient", mock_pc):
            total, files = run_chunk(db, cfg, rechunk=True)

        assert files == 1
        assert total >= 1
        mock_coll.delete.assert_called_once()
        assert mock_coll.delete.call_args.kwargs["ids"] == [f"c_{old_chunk_id}"]
        # Old chunk row gone, new chunks present.
        assert list(db["chunk"].rows_where("id = ?", [old_chunk_id])) == []
        assert db["chunk"].count >= 1


class TestSyncAllUnreadableFileGuard:
    """OCR #3: transiently unreadable files must not be treated as deleted."""

    def _env(self, tmp_path):
        from tests.test_indexer import _make_app_config, _make_db

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        cache = corpus / ".rag-cache"
        cache.mkdir()
        chroma = tmp_path / "chroma"
        chroma.mkdir()
        return corpus, _make_app_config(corpus, cache, chroma), _make_db(tmp_path)

    def test_unreadable_file_keeps_index(self, tmp_path):
        from rag.indexer import IncrementalIndexer
        from tests.test_indexer import _make_chroma_mock

        corpus, cfg, db = self._env(tmp_path)
        f = corpus / "locked.txt"
        f.write_text("important content " * 10, encoding="utf-8")

        fid = seed_test_files(
            db,
            files=[
                {
                    "name": "locked.txt",
                    "path": str(f.resolve()),
                    "rel_path": "locked.txt",
                    "sha256": "oldhash",
                    "mtime": "2000-01-01T00:00:00Z",  # forces re-hash
                }
            ],
        )[0]
        db["chunk"].insert(
            {"file_id": fid, "text": "indexed content", "chunk_index": 0,
             "token_count": 2, "metadata_json": "{}"}
        )
        db.conn.commit()

        chroma_client, _ = _make_chroma_mock()
        indexer = IncrementalIndexer(db, cfg, chroma_client=chroma_client)

        with patch("rag.indexer._sha256_file", side_effect=OSError("file locked")):
            result = indexer.sync_all()

        assert result["deleted"] == 0
        assert result["removed"] == 0
        assert db["file"].get(fid)["path"] == str(f.resolve())
        assert db["chunk"].count_where("file_id = ?", [fid]) == 1

    def test_existing_file_outside_scan_root_not_removed(self, tmp_path):
        from rag.indexer import IncrementalIndexer
        from tests.test_indexer import _make_chroma_mock

        _corpus, cfg, db = self._env(tmp_path)
        outside = tmp_path / "elsewhere.txt"
        outside.write_text("still here", encoding="utf-8")

        fid = seed_test_files(
            db,
            files=[
                {
                    "name": "elsewhere.txt",
                    "path": str(outside.resolve()),
                    "rel_path": "elsewhere.txt",
                }
            ],
        )[0]

        chroma_client, _ = _make_chroma_mock()
        indexer = IncrementalIndexer(db, cfg, chroma_client=chroma_client)
        result = indexer.sync_all()

        # The exists() guard must prevent removal even though the scan
        # never saw this path (corpus root reconfigured scenario).
        assert result["removed"] == 0
        assert db["file"].get(fid) is not None


class TestDeleteFileFKSafety:
    """OCR #4: DELETE /files must be atomic, FK-safe, and disk-last."""

    def test_delete_archive_file_with_fk_references(self, tmp_path):
        from fastapi.testclient import TestClient

        from api.main import app
        from tests.test_file_management import (
            _get_last_file_id,
            _make_cfg,
            _make_db,
            _seed_folder,
        )

        db = _make_db(tmp_path)
        cfg = _make_cfg(tmp_path)
        folder_id = _seed_folder(db)

        archive = cfg.corpus_root_path / "bundle.zip"
        archive.write_bytes(b"PK fake archive")
        db["file"].insert(
            {
                "folder_id": folder_id, "path": str(archive), "rel_path": "bundle.zip",
                "name": "bundle.zip", "size_bytes": 13, "mtime": "2026-01-01T00:00:00",
                "hash_status": "done", "identify_status": "done", "triage_status": "done",
                "is_dup_primary": 1, "excluded": 0,
            }
        )
        archive_id = _get_last_file_id(db)

        # Non-cascade FK references that used to make DELETE FROM file fail.
        db["archive_expansion"].insert(
            {
                "archive_file_id": archive_id, "extracted_to_path": "bundle.extracted",
                "tool": "patool", "succeeded": 1,
            }
        )
        folder_table = db["folder"]
        folder_table.insert(
            {
                "path": "bundle.extracted", "rel_path": "bundle.extracted",
                "name": "bundle.extracted", "depth": 1, "parent_id": folder_id,
                "source_archive_file_id": archive_id,
            }
        )
        child_folder_id = folder_table.last_rowid
        db["file"].insert(
            {
                "folder_id": child_folder_id, "path": str(tmp_path / "child.txt"),
                "rel_path": "bundle.extracted/child.txt", "name": "child.txt",
                "size_bytes": 1, "mtime": "2026-01-01T00:00:00",
                "hash_status": "done", "identify_status": "done", "triage_status": "done",
                "is_dup_primary": 1, "excluded": 0,
                "extracted_from_archive_id": archive_id,
            }
        )
        child_id = _get_last_file_id(db)
        db.conn.commit()

        mock_chroma = MagicMock()
        mock_chroma.__enter__.return_value = mock_chroma
        mock_chroma.get_collection.side_effect = Exception("no collections")

        with (
            patch("api.main._load_config", return_value=cfg),
            patch("api.main._get_db", return_value=db),
            patch("api.main.chromadb.PersistentClient", return_value=mock_chroma),
        ):
            response = TestClient(app).delete(f"/files/{archive_id}")

        assert response.status_code == 200
        assert not archive.exists()
        assert list(db.query("SELECT id FROM file WHERE id = ?", [archive_id])) == []
        assert list(db.query("SELECT id FROM archive_expansion WHERE archive_file_id = ?", [archive_id])) == []
        # Child file survives with its provenance link nulled.
        child = db["file"].get(child_id)
        assert child["extracted_from_archive_id"] is None
        assert db["folder"].get(child_folder_id)["source_archive_file_id"] is None

    def test_db_failure_keeps_disk_file_and_rolls_back(self, tmp_path):
        import sqlite3

        from fastapi.testclient import TestClient

        from api.main import app
        from tests.test_file_management import (
            _get_last_file_id,
            _make_cfg,
            _make_db,
            _seed_folder,
        )

        db = _make_db(tmp_path)
        cfg = _make_cfg(tmp_path)
        folder_id = _seed_folder(db)
        victim = cfg.corpus_root_path / "victim.txt"
        victim.write_bytes(b"precious original")
        db["file"].insert(
            {
                "folder_id": folder_id, "path": str(victim), "rel_path": "victim.txt",
                "name": "victim.txt", "size_bytes": 17, "mtime": "2026-01-01T00:00:00",
                "hash_status": "done", "identify_status": "done", "triage_status": "done",
                "is_dup_primary": 1, "excluded": 0,
            }
        )
        file_id = _get_last_file_id(db)
        db["chunk"].insert(
            {"file_id": file_id, "text": "chunk", "chunk_index": 0,
             "token_count": 1, "metadata_json": "{}"}
        )
        db.conn.commit()

        with (
            patch("api.main._load_config", return_value=cfg),
            patch("api.main._get_db", return_value=db),
            patch(
                "rag.tokenization.remove_zh_index_for_file",
                side_effect=sqlite3.OperationalError("simulated mid-delete failure"),
            ),
            pytest.raises(sqlite3.OperationalError),
        ):
            TestClient(app).delete(f"/files/{file_id}")

        # Disk original intact, DB deletes rolled back.
        assert victim.exists()
        assert db["file"].get(file_id) is not None
        assert db["chunk"].count_where("file_id = ?", [file_id]) == 1


def _extract_js_function(source: str, name: str) -> str:
    """Extract a top-level JS function body from the embedded UI script."""
    start = source.index(f"function {name}(")
    i = source.index("{", start)
    depth = 0
    for j in range(i, len(source)):
        if source[j] == "{":
            depth += 1
        elif source[j] == "}":
            depth -= 1
            if depth == 0:
                return source[start : j + 1]
    raise AssertionError(f"unbalanced braces in {name}")


def _ui_js_source() -> str:
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "ui" / "gradio_app.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "function _md(" in node.value
        ):
            return node.value
    raise AssertionError("UI JS blob not found")


@pytest.mark.skipif(
    __import__("shutil").which("node") is None, reason="node not installed"
)
class TestUiXssFixes:
    """OCR #5/#6: execute the real UI JS against attack payloads via node."""

    def _run_node(self, script: str):
        import subprocess

        proc = subprocess.run(
            ["node", "-e", script], capture_output=True, text=True, timeout=30
        )
        assert proc.returncode == 0, proc.stderr

    def test_md_links_block_unsafe_schemes_and_quote_breakout(self):
        js = _ui_js_source()
        md = _extract_js_function(js, "_md")
        script = (
            md
            + """
var out = _md('[click](javascript:alert(1))');
if(out.indexOf('<a') !== -1) throw new Error('javascript: scheme rendered as link');
var out2 = _md('[x](https://ok.example/a"b)');
var m = out2.match(/href="([^"]*)"/);
if(!m || m[1] !== 'https://ok.example/a&quot;b') throw new Error('quote breakout: ' + out2);
var out3 = _md('[ok](https://good.example/p?q=1)');
if(out3.indexOf('<a href="https://good.example/p?q=1"') === -1) throw new Error('safe link broken: ' + out3);
var out4 = _md('[x](data:text/html,<script>alert(1)</script>)');
if(out4.indexOf('<a') !== -1) throw new Error('data: scheme rendered as link');
"""
        )
        self._run_node(script)

    def test_conv_list_escapes_client_supplied_id(self):
        js = _ui_js_source()
        esc = _extract_js_function(js, "_escHtml")
        render = _extract_js_function(js, "_renderConvList")
        script = (
            esc
            + render
            + """
var _lastHTML = '';
var _currentConvId = null;
var document = {
  createElement: function(){
    return {
      _t: '',
      set textContent(v){ this._t = String(v); },
      get innerHTML(){ return this._t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
    };
  },
  getElementById: function(){ return { set innerHTML(v){ _lastHTML = v; } }; }
};
_renderConvList([{id: '"><img src=x onerror=alert(1)>', title: 't'}]);
if(_lastHTML.indexOf('<img') !== -1) throw new Error('stored XSS payload rendered as markup');
if(_lastHTML.indexOf('&quot;&gt;&lt;img') === -1) throw new Error('id not attribute-escaped: ' + _lastHTML);
"""
        )
        self._run_node(script)


class TestTranscribeRealApi:
    """OCR #7: transcription must use the real pywhispercpp API, offline."""

    def test_missing_model_fails_loudly_without_download(self, tmp_path):
        from rag.phase8b_transcribe import _transcribe_file

        cache = tmp_path / "cache"
        cache.mkdir()
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"fake audio")

        with patch("pywhispercpp.model.Model") as mock_model:
            succeeded, path, err = _transcribe_file(
                str(audio), "abc123", cache, model_name="large-v3"
            )

        assert succeeded is False
        assert path is None
        assert "not found" in err["error"]
        mock_model.assert_not_called()  # no model load, no network download

    def test_transcribe_uses_real_api_and_converts_centiseconds(self, tmp_path):
        import json
        from types import SimpleNamespace

        from rag.phase8b_transcribe import _transcribe_file

        cache = tmp_path / "cache"
        (cache / "models").mkdir(parents=True)
        model_file = cache / "models" / "large-v3.bin"
        model_file.write_bytes(b"ggml fake weights")
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"fake audio")

        seg = SimpleNamespace(t0=150, t1=275, text=" hello world ")
        with patch("pywhispercpp.model.Model") as mock_model:
            mock_model.return_value.transcribe.return_value = [seg]
            succeeded, path, err = _transcribe_file(
                str(audio), "abc123", cache, model_name="large-v3", threads=4
            )

        assert succeeded is True, err
        assert err is None
        # Model constructed with the resolved local file, real kwargs.
        kwargs = mock_model.call_args.kwargs
        assert kwargs["model"] == str(model_file.resolve())
        assert kwargs["n_threads"] == 4
        mock_model.return_value.transcribe.assert_called_once_with(str(audio))
        # Centiseconds converted to seconds in the stored transcript.
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert data["segments"] == [{"text": " hello world ", "start": 1.5, "end": 2.75}]


class TestVaultHome:
    """OCR #8: PROJECT_ROOT must be anchorable for pip-installed cli.py."""

    def test_project_root_honors_vault_home(self, tmp_path, monkeypatch):
        import importlib

        import cli

        monkeypatch.setenv("VAULT_HOME", str(tmp_path))
        try:
            importlib.reload(cli)
            assert tmp_path.resolve() == cli.PROJECT_ROOT
        finally:
            monkeypatch.delenv("VAULT_HOME")
            importlib.reload(cli)

    def test_project_root_defaults_to_cli_dir(self, monkeypatch):
        import importlib

        import cli

        monkeypatch.delenv("VAULT_HOME", raising=False)
        importlib.reload(cli)
        assert Path(cli.__file__).resolve().parent == cli.PROJECT_ROOT

    def test_dockerfile_anchors_vault_home(self):
        from pathlib import Path

        dockerfile = Path(__file__).resolve().parent.parent / "Dockerfile"
        assert "ENV VAULT_HOME=/app" in dockerfile.read_text(encoding="utf-8")
