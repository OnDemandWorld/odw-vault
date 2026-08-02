"""Tests for the V1.6 distributed-tracing spans (F-2, tasks VS1-VS3).

The span layer is strictly additive and best-effort: it never changes a
business response body or status code. The unit tests exercise the span model
(tree/parenting, sampling, no-op) and the exporters (console output, OTLP
failure degradation). The integration tests drive ``/query`` over HTTP with the
same mocks used by ``tests/test_api_query.py`` (no Ollama / Chroma / network)
and capture the produced spans via a test exporter and via the console exporter.
"""

import logging
import sqlite3
from unittest.mock import MagicMock, patch

import sqlite_utils
from fastapi.testclient import TestClient

from api.main import app
from api.spans import (
    ConsoleSpanExporter,
    OtlpHttpSpanExporter,
    Span,
    SpanExporter,
    _NullExporter,
    get_current_span,
    get_exporter,
    set_exporter,
    span,
    start_span,
)
from api.tracing import TRACE_HEADER, trace_id_var
from pipeline.db import migrate
from rag.retrieval import Hit


class _CapturingExporter(SpanExporter):
    """Test exporter that records every exported span dict."""

    def __init__(self):
        self.spans: list[dict] = []

    def export(self, span_obj: Span) -> None:
        self.spans.append(span_obj.to_dict())


# ---------------------------------------------------------------------------
# VS1 — span model: tree/parenting, sampling, no-op
# ---------------------------------------------------------------------------


class TestSpanModel:
    def test_to_dict_has_documented_fields(self):
        sp = start_span("op")
        try:
            d = sp.to_dict()
            assert set(d) == {
                "name",
                "trace_id",
                "span_id",
                "parent_span_id",
                "start_ms",
                "duration_ms",
                "attrs",
                "status",
            }
        finally:
            sp.end()

    def test_root_span_reuses_bound_trace_id_and_has_no_parent(self):
        token = trace_id_var.set("trace-root-1")
        try:
            sp = start_span("root")
            try:
                assert sp.parent_span_id is None
                assert sp.trace_id == "trace-root-1"
                assert sp.sampled is True
            finally:
                sp.end()
        finally:
            trace_id_var.reset(token)

    def test_child_span_auto_parents_and_shares_trace_id(self):
        token = trace_id_var.set("trace-tree-1")
        try:
            root = start_span("root")
            child = start_span("child")
            grandchild = start_span("grandchild")
            try:
                assert get_current_span() is grandchild
                assert child.parent_span_id == root.span_id
                assert grandchild.parent_span_id == child.span_id
                # trace_id propagates down the tree
                assert child.trace_id == root.trace_id == "trace-tree-1"
                assert grandchild.trace_id == "trace-tree-1"
                # span ids are distinct
                assert len({root.span_id, child.span_id, grandchild.span_id}) == 3
            finally:
                grandchild.end()
                child.end()
                root.end()
            # stack fully unwound
            assert get_current_span() is None
        finally:
            trace_id_var.reset(token)

    def test_span_context_manager_records_ok_and_duration(self):
        with span("op") as sp:
            sp.set_attr("k", "v")
        assert sp.status == "ok"
        assert sp.duration_ms is not None
        assert sp.duration_ms >= 0
        assert sp.attrs["k"] == "v"

    def test_span_context_manager_records_error_and_reraises(self):
        with patch("api.spans.export_span"):
            try:
                with span("boom") as sp:
                    raise ValueError("kaboom")
            except ValueError:
                pass
            else:  # pragma: no cover
                raise AssertionError("expected ValueError to propagate")
        assert sp.status == "error"

    def test_sampling_rate_zero_yields_noop_span(self, monkeypatch):
        monkeypatch.setenv("TRACE_SAMPLE_RATE", "0")
        capturing = _CapturingExporter()
        set_exporter(capturing)
        try:
            with span("unsampled") as sp:
                assert sp.is_noop is True
                assert sp.sampled is False
        finally:
            set_exporter(None)
        # nothing exported for an unsampled span
        assert capturing.spans == []

    def test_child_inherits_unsampled_decision(self, monkeypatch):
        monkeypatch.setenv("TRACE_SAMPLE_RATE", "0")
        capturing = _CapturingExporter()
        set_exporter(capturing)
        try:
            with span("root"), span("child") as child:
                assert child.is_noop is True
        finally:
            set_exporter(None)
        assert capturing.spans == []

    def test_sampling_rate_one_samples_everything(self, monkeypatch):
        monkeypatch.setenv("TRACE_SAMPLE_RATE", "1.0")
        capturing = _CapturingExporter()
        set_exporter(capturing)
        try:
            with span("sampled"):
                pass
        finally:
            set_exporter(None)
        assert [s["name"] for s in capturing.spans] == ["sampled"]

    def test_invalid_sample_rate_defaults_to_full_sampling(self, monkeypatch):
        monkeypatch.setenv("TRACE_SAMPLE_RATE", "not-a-number")
        capturing = _CapturingExporter()
        set_exporter(capturing)
        try:
            with span("op"):
                pass
        finally:
            set_exporter(None)
        assert len(capturing.spans) == 1


# ---------------------------------------------------------------------------
# VS2 — exporters: console output + OTLP best-effort degradation
# ---------------------------------------------------------------------------


class TestExporters:
    def test_console_exporter_logs_span_dict(self, caplog):
        with span("console.op") as sp:
            sp.set_attr("answer_model", "test-gen")
        # Re-export through a fresh console exporter to capture deterministically.
        with caplog.at_level(logging.INFO, logger="api.spans"):
            ConsoleSpanExporter().export(sp)
        messages = [r.getMessage() for r in caplog.records]
        assert any("console.op" in m for m in messages)
        assert any("test-gen" in m for m in messages)

    def test_exporter_resolution_defaults_and_overrides(self, monkeypatch):
        set_exporter(None)
        monkeypatch.delenv("TRACE_EXPORTER", raising=False)
        assert isinstance(get_exporter(), ConsoleSpanExporter)

        monkeypatch.setenv("TRACE_EXPORTER", "none")
        assert isinstance(get_exporter(), _NullExporter)

        monkeypatch.setenv("TRACE_EXPORTER", "otlp")
        assert isinstance(get_exporter(), OtlpHttpSpanExporter)

        override = _CapturingExporter()
        set_exporter(override)
        try:
            assert get_exporter() is override
        finally:
            set_exporter(None)

    def test_otlp_exporter_no_endpoint_is_silent_noop(self):
        sp = start_span("otlp.noop")
        sp.end()
        # No endpoint configured -> export returns without raising.
        OtlpHttpSpanExporter(endpoint="").export(sp)

    def test_otlp_exporter_failure_degrades_silently(self):
        sp = start_span("otlp.fail")
        sp.end()
        # Port 1 refuses connections immediately; export must NOT raise.
        OtlpHttpSpanExporter(endpoint="http://127.0.0.1:1/v1/traces", timeout=0.5).export(sp)

    def test_otlp_exporter_posts_payload(self, monkeypatch):
        calls = {}

        class _Resp:
            def read(self):
                return b""

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(req, timeout=None):
            calls["url"] = req.full_url
            calls["data"] = req.data
            calls["timeout"] = timeout
            return _Resp()

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        sp = start_span("otlp.ok")
        sp.end()
        OtlpHttpSpanExporter(endpoint="http://collector:4318/v1/traces", timeout=1.5).export(sp)

        assert calls["url"] == "http://collector:4318/v1/traces"
        assert calls["timeout"] == 1.5
        assert b"resourceSpans" in calls["data"]
        assert b"otlp.ok" in calls["data"]

    def test_export_span_swallows_exporter_errors(self):
        class _Broken(SpanExporter):
            def export(self, span_obj):
                raise RuntimeError("exporter exploded")

        set_exporter(_Broken())
        try:
            sp = start_span("resilient")
            # Must not raise even though the exporter blows up.
            sp.end()
        finally:
            set_exporter(None)


# ---------------------------------------------------------------------------
# VS3 — /query produces spans (integration, captured via exporters)
# ---------------------------------------------------------------------------


def _make_test_db(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    db = sqlite_utils.Database(conn)
    migrate(db)
    return db


def _make_cfg(tmp_path):
    cfg = MagicMock()
    cfg.ollama.host = "http://localhost:11434"
    cfg.chroma_root_path = str(tmp_path / "chroma")
    cfg.models.embedding.collection_suffix = "test"
    cfg.models.embedding.name = "test-embed"
    cfg.models.generation.name = "test-gen"
    cfg.models.generation.thinking = False
    cfg.models.reranker.enabled = False
    cfg.models.contextual_retrieval.enabled = False
    return cfg


def _make_hit():
    return Hit(
        chunk_id=1,
        file_id=1,
        folder_id=1,
        rel_path="test/doc.txt",
        page_start=None,
        text="The Gleneagles deployment uses robot platform X [1].",
        dense_score=0.9,
        bm25_score=0.8,
        fused_score=0.85,
    )


def _gen_result():
    return {
        "answer": "The deployment uses robot platform X [1].",
        "citations": [
            {
                "citation_number": 1,
                "file_id": 1,
                "rel_path": "test/doc.txt",
                "page_start": None,
                "chunk_id": 1,
                "snippet": "robot platform X",
            }
        ],
        "generation_ms": 12.0,
        "model": "test-gen",
        "refused": False,
    }


def _patch_query_seams(tmp_path):
    """Return a list of started patchers mocking the /query seams (no network)."""
    patchers = [
        patch("api.main._load_config", return_value=_make_cfg(tmp_path)),
        patch("api.main.ollama.Client"),
        patch("api.main.chromadb.PersistentClient"),
        patch(
            "api.main.retrieve",
            return_value=([_make_hit()], {"retrieval_ms": 5.0, "query_lang": "en"}),
        ),
        patch("api.main.generate_answer", return_value=_gen_result()),
        patch("api.main._get_db", return_value=_make_test_db(tmp_path)),
    ]
    started = []
    for p in patchers:
        mock = p.start()
        started.append((p, mock))
    # Wire the ollama/chroma reachability checks to succeed.
    for p, mock in started:
        if p.attribute == "Client":
            mock.return_value.list.return_value = {"models": []}
        if p.attribute == "PersistentClient":
            mock.return_value.get_collection.return_value = MagicMock()
    return [p for p, _ in started]


class TestQuerySpans:
    def test_query_produces_span_tree(self, tmp_path):
        patchers = _patch_query_seams(tmp_path)
        capturing = _CapturingExporter()
        set_exporter(capturing)
        try:
            client = TestClient(app)
            response = client.post(
                "/query",
                json={"query": "What robot platform?"},
                headers={TRACE_HEADER: "trace-span-1"},
            )
        finally:
            for p in patchers:
                p.stop()
            set_exporter(None)

        # Business response is unchanged.
        assert response.status_code == 200
        assert response.json()["answer"] == "The deployment uses robot platform X [1]."

        names = [s["name"] for s in capturing.spans]
        assert "vault.query" in names
        assert "vault.query.retrieve" in names
        assert "vault.query.generate" in names

        by_name = {s["name"]: s for s in capturing.spans}
        root = by_name["vault.query"]
        # Root span: no parent, carries the inbound trace id.
        assert root["parent_span_id"] is None
        assert root["trace_id"] == "trace-span-1"
        assert root["status"] == "ok"
        # Children auto-parent to the root and share the trace id.
        for child_name in ("vault.query.retrieve", "vault.query.generate"):
            assert by_name[child_name]["parent_span_id"] == root["span_id"]
            assert by_name[child_name]["trace_id"] == "trace-span-1"
        # Retrieval / generation attrs were recorded.
        assert by_name["vault.query.retrieve"]["attrs"]["n_chunks"] == 1
        assert by_name["vault.query.retrieve"]["attrs"]["retrieval_ms"] == 5.0
        assert by_name["vault.query.generate"]["attrs"]["model"] == "test-gen"
        assert root["attrs"]["total_ms"] is not None

    def test_query_span_emitted_via_console_exporter(self, tmp_path, caplog):
        patchers = _patch_query_seams(tmp_path)
        set_exporter(None)  # use the default console exporter
        try:
            with caplog.at_level(logging.INFO, logger="api.spans"):
                client = TestClient(app)
                response = client.post("/query", json={"query": "What platform?"})
        finally:
            for p in patchers:
                p.stop()

        assert response.status_code == 200
        messages = [r.getMessage() for r in caplog.records]
        assert any("vault.query" in m for m in messages)

    def test_query_error_still_returns_503_and_records_error_span(self, tmp_path):
        patchers = _patch_query_seams(tmp_path)
        # Force a retrieval failure -> handler raises HTTPException(503).
        # Swap the retrieve mock for one that raises (avoid stopping it twice).
        remaining = []
        for p in patchers:
            if p.attribute == "retrieve":
                p.stop()
            else:
                remaining.append(p)
        capturing = _CapturingExporter()
        set_exporter(capturing)
        retrieve_patcher = patch("api.main.retrieve", side_effect=RuntimeError("Chroma missing"))
        retrieve_patcher.start()
        remaining.append(retrieve_patcher)
        try:
            client = TestClient(app)
            response = client.post("/query", json={"query": "boom"})
        finally:
            for p in remaining:
                p.stop()
            set_exporter(None)

        # Status code unchanged (503), and the retrieve span recorded an error.
        assert response.status_code == 503
        by_name = {s["name"]: s for s in capturing.spans}
        assert by_name["vault.query.retrieve"]["status"] == "error"
        assert by_name["vault.query"]["status"] == "error"
