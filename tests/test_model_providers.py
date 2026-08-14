"""Tests for the multi-provider LLM settings feature in ui/gradio_app.py.

Covers the provider registry (llm_providers.json), key masking, active
provider resolution, and the three wire-protocol stream adapters
(ollama / openai-compatible / anthropic) with mocked httpx transports.
"""

import json
from unittest.mock import patch

import pytest
from fastapi import Request  # module-level: mirrors the fix in ui/gradio_app.py

import ui.gradio_app as ga


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


class TestMaskKey:
    def test_empty(self):
        assert ga._mask_key("") == ""

    def test_short(self):
        assert ga._mask_key("abc") == "\u2022\u2022\u2022"

    def test_long(self):
        masked = ga._mask_key("sk-1234567890abcdef")
        assert masked.startswith("sk-1")
        assert masked.endswith("ef")
        assert "\u2022\u2022\u2022" in masked


class TestProviderRegistry:
    def _reg_file(self, tmp_path):
        return tmp_path / "llm_providers.json"

    def test_seed_creates_default_from_cfg(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ga, "_PROVIDERS_FILE", self._reg_file(tmp_path))
        monkeypatch.setattr(ga, "_cfg", None)
        monkeypatch.setattr(ga, "_ollama_host", "http://localhost:99999")

        reg = ga._seed_default_registry()
        assert reg["version"] == 1
        assert len(reg["providers"]) == 1
        entry = reg["providers"][0]
        assert entry["protocol"] == "ollama"
        assert entry["base_url"] == "http://localhost:99999"
        assert reg["active_id"] == entry["id"]
        # Persisted to disk
        assert self._reg_file(tmp_path).exists()

    def test_load_returns_saved_registry(self, tmp_path, monkeypatch):
        f = self._reg_file(tmp_path)
        data = {"version": 1, "active_id": "p2", "providers": [
            {"id": "p1", "name": "A", "protocol": "ollama",
             "base_url": "http://x", "api_key": "", "model": "m1"},
            {"id": "p2", "name": "B", "protocol": "openai",
             "base_url": "https://y/v1", "api_key": "sk-zzz", "model": "m2"},
        ]}
        f.write_text(json.dumps(data))
        monkeypatch.setattr(ga, "_PROVIDERS_FILE", f)
        reg = ga._load_provider_registry()
        assert [p["id"] for p in reg["providers"]] == ["p1", "p2"]

    def test_load_corrupt_file_reseeds(self, tmp_path, monkeypatch):
        f = self._reg_file(tmp_path)
        f.write_text("{not valid json")
        monkeypatch.setattr(ga, "_PROVIDERS_FILE", f)
        monkeypatch.setattr(ga, "_cfg", None)
        reg = ga._load_provider_registry()
        assert isinstance(reg.get("providers"), list)
        assert len(reg["providers"]) >= 1

    def test_get_active_provider_prefers_active_id(self, tmp_path, monkeypatch):
        f = self._reg_file(tmp_path)
        data = {"version": 1, "active_id": "p2", "providers": [
            {"id": "p1", "name": "A", "protocol": "ollama",
             "base_url": "http://x", "api_key": "", "model": "m1"},
            {"id": "p2", "name": "B", "protocol": "openai",
             "base_url": "https://y/v1", "api_key": "k", "model": "m2"},
        ]}
        f.write_text(json.dumps(data))
        monkeypatch.setattr(ga, "_PROVIDERS_FILE", f)
        assert ga._get_active_provider()["id"] == "p2"

    def test_get_active_provider_falls_back_to_first(self, tmp_path, monkeypatch):
        f = self._reg_file(tmp_path)
        data = {"version": 1, "active_id": "gone", "providers": [
            {"id": "p1", "name": "A", "protocol": "ollama",
             "base_url": "http://x", "api_key": "", "model": "m1"},
        ]}
        f.write_text(json.dumps(data))
        monkeypatch.setattr(ga, "_PROVIDERS_FILE", f)
        assert ga._get_active_provider()["id"] == "p1"

    def test_get_active_provider_empty(self, tmp_path, monkeypatch):
        f = self._reg_file(tmp_path)
        data = {"version": 1, "active_id": None, "providers": []}
        f.write_text(json.dumps(data))
        monkeypatch.setattr(ga, "_PROVIDERS_FILE", f)
        assert ga._get_active_provider() is None


# ---------------------------------------------------------------------------
# Protocol adapters (mocked httpx)
# ---------------------------------------------------------------------------


class _FakeStreamResponse:
    def __init__(self, lines, status_code=200):
        self._lines = lines
        self.status_code = status_code

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def iter_lines(self):
        return iter(self._lines)

    def read(self):
        return b'{"error": "boom"}'


class _FakeHttpClient:
    def __init__(self, response):
        self._response = response
        self.captured = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def stream(self, method, url, headers=None, json=None):
        self.captured = {"method": method, "url": url, "headers": headers, "json": json}
        return self._response


def _run_stream(entry, lines, messages=None):
    messages = messages or [{"role": "user", "content": "hi"}]
    fake_client = _FakeHttpClient(_FakeStreamResponse(lines))
    with patch("httpx.Client", return_value=fake_client):
        tokens = list(ga._provider_chat_stream(entry, messages))
    return "".join(tokens), fake_client.captured


MESSAGES = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "hi"},
]


class TestOllamaStream:
    ENTRY = {"protocol": "ollama", "base_url": "http://localhost:11434",
             "api_key": "", "model": "gemma4:latest"}

    def test_tokens_and_payload(self):
        lines = [
            '{"message": {"content": "Hel"}}',
            '{"message": {"content": "lo"}}',
            '{"done": true}',
        ]
        text, captured = _run_stream(self.ENTRY, lines, MESSAGES)
        assert text == "Hello"
        assert captured["url"] == "http://localhost:11434/api/chat"
        body = captured["json"]
        assert body["model"] == "gemma4:latest"
        assert body["stream"] is True
        assert body["options"]["temperature"] == 0.5
        assert body["messages"] == MESSAGES

    def test_bearer_header_when_key_set(self):
        entry = dict(self.ENTRY, api_key="secret")
        _, captured = _run_stream(entry, [])
        assert captured["headers"]["Authorization"] == "Bearer secret"


class TestOpenAIStream:
    ENTRY = {"protocol": "openai", "base_url": "https://api.deepseek.com/v1",
             "api_key": "sk-test", "model": "deepseek-chat"}

    def test_tokens_and_payload(self):
        lines = [
            'data: {"choices": [{"delta": {"content": "Wo"}}]}',
            'event: ping',
            'data: {"choices": [{"delta": {"content": "rld"}}]}',
            'data: [DONE]',
        ]
        text, captured = _run_stream(self.ENTRY, lines, MESSAGES)
        assert text == "World"
        assert captured["url"] == "https://api.deepseek.com/v1/chat/completions"
        assert captured["headers"]["Authorization"] == "Bearer sk-test"
        body = captured["json"]
        assert body["model"] == "deepseek-chat"
        assert body["stream"] is True
        assert body["messages"] == MESSAGES

    def test_http_error_raises(self):
        fake_client = _FakeHttpClient(_FakeStreamResponse([], status_code=401))
        with patch("httpx.Client", return_value=fake_client):
            with pytest.raises(RuntimeError, match="401"):
                list(ga._provider_chat_stream(self.ENTRY, MESSAGES))


class TestAnthropicStream:
    ENTRY = {"protocol": "anthropic", "base_url": "https://api.anthropic.com",
             "api_key": "ak-test", "model": "claude-sonnet-4-20250514"}

    def test_tokens_system_extraction_and_payload(self):
        lines = [
            'event: message_start',
            'data: {"type": "content_block_delta", "delta": {"text": "Fo"}}',
            'data: {"type": "content_block_delta", "delta": {"text": "o"}}',
            'data: {"type": "message_stop"}',
        ]
        text, captured = _run_stream(self.ENTRY, lines, MESSAGES)
        assert text == "Foo"
        assert captured["url"] == "https://api.anthropic.com/v1/messages"
        assert captured["headers"]["x-api-key"] == "ak-test"
        assert captured["headers"]["anthropic-version"] == "2023-06-01"
        body = captured["json"]
        # System content is hoisted out of the message list
        assert body["system"] == "You are helpful."
        assert all(m["role"] != "system" for m in body["messages"])
        assert body["max_tokens"] > 0
        assert body["stream"] is True


# ---------------------------------------------------------------------------
# Model list parsing + health check
# ---------------------------------------------------------------------------


class TestParseModelList:
    def test_ollama(self):
        body = json.dumps({"models": [{"name": "gemma4:latest"}, {"name": "qwen:7b"}]})
        assert ga._parse_model_list("ollama", body) == ["gemma4:latest", "qwen:7b"]

    def test_openai(self):
        body = json.dumps({"data": [{"id": "gpt-4o"}, {"id": "gpt-4o-mini"}]})
        assert ga._parse_model_list("openai", body) == ["gpt-4o", "gpt-4o-mini"]

    def test_anthropic(self):
        body = json.dumps({"data": [{"id": "claude-sonnet-4-20250514"}]})
        assert ga._parse_model_list("anthropic", body) == ["claude-sonnet-4-20250514"]

    def test_invalid_json(self):
        assert ga._parse_model_list("openai", "not json") == []


class TestProviderTest:
    def _run(self, entry):
        ok, detail = ga._provider_test(entry)
        return ok, detail

    def test_empty_base_url(self):
        ok, detail = self._run({"protocol": "openai", "base_url": "", "api_key": "", "model": "m"})
        assert ok is False

    def test_ollama_ok(self):
        class _R:
            status_code = 200
            text = json.dumps({"models": [{"name": "gemma4:latest"}]})

        class _C:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def get(self, url, headers=None):
                assert url.endswith("/api/tags")
                return _R()

        with patch("httpx.Client", return_value=_C()):
            ok, detail = self._run({"protocol": "ollama", "base_url": "http://localhost:11434",
                                    "api_key": "", "model": "m"})
        assert ok is True
        assert "1 models visible" in detail

    def test_openai_http_error(self):
        class _R:
            status_code = 401
            text = "unauthorized"

        class _C:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def get(self, url, headers=None):
                return _R()

        with patch("httpx.Client", return_value=_C()):
            ok, detail = self._run({"protocol": "openai", "base_url": "https://api.x/v1",
                                    "api_key": "bad", "model": "m"})
        assert ok is False
        assert "401" in detail


# ---------------------------------------------------------------------------
# FastAPI route annotations — regression guard for the 422 bug
#
# ui/gradio_app.py uses `from __future__ import annotations`, so route params
# like `request: Request` are strings that FastAPI resolves from module
# globals. If `Request` is only imported inside launch_ui(), resolution fails
# and FastAPI demands a *query* param named "request" — every POST body route
# then returns HTTP 422. These tests pin the module-level import.
# ---------------------------------------------------------------------------


class TestRequestAnnotationResolvable:
    def test_module_exports_request(self):
        from fastapi import Request

        assert ga.Request is Request

    def test_route_annotations_resolve_via_module_globals(self):
        import typing

        def _route_like(conv_id: str, request: "Request"):  # noqa: F821 — mirrors launch_ui
            return conv_id

        hints = typing.get_type_hints(_route_like, ga.__dict__)
        assert hints["request"] is not str

    def test_post_body_route_accepts_json(self):
        """Functional check: a FastAPI route using the same annotation pattern
        resolves `request` to the Request object (not a query parameter) when
        `Request` lives in the defining module's globals — as in gradio_app."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()

        @app.post("/echo")
        async def echo(request: "Request"):  # same pattern as launch_ui routes
            payload = await request.json()
            return {"ok": True, "name": payload.get("name")}

        client = TestClient(app)
        resp = client.post("/echo", json={"name": "vault"})
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "name": "vault"}

    def test_unresolvable_annotation_still_422s(self):
        """Documents the original bug: when the defining module's globals do
        NOT contain `Request`, FastAPI demands a query param and returns 422."""
        import types

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        ns = types.ModuleType("module_without_request").__dict__  # no Request here
        ns["app"] = app
        exec(
            "@app.post('/echo')\n"
            "async def echo(request: 'Request'):\n"
            "    payload = await request.json()\n"
            "    return {'ok': True}\n",
            ns,
        )
        client = TestClient(app)
        assert client.post("/echo", json={"name": "vault"}).status_code == 422
