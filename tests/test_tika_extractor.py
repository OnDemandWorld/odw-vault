"""Tests for rag/extractors/tika_extractor.py — offline-first server handling."""

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from rag.extractors import tika_extractor as te

URL = "http://localhost:9998"


@pytest.fixture(autouse=True)
def _reset_ready():
    te._READY_ENDPOINTS.clear()
    yield
    te._READY_ENDPOINTS.clear()


class TestServerReachable:
    def test_uses_existing_server_without_startup(self):
        with (
            patch.object(te, "_probe", return_value=True) as probe,
            patch("tika.tika.startServer") as start,
            patch("tika.parser.from_file", return_value={"content": "hello", "metadata": {}}) as pf,
        ):
            text, _, ok, err = te.extract_tika("f.md", tika_url=URL)
        assert ok and text == "hello" and err is None
        start.assert_not_called()
        pf.assert_called_once_with("f.md", serverEndpoint=URL)
        assert probe.call_count == 1

    def test_client_only_mode_enabled(self):
        import tika.tika as tika_core

        with (
            patch.object(te, "_probe", return_value=True),
            patch("tika.parser.from_file", return_value={"content": "x", "metadata": {}}),
        ):
            te.extract_tika("f.md", tika_url=URL)
        assert tika_core.TikaClientOnly is True


class TestServerUnavailable:
    def test_no_brute_force_fails_fast(self):
        with (
            patch.object(te, "_probe", return_value=False),
            patch("tika.tika.startServer") as start,
            patch("tika.parser.from_file") as pf,
        ):
            text, _, ok, err = te.extract_tika("f.md", tika_url=URL, brute_force=False)
        assert not ok and text is None
        assert "unavailable" in err
        start.assert_not_called()
        pf.assert_not_called()

    def test_no_local_jar_never_downloads(self):
        with (
            patch.object(te, "_probe", return_value=False),
            patch.object(te, "_local_jar", return_value=None),
            patch("tika.tika.startServer") as start,
            patch("tika.tika.getRemoteJar") as download,
            patch("tika.parser.from_file") as pf,
        ):
            _, _, ok, err = te.extract_tika("f.md", tika_url=URL, brute_force=True)
        assert not ok
        assert "never downloaded" in err
        start.assert_not_called()
        download.assert_not_called()
        pf.assert_not_called()


class TestLocalStartup:
    def test_starts_from_local_jar_then_extracts(self):
        with (
            patch.object(te, "_probe", side_effect=[False, False, True]),
            patch.object(te, "_local_jar", return_value="/opt/tika-server.jar"),
            patch("tika.tika.startServer", return_value=True) as start,
            patch("tika.parser.from_file", return_value={"content": "body", "metadata": {}}),
        ):
            text, _, ok, err = te.extract_tika("f.doc", tika_url=URL)
        assert ok and text == "body" and err is None
        start.assert_called_once()
        assert start.call_args.args[0] == "/opt/tika-server.jar"
        assert URL in te._READY_ENDPOINTS

    def test_startup_timeout_returns_error(self):
        with (
            patch.object(te, "_probe", return_value=False),
            patch.object(te, "_local_jar", return_value="/opt/tika-server.jar"),
            patch("tika.tika.startServer", return_value=False),
            patch.object(te, "_STARTUP_DEADLINE_SECONDS", 0.0),
            patch("tika.parser.from_file") as pf,
        ):
            _, _, ok, err = te.extract_tika("f.doc", tika_url=URL)
        assert not ok and "did not become ready" in err
        pf.assert_not_called()

    def test_concurrent_workers_start_server_once(self):
        started = threading.Event()

        def fake_start(*args, **kwargs):
            started.set()
            return True

        def fake_probe(url, timeout=3.0):
            return started.is_set()

        with (
            patch.object(te, "_probe", side_effect=fake_probe),
            patch.object(te, "_local_jar", return_value="/opt/tika-server.jar"),
            patch("tika.tika.startServer", side_effect=fake_start) as start,
            patch("tika.parser.from_file", return_value={"content": "ok", "metadata": {}}),
            ThreadPoolExecutor(max_workers=8) as pool,
        ):
            results = list(pool.map(lambda _: te.extract_tika("f.doc", tika_url=URL), range(8)))
        assert all(ok for _, _, ok, _ in results)
        assert start.call_count == 1


class TestFailureRecovery:
    def test_connection_error_evicts_ready_endpoint(self):
        te._READY_ENDPOINTS.add(URL)
        with (
            patch.object(te, "_probe", return_value=True) as probe,
            patch("tika.parser.from_file", side_effect=ConnectionError("server died")),
        ):
            _, _, ok, err = te.extract_tika("f.doc", tika_url=URL)
        assert not ok and "server died" in err
        assert URL not in te._READY_ENDPOINTS
        probe.assert_not_called()

    def test_empty_text_reported(self):
        with (
            patch.object(te, "_probe", return_value=True),
            patch("tika.parser.from_file", return_value={"content": "  ", "metadata": {}}),
        ):
            _, _, ok, err = te.extract_tika("f.bin", tika_url=URL)
        assert not ok and "empty text" in err


class TestLocalJarDiscovery:
    def test_env_override(self, tmp_path):
        jar = tmp_path / "tika.jar"
        import zipfile

        with zipfile.ZipFile(jar, "w") as zf:
            zf.writestr("META-INF/MANIFEST.MF", "Manifest-Version: 1.0\n")
        with patch.dict("os.environ", {"TIKA_SERVER_JAR": str(jar)}):
            assert te._local_jar() == str(jar)

    def test_truncated_jar_rejected(self, tmp_path):
        jar = tmp_path / "tika.jar"
        jar.write_bytes(b"PK\x03\x04partial-download")
        with (
            patch.dict("os.environ", {"TIKA_SERVER_JAR": str(jar)}),
            patch("tika.tika.TikaJarPath", str(tmp_path / "elsewhere")),
        ):
            assert te._local_jar() is None
