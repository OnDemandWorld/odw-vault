"""Tests for ui/gradio_app.py — UI helpers and mocked interactions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rag.retrieval import Hit


class TestCitationsHtml:
    """Tests for _citations_html."""

    def _import(self):
        from ui.gradio_app import _citations_html
        return _citations_html

    def test_empty_returns_empty(self):
        fn = self._import()
        assert fn([]) == ""

    def test_single_citation(self):
        fn = self._import()
        citations = [
            {
                "citation_number": 1,
                "rel_path": "doc.txt",
                "page_start": None,
                "snippet": "test snippet",
            }
        ]
        result = fn(citations)
        assert "[1]" in result
        assert "doc.txt" in result
        assert "test snippet" in result
        assert "Sources:" in result

    def test_citation_with_page(self):
        fn = self._import()
        citations = [
            {
                "citation_number": 2,
                "rel_path": "report.pdf",
                "page_start": 5,
                "snippet": "page content",
            }
        ]
        result = fn(citations)
        assert "page 5" in result

    def test_multiple_citations(self):
        fn = self._import()
        citations = [
            {"citation_number": 1, "rel_path": "a.txt", "page_start": None, "snippet": "s1"},
            {"citation_number": 2, "rel_path": "b.txt", "page_start": 3, "snippet": "s2"},
        ]
        result = fn(citations)
        assert "[1]" in result
        assert "[2]" in result
        assert "a.txt" in result
        assert "b.txt" in result


class TestOnFolderChange:
    """Tests for _on_folder_change."""

    def _import(self):
        from ui.gradio_app import _on_folder_change
        return _on_folder_change

    def test_all_folders(self):
        fn = self._import()
        assert fn("All folders") == "Searching all folders."

    def test_specific_folder(self):
        fn = self._import()
        assert "Scoped to: Project/Alpha" in fn("Project/Alpha")


class TestFormatChunksForPrompt:
    """Tests for _format_chunks_for_prompt."""

    def _import(self):
        from ui.gradio_app import _format_chunks_for_prompt
        return _format_chunks_for_prompt

    def test_empty_hits(self):
        fn = self._import()
        assert fn([]) == "(no context available)"

    def test_single_hit(self):
        fn = self._import()
        hit = Hit(
            chunk_id=1,
            file_id=1,
            folder_id=1,
            rel_path="doc.txt",
            page_start=None,
            text="Hello world.",
        )
        result = fn([hit])
        assert "[1]" in result
        assert "doc.txt" in result
        assert "Hello world." in result

    def test_multiple_hits(self):
        fn = self._import()
        hits = [
            Hit(chunk_id=1, file_id=1, folder_id=1, rel_path="a.txt", page_start=None, text="First."),
            Hit(chunk_id=2, file_id=2, folder_id=1, rel_path="b.txt", page_start=3, text="Second."),
        ]
        result = fn(hits)
        assert "[1]" in result
        assert "[2]" in result
        assert "a.txt" in result
        assert "b.txt" in result
        assert "(page 3)" in result

    def test_separator_between_blocks(self):
        fn = self._import()
        hits = [
            Hit(chunk_id=1, file_id=1, folder_id=1, rel_path="a", page_start=None, text="X"),
            Hit(chunk_id=2, file_id=2, folder_id=1, rel_path="b", page_start=None, text="Y"),
        ]
        result = fn(hits)
        assert "\n\n" in result


class TestCheckOllama:
    """Tests for _check_ollama."""

    def _import(self):
        from ui.gradio_app import _check_ollama
        return _check_ollama

    def test_reachable(self):
        fn = self._import()
        with patch("ui.gradio_app.ollama.Client") as MockClient:
            instance = MockClient.return_value
            instance.list.return_value = {"models": []}
            assert fn() is True

    def test_unreachable(self):
        fn = self._import()
        with patch("ui.gradio_app.ollama.Client") as MockClient:
            MockClient.side_effect = Exception("Connection refused")
            assert fn() is False


class TestCheckChroma:
    """Tests for _check_chroma."""

    def _import(self):
        from ui.gradio_app import _check_chroma
        return _check_chroma

    def test_collection_exists(self):
        fn = self._import()
        mock_coll = MagicMock()
        mock_coll.name = "chunks__test"
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_coll

        with patch("chromadb.PersistentClient", return_value=mock_client):
            with patch("ui.gradio_app._cfg", create=True) as mock_cfg:
                mock_cfg.models.embedding.collection_suffix = "test"
                ok, msg = fn()
                assert ok is True
                assert "chunks__test" in msg

    def test_collection_missing(self):
        fn = self._import()
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = Exception("Collection not found")

        with patch("chromadb.PersistentClient", return_value=mock_client):
            with patch("ui.gradio_app._cfg", None):
                ok, _msg = fn()
                assert ok is False


class TestBuildStatusHtml:
    """Tests for _build_status_html."""

    def _import(self):
        from ui.gradio_app import _build_status_html
        return _build_status_html

    def test_contains_models(self):
        fn = self._import()
        mock_cfg = MagicMock()
        mock_cfg.ollama.host = "http://localhost:11434"
        mock_cfg.models.generation.name = "gemma4:latest"
        mock_cfg.models.embedding.name = "nomic-embed"
        mock_cfg.paths.chroma_root = "/tmp/chroma"

        with patch("ui.gradio_app._cfg", mock_cfg):
            with patch("ui.gradio_app._ollama_host", "http://localhost:11434"):
                with patch("ui.gradio_app._check_ollama", return_value=True):
                    with patch("ui.gradio_app._check_chroma", return_value=(True, "chunks__test")):
                        result = fn()
                        assert "gemma4:latest" in result
                        assert "nomic-embed" in result
                        assert "reachable" in result

    def test_unreachable_shown(self):
        fn = self._import()
        mock_cfg = MagicMock()
        mock_cfg.ollama.host = "http://localhost:11434"
        mock_cfg.models.generation.name = "test-gen"
        mock_cfg.models.embedding.name = "test-embed"

        with patch("ui.gradio_app._cfg", mock_cfg):
            with patch("ui.gradio_app._ollama_host", "http://localhost:11434"):
                with patch("ui.gradio_app._check_ollama", return_value=False):
                    with patch("ui.gradio_app._check_chroma", return_value=(False, "error")):
                        result = fn()
                        assert "unreachable" in result
                        assert "Chroma unavailable" in result
