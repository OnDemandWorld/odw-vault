"""Tests for pipeline/config.py."""

from pathlib import Path

import pytest

from pipeline.config import (
    DEFAULT_CONFIG_TOML,
    ArchivesConfig,
    Config,
    FolderMetaConfig,
    IdentifyConfig,
    OllamaConfig,
    PathsConfig,
    TriageConfig,
    WalkConfig,
    load_config,
)


class TestConfigModels:
    def test_paths_config(self):
        p = PathsConfig(corpus_root="/tmp/corpus", cache_root="/tmp/cache")
        assert p.corpus_root == "/tmp/corpus"
        assert p.cache_root == "/tmp/cache"

    def test_walk_config_defaults(self):
        w = WalkConfig()
        assert w.max_file_size_bytes == 5_368_709_120
        assert w.hash_workers == 0

    def test_archives_config_defaults(self):
        a = ArchivesConfig()
        assert a.max_depth == 5
        assert ".zip" in a.expand_extensions

    def test_identify_config_defaults(self):
        i = IdentifyConfig()
        assert i.siegfried_path == "sf"
        assert i.siegfried_workers == 32

    def test_triage_config_defaults(self):
        t = TriageConfig()
        assert t.pdf_sample_pages == 3
        assert t.pdf_text_threshold_chars_per_page == 50

    def test_ollama_config_defaults(self):
        o = OllamaConfig()
        assert o.host == "http://localhost:11434"
        assert o.model == "gemma4:latest"

    def test_folder_meta_config_defaults(self):
        f = FolderMetaConfig()
        assert f.max_filenames_in_prompt == 30
        assert f.min_files_to_infer == 1


class TestConfigRoot:
    def test_corpus_root_path(self):
        c = Config(paths=PathsConfig(corpus_root="/tmp/corpus", cache_root="/tmp/cache"))
        assert isinstance(c.corpus_root_path, Path)
        assert str(c.corpus_root_path) == "/tmp/corpus"

    def test_cache_root_path(self):
        c = Config(paths=PathsConfig(corpus_root="/tmp/corpus", cache_root="/tmp/cache"))
        assert isinstance(c.cache_root_path, Path)
        assert str(c.cache_root_path) == "/tmp/cache"

    def test_defaults_for_sub_configs(self):
        c = Config(paths=PathsConfig(corpus_root="/tmp/corpus", cache_root="/tmp/cache"))
        assert c.identify.siegfried_workers == 32
        assert c.triage.pdf_sample_pages == 3
        assert c.ollama.model == "gemma4:latest"


class TestLoadConfig:
    def test_load_valid_toml(self, tmp_path):
        toml = tmp_path / "config.toml"
        toml.write_text(
            DEFAULT_CONFIG_TOML.replace("/path/to/corpus", str(tmp_path / "corpus")),
            encoding="utf-8",
        )
        (tmp_path / "corpus").mkdir()
        cfg = load_config(toml)
        assert cfg.paths.corpus_root == str(tmp_path / "corpus")

    def test_load_custom_values(self, tmp_path):
        toml = tmp_path / "config.toml"
        content = f"""
[paths]
corpus_root = "{tmp_path}"
cache_root = "{tmp_path}/.rag-cache"

[identify]
siegfried_path = "/usr/local/bin/sf"
siegfried_workers = 8

[triage]
pdf_sample_pages = 5
"""
        toml.write_text(content, encoding="utf-8")
        cfg = load_config(toml)
        assert cfg.identify.siegfried_workers == 8
        assert cfg.triage.pdf_sample_pages == 5

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.toml")

    def test_load_invalid_toml(self, tmp_path):
        toml = tmp_path / "bad.toml"
        toml.write_text("this is not valid toml {{{{", encoding="utf-8")
        with pytest.raises(Exception):
            load_config(toml)


class TestDefaultConfigToml:
    def test_is_parsable(self):
        import tomllib

        data = tomllib.loads(DEFAULT_CONFIG_TOML)
        assert "paths" in data
        assert "walk" in data
        assert "identify" in data
