"""Configuration parsing via Pydantic. Reads config.toml from project root."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# ============================================================
# Pre-flight configs (preserve backward compatibility)
# ============================================================


class PathsConfig(BaseModel):
    corpus_root: str
    cache_root: str
    chroma_root: str = "./chroma"


class WalkConfig(BaseModel):
    max_file_size_bytes: int = 5_368_709_120  # 5 GiB
    skip_patterns: list[str] = Field(
        default_factory=lambda: [".DS_Store", "Thumbs.db", "__MACOSX", "*.tmp"]
    )
    hash_workers: int = 0  # 0 = os.cpu_count()


class ArchivesConfig(BaseModel):
    max_depth: int = 5
    expand_extensions: list[str] = Field(
        default_factory=lambda: [".zip", ".rar", ".7z", ".tar", ".tar.gz", ".tgz", ".tar.bz2"]
    )
    exclude_extensions: list[str] = Field(
        default_factory=lambda: [
            ".pages",
            ".numbers",
            ".key",
            ".docx",
            ".xlsx",
            ".pptx",
            ".epub",
            ".jar",
        ]
    )


class IdentifyConfig(BaseModel):
    siegfried_path: str = "sf"
    siegfried_workers: int = 32


class TriageConfig(BaseModel):
    pdf_sample_pages: int = 3
    pdf_text_threshold_chars_per_page: int = 50


class OllamaConfig(BaseModel):
    host: str = "http://localhost:11434"
    model: str = "gemma4:latest"
    timeout_seconds: int = 120
    max_retries: int = 3
    keep_alive: str = "10m"


class FolderMetaConfig(BaseModel):
    max_filenames_in_prompt: int = 30
    min_files_to_infer: int = 1


# ============================================================
# Post pre-flight model configs
# ============================================================


class EmbeddingConfig(BaseModel):
    name: str
    collection_suffix: str
    batch_size: int = 32
    normalize: bool = True
    truncate_dim: int = 0  # 0 means no truncation


class EmbeddingAlternativesConfig(BaseModel):
    name: str
    suffix: str


class SummarizationConfig(BaseModel):
    name: str
    temperature: float = 0.3
    max_tokens: int = 400
    prompt_version: str = "v1"


class ContextualRetrievalConfig(BaseModel):
    enabled: bool = True
    name: str
    temperature: float = 0.1
    max_context_tokens: int = 16384
    prompt_version: str = "v1"


class GenerationEndpointConfig(BaseModel):
    host: str = "http://localhost:11434"
    api_key: str = ""
    retries: int = 3


class GenerationConfig(BaseModel):
    name: str
    fallback_name: str
    alternate_name: str
    temperature: float = 0.5
    top_p: float = 0.95
    top_k: int = 64
    max_context_tokens: int = 16384
    prompt_version: str = "v1"
    thinking: bool = False
    endpoint: GenerationEndpointConfig = Field(default_factory=GenerationEndpointConfig)


class RerankerConfig(BaseModel):
    enabled: bool = False
    name: str = ""
    top_n_in: int = 50
    top_n_out: int = 8


class TranscriptionConfig(BaseModel):
    backend: str = "whisper.cpp"
    model: str = "large-v3"
    language: str = "auto"
    threads: int = 8
    word_timestamps: bool = True
    opt_in_globs: list[str] = Field(default_factory=list)


class LanguageIdConfig(BaseModel):
    backend: str = "fasttext"
    model_path: str = ".rag-cache/models/lid.176.bin"


class ModelsConfig(BaseModel):
    embedding: EmbeddingConfig
    summarization: SummarizationConfig
    contextual_retrieval: ContextualRetrievalConfig
    generation: GenerationConfig
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    language_id: LanguageIdConfig = Field(default_factory=LanguageIdConfig)
    alternatives: dict[str, EmbeddingAlternativesConfig] = Field(default_factory=dict)


# ============================================================
# Pipeline parameter configs
# ============================================================


class ExtractConfig(BaseModel):
    docling_workers: int = 4
    tika_url: str = "http://localhost:9998"
    size_threshold_for_summary: int = 500
    tika_brute_force_fallback: bool = True


class ChunkConfig(BaseModel):
    chunker: str = "sentence-window"
    window_size: int = 5
    target_tokens: int = 512
    chunker_version: str = "1"


class RetrievalConfig(BaseModel):
    top_k_chunks: int = 8
    top_k_documents: int = 20
    top_k_folders: int = 5
    dense_candidates: int = 50
    bm25_candidates: int = 50
    hierarchical: bool = True
    rrf_k: int = 60


class GenerationRuntimeConfig(BaseModel):
    require_citations: bool = True
    refuse_on_empty_context: bool = True


class ApiConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765


class UiConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 7860


# ============================================================
# Composed configs
# ============================================================


class Config(BaseModel):
    """Pre-flight config composition (backward compatible)."""

    paths: PathsConfig
    walk: WalkConfig = Field(default_factory=WalkConfig)
    archives: ArchivesConfig = Field(default_factory=ArchivesConfig)
    identify: IdentifyConfig = Field(default_factory=IdentifyConfig)
    triage: TriageConfig = Field(default_factory=TriageConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    folder_meta: FolderMetaConfig = Field(default_factory=FolderMetaConfig)

    @property
    def corpus_root_path(self) -> Path:
        return Path(self.paths.corpus_root)

    @property
    def cache_root_path(self) -> Path:
        return Path(self.paths.cache_root)


class AppConfig(BaseModel):
    """Post pre-flight config composition (phases 8-14)."""

    paths: PathsConfig
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    models: ModelsConfig
    extract: ExtractConfig = Field(default_factory=ExtractConfig)
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation_runtime: GenerationRuntimeConfig = Field(default_factory=GenerationRuntimeConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    ui: UiConfig = Field(default_factory=UiConfig)

    @property
    def corpus_root_path(self) -> Path:
        return Path(self.paths.corpus_root)

    @property
    def cache_root_path(self) -> Path:
        return Path(self.paths.cache_root)

    @property
    def chroma_root_path(self) -> Path:
        return Path(self.paths.chroma_root)


# ============================================================
# Config hashing
# ============================================================


def _canonical_json(obj: Any) -> str:
    """Produce deterministic JSON for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def config_hash(
    role_block: dict[str, Any],
    chunk_block: dict[str, Any],
    extract_block: dict[str, Any],
) -> str:
    """SHA-256 of the canonical JSON of combined config blocks."""
    combined = {
        "role": role_block,
        "chunk": chunk_block,
        "extract": extract_block,
    }
    return hashlib.sha256(_canonical_json(combined).encode("utf-8")).hexdigest()


def embedding_config_hash(cfg: EmbeddingConfig) -> str:
    """SHA-256 of the embedding-relevant config block."""
    return hashlib.sha256(_canonical_json(cfg.model_dump()).encode("utf-8")).hexdigest()


# ============================================================
# Loading
# ============================================================


def load_config(config_path: Path) -> Config:
    """Load and validate pre-flight config from a TOML file."""
    with open(config_path, "rb") as f:
        raw: dict[str, Any] = tomllib.load(f)
    return Config(**raw)


def load_app_config(config_path: Path) -> AppConfig:
    """Load and validate post pre-flight config from a TOML file."""
    with open(config_path, "rb") as f:
        raw: dict[str, Any] = tomllib.load(f)

    # [models.embedding.alternatives] parses as nested under embedding;
    # lift it to models.alternatives for AppConfig composition.
    models_block = raw.get("models", {})
    emb_block = models_block.get("embedding", {})
    if "alternatives" in emb_block:
        models_block["alternatives"] = emb_block.pop("alternatives")

    return AppConfig(**raw)


DEFAULT_CONFIG_TOML = """\
[paths]
corpus_root = "/path/to/corpus"
cache_root = "/path/to/corpus/.rag-cache"
chroma_root = "./chroma"

[walk]
max_file_size_bytes = 5_368_709_120
skip_patterns = [".DS_Store", "Thumbs.db", "__MACOSX", "*.tmp"]
hash_workers = 0

[archives]
max_depth = 5
expand_extensions = [".zip", ".rar", ".7z", ".tar", ".tar.gz", ".tgz", ".tar.bz2"]
exclude_extensions = [".pages", ".numbers", ".key", ".docx", ".xlsx", ".pptx", ".epub", ".jar"]

[identify]
siegfried_path = "sf"
siegfried_workers = 32

[triage]
pdf_sample_pages = 3
pdf_text_threshold_chars_per_page = 50

[ollama]
host = "http://localhost:11434"
model = "gemma4:latest"
timeout_seconds = 120
max_retries = 3
keep_alive = "10m"

[folder_meta]
max_filenames_in_prompt = 30
min_files_to_infer = 1

[models.embedding]
name = "qwen3-embedding:8b"
collection_suffix = "qwen3emb8b"
batch_size = 32
normalize = true
truncate_dim = 0

[models.summarization]
name = "gemma4:latest"
temperature = 0.3
max_tokens = 400
prompt_version = "v1"

[models.contextual_retrieval]
enabled = true
name = "gemma4:latest"
temperature = 0.1
max_context_tokens = 16384
prompt_version = "v1"

[models.generation]
name = "gpt-oss:20b"
fallback_name = "gemma4:latest"
alternate_name = "gemma4:26b"
temperature = 0.5
top_p = 0.95
top_k = 64
max_context_tokens = 16384
prompt_version = "v1"
thinking = false

[models.generation.endpoint]
host = "https://ollama.com"
api_key = ""
retries = 3

[models.reranker]
enabled = false
name = "dengcao/Qwen3-Reranker-0.6B"
top_n_in = 50
top_n_out = 8

[models.transcription]
backend = "whisper.cpp"
model = "large-v3"
language = "auto"
threads = 8
word_timestamps = true
opt_in_globs = []

[models.language_id]
backend = "fasttext"
model_path = ".rag-cache/models/lid.176.bin"

[extract]
docling_workers = 4
tika_url = "http://localhost:9998"
size_threshold_for_summary = 500
tika_brute_force_fallback = true

[chunk]
chunker = "sentence-window"
window_size = 5
target_tokens = 512
chunker_version = "1"

[retrieval]
top_k_chunks = 8
top_k_documents = 20
top_k_folders = 5
dense_candidates = 50
bm25_candidates = 50
hierarchical = true
rrf_k = 60

[generation_runtime]
require_citations = true
refuse_on_empty_context = true

[api]
host = "127.0.0.1"
port = 8765

[ui]
host = "127.0.0.1"
port = 7860
"""
