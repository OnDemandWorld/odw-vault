"""Shared test fixtures and helpers for the RAG pre-flight test suite."""

from __future__ import annotations

import csv
import hashlib
import json
import struct
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz
import pytest
import sqlite_utils

from pipeline.config import Config, PathsConfig
from pipeline.db import migrate, open_db
from pipeline.phase5_folder_meta import FolderInference

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEEDS_CSV = PROJECT_ROOT / "seeds" / "format_policy.csv"


# ---------------------------------------------------------------------------
# Corpus factory
# ---------------------------------------------------------------------------


def create_test_corpus(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    """Create a synthetic test corpus. Returns (corpus_root, path_dict)."""
    root = tmp_path / "corpus"
    root.mkdir()

    # Plain text files
    (root / "readme.txt").write_text(
        "Hello world, this is a test document.\n" * 10, encoding="utf-8"
    )
    (root / "notes.txt").write_text(
        "Another text file with enough content for language detection to work properly.\n" * 10,
        encoding="utf-8",
    )
    (root / "notes_copy.txt").write_text(
        "Another text file with enough content for language detection to work properly.\n" * 10,
        encoding="utf-8",
    )  # duplicate

    # Data files
    (root / "data.csv").write_text(
        "id,name,value\n1,alpha,100\n2,beta,200\n3,gamma,300\n", encoding="utf-8"
    )
    (root / "config.json").write_text(
        '{"database": {"host": "localhost", "port": 5432}, "debug": true}\n', encoding="utf-8"
    )

    # PDFs
    _make_text_pdf(root / "hello.pdf")
    _make_scanned_pdf(root / "scanned.pdf")
    _make_encrypted_pdf(root / "encrypted.pdf")
    _make_empty_pdf(root / "empty.pdf")

    # Image — minimal valid PNG (1x1 white pixel)
    _make_minimal_png(root / "image.png")

    # Archives
    _make_zip(root / "archive.zip", {"inner_file.txt": "content from inner file\n"})
    _make_nested_zip(tmp_path, root / "nested.zip")

    # DOC_ARCHIVE (should NOT be expanded by phase0)
    (root / "tool.docx").write_bytes(b"PK\x03\x04" + b"\x00" * 20)

    # Hidden file (should be skipped by walk)
    (root / ".DS_Store").write_text("", encoding="utf-8")

    # Subdirectory
    subdir = root / "subdir"
    subdir.mkdir()
    (subdir / "deep.txt").write_text(
        "This is a file in a subdirectory for testing walk depth.\n" * 10, encoding="utf-8"
    )

    # .rag-cache (should be skipped by walk)
    cache_dir = root / ".rag-cache" / "logs"
    cache_dir.mkdir(parents=True)
    (cache_dir / "test.jsonl").write_text('{"msg": "test"}\n', encoding="utf-8")

    paths = {
        "readme.txt": root / "readme.txt",
        "notes.txt": root / "notes.txt",
        "notes_copy.txt": root / "notes_copy.txt",
        "data.csv": root / "data.csv",
        "config.json": root / "config.json",
        "hello.pdf": root / "hello.pdf",
        "scanned.pdf": root / "scanned.pdf",
        "encrypted.pdf": root / "encrypted.pdf",
        "empty.pdf": root / "empty.pdf",
        "image.png": root / "image.png",
        "archive.zip": root / "archive.zip",
        "nested.zip": root / "nested.zip",
        "tool.docx": root / "tool.docx",
        "subdir/deep.txt": subdir / "deep.txt",
    }
    return root, paths


def _make_text_pdf(path: Path) -> None:
    """Create a PDF with a text layer."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72), "This is a test PDF document with a text layer for triage testing.", fontsize=12
    )
    doc.save(str(path))
    doc.close()


def _make_scanned_pdf(path: Path) -> None:
    """Create a PDF with image-only pages (no text layer)."""
    # Create a minimal 1x1 pixel PNG in memory
    png_bytes = _make_png_bytes(1, 1, (255, 255, 255))
    doc = fitz.open()
    page = doc.new_page()
    # Insert image (will be a page with an image, no text)
    rect = fitz.Rect(0, 0, 100, 100)
    page.insert_image(rect, stream=png_bytes)
    doc.save(str(path))
    doc.close()


def _make_encrypted_pdf(path: Path) -> None:
    """Create an encrypted PDF."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "This is an encrypted PDF for testing.", fontsize=12)
    doc.save(str(path), encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user")
    doc.close()


def _make_empty_pdf(path: Path) -> None:
    """Create a minimal 1-page blank PDF (no text)."""
    doc = fitz.open()
    doc.new_page()
    doc.save(str(path))
    doc.close()


def _make_png_bytes(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """Generate a minimal valid PNG with the given dimensions and color."""
    import zlib

    # PNG signature
    signature = b"\x89PNG\r\n\x1a\n"
    # IHDR
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
    ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
    # IDAT
    raw = b""
    for _ in range(height):
        raw += b"\x00" + bytes(list(rgb)) * width
    compressed = zlib.compress(raw)
    idat_crc = zlib.crc32(b"IDAT" + compressed) & 0xFFFFFFFF
    idat = struct.pack(">I", len(compressed)) + b"IDAT" + compressed + struct.pack(">I", idat_crc)
    # IEND
    iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
    iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
    return signature + ihdr + idat + iend


def _make_minimal_png(path: Path) -> None:
    path.write_bytes(_make_png_bytes(1, 1, (255, 255, 255)))


def _make_zip(path: Path, contents: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in contents.items():
            zf.writestr(name, data)


def _make_nested_zip(tmp_path: Path, path: Path) -> None:
    inner = tmp_path / "inner.zip"
    _make_zip(inner, {"nested_inner.txt": "content inside nested archive\n"})
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(inner, "inner.zip")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_corpus(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    """Create a synthetic corpus. Yields (corpus_root, path_dict)."""
    return create_test_corpus(tmp_path)


@pytest.fixture
def test_db(tmp_path: Path) -> sqlite_utils.Database:
    """Fresh database with full schema and seeded format_policy."""
    db_path = tmp_path / "test.db"
    db = open_db(db_path)
    migrate(db)
    db.path = db_path  # Attach path for test access
    # Seed format_policy from CSV
    if SEEDS_CSV.exists():
        import csv

        with open(SEEDS_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            db["format_policy"].insert_all(reader)
    return db


@pytest.fixture
def test_config(test_corpus: tuple[Path, dict]) -> Config:
    """Config pointing to the test corpus."""
    root, _ = test_corpus
    cache = root / ".rag-cache"
    cache.mkdir(exist_ok=True)
    return Config(
        paths=PathsConfig(corpus_root=str(root), cache_root=str(cache)),
    )


@pytest.fixture
def mock_plog() -> MagicMock:
    """No-op PhaseLogger replacement."""
    plog = MagicMock()
    plog.info = MagicMock()
    plog.warning = MagicMock()
    plog.error = MagicMock()
    plog.debug = MagicMock()
    return plog


@pytest.fixture
def mock_ollama() -> MagicMock:
    """Mock _call_ollama to return a valid FolderInference."""
    result = FolderInference(
        category="client-project",
        label="Test Project",
        tags=["test", "sample"],
        summary="A test folder for validation.",
    )
    with patch("pipeline.phase5_folder_meta._call_ollama", return_value=result) as m:
        yield m


def build_sf_response(file_paths: list[str], corpus_root: Path) -> str:
    """Build a Siegfried-style JSON response for given absolute paths."""
    files = []
    for fp in file_paths:
        ext = Path(fp).suffix.lower()
        # Map common extensions to plausible PRONOM IDs
        mapping = {
            ".txt": ("x-fmt/111", "Plain Text File", "text/plain", "document", "tika"),
            ".csv": ("x-fmt/18", "Comma Separated Values", "text/csv", "data", "tika"),
            ".json": ("fmt/817", "JSON", "application/json", "data", "tika"),
            ".pdf": ("fmt/16", "Acrobat PDF 1.2", "application/pdf", "pdf-text", "docling"),
            ".png": ("fmt/13", "Portable Network Graphics", "image/png", "image", "metadata-only"),
            ".docx": (
                "fmt/412",
                "Microsoft Word OOXML",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "document",
                "docling",
            ),
            ".zip": ("fmt/189", "ZIP Format", "application/zip", "archive", "skip"),
        }
        info = mapping.get(ext, ("UNKNOWN", "Unknown", "", "unknown", "manual"))
        files.append(
            {
                "filename": fp,
                "filesize": Path(fp).stat().st_size if Path(fp).exists() else 100,
                "modified": "2026-01-01T00:00:00Z",
                "matches": [
                    {
                        "ns": "pronom",
                        "id": info[0],
                        "format": info[1],
                        "mime": info[2],
                        "version": "",
                    }
                ],
            }
        )
    return json.dumps({"siegfried": "v1.11.4", "files": files}, ensure_ascii=False)


@pytest.fixture
def mock_siegfried(test_corpus: tuple[Path, dict]) -> MagicMock:
    """Mock subprocess.run to return Siegfried JSON."""
    root, paths = test_corpus
    all_files = [str(p.resolve()) for p in paths.values() if p.exists()]
    response = build_sf_response(all_files, root)
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=response, stderr="")
        yield mock_run


@pytest.fixture
def mock_patool() -> MagicMock:
    """Mock patoolib.extract_archive to simulate extraction."""

    def fake_extract(archive_path: str, outdir: str, verbosity: int = -1):
        Path(outdir).mkdir(parents=True, exist_ok=True)
        # Create a dummy file based on the archive name
        archive_name = Path(archive_path).stem
        (Path(outdir) / f"{archive_name}_content.txt").write_text(
            f"Extracted content from {archive_name}\n", encoding="utf-8"
        )

    with patch("patoolib.extract_archive", side_effect=fake_extract) as m:
        yield m


def make_config(corpus_root: Path, cache_root: Path | None = None) -> Config:
    """Build a Config with given paths."""
    if cache_root is None:
        cache_root = corpus_root / ".rag-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    return Config(paths=PathsConfig(corpus_root=str(corpus_root), cache_root=str(cache_root)))


def seed_format_policy(db: sqlite_utils.Database) -> None:
    """Load seeds/format_policy.csv into the format_policy table."""
    if SEEDS_CSV.exists():
        with open(SEEDS_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            db["format_policy"].insert_all(reader)


# ---------------------------------------------------------------------------
# Part 2 fixtures (RAG pipeline: rag/, api/, eval/, ui/)
# ---------------------------------------------------------------------------


@pytest.fixture
def hit_factory():
    """Factory for rag.retrieval.Hit dataclass instances."""
    from rag.retrieval import Hit

    def _make_hit(
        chunk_id: int = 1,
        file_id: int = 1,
        folder_id: int = 1,
        rel_path: str = "test/doc.txt",
        page_start: int | None = None,
        text: str = "test content",
        dense_score: float | None = None,
        bm25_score: float | None = None,
        rerank_score: float | None = None,
        fused_score: float | None = None,
    ) -> Hit:
        return Hit(
            chunk_id=chunk_id,
            file_id=file_id,
            folder_id=folder_id,
            rel_path=rel_path,
            page_start=page_start,
            text=text,
            dense_score=dense_score,
            bm25_score=bm25_score,
            rerank_score=rerank_score,
            fused_score=fused_score,
        )

    return _make_hit


@pytest.fixture
def mock_ollama_chat():
    """Mock ollama.Client for chat responses.

    Returns the mock client instance. Configure .chat.return_value to control
    the response. Default: {"message": {"content": "Test response."}}
    """
    with patch("ollama.Client") as MockClient:
        instance = MockClient.return_value
        instance.chat.return_value = {"message": {"content": "Test response."}}
        instance.list.return_value = {"models": []}
        instance.embed.return_value = {"embeddings": [[0.1] * 4096]}
        yield instance


@pytest.fixture
def mock_chroma_collection():
    """Mock chromadb.PersistentClient with a mock collection.

    Yields (client, collection) tuple.
    """
    coll = MagicMock()
    coll.name = "chunks__test"
    coll.metadata = {
        "embedding_model": "test-model",
        "hnsw:space": "cosine",
        "dim": 4096,
        "config_hash": "test",
        "source_db_path": "/test.db",
        "contextual_augmentation": "false",
    }

    client = MagicMock()
    client.get_collection.return_value = coll
    client.list_collections.return_value = [coll]
    client.create_collection.return_value = coll
    client.delete_collection = MagicMock()

    with patch("chromadb.PersistentClient", return_value=client):
        yield client, coll


@pytest.fixture
def app_config(tmp_path: Path):
    """Build a minimal AppConfig for Part 2 tests."""
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
    corpus.mkdir()
    cache = corpus / ".rag-cache"
    cache.mkdir()
    chroma = tmp_path / "chroma"
    chroma.mkdir()

    return AppConfig(
        paths=PathsConfig(
            corpus_root=str(corpus),
            cache_root=str(cache),
            chroma_root=str(chroma),
        ),
        ollama=OllamaConfig(),
        models=ModelsConfig(
            embedding=EmbeddingConfig(name="test-embed", collection_suffix="test"),
            summarization=SummarizationConfig(name="gemma4:latest"),
            contextual_retrieval=ContextualRetrievalConfig(
                enabled=False, name="gemma4:latest"
            ),
            generation=GenerationConfig(
                name="gemma4:latest",
                fallback_name="gemma4:latest",
                alternate_name="gemma4:26b",
            ),
            reranker=RerankerConfig(enabled=False),
        ),
        generation_runtime=GenerationRuntimeConfig(refuse_on_empty_context=True),
        chunk=ChunkConfig(),
        retrieval=RetrievalConfig(),
        extract=ExtractConfig(),
    )


def seed_test_files(
    db: sqlite_utils.Database,
    files: list[dict] | None = None,
    folders: list[dict] | None = None,
) -> list[int]:
    """Insert test rows into folder/file tables. Returns file IDs.

    Args:
        db: sqlite_utils Database with migrated schema.
        files: List of dicts with keys like name, rel_path, folder_id, sha256,
               extract_strategy, is_dup_primary, excluded, etc.
        folders: List of dicts with keys like rel_path, parent_id, etc.

    Returns:
        List of inserted file IDs.
    """
    if folders:
        for folder in folders:
            db["folder"].insert(folder)
        db.conn.commit()

    if not files:
        # Ensure a default folder exists
        folder_count = next(iter(db.query("SELECT COUNT(*) as c FROM folder")))["c"]
        if folder_count == 0:
            db["folder"].insert({"path": "test", "rel_path": "test", "name": "test", "depth": 0})
            db.conn.commit()
        default_folder_id = next(iter(db.query("SELECT id FROM folder LIMIT 1")))["id"]
        files = [
            {
                "name": "test.txt",
                "path": "/tmp/corpus/test.txt",
                "rel_path": "test.txt",
                "folder_id": default_folder_id,
                "sha256": "abc123",
                "size_bytes": 100,
                "mtime": "2026-01-01T00:00:00Z",
                "hash_status": "done",
                "identify_status": "done",
                "triage_status": "pending",
                "category": "document",
                "extract_strategy": "textutil",
                "is_dup_primary": 1,
                "excluded": 0,
            },
        ]

    file_ids = []
    file_table = db["file"]
    for f in files:
        row = dict(f)
        row.setdefault("path", f"/tmp/corpus/{row['name']}")
        row.setdefault("rel_path", row["name"])
        row.setdefault("size_bytes", 100)
        row.setdefault("mtime", "2026-01-01T00:00:00Z")
        row.setdefault("triage_status", "pending")
        # Ensure a folder exists for this file
        if "folder_id" not in row:
            folder_count = next(iter(db.query("SELECT COUNT(*) as c FROM folder")))["c"]
            if folder_count == 0:
                db["folder"].insert({"path": "test", "rel_path": "test", "name": "test", "depth": 0})
                db.conn.commit()
            row["folder_id"] = next(iter(db.query("SELECT id FROM folder LIMIT 1")))["id"]
        row.setdefault("sha256", hashlib.sha256(row["name"].encode()).hexdigest())
        row.setdefault("hash_status", "done")
        row.setdefault("identify_status", "done")
        row.setdefault("category", "document")
        row.setdefault("extract_strategy", "textutil")
        row.setdefault("is_dup_primary", 1)
        row.setdefault("excluded", 0)
        file_table.insert(row)
        file_ids.append(file_table.last_rowid)
    db.conn.commit()
    return file_ids


def seed_test_extractions(
    db: sqlite_utils.Database,
    file_ids: list[int],
    text: str = "Sample extracted text for testing.",
) -> list[int]:
    """Insert extraction rows for given file IDs. Returns extraction IDs."""
    ext_ids = []
    ext_table = db["extraction"]
    for fid in file_ids:
        ext_table.insert(
            {
                "file_id": fid,
                "text_extracted": text,
                "tool": "test",
                "succeeded": 1,
                "char_count": len(text),
            }
        )
        ext_ids.append(ext_table.last_rowid)
    db.conn.commit()
    return ext_ids
