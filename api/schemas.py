"""Pydantic request/response schemas for the FastAPI service."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FolderFilter(BaseModel):
    path_prefix: str | None = None
    folder_id: int | None = None
    inferred_category: str | None = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    folder_filter: FolderFilter | None = None
    top_k_chunks: int | None = Field(None, ge=1, le=20)
    use_reranker: bool | None = None
    use_augmentation: bool | None = None
    thinking: bool | None = None
    stream: bool | None = False
    user: str | None = None


class Citation(BaseModel):
    marker: str
    file_id: int
    rel_path: str
    page: int | None
    chunk_id: int
    snippet: str


class Metrics(BaseModel):
    retrieval_ms: float | None
    generation_ms: float | None
    total_ms: float


class ModelInfo(BaseModel):
    embedding: str
    generation: str
    reranker: str | None
    contextual_augmentation: str | None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[dict]
    metrics: Metrics
    models: ModelInfo
    query_log_id: int


class FeedbackRequest(BaseModel):
    query_log_id: int
    feedback: str  # "up" or "down"
    note: str | None = None


class HealthResponse(BaseModel):
    ollama: bool
    chroma: bool
    database: bool
    fasttext: bool


class FileResponse(BaseModel):
    id: int
    rel_path: str
    name: str
    category: str | None
    format_name: str | None
    page_count: int | None
    folder_id: int
    parent_folder: str | None
    summary: str | None
    extraction_path: str | None


class FolderNode(BaseModel):
    id: int
    rel_path: str
    name: str
    inferred_category: str | None
    inferred_label: str | None
    children: list[FolderNode] = Field(default_factory=list)
