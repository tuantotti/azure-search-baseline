"""Index and embedding operation contracts."""

from typing import Any

from pydantic import Field

from .search import AzureResponse, StrictModel


class BaselineIndexRequest(StrictModel):
    dimensions: int = Field(ge=2)
    description: str = "Azure AI Search baseline"


class IndexBatchRequest(StrictModel):
    value: list[dict[str, Any]] = Field(min_length=1, max_length=1000)


class EmbeddingRequest(StrictModel):
    input: list[str] = Field(min_length=1, max_length=32)


class Chunk(StrictModel):
    chunk_id: str = Field(pattern=r"^[A-Za-z0-9_=-]+$")
    document_id: str = Field(min_length=1)
    title: str
    content: str = Field(min_length=1)
    domain: str
    source: str
    document_type: str
    classification: str
    source_uri: str
    page_start: int | None = Field(default=None, ge=1)
    is_public: bool
    allowed_group_ids: list[str]


class EmbedChunksRequest(StrictModel):
    documents: list[Chunk] = Field(min_length=1, max_length=32)


class IndexBatchResponse(AzureResponse):
    all_succeeded: bool


class EmbedChunksResponse(IndexBatchResponse):
    embedding_ms: float
