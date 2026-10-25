"""Index schema and document action endpoints."""

from typing import Annotated, Any

from fastapi import APIRouter, Body

from app.api.dependencies import ApiVersion, IndexName, ServiceDependency
from app.schemas.indexing import (
    BaselineIndexRequest,
    EmbedChunksRequest,
    EmbedChunksResponse,
    IndexBatchRequest,
    IndexBatchResponse,
)
from app.schemas.search import AzureResponse

router = APIRouter(prefix="/indexes/{index_name}", tags=["Indexes"])


@router.get("", response_model=AzureResponse)
async def get_index(
    index_name: IndexName, services: ServiceDependency, api_version: ApiVersion = None
) -> dict:
    return await services.repository.get_index(index_name, api_version)


@router.post("", status_code=201, response_model=AzureResponse)
async def create_index(
    index_name: IndexName,
    services: ServiceDependency,
    definition: Annotated[
        dict[str, Any], Body(description="Azure index schema supported by pinned SDK.")
    ],
    api_version: ApiVersion = None,
) -> dict:
    """Create an index; never silently overwrite an existing index."""
    return await services.indexing.create(index_name, definition, api_version)


@router.post("/baseline-schema", status_code=201, response_model=AzureResponse)
async def baseline_schema(
    index_name: IndexName,
    data: BaselineIndexRequest,
    services: ServiceDependency,
    api_version: ApiVersion = None,
) -> dict:
    return await services.indexing.create_baseline(index_name, data, api_version)


@router.post("/documents", response_model=IndexBatchResponse)
async def index_documents(
    index_name: IndexName,
    data: IndexBatchRequest,
    services: ServiceDependency,
    api_version: ApiVersion = None,
) -> dict:
    """SDK batch indexing. Check all_succeeded and each per-document status."""
    return await services.repository.index_documents(index_name, data.value, api_version)


@router.post("/documents/embed", response_model=EmbedChunksResponse)
async def embed_documents(
    index_name: IndexName,
    data: EmbedChunksRequest,
    services: ServiceDependency,
    api_version: ApiVersion = None,
) -> dict:
    return await services.indexing.embed_chunks(index_name, data, api_version)
