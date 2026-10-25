"""Thin search HTTP endpoints; behavior resides in services/adapters."""

from fastapi import APIRouter

from app.api.dependencies import ApiVersion, IndexName, ServiceDependency, SourceToken
from app.schemas.search import (
    AzureResponse,
    CompareRequest,
    CompareResponse,
    RawSearchRequest,
    SearchOptions,
    TextSearchRequest,
    TextSearchResponse,
)

router = APIRouter(prefix="/indexes/{index_name}", tags=["Search"])


@router.post("/search", response_model=AzureResponse)
async def raw_search(
    index_name: IndexName,
    data: RawSearchRequest,
    services: ServiceDependency,
    api_version: ApiVersion = None,
    source_token: SourceToken = None,
) -> dict:
    """Azure REST body (camelCase), executed via SearchClient.send_request()."""
    return await services.repository.raw_search(
        index_name, data.model_dump(exclude_none=True), api_version, source_token
    )


@router.post("/search/sdk", response_model=AzureResponse)
async def sdk_search(
    index_name: IndexName,
    data: SearchOptions,
    services: ServiceDependency,
    api_version: ApiVersion = None,
    source_token: SourceToken = None,
) -> dict:
    """Typed SearchClient.search options (snake_case)."""
    return await services.repository.search(index_name, data, api_version, source_token)


@router.post("/search/text", response_model=TextSearchResponse)
async def text_search(
    index_name: IndexName,
    data: TextSearchRequest,
    services: ServiceDependency,
    api_version: ApiVersion = None,
    source_token: SourceToken = None,
) -> dict:
    """Generate an embedding inside UAT, then execute the selected search mode."""
    return await services.retrieval.text_search(index_name, data, api_version, source_token)


@router.post("/experiments/compare", response_model=CompareResponse, tags=["Experiments"])
async def compare(
    index_name: IndexName,
    data: CompareRequest,
    services: ServiceDependency,
    api_version: ApiVersion = None,
    source_token: SourceToken = None,
) -> dict:
    """Compare retrieval modes for one labeled query using a shared embedding."""
    return await services.experiments.compare(index_name, data, api_version, source_token)
