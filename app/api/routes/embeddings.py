"""Embedding endpoint using the configured embedding provider."""

from fastapi import APIRouter

from app.api.dependencies import ServiceDependency
from app.schemas.indexing import EmbeddingRequest
from app.schemas.search import AzureResponse

router = APIRouter(tags=["Embeddings"])


@router.post("/embeddings", response_model=AzureResponse)
async def embed(data: EmbeddingRequest, services: ServiceDependency) -> dict:
    return await services.embeddings.embed(data.input)
