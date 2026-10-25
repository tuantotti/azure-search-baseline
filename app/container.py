"""Composition root and deterministic cleanup of async resources."""

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

import aiohttp
from azure.core.credentials import AzureKeyCredential
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from openai import AsyncAzureOpenAI, AsyncOpenAI

from app.core.config import Settings
from app.infrastructure.agentic import AzureAgenticRepository
from app.infrastructure.azure_search import AzureSearchRepository
from app.infrastructure.embeddings import AzureEmbeddingProvider, OpenAIEmbeddingProvider
from app.infrastructure.indexers import AzureIndexerRepository
from app.ports import AgenticRepository, EmbeddingProvider, IndexerRepository, SearchRepository
from app.services.agentic import AgenticService
from app.services.enrichment import EnrichmentService
from app.services.experiments import ExperimentService
from app.services.indexing import IndexingService
from app.services.retrieval import RetrievalService


@dataclass(frozen=True)
class Services:
    repository: SearchRepository
    embeddings: EmbeddingProvider
    retrieval: RetrievalService
    indexing: IndexingService
    experiments: ExperimentService
    agentic: AgenticService
    enrichment: EnrichmentService


def compose_services(
    repository: SearchRepository,
    embeddings: EmbeddingProvider,
    agentic_repository: AgenticRepository | None = None,
    indexer_repository: IndexerRepository | None = None,
) -> Services:
    retrieval = RetrievalService(repository, embeddings)
    return Services(
        repository,
        embeddings,
        retrieval,
        IndexingService(repository, embeddings),
        ExperimentService(retrieval),
        AgenticService(agentic_repository),
        EnrichmentService(indexer_repository),
    )


@asynccontextmanager
async def build_services(settings: Settings) -> AsyncIterator[Services]:
    async with AsyncExitStack() as stack:
        session = await stack.enter_async_context(aiohttp.ClientSession())
        identity = None
        if not settings.azure_search_api_key or (
            settings.embedding_provider == "azure_openai"
            and settings.azure_openai_endpoint
            and settings.azure_openai_embedding_deployment
            and not settings.azure_openai_api_key
        ):
            identity = await stack.enter_async_context(DefaultAzureCredential())
        credential = (
            AzureKeyCredential(settings.azure_search_api_key.get_secret_value())
            if settings.azure_search_api_key
            else identity
        )
        repository = AzureSearchRepository(settings, credential, session)
        client = None
        if (
            settings.embedding_provider == "azure_openai"
            and settings.azure_openai_endpoint
            and settings.azure_openai_embedding_deployment
        ):
            authentication = (
                {"api_key": settings.azure_openai_api_key.get_secret_value()}
                if settings.azure_openai_api_key
                else {
                    "azure_ad_token_provider": get_bearer_token_provider(
                        identity, "https://cognitiveservices.azure.com/.default"
                    )
                }
            )
            client = await stack.enter_async_context(
                AsyncAzureOpenAI(
                    azure_endpoint=settings.azure_openai_endpoint,
                    api_version=settings.openai_api_version,
                    timeout=settings.request_timeout_seconds,
                    max_retries=settings.azure_retry_total,
                    **authentication,
                )
            )
        embeddings = AzureEmbeddingProvider(client, settings.azure_openai_embedding_deployment)
        if settings.embedding_provider == "openai":
            if settings.openai_api_key and settings.openai_embedding_model:
                client = await stack.enter_async_context(
                    AsyncOpenAI(
                        api_key=settings.openai_api_key.get_secret_value(),
                        timeout=settings.request_timeout_seconds,
                        max_retries=settings.azure_retry_total,
                    )
                )
            embeddings = OpenAIEmbeddingProvider(client, settings.openai_embedding_model)
        agentic_repository = AzureAgenticRepository(settings, credential, session)
        yield compose_services(
            repository,
            embeddings,
            agentic_repository,
            AzureIndexerRepository(settings, credential, session),
        )
