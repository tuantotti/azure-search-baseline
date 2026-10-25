"""Provider wiring, SDK arguments, and shared indexing/retrieval behavior."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
from fastapi.testclient import TestClient
from openai import AuthenticationError

from app.container import build_services, compose_services
from app.core.config import Settings
from app.core.errors import ApplicationError
from app.infrastructure.embeddings import AzureEmbeddingProvider, OpenAIEmbeddingProvider
from app.main import create_app
from app.schemas.indexing import EmbedChunksRequest
from app.schemas.search import TextSearchRequest


def settings(**kwargs):
    return Settings(
        _env_file=None,
        azure_search_endpoint="https://example.search.windows.net",
        azure_search_api_key="test",
        **kwargs,
    )


class EmbeddingTests(unittest.IsolatedAsyncioTestCase):
    async def test_both_adapters_sort_and_request_float_vectors(self):
        for adapter in (OpenAIEmbeddingProvider, AzureEmbeddingProvider):
            with self.subTest(adapter=adapter):
                create = AsyncMock(
                    return_value=Mock(
                        model_dump=Mock(
                            return_value={
                                "data": [
                                    {"index": 1, "embedding": [2.0]},
                                    {"index": 0, "embedding": [1.0]},
                                ]
                            }
                        )
                    )
                )
                provider = adapter(
                    SimpleNamespace(embeddings=SimpleNamespace(create=create)), "model"
                )
                result = await provider.embed(["a", "b"])
                self.assertEqual([v["index"] for v in result["azure"]["data"]], [0, 1])
                create.assert_awaited_once_with(
                    model="model", input=["a", "b"], encoding_format="float"
                )
                with self.assertRaises(ApplicationError) as error:
                    await provider.embed([" "])
                self.assertEqual(error.exception.status_code, 422)

    async def test_provider_selection_and_cleanup(self):
        for provider in ("openai", "azure_openai"):
            client = AsyncMock()
            client.__aenter__.return_value = client
            with (
                patch("app.container.AsyncOpenAI", return_value=client) as public,
                patch("app.container.AsyncAzureOpenAI", return_value=client) as azure,
            ):
                configured = settings(
                    embedding_provider=provider,
                    openai_api_key="public-key",
                    openai_embedding_model="public-model",
                    azure_openai_api_key="azure-key",
                    azure_openai_endpoint="https://example.openai.azure.com",
                    azure_openai_embedding_deployment="azure-model",
                )
                async with build_services(configured) as services:
                    self.assertEqual(services.embeddings.provider, provider)
                    selected, other = (public, azure) if provider == "openai" else (azure, public)
                    selected.assert_called_once()
                    other.assert_not_called()
                    self.assertIs(services.retrieval.embeddings, services.embeddings)
                    self.assertIs(services.indexing._embeddings, services.embeddings)
                client.__aexit__.assert_awaited_once()

    async def test_unconfigured_provider_does_not_block_bm25(self):
        for provider in ("openai", "azure_openai"):
            async with build_services(settings(embedding_provider=provider)) as services:
                with self.assertRaises(ApplicationError) as error:
                    await services.embeddings.embed(["text"])
                self.assertEqual(error.exception.status_code, 503)
                services.retrieval.repository = AsyncMock()
                services.retrieval.repository.search.return_value = {"azure": {"value": []}}
                await services.retrieval.text_search(
                    "demo", TextSearchRequest(query="a", mode="bm25")
                )

    async def test_incomplete_public_configuration(self):
        for options in ({"openai_api_key": "key"}, {"openai_embedding_model": "model"}):
            async with build_services(settings(embedding_provider="openai", **options)) as services:
                with self.assertRaises(ApplicationError) as error:
                    await services.embeddings.embed(["text"])
                self.assertEqual(error.exception.status_code, 503)

    async def test_upload_uses_selected_embeddings(self):
        for adapter in (OpenAIEmbeddingProvider, AzureEmbeddingProvider):
            create = AsyncMock(
                return_value=Mock(
                    model_dump=Mock(return_value={"data": [{"index": 0, "embedding": [0.1, 0.2]}]})
                )
            )
            provider = adapter(SimpleNamespace(embeddings=SimpleNamespace(create=create)), "model")
            repository = AsyncMock()
            repository.index_documents.return_value = {
                "azure": {"value": []},
                "all_succeeded": True,
            }
            service = compose_services(repository, provider)
            request = EmbedChunksRequest(
                documents=[
                    {
                        "chunk_id": "a",
                        "document_id": "doc",
                        "title": "Title",
                        "content": "Text",
                        "domain": "test",
                        "source": "test",
                        "document_type": "text",
                        "classification": "public",
                        "source_uri": "https://example.com",
                        "is_public": True,
                        "allowed_group_ids": [],
                    }
                ]
            )
            await service.indexing.embed_chunks("demo", request)
            actions = repository.index_documents.call_args.args[1]
            self.assertEqual(actions[0]["content_vector"], [0.1, 0.2])
            self.assertEqual(create.call_args.kwargs["input"], ["Title\nText"])


class EmbeddingErrorTests(unittest.TestCase):
    def test_error_identifies_provider_without_exposing_upstream_body(self):
        for provider in ("openai", "azure_openai"):
            embeddings = AsyncMock()
            embeddings.embed.side_effect = AuthenticationError(
                "secret",
                response=httpx.Response(401, request=httpx.Request("POST", "https://example.com")),
                body={"api_key": "secret"},
            )
            with TestClient(
                create_app(
                    settings(embedding_provider=provider, log_level="ERROR"),
                    compose_services(AsyncMock(), embeddings),
                )
            ) as client:
                response = client.post("/embeddings", json={"input": ["hello"]})
            self.assertEqual(response.status_code, 502)
            self.assertEqual(response.json()["detail"]["service"], provider)
            self.assertNotIn("secret", response.text)
