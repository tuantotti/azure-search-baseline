"""OpenAI-compatible SDK embedding adapters."""

from time import perf_counter
from typing import Any

from openai import AsyncOpenAI

from app.core.errors import ApplicationError


class OpenAIEmbeddingProvider:
    def __init__(
        self, client: AsyncOpenAI | None, deployment: str | None, provider: str = "openai"
    ) -> None:
        self._client = client
        self._deployment = deployment
        self.provider = provider

    async def embed(self, texts: list[str]) -> dict[str, Any]:
        if not self._client or not self._deployment:
            raise ApplicationError(f"{self.provider} embedding settings are not configured", 503)
        if any(not text.strip() for text in texts):
            raise ApplicationError("Embedding input must contain nonempty text", 422)
        start = perf_counter()
        result = await self._client.embeddings.create(
            model=self._deployment, input=texts, encoding_format="float"
        )
        payload = result.model_dump()
        payload["data"].sort(key=lambda item: item["index"])
        return {"elapsed_ms": (perf_counter() - start) * 1000, "azure": payload}


class AzureEmbeddingProvider(OpenAIEmbeddingProvider):
    def __init__(self, client: AsyncOpenAI | None, deployment: str | None) -> None:
        super().__init__(client, deployment, "azure_openai")
