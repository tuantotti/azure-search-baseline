"""Indexer management through the Azure SDK, isolated from search API versions."""

import json
from typing import Any
from urllib.parse import quote

from azure.core import MatchConditions
from azure.core.exceptions import HttpResponseError
from azure.core.rest import HttpRequest
from azure.search.documents.indexes.aio import SearchIndexerClient
from azure.search.documents.indexes.models import (
    ChatCompletionSkill,
    SearchIndex,
    SearchIndexer,
)

from app.core.errors import ApplicationError
from app.infrastructure.azure_search import AzureSearchRepository


class AzureIndexerRepository(AzureSearchRepository):
    def _management_client(self) -> SearchIndexerClient:
        return SearchIndexerClient(
            self._settings.azure_search_endpoint,
            self._credential,
            **self._options(self._settings.azure_search_skill_api_version),
        )

    def summary_skill(self, prompt: str) -> dict[str, Any]:
        settings = self._settings
        if not (settings.azure_openai_endpoint and settings.azure_openai_chat_deployment):
            raise ApplicationError(
                "LLM skills require AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_CHAT_DEPLOYMENT", 503
            )
        return ChatCompletionSkill(
            {
                "@odata.type": "#Microsoft.Skills.Custom.ChatCompletionSkill",
                "name": "document-summary",
                "context": "/document",
                "uri": f"{settings.azure_openai_endpoint}/openai/deployments/"
                f"{quote(settings.azure_openai_chat_deployment, safe='')}/chat/completions"
                f"?api-version={settings.openai_api_version}",
                "inputs": [
                    {"name": "systemMessage", "source": "='You summarize documents faithfully.'"},
                    {"name": "userMessage", "source": "=" + json.dumps(prompt)},
                    {"name": "text", "source": "/document/content"},
                ],
                "outputs": [{"name": "response", "targetName": "summary"}],
                "responseFormat": {"type": "text"},
                **(
                    {"apiKey": settings.azure_openai_api_key.get_secret_value()}
                    if settings.azure_openai_api_key
                    else {}
                ),
            }
        ).as_dict()

    async def get_indexer(self, name: str) -> dict[str, Any]:
        async with self._management_client() as client:
            return (await client.get_indexer(name)).as_dict()

    def billing_connection(self) -> dict[str, Any] | None:
        settings = self._settings
        if settings.azure_ai_services_key and not settings.azure_ai_services_endpoint:
            raise ApplicationError("AZURE_AI_SERVICES_KEY requires AZURE_AI_SERVICES_ENDPOINT", 503)
        if not settings.azure_ai_services_endpoint:
            return None
        connection = {"subdomainUrl": settings.azure_ai_services_endpoint}
        if settings.azure_ai_services_key:
            connection.update(
                {
                    "@odata.type": "#Microsoft.Azure.Search.AIServicesByKey",
                    "key": settings.azure_ai_services_key.get_secret_value(),
                }
            )
        else:
            connection.update(
                {"@odata.type": "#Microsoft.Azure.Search.AIServicesByIdentity", "identity": None}
            )
        return connection

    async def _skillset_request(
        self, method: str, path: str, definition: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        request = HttpRequest(
            method,
            f"{self._settings.azure_search_endpoint}/{path}",
            params={"api-version": self._settings.azure_search_skill_api_version},
            **({"json": definition} if definition is not None else {}),
        )
        async with self._management_client() as client:
            response = await client.send_request(request)
            try:
                if response.status_code >= 400:
                    status = response.status_code
                    if status in (401, 403) or status >= 500:
                        status = 502
                    # Never echo Azure's body: it can contain submitted credentials.
                    raise ApplicationError(
                        f"Azure skillset request failed (HTTP {response.status_code})", status
                    )
                return response.json()
            finally:
                await response.close()

    async def get_skillset(self, name: str) -> dict[str, Any]:
        return await self._skillset_request("GET", f"skillsets/{quote(name, safe='')}")

    async def read_index(self, name: str) -> dict[str, Any]:
        async with self._index_client(self._settings.azure_search_skill_api_version) as client:
            return (await client.get_index(name)).as_dict()

    async def create_skillset(self, definition: dict[str, Any]) -> None:
        # POST is create-only; raw JSON preserves future/unknown skill parameters.
        await self._skillset_request("POST", "skillsets", definition)

    async def update_index(self, definition: dict[str, Any]) -> None:
        async with self._index_client(self._settings.azure_search_skill_api_version) as client:
            await client.create_or_update_index(
                SearchIndex(definition), match_condition=MatchConditions.IfNotModified
            )

    async def update_indexer(self, definition: dict[str, Any]) -> None:
        # 12.1.0b2 forwards None-valued preview flags in create_or_update_indexer,
        # causing its version validator to reject stable 2026-04-01. Use the
        # public SDK pipeline with the typed model and conditional header.
        model = SearchIndexer(definition)
        request = HttpRequest(
            "PUT",
            f"{self._settings.azure_search_endpoint}/indexers/{quote(model.name, safe='')}",
            params={"api-version": self._settings.azure_search_skill_api_version},
            headers={"If-Match": model.e_tag},
            json=model.as_dict(),
        )
        async with self._management_client() as client:
            response = await client.send_request(request)
            try:
                if response.status_code >= 400:
                    raise HttpResponseError(response=response)
            finally:
                await response.close()

    async def indexer_operation(self, name: str, operation: str) -> dict[str, Any]:
        async with self._management_client() as client:
            if operation == "status":
                return (await client.get_indexer_status(name)).as_dict()
            if operation == "run":
                await client.run_indexer(name)
            elif operation == "reset":
                await client.reset_indexer(name)
            else:
                raise ValueError("Unknown indexer operation")
        return {"name": name, "operation": operation, "accepted": True}
