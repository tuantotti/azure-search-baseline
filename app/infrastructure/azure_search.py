"""Azure Search SDK adapter. Shared HTTP session, operation-scoped SDK clients."""

from time import perf_counter
from typing import Any

import aiohttp
from azure.core.credentials import AzureKeyCredential
from azure.core.credentials_async import AsyncTokenCredential
from azure.core.exceptions import HttpResponseError
from azure.core.pipeline.transport import AioHttpTransport
from azure.core.rest import HttpRequest
from azure.search.documents import IndexDocumentsBatch
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex
from azure.search.documents.models import VectorizableTextQuery, VectorizedQuery

from app.core.config import Settings
from app.core.errors import ApplicationError
from app.schemas.search import SearchOptions


def serialize_model(value: Any) -> Any:
    """Serialize SDK models to Azure wire field names."""
    if hasattr(value, "as_dict"):
        return value.as_dict()
    return value


class AzureSearchRepository:
    def __init__(
        self,
        settings: Settings,
        credential: AzureKeyCredential | AsyncTokenCredential,
        session: aiohttp.ClientSession,
    ) -> None:
        self._settings = settings
        self._credential = credential
        self._session = session

    def _options(self, version: str | None) -> dict[str, Any]:
        return {
            "api_version": version or self._settings.azure_search_api_version,
            "transport": AioHttpTransport(session=self._session, session_owner=False),
            "retry_total": self._settings.azure_retry_total,
            "connection_timeout": self._settings.request_timeout_seconds,
            "read_timeout": self._settings.request_timeout_seconds,
            "logging_enable": False,
        }

    def _search_client(self, index: str, version: str | None) -> SearchClient:
        return SearchClient(
            self._settings.azure_search_endpoint, index, self._credential, **self._options(version)
        )

    def _index_client(self, version: str | None) -> SearchIndexClient:
        return SearchIndexClient(
            self._settings.azure_search_endpoint, self._credential, **self._options(version)
        )

    async def search(
        self,
        index: str,
        options: SearchOptions,
        version: str | None = None,
        source_token: str | None = None,
    ) -> dict[str, Any]:
        parameters = options.model_dump(exclude_none=True, exclude={"vector_queries"})
        if options.vector_queries is not None:
            parameters["vector_queries"] = []
            for query in options.vector_queries:
                values = query.model_dump(exclude_none=True)
                query_type = VectorizedQuery if query.vector is not None else VectorizableTextQuery
                parameters["vector_queries"].append(query_type(**values))
        if source_token:
            parameters["headers"] = {"x-ms-query-source-authorization": source_token}
        start = perf_counter()
        async with self._search_client(index, version) as client:
            results = await client.search(**parameters)
            documents = [document async for document in results]
            payload: dict[str, Any] = {"value": documents}
            count = await results.get_count()
            facets = await results.get_facets()
            answers = await results.get_answers()
            coverage = await results.get_coverage()
            if count is not None:
                payload["@odata.count"] = count
            if facets is not None:
                payload["@search.facets"] = facets
            if answers is not None:
                payload["@search.answers"] = [serialize_model(answer) for answer in answers]
            if coverage is not None:
                payload["@search.coverage"] = coverage
        return {"elapsed_ms": (perf_counter() - start) * 1000, "azure": payload}

    async def raw_search(
        self,
        index: str,
        body: dict[str, Any],
        version: str | None = None,
        source_token: str | None = None,
    ) -> dict[str, Any]:
        """Advanced REST parameters still travel through the public SDK pipeline."""
        url = f"{self._settings.azure_search_endpoint}/indexes/{index}/docs/search"
        headers = {"x-ms-query-source-authorization": source_token} if source_token else {}
        request = HttpRequest(
            "POST",
            url,
            params={"api-version": version or self._settings.azure_search_api_version},
            json=body,
            headers=headers,
        )
        start = perf_counter()
        async with self._search_client(index, version) as client:
            response = await client.send_request(request)
            try:
                if response.status_code >= 400:
                    raise HttpResponseError(response=response)
                payload = response.json()
            finally:
                await response.close()
        return {"elapsed_ms": (perf_counter() - start) * 1000, "azure": payload}

    async def get_index(self, index: str, version: str | None = None) -> dict[str, Any]:
        start = perf_counter()
        async with self._index_client(version) as client:
            result = await client.get_index(index)
        return {"elapsed_ms": (perf_counter() - start) * 1000, "azure": result.as_dict()}

    async def create_index(
        self, definition: dict[str, Any], version: str | None = None
    ) -> dict[str, Any]:
        try:
            model = SearchIndex(definition)
        except (ValueError, TypeError, KeyError) as error:
            raise ApplicationError("Invalid SDK index definition", 422) from error
        start = perf_counter()
        async with self._index_client(version) as client:
            result = await client.create_index(model)
        return {"elapsed_ms": (perf_counter() - start) * 1000, "azure": result.as_dict()}

    async def index_documents(
        self, index: str, actions: list[dict[str, Any]], version: str | None = None
    ) -> dict[str, Any]:
        batch = IndexDocumentsBatch()
        methods = {
            "upload": batch.add_upload_actions,
            "merge": batch.add_merge_actions,
            "mergeOrUpload": batch.add_merge_or_upload_actions,
            "delete": batch.add_delete_actions,
        }
        for action in actions:
            document = dict(action)
            kind = document.pop("@search.action", "upload")
            if kind not in methods:
                raise ApplicationError(f"Unsupported indexing action: {kind}", 422)
            methods[kind](document)
        start = perf_counter()
        async with self._search_client(index, version) as client:
            results = await client.index_documents(batch)
        values = [
            {
                "key": result.key,
                "status": result.succeeded,
                "statusCode": result.status_code,
                "errorMessage": result.error_message,
            }
            for result in results
        ]
        return {
            "elapsed_ms": (perf_counter() - start) * 1000,
            "azure": {"value": values},
            "all_succeeded": all(result.succeeded for result in results),
        }
