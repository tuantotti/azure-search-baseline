"""Async Azure SDK adapter for knowledge sources, bases, and retrieval."""

from time import perf_counter
from typing import Any

from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.indexes.models import (
    AzureOpenAIVectorizerParameters,
    KnowledgeBase,
    KnowledgeBaseAzureOpenAIModel,
    KnowledgeSourceReference,
    SearchIndexFieldReference,
    SearchIndexKnowledgeSource,
    SearchIndexKnowledgeSourceParameters,
)
from azure.search.documents.knowledgebases.aio import KnowledgeBaseRetrievalClient
from azure.search.documents.knowledgebases.models import (
    KnowledgeBaseMessage,
    KnowledgeBaseMessageTextContent,
    KnowledgeBaseRetrievalRequest,
    KnowledgeRetrievalLowReasoningEffort,
    SearchIndexKnowledgeSourceParams,
)

from app.core.errors import ApplicationError
from app.core.redaction import redact_credentials
from app.infrastructure.azure_search import AzureSearchRepository
from app.schemas.agentic import (
    AgenticRetrievalRequest,
    KnowledgeBaseRequest,
    KnowledgeSourceRequest,
)


class AzureAgenticRepository(AzureSearchRepository):
    def _management_client(self) -> SearchIndexClient:
        return self._index_client(self._settings.azure_search_agentic_api_version)

    def _retrieval_client(self, name: str) -> KnowledgeBaseRetrievalClient:
        return KnowledgeBaseRetrievalClient(
            endpoint=self._settings.azure_search_endpoint,
            knowledge_base_name=name,
            credential=self._credential,
            **self._options(self._settings.azure_search_agentic_api_version),
        )

    @staticmethod
    def _response(start: float, result: Any, *, redact: bool = True) -> dict[str, Any]:
        payload = result.as_dict()
        return {
            "elapsed_ms": (perf_counter() - start) * 1000,
            "azure": redact_credentials(payload) if redact else payload,
        }

    async def create_knowledge_source(
        self, name: str, request: KnowledgeSourceRequest
    ) -> dict[str, Any]:
        parameters = SearchIndexKnowledgeSourceParameters(
            search_index_name=request.index_name,
            semantic_configuration_name=request.semantic_configuration_name,
            search_fields=(
                [SearchIndexFieldReference(name=field) for field in request.search_fields]
                if request.search_fields
                else None
            ),
            source_data_fields=(
                [SearchIndexFieldReference(name=field) for field in request.source_data_fields]
                if request.source_data_fields
                else None
            ),
        )
        model = SearchIndexKnowledgeSource(
            name=name, description=request.description, search_index_parameters=parameters
        )
        start = perf_counter()
        async with self._management_client() as client:
            result = await client.create_knowledge_source(model)
        return self._response(start, result)

    async def get_knowledge_source(self, name: str) -> dict[str, Any]:
        start = perf_counter()
        async with self._management_client() as client:
            result = await client.get_knowledge_source(name)
        return self._response(start, result)

    async def create_knowledge_base(
        self, name: str, request: KnowledgeBaseRequest
    ) -> dict[str, Any]:
        settings = self._settings
        if not (
            settings.azure_openai_endpoint
            and settings.azure_openai_chat_deployment
            and settings.azure_openai_chat_model
        ):
            raise ApplicationError(
                "Knowledge base creation requires AZURE_OPENAI_ENDPOINT, "
                "AZURE_OPENAI_CHAT_DEPLOYMENT and AZURE_OPENAI_CHAT_MODEL",
                503,
            )
        connection = AzureOpenAIVectorizerParameters(
            resource_url=settings.azure_openai_endpoint,
            deployment_name=settings.azure_openai_chat_deployment,
            model_name=settings.azure_openai_chat_model,
            api_key=(
                settings.azure_openai_api_key.get_secret_value()
                if settings.azure_openai_api_key
                else None
            ),
        )
        model = KnowledgeBase(
            name=name,
            description=request.description,
            knowledge_sources=[
                KnowledgeSourceReference(name=item) for item in request.knowledge_source_names
            ],
            models=[KnowledgeBaseAzureOpenAIModel(azure_open_ai_parameters=connection)],
            retrieval_reasoning_effort=KnowledgeRetrievalLowReasoningEffort(),
            output_mode="extractiveData",
        )
        start = perf_counter()
        async with self._management_client() as client:
            result = await client.create_knowledge_base(model)
        return self._response(start, result)

    async def get_knowledge_base(self, name: str) -> dict[str, Any]:
        start = perf_counter()
        async with self._management_client() as client:
            result = await client.get_knowledge_base(name)
        return self._response(start, result)

    async def retrieve(
        self, name: str, request: AgenticRetrievalRequest, source_token: str | None = None
    ) -> dict[str, Any]:
        start = perf_counter()
        # Resolve every source so reference data is enabled even when no filter is supplied.
        async with self._management_client() as client:
            base = await client.get_knowledge_base(name)
        if not base.knowledge_sources:
            raise ApplicationError("Azure knowledge base contains no knowledge sources", 502)
        source_names = [item.name for item in base.knowledge_sources]
        filters = {
            item.knowledge_source_name: item.filter_add_on
            for item in request.knowledge_source_params or []
        }
        if set(filters) - set(source_names):
            raise ApplicationError(
                "Filters must reference knowledge sources in this knowledge base", 422
            )
        model = KnowledgeBaseRetrievalRequest(
            **({"max_output_documents": request.max_output_documents}
               if request.max_output_documents is not None else {}),
            messages=[
                KnowledgeBaseMessage(
                    role=message.role,
                    content=[KnowledgeBaseMessageTextContent(text=message.content)],
                )
                for message in request.messages
            ],
            retrieval_reasoning_effort=KnowledgeRetrievalLowReasoningEffort(),
            output_mode="extractiveData",
            include_activity=True,
            knowledge_source_params=[
                SearchIndexKnowledgeSourceParams(
                    knowledge_source_name=source,
                    filter_add_on=filters.get(source),
                    include_references=True,
                    include_reference_source_data=True,
                )
                for source in source_names
            ],
        )
        async with self._retrieval_client(name) as client:
            result = await client.retrieve(model, query_source_authorization=source_token)
        return self._response(start, result, redact=False)
