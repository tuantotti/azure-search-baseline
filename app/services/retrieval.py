"""Retrieval orchestration depends on interfaces, not Azure or FastAPI."""

from typing import Any

from app.core.errors import ApplicationError
from app.ports import EmbeddingProvider, SearchRepository
from app.schemas.search import SearchMode, SearchOptions, TextSearchRequest, VectorQuery


class RetrievalService:
    def __init__(self, repository: SearchRepository, embeddings: EmbeddingProvider) -> None:
        self.repository = repository
        self.embeddings = embeddings

    @staticmethod
    def build_options(
        request: TextSearchRequest, mode: SearchMode, vector: list[float] | None
    ) -> SearchOptions:
        controlled = {
            "search_text",
            "vector_queries",
            "query_type",
            "semantic_query",
            "semantic_configuration_name",
            "semantic_error_mode",
            "query_answer",
            "query_caption",
            "order_by",
        }
        conflicts = controlled & request.parameters.model_dump(exclude_none=True).keys()
        if conflicts:
            raise ApplicationError(
                f"Use /search/sdk for controlled parameters: {sorted(conflicts)}", 422
            )
        options = request.parameters.model_copy(deep=True)
        if mode != "vector":
            options.search_text = request.query
            options.query_type = "simple"
        if mode != "bm25" and vector is not None:
            options.vector_filter_mode = options.vector_filter_mode or "preFilter"
            options.vector_queries = [
                VectorQuery(
                    vector=vector,
                    fields=request.vector_field,
                    k_nearest_neighbors=request.k,
                    exhaustive=request.exhaustive,
                )
            ]
        if mode == "hybrid_semantic":
            options.query_type = "semantic"
            options.semantic_configuration_name = request.semantic_configuration
            options.semantic_error_mode = "fail"
        return options

    @staticmethod
    def verify_semantic(mode: SearchMode, result: dict[str, Any]) -> None:
        if mode == "hybrid_semantic" and any(
            hit.get("@search.rerankerScore") is None for hit in result["azure"]["value"]
        ):
            raise ApplicationError("Azure response is missing semantic reranking scores", 502)

    async def text_search(
        self,
        index: str,
        request: TextSearchRequest,
        version: str | None = None,
        source_token: str | None = None,
    ) -> dict[str, Any]:
        self.build_options(request, request.mode, None)
        vector, embedding_ms = None, 0.0
        if request.mode != "bm25":
            response = await self.embeddings.embed([request.query])
            vector = response["azure"]["data"][0]["embedding"]
            embedding_ms = response["elapsed_ms"]
        result = await self.repository.search(
            index, self.build_options(request, request.mode, vector), version, source_token
        )
        self.verify_semantic(request.mode, result)
        return {"mode": request.mode, "embedding_ms": embedding_ms, **result}
