"""Index creation and chunk embedding without HTTP-framework dependencies."""

from typing import Any

from app.core.errors import ApplicationError
from app.domain.index_schema import schema
from app.ports import EmbeddingProvider, SearchRepository
from app.schemas.indexing import BaselineIndexRequest, EmbedChunksRequest


class IndexingService:
    def __init__(self, repository: SearchRepository, embeddings: EmbeddingProvider) -> None:
        self.repository = repository
        self._embeddings = embeddings

    async def create(
        self, index: str, definition: dict[str, Any], version: str | None = None
    ) -> dict[str, Any]:
        if definition.get("name", index) != index:
            raise ApplicationError("Schema name must match index_name", 422)
        return await self.repository.create_index({**definition, "name": index}, version)

    async def create_baseline(
        self, index: str, request: BaselineIndexRequest, version: str | None = None
    ) -> dict[str, Any]:
        return await self.create(
            index, schema(index, request.dimensions, request.description), version
        )

    async def embed_chunks(
        self, index: str, request: EmbedChunksRequest, version: str | None = None
    ) -> dict[str, Any]:
        ids = [document.chunk_id for document in request.documents]
        if len(ids) != len(set(ids)):
            raise ApplicationError("Duplicate chunk IDs in batch", 422)
        embedded = await self._embeddings.embed(
            [f"{document.title}\n{document.content}" for document in request.documents]
        )
        vectors = embedded["azure"]["data"]
        if len(vectors) != len(request.documents):
            raise ApplicationError("Embedding response count does not match input", 502)
        actions = [
            {
                **document.model_dump(),
                "@search.action": "upload",
                "content_vector": vector["embedding"],
            }
            for document, vector in zip(request.documents, vectors, strict=True)
        ]
        result = await self.repository.index_documents(index, actions, version)
        return {"embedding_ms": embedded["elapsed_ms"], **result}
