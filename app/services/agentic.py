"""Agentic workflows delegated to Azure; no local embedding or answer generation."""

from typing import Any

from app.core.errors import ApplicationError
from app.ports import AgenticRepository
from app.schemas.agentic import (
    AgenticRetrievalRequest,
    KnowledgeBaseRequest,
    KnowledgeSourceRequest,
)


class AgenticService:
    def __init__(self, repository: AgenticRepository | None) -> None:
        self._repository = repository

    @property
    def repository(self) -> AgenticRepository:
        if self._repository is None:
            raise ApplicationError("Agentic retrieval is not configured", 503)
        return self._repository

    async def create_source(self, name: str, request: KnowledgeSourceRequest) -> dict[str, Any]:
        return await self.repository.create_knowledge_source(name, request)

    async def get_source(self, name: str) -> dict[str, Any]:
        return await self.repository.get_knowledge_source(name)

    async def create_base(self, name: str, request: KnowledgeBaseRequest) -> dict[str, Any]:
        return await self.repository.create_knowledge_base(name, request)

    async def get_base(self, name: str) -> dict[str, Any]:
        return await self.repository.get_knowledge_base(name)

    async def retrieve(
        self, name: str, request: AgenticRetrievalRequest, source_token: str | None = None
    ) -> dict[str, Any]:
        return await self.repository.retrieve(name, request, source_token)
