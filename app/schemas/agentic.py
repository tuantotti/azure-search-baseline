"""Public, credential-free contracts for Azure agentic retrieval."""

from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from .search import StrictModel

ResourceName = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9-]{1,126}[a-z0-9]$")]
NonemptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class KnowledgeSourceRequest(StrictModel):
    index_name: ResourceName
    description: NonemptyText | None = None
    semantic_configuration_name: NonemptyText | None = None
    search_fields: list[NonemptyText] | None = Field(default=None, min_length=1)
    source_data_fields: list[NonemptyText] | None = Field(default=None, min_length=1)


class KnowledgeBaseRequest(StrictModel):
    knowledge_source_names: list[ResourceName] = Field(min_length=1)
    description: NonemptyText | None = None

    @model_validator(mode="after")
    def unique_sources(self) -> "KnowledgeBaseRequest":
        if len(set(self.knowledge_source_names)) != len(self.knowledge_source_names):
            raise ValueError("knowledge_source_names must be unique")
        return self


class RetrievalMessage(StrictModel):
    role: Literal["user", "assistant"]
    content: NonemptyText


class KnowledgeSourceFilter(StrictModel):
    knowledge_source_name: ResourceName
    filter_add_on: NonemptyText | None = None


class AgenticRetrievalRequest(StrictModel):
    max_output_documents: int | None = Field(default=None, ge=1, le=1000)
    messages: list[RetrievalMessage] = Field(min_length=1)
    knowledge_source_params: list[KnowledgeSourceFilter] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_conversation(self) -> "AgenticRetrievalRequest":
        if not any(message.role == "user" for message in self.messages):
            raise ValueError("messages must contain a user message")
        names = [item.knowledge_source_name for item in self.knowledge_source_params or []]
        if len(names) != len(set(names)):
            raise ValueError("knowledge_source_params must contain unique sources")
        return self
