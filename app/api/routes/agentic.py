"""Knowledge source setup, knowledge base setup, and stateless context retrieval."""

from typing import Annotated

from fastapi import APIRouter, Body

from app.api.dependencies import IndexName, ServiceDependency, SourceToken
from app.schemas.agentic import (
    AgenticRetrievalRequest,
    KnowledgeBaseRequest,
    KnowledgeSourceRequest,
)
from app.schemas.search import AzureResponse

router = APIRouter(tags=["Agentic retrieval"])


@router.post("/knowledge-sources/{source_name}", status_code=201, response_model=AzureResponse)
async def create_source(
    source_name: IndexName,
    data: Annotated[
        KnowledgeSourceRequest,
        Body(
            openapi_examples={
                "baseline": {
                    "value": {
                        "index_name": "readme-baseline-20261019",
                        "semantic_configuration_name": "semantic-config",
                        "source_data_fields": [
                            "chunk_id",
                            "document_id",
                            "title",
                            "content",
                            "source_uri",
                        ],
                    }
                }
            }
        ),
    ],
    services: ServiceDependency,
) -> dict:
    """Create an index-backed knowledge source without overwriting an existing one."""
    return await services.agentic.create_source(source_name, data)


@router.get("/knowledge-sources/{source_name}", response_model=AzureResponse)
async def get_source(source_name: IndexName, services: ServiceDependency) -> dict:
    return await services.agentic.get_source(source_name)


@router.post(
    "/knowledge-bases/{knowledge_base_name}", status_code=201, response_model=AzureResponse
)
async def create_base(
    knowledge_base_name: IndexName,
    data: Annotated[
        KnowledgeBaseRequest,
        Body(
            openapi_examples={
                "baseline": {
                    "value": {
                        "knowledge_source_names": ["readme-agentic-source"],
                        "description": "Context retrieval with Azure query planning",
                    }
                }
            }
        ),
    ],
    services: ServiceDependency,
) -> dict:
    """Create an extractive knowledge base using the server's chat deployment."""
    return await services.agentic.create_base(knowledge_base_name, data)


@router.get("/knowledge-bases/{knowledge_base_name}", response_model=AzureResponse)
async def get_base(knowledge_base_name: IndexName, services: ServiceDependency) -> dict:
    return await services.agentic.get_base(knowledge_base_name)


@router.post("/knowledge-bases/{knowledge_base_name}/retrieve", response_model=AzureResponse)
async def retrieve(
    knowledge_base_name: IndexName,
    data: Annotated[
        AgenticRetrievalRequest,
        Body(
            openapi_examples={
                "leave": {
                    "value": {
                        "messages": [{"role": "user", "content": "How do I request annual leave?"}],
                        "knowledge_source_params": [
                            {
                                "knowledge_source_name": "readme-agentic-source",
                                "filter_add_on": "is_public eq true",
                            }
                        ],
                    }
                }
            }
        ),
    ],
    services: ServiceDependency,
    source_token: SourceToken = None,
) -> dict:
    """Return context, references, and activity; callers supply conversation history."""
    return await services.agentic.retrieve(knowledge_base_name, data, source_token)
