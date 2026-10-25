"""Explicit configuration and processing controls for existing indexers."""

from typing import Annotated

from fastapi import APIRouter, Body

from app.api.dependencies import IndexName, ServiceDependency
from app.schemas.enrichment import AttachSkillsetRequest, SkillsRequest, SummarySkillRequest
from app.schemas.search import AzureResponse

router = APIRouter(prefix="/indexers/{indexer_name}", tags=["Indexers"])


@router.post("/summary-skill", response_model=AzureResponse)
async def summary_skill(
    indexer_name: IndexName, data: SummarySkillRequest, services: ServiceDependency
) -> dict:
    return await services.enrichment.configure_summary(indexer_name, data)


@router.post("/skills", response_model=AzureResponse)
async def configure_skills(
    indexer_name: IndexName,
    data: Annotated[
        SkillsRequest,
        Body(
            openapi_examples={
                "classification": {
                    "summary": "LLM classification",
                    "value": {
                        "skillset_name": "uat-classification-v1",
                        "skills": [
                            {
                                "kind": "classification",
                                "categories": ["Finance", "HR", "Technology", "Other"],
                            }
                        ],
                    },
                },
                "entities": {
                    "summary": "Entity Recognition V3",
                    "value": {
                        "skillset_name": "uat-entities-v1",
                        "skills": [
                            {
                                "kind": "entities",
                                "categories": ["Person", "Organization", "Location"],
                                "minimum_precision": 0.5,
                            }
                        ],
                    },
                },
                "combined": {
                    "summary": "All six text presets",
                    "value": {
                        "skillset_name": "uat-combined-v1",
                        "skills": [
                            {
                                "kind": "classification",
                                "categories": ["Finance", "HR", "Technology", "Other"],
                            },
                            {"kind": "entities"},
                            {"kind": "key_phrases"},
                            {"kind": "language_detection"},
                            {"kind": "sentiment"},
                            {"kind": "summary"},
                        ],
                    },
                },
            }
        ),
    ],
    services: ServiceDependency,
) -> dict:
    """Clone the current skillset and replace selected app-managed text presets."""
    return await services.enrichment.configure_skills(indexer_name, data)


@router.post("/skillset", response_model=AzureResponse)
async def attach_skillset(
    indexer_name: IndexName, data: AttachSkillsetRequest, services: ServiceDependency
) -> dict:
    """Attach an existing skillset with explicit additional fields and mappings."""
    return await services.enrichment.attach_skillset(indexer_name, data)


@router.post("/run", status_code=202, response_model=AzureResponse)
async def run(indexer_name: IndexName, services: ServiceDependency) -> dict:
    return await services.enrichment.operation(indexer_name, "run")


@router.post("/reset", response_model=AzureResponse)
async def reset(indexer_name: IndexName, services: ServiceDependency) -> dict:
    return await services.enrichment.operation(indexer_name, "reset")


@router.get("/status", response_model=AzureResponse)
async def status(indexer_name: IndexName, services: ServiceDependency) -> dict:
    return await services.enrichment.operation(indexer_name, "status")
