"""Advanced create-only Azure skillset experiments."""

from typing import Annotated, Any

from fastapi import APIRouter, Body

from app.api.dependencies import IndexName, ServiceDependency
from app.schemas.search import AzureResponse

router = APIRouter(prefix="/skillsets/{skillset_name}", tags=["Skillsets"])


@router.post("", status_code=201, response_model=AzureResponse)
async def create_skillset(
    skillset_name: IndexName,
    services: ServiceDependency,
    definition: Annotated[
        dict[str, Any],
        Body(description="Azure wire JSON, including @odata.type. Create only; never overwrites."),
    ],
) -> dict:
    return await services.enrichment.create_skillset(skillset_name, definition)


@router.get("", response_model=AzureResponse)
async def get_skillset(skillset_name: IndexName, services: ServiceDependency) -> dict:
    return await services.enrichment.inspect_skillset(skillset_name)
