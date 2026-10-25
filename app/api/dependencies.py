"""Typed FastAPI dependencies and reusable path/query contracts."""

from typing import Annotated

from fastapi import Depends, Header, Path, Query, Request

from app.container import Services

IndexName = Annotated[str, Path(pattern=r"^[a-z0-9][a-z0-9-]{1,126}[a-z0-9]$")]
ApiVersion = Annotated[str | None, Query(pattern=r"^\d{4}-\d{2}-\d{2}(-preview)?$")]
SourceToken = Annotated[
    str | None,
    Header(
        alias="X-MS-Query-Source-Authorization",
        description="Optional end-user token for configured Azure native document permissions. Not required for ordinary search.",
    ),
]


def get_services(request: Request) -> Services:
    return request.app.state.services


ServiceDependency = Annotated[Services, Depends(get_services)]
