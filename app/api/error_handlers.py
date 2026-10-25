"""Central exception-to-HTTP mapping; never echo tokens in validation errors."""

from azure.core.exceptions import HttpResponseError, ServiceRequestError, ServiceResponseError
from azure.identity import CredentialUnavailableError
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from openai import APIConnectionError, APIStatusError, APITimeoutError

from app.core.errors import ApplicationError
from app.core.logging import log_error


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApplicationError)
    async def application_error(request: Request, error: ApplicationError) -> JSONResponse:
        log_error(error, service="application", method=request.method, path=request.url.path)
        return JSONResponse(status_code=error.status_code, content={"detail": error.message})

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {key: value for key, value in item.items() if key in ("type", "loc", "msg")}
                    for item in error.errors()
                ]
            },
        )

    @app.exception_handler(HttpResponseError)
    async def search_error(request: Request, error: HttpResponseError) -> JSONResponse:
        log_error(error, service="azure_search", method=request.method, path=request.url.path)
        upstream = error.status_code or 502
        # Azure auth failures concern the server's credentials, not caller login.
        status = 502 if upstream in (401, 403) or upstream >= 500 else upstream
        if upstream == 400 and getattr(error.error, "code", None) == "ResourceNameAlreadyInUse":
            status = 409
        headers = {}
        if upstream == 429 and error.response:
            headers["Retry-After"] = error.response.headers.get("Retry-After", "1")
        return JSONResponse(
            status_code=status,
            headers=headers,
            content={
                "detail": {
                    "service": "azure_search",
                    "upstream_status": upstream,
                    "message": error.message,
                }
            },
        )

    @app.exception_handler(APIStatusError)
    async def embedding_error(request: Request, error: APIStatusError) -> JSONResponse:
        log_error(error, service="embedding", method=request.method, path=request.url.path)
        status = (
            502
            if error.status_code in (401, 403) or error.status_code >= 500
            else error.status_code
        )
        return JSONResponse(
            status_code=status,
            content={
                "detail": {
                    "service": request.app.state.embedding_provider,
                    "upstream_status": error.status_code,
                    "message": "Embedding request failed; verify model, input and server credentials.",
                }
            },
        )

    @app.exception_handler(APITimeoutError)
    async def timeout_error(request: Request, error: APITimeoutError) -> JSONResponse:
        log_error(error, service="embedding", method=request.method, path=request.url.path)
        return JSONResponse(
            status_code=504,
            content={
                "detail": f"{request.app.state.embedding_provider} embedding request timed out"
            },
        )

    async def connectivity_error(request: Request, error: Exception) -> JSONResponse:
        log_error(error, service="upstream", method=request.method, path=request.url.path)
        return JSONResponse(
            status_code=502,
            content={"detail": "Upstream connection or credential resolution failed"},
        )

    for error_type in (
        ServiceRequestError,
        ServiceResponseError,
        APIConnectionError,
        CredentialUnavailableError,
    ):
        app.add_exception_handler(error_type, connectivity_error)
