"""Application factory: lifecycle, middleware and router registration only."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from loguru import logger

from app.api.error_handlers import install_error_handlers
from app.api.routes import agentic, embeddings, indexers, indexes, search, skillsets
from app.container import Services, build_services
from app.core.config import Settings
from app.core.logging import configure_logging, log_error


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    """Optional dependencies support isolated tests; normal startup uses Azure SDKs."""
    settings = settings or Settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        if services is not None:
            application.state.services = services
            yield
        else:
            async with build_services(settings) as configured:
                application.state.services = configured
                yield

    application = FastAPI(
        title="Azure AI Search Experiment API",
        version="3.0.0",
        root_path=settings.root_path,
        lifespan=lifespan,
    )
    application.state.embedding_provider = settings.embedding_provider
    install_error_handlers(application)

    @application.middleware("http")
    async def request_logging(request: Request, call_next) -> Response:
        request_id = str(uuid4())
        start = perf_counter()
        with logger.contextualize(request_id=request_id):
            try:
                response = await call_next(request)
            except Exception as error:
                log_error(error, service="application", method=request.method,
                          path=request.url.path)
                raise
        logger.bind(request_id=request_id).info(
            "{} {} status={} elapsed_ms={:.2f}",
            request.method,
            request.url.path,
            response.status_code,
            (perf_counter() - start) * 1000,
        )
        response.headers["X-Request-ID"] = request_id
        return response

    @application.get("/health", tags=["Health"])
    async def health() -> dict[str, str]:
        """Liveness only. Read an index schema to check Azure connectivity."""
        return {"status": "ok"}

    application.include_router(skillsets.router)
    application.include_router(indexers.router)
    application.include_router(indexes.router)
    application.include_router(search.router)
    application.include_router(embeddings.router)
    application.include_router(agentic.router)
    return application
