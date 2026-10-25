"""Environment-backed configuration; secrets never enter request models."""

from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    azure_search_endpoint: str
    azure_search_api_key: SecretStr | None = None
    azure_search_api_version: str = "2025-09-01"
    azure_search_agentic_api_version: str = "2026-08-01-preview"
    embedding_provider: Literal["azure_openai", "openai"] = "azure_openai"
    openai_api_key: SecretStr | None = None
    openai_embedding_model: str | None = None
    azure_search_skill_api_version: str = "2026-04-01"
    azure_ai_services_endpoint: str | None = None
    azure_ai_services_key: SecretStr | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: SecretStr | None = None
    azure_openai_embedding_deployment: str | None = None
    azure_openai_chat_deployment: str | None = None
    azure_openai_chat_model: str | None = None
    openai_api_version: str = "2024-10-21"
    request_timeout_seconds: float = Field(default=60, gt=0, le=300)
    azure_retry_total: int = Field(default=2, ge=0, le=5)
    root_path: str = ""
    log_level: str = "INFO"

    @field_validator("azure_search_endpoint", "azure_openai_endpoint", "azure_ai_services_endpoint")
    @classmethod
    def validate_endpoint(cls, value: str | None) -> str | None:
        if value is None:
            return None
        from urllib.parse import urlsplit

        parts = urlsplit(value)
        if (
            parts.scheme != "https"
            or not parts.hostname
            or parts.username
            or parts.query
            or parts.fragment
        ):
            raise ValueError("Use an HTTPS Azure resource endpoint")
        return value.rstrip("/")
