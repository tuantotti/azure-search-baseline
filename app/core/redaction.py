"""Resource-definition redaction shared by enrichment and agentic APIs."""

from typing import Any

SECRET_KEYS = {
    "apikey",
    "key",
    "accesstoken",
    "secret",
    "password",
    "connectionstring",
    "authorization",
    "httpheaders",
    "encryptionkey",
}


def redact_credentials(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: redact_credentials(item)
            for key, item in value.items()
            if key.replace("_", "").replace("-", "").lower() not in SECRET_KEYS
        }
    if isinstance(value, list):
        return [redact_credentials(item) for item in value]
    return value
