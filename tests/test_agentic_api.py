"""Agentic HTTP contracts and upstream error mappings."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from azure.core.exceptions import HttpResponseError, ServiceRequestError
from fastapi.testclient import TestClient

from app.container import compose_services
from app.core.config import Settings
from app.core.errors import ApplicationError
from app.main import create_app


class AgenticApiTests(unittest.TestCase):
    def setUp(self):
        self.repository = AsyncMock()
        for name in [
            "create_knowledge_source",
            "get_knowledge_source",
            "create_knowledge_base",
            "get_knowledge_base",
            "retrieve",
        ]:
            getattr(self.repository, name).return_value = {
                "elapsed_ms": 1,
                "azure": {"name": "demo"},
            }
        settings = Settings(
            _env_file=None,
            azure_search_endpoint="https://example.search.windows.net",
            log_level="ERROR",
        )
        self.client = TestClient(
            create_app(settings, compose_services(AsyncMock(), AsyncMock(), self.repository))
        )
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def test_all_routes_and_swagger_examples(self):
        cases = [
            ("POST", "/knowledge-sources/demo-source", {"index_name": "demo"}, 201),
            ("GET", "/knowledge-sources/demo-source", None, 200),
            (
                "POST",
                "/knowledge-bases/demo-base",
                {"knowledge_source_names": ["demo-source"]},
                201,
            ),
            ("GET", "/knowledge-bases/demo-base", None, 200),
            (
                "POST",
                "/knowledge-bases/demo-base/retrieve",
                {"messages": [{"role": "user", "content": "Question"}]},
                200,
            ),
        ]
        for method, path, body, status in cases:
            with self.subTest(path=path, method=method):
                response = self.client.request(method, path, json=body)
                self.assertEqual(response.status_code, status, response.text)
        spec = self.client.get("/openapi.json").json()
        for path in [
            "/knowledge-sources/{source_name}",
            "/knowledge-bases/{knowledge_base_name}",
            "/knowledge-bases/{knowledge_base_name}/retrieve",
        ]:
            self.assertIn(
                "examples",
                spec["paths"][path]["post"]["requestBody"]["content"]["application/json"],
            )

    def test_invalid_requests_never_reach_repository(self):
        cases = [
            ("/knowledge-sources/demo-source", {"index_name": "demo", "api_key": "secret"}),
            ("/knowledge-sources/demo-source", {"index_name": "bad/name"}),
            ("/knowledge-sources/demo-source", {"index_name": "demo", "search_fields": []}),
            ("/knowledge-bases/demo-base", {"knowledge_source_names": []}),
            (
                "/knowledge-bases/demo-base",
                {"knowledge_source_names": ["demo-source", "demo-source"]},
            ),
            (
                "/knowledge-bases/demo-base",
                {"knowledge_source_names": ["demo-source"], "models": []},
            ),
            ("/knowledge-bases/demo-base/retrieve", {"messages": []}),
            (
                "/knowledge-bases/demo-base/retrieve",
                {"messages": [{"role": "user", "content": "  "}]},
            ),
            (
                "/knowledge-bases/demo-base/retrieve",
                {"messages": [{"role": "assistant", "content": "Answer"}]},
            ),
            (
                "/knowledge-bases/demo-base/retrieve",
                {"messages": [{"role": "system", "content": "Instructions"}]},
            ),
        ]
        for path, body in cases:
            with self.subTest(body=body):
                response = self.client.post(path, json=body)
                self.assertEqual(response.status_code, 422, response.text)
                self.assertNotIn("secret", response.text)
        self.assertEqual(self.repository.mock_calls, [])

    def test_source_authorization_and_filters_are_forwarded(self):
        body = {
            "messages": [{"role": "user", "content": "Question"}],
            "knowledge_source_params": [
                {"knowledge_source_name": "demo-source", "filter_add_on": "is_public eq true"}
            ],
        }
        response = self.client.post(
            "/knowledge-bases/demo-base/retrieve",
            json=body,
            headers={"X-MS-Query-Source-Authorization": "source-token"},
        )
        self.assertEqual(response.status_code, 200)
        name, request, token = self.repository.retrieve.call_args.args
        self.assertEqual(name, "demo-base")
        self.assertEqual(token, "source-token")
        self.assertEqual(request.knowledge_source_params[0].filter_add_on, "is_public eq true")

    def test_upstream_errors_and_missing_configuration(self):
        for status, expected in [
            (400, 400),
            (401, 502),
            (403, 502),
            (404, 404),
            (409, 409),
            (429, 429),
            (500, 502),
        ]:
            error = HttpResponseError(message="Upstream failed")
            error.status_code = status
            self.repository.get_knowledge_base.side_effect = error
            response = self.client.get("/knowledge-bases/demo-base")
            self.assertEqual(response.status_code, expected)
            self.assertEqual(response.json()["detail"]["upstream_status"], status)
        self.repository.create_knowledge_base.side_effect = ApplicationError(
            "Chat configuration missing", 503
        )
        response = self.client.post(
            "/knowledge-bases/demo-base", json={"knowledge_source_names": ["demo-source"]}
        )
        self.assertEqual(response.status_code, 503)
        self.repository.retrieve.side_effect = ServiceRequestError("Network unavailable")
        response = self.client.post(
            "/knowledge-bases/demo-base/retrieve",
            json={"messages": [{"role": "user", "content": "Question"}]},
        )
        self.assertEqual(response.status_code, 502)

    def test_azure_duplicate_resource_400_is_exposed_as_conflict(self):
        error = HttpResponseError(message="Resource already exists")
        error.status_code = 400
        error.error = SimpleNamespace(code="ResourceNameAlreadyInUse")
        self.repository.create_knowledge_source.side_effect = error
        response = self.client.post("/knowledge-sources/demo-source", json={"index_name": "demo"})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"]["upstream_status"], 400)
