"""Run the full enrichment service through the real SDK and a fake transport."""

import json
import unittest
from copy import deepcopy
from unittest.mock import AsyncMock, patch

import aiohttp
from azure.core.credentials import AzureKeyCredential
from fastapi.testclient import TestClient
from test_sdk import FakeResponse, FakeTransport

from app.container import compose_services
from app.core.config import Settings
from app.core.errors import ApplicationError
from app.infrastructure.indexers import AzureIndexerRepository
from app.main import create_app
from app.schemas.enrichment import SummarySkillRequest
from app.services.enrichment import EnrichmentService


class IndexerTransport(FakeTransport):
    def __init__(self):
        super().__init__()
        self.indexer = {
            "name": "demo-indexer",
            "targetIndexName": "demo",
            "dataSourceName": "source",
            "skillsetName": "old-skills",
            "@odata.etag": '"indexer-v1"',
            "schedule": {"interval": "PT2H"},
            "parameters": {"batchSize": 10},
            "fieldMappings": [{"sourceFieldName": "body", "targetFieldName": "content"}],
            "outputFieldMappings": [
                {"sourceFieldName": "/document/title", "targetFieldName": "title"}
            ],
        }
        self.index = {
            "name": "demo",
            "@odata.etag": '"index-v1"',
            "fields": [
                {"name": "id", "type": "Edm.String", "key": True},
                {"name": "content", "type": "Edm.String", "searchable": True},
            ],
        }
        self.skillset = {
            "name": "old-skills",
            "@odata.etag": '"skills-v1"',
            "description": "Preserve me",
            "skills": [
                {
                    "@odata.type": "#Microsoft.Skills.Text.SplitSkill",
                    "name": "split",
                    "context": "/document",
                    "textSplitMode": "pages",
                    "maximumPageLength": 1000,
                    "inputs": [{"name": "text", "source": "/document/content"}],
                    "outputs": [{"name": "textItems", "targetName": "pages"}],
                }
            ],
        }
        self.exists = False
        self.fail = None

    async def send(self, request, **kwargs):
        self.requests.append(request)
        body = json.loads(request.body) if request.body else {}
        code = 200
        if request.method == "GET":
            if "/indexers(" in request.url:
                payload = (
                    {"status": "running", "lastResult": {"status": "success"}}
                    if "/search.status" in request.url
                    else self.indexer
                )
            elif "/indexes(" in request.url:
                payload = self.index
            elif "old-skills" in request.url:
                payload = self.skillset
            elif self.exists:
                payload = {"name": "summary-skills", "skills": []}
            else:
                code, payload = 404, {"error": {"code": "NotFound", "message": "not found"}}
        else:
            payload = body
            code = 201 if request.method == "POST" and "/skillsets" in request.url else 200
            if self.fail and self.fail[0] in request.url:
                code, payload = (
                    self.fail[1],
                    {"error": {"code": "Failure", "message": "secret-key"}},
                )
        if request.url.split("?")[0].endswith("/search.run"):
            code = 202
        elif request.url.split("?")[0].endswith("/search.reset"):
            code = 204
        response = FakeResponse(request, deepcopy(payload))
        response.status_code = code
        return response


class EnrichmentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.transport = IndexerTransport()
        self.session = aiohttp.ClientSession()
        self.settings = Settings(
            _env_file=None,
            azure_search_endpoint="https://example.search.windows.net",
            azure_search_api_key="search-key",
            azure_openai_endpoint="https://example.openai.azure.com",
            azure_openai_chat_deployment="chat",
            azure_openai_api_key="secret-key",
            azure_retry_total=0,
        )
        self.repository = AzureIndexerRepository(
            self.settings, AzureKeyCredential("key"), self.session
        )
        self.service = EnrichmentService(self.repository)
        self.patch = patch(
            "app.infrastructure.azure_search.AioHttpTransport", return_value=self.transport
        )
        self.patch.start()
        self.request = SummarySkillRequest(skillset_name="summary-skills")

    async def asyncTearDown(self):
        self.patch.stop()
        await self.session.close()

    def writes(self):
        return [r for r in self.transport.requests if r.method != "GET"]

    async def test_clone_extend_mapping_and_conditional_writes(self):
        result = await self.service.configure_summary("demo-indexer", self.request)
        self.assertNotIn("secret-key", json.dumps(result))
        writes = self.writes()
        self.assertEqual(len(writes), 3)
        skillset, index, indexer = [json.loads(r.body) for r in writes]
        self.assertEqual(skillset["name"], "summary-skills")
        self.assertEqual(skillset["description"], "Preserve me")
        self.assertEqual(skillset["skills"][0]["name"], "split")
        skill = skillset["skills"][1]
        self.assertEqual(skill["@odata.type"], "#Microsoft.Skills.Custom.ChatCompletionSkill")
        self.assertEqual(skill["apiKey"], "secret-key")
        self.assertEqual(skill["inputs"][2]["source"], "/document/content")
        self.assertEqual(skill["outputs"][0]["targetName"], "summary")
        self.assertIn("/deployments/chat/chat/completions", skill["uri"])
        self.assertEqual(index["fields"][-1]["name"], "summary")
        self.assertEqual(indexer["dataSourceName"], "source")
        self.assertEqual(indexer["parameters"]["batchSize"], 10)
        self.assertEqual(indexer["schedule"]["interval"], "PT2H")
        self.assertEqual(indexer["fieldMappings"][0]["sourceFieldName"], "body")
        self.assertEqual(len(indexer["outputFieldMappings"]), 2)
        self.assertEqual(writes[1].headers["If-Match"], '"index-v1"')
        self.assertEqual(writes[2].headers["If-Match"], '"indexer-v1"')
        self.assertTrue(all("api-version=2026-04-01" in r.url for r in self.transport.requests))
        self.assertFalse(result["azure"]["processing_started"])

    async def test_summary_only_and_existing_field(self):
        self.transport.indexer.pop("skillsetName")
        self.transport.index["fields"].append(
            {"name": "summary", "type": "Edm.String", "searchable": True, "retrievable": True}
        )
        await self.service.configure_summary("demo-indexer", self.request)
        self.assertEqual(len(self.writes()), 2)
        self.assertEqual(len(json.loads(self.writes()[0].body)["skills"]), 1)

    async def test_preflight_conflicts_do_not_write(self):
        cases = [
            lambda t: t.index["fields"].append({"name": "summary", "type": "Edm.Int32"}),
            lambda t: t.skillset.update(
                indexProjections={"selectors": [{"targetIndexName": "demo"}]}
            ),
            lambda t: t.skillset.update(
                cognitiveServices={
                    "@odata.type": "#Microsoft.Azure.Search.CognitiveServicesByKey",
                    "key": "***",
                }
            ),
            lambda t: t.indexer["outputFieldMappings"].append(
                {"sourceFieldName": "/document/other", "targetFieldName": "summary"}
            ),
            lambda t: t.indexer.pop("@odata.etag"),
            lambda t: setattr(t, "exists", True),
            lambda t: t.skillset["skills"].append(
                {
                    "@odata.type": "#Microsoft.Skills.Text.SplitSkill",
                    "name": "document-summary",
                    "inputs": [],
                    "outputs": [],
                }
            ),
        ]
        for mutate in cases:
            with self.subTest(mutate=mutate):
                original = self.transport
                transport = IndexerTransport()
                mutate(transport)
                self.patch.stop()
                self.patch = patch(
                    "app.infrastructure.azure_search.AioHttpTransport", return_value=transport
                )
                self.patch.start()
                self.transport = transport
                with self.assertRaises(ApplicationError):
                    await self.service.configure_summary("demo-indexer", self.request)
                self.assertEqual(self.writes(), [])
                self.transport = original

    async def test_partial_failure_and_concurrent_update(self):
        for path, completed in (
            ("/indexes(", "skillset summary-skills created"),
            ("/indexers/", "index demo updated"),
        ):
            self.transport.requests.clear()
            self.transport.fail = (path, 412)
            with self.assertRaises(ApplicationError) as error:
                await self.service.configure_summary("demo-indexer", self.request)
            self.assertEqual(error.exception.status_code, 412)
            self.assertIn(completed, error.exception.message)
            self.assertNotIn("secret-key", error.exception.message)
            self.assertFalse(any(r.method == "DELETE" for r in self.transport.requests))

    async def test_create_conflict_does_not_overwrite(self):
        self.transport.fail = ("/skillsets", 409)
        with self.assertRaises(ApplicationError) as error:
            await self.service.configure_summary("demo-indexer", self.request)
        self.assertEqual(error.exception.status_code, 409)
        self.assertEqual(len(self.writes()), 1)
        self.assertEqual(self.writes()[0].method, "POST")

    async def test_prompt_literal_and_status_credentials(self):
        prompt = 'Summarize the author\'s "quoted" text.\nKeep \\ paths literal.'
        skill = self.repository.summary_skill(prompt)
        expression = skill["inputs"][1]["source"]
        self.assertEqual(json.loads(expression[1:]), prompt)
        repository = AsyncMock()
        repository.indexer_operation.return_value = {
            "status": "running",
            "nested": {"apiKey": "hidden", "api_key": "also-hidden"},
        }
        result = await EnrichmentService(repository).operation("demo-indexer", "status")
        self.assertNotIn("hidden", json.dumps(result))

    async def test_identity_and_missing_chat_config(self):
        self.settings.azure_openai_api_key = None
        self.assertNotIn("apiKey", self.repository.summary_skill("Summary"))
        self.settings.azure_openai_chat_deployment = None
        with self.assertRaises(ApplicationError) as error:
            await self.service.configure_summary("demo-indexer", self.request)
        self.assertEqual(error.exception.status_code, 503)
        self.assertEqual(self.transport.requests, [])

    async def test_run_reset_and_status_use_sdk(self):
        for operation in ("run", "reset", "status"):
            result = await self.service.operation("demo-indexer", operation)
            request = self.transport.requests[-1]
            self.assertIn("/search." + operation, request.url)
            self.assertEqual(request.method, "GET" if operation == "status" else "POST")
            if operation == "status":
                self.assertEqual(result["azure"]["lastResult"]["status"], "success")


class IndexerApiTests(unittest.TestCase):
    def test_routes_validation_and_no_automatic_run(self):
        repository = AsyncMock()
        repository.summary_skill = lambda prompt: {
            "name": "document-summary",
            "outputs": [],
            "inputs": [
                {"name": "systemMessage", "source": "='System'"},
                {"name": "userMessage", "source": "='Prompt'"},
                {"name": "text", "source": "/document/content"},
            ],
        }
        repository.billing_connection = lambda: None
        repository.get_indexer.return_value = {
            "name": "demo-indexer",
            "targetIndexName": "demo",
            "@odata.etag": "v1",
        }
        repository.read_index.return_value = {"name": "demo", "fields": [], "@odata.etag": "v1"}
        from azure.core.exceptions import ResourceNotFoundError

        missing = ResourceNotFoundError()
        missing.status_code = 404
        repository.get_skillset.side_effect = missing
        repository.indexer_operation.return_value = {"accepted": True}
        configured = Settings(
            _env_file=None,
            azure_search_endpoint="https://example.search.windows.net",
            log_level="ERROR",
        )
        with TestClient(
            create_app(
                configured,
                compose_services(AsyncMock(), AsyncMock(), indexer_repository=repository),
            )
        ) as client:
            self.assertEqual(
                client.post("/indexers/demo-indexer/summary-skill", json={}).status_code, 422
            )
            response = client.post(
                "/indexers/demo-indexer/summary-skill", json={"skillset_name": "new-skills"}
            )
            self.assertEqual(response.status_code, 200, response.text)
            repository.indexer_operation.assert_not_called()
            for operation in ("run", "reset", "status"):
                response = (client.get if operation == "status" else client.post)(
                    "/indexers/demo-indexer/" + operation
                )
                self.assertEqual(response.status_code, 202 if operation == "run" else 200)
