"""Verify real SDK requests and response models without Azure resources."""

import json
import unittest
from unittest.mock import patch

import aiohttp
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from test_sdk import FakeResponse, FakeTransport

from app.core.config import Settings
from app.core.errors import ApplicationError
from app.infrastructure.agentic import AzureAgenticRepository, redact_credentials
from app.schemas.agentic import (
    AgenticRetrievalRequest,
    KnowledgeBaseRequest,
    KnowledgeSourceRequest,
)


class AgenticTransport(FakeTransport):
    def __init__(self):
        super().__init__()
        self.status = None
        self.empty_sources = False

    async def send(self, request, **kwargs):
        self.requests.append(request)
        body = json.loads(request.body) if request.body else {}
        if request.url.split("?")[0].endswith("/retrieve"):
            payload = {
                "response": [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Grounding context"}],
                    }
                ],
                "activity": [
                    {"type": "searchIndex", "id": 0, "knowledgeSourceName": "demo-source"}
                ],
                "references": [
                    {
                        "type": "searchIndex",
                        "id": "0",
                        "activitySource": 0,
                        "docKey": "chunk-1",
                        "sourceData": {"content": "Annual leave"},
                    }
                ],
            }
        elif request.method == "GET" and "/knowledgebases(" in request.url:
            payload = {
                "name": "demo-base",
                "knowledgeSources": [{"name": "demo-source"}, {"name": "other-source"}],
                "models": [{"kind": "azureOpenAI", "azureOpenAIParameters": {"apiKey": "hidden"}}],
            }
        elif request.method == "GET":
            payload = {
                "name": "demo-source",
                "kind": "searchIndex",
                "searchIndexParameters": {"searchIndexName": "demo"},
            }
        else:
            payload = body
        if self.empty_sources and request.method == "GET" and "/knowledgebases(" in request.url:
            payload["knowledgeSources"] = []
        response = FakeResponse(request, payload)
        response.status_code = self.status or (
            201 if request.method == "POST" and "/retrieve" not in request.url else 200
        )
        if self.status:
            response._body = json.dumps(
                {"error": {"code": "Conflict", "message": "Resource already exists"}}
            ).encode()
        return response


class AgenticSdkTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.transport = AgenticTransport()
        self.session = aiohttp.ClientSession()
        self.settings = Settings(
            _env_file=None,
            azure_search_endpoint="https://example.search.windows.net",
            azure_openai_endpoint="https://example.openai.azure.com",
            azure_openai_chat_deployment="chat-deployment",
            azure_openai_chat_model="gpt-4.1-mini",
            azure_openai_api_key="test-openai-key",
        )
        self.repository = AzureAgenticRepository(
            self.settings, AzureKeyCredential("search-key"), self.session
        )
        self.patch = patch(
            "app.infrastructure.azure_search.AioHttpTransport", return_value=self.transport
        )
        self.patch.start()

    async def asyncTearDown(self):
        self.patch.stop()
        await self.session.close()

    def assert_agentic_version(self):
        for request in self.transport.requests:
            self.assertIn("api-version=2026-08-01-preview", request.url)
            self.assertEqual(request.headers["api-key"], "search-key")

    async def test_source_create_and_read_use_typed_sdk(self):
        result = await self.repository.create_knowledge_source(
            "demo-source",
            KnowledgeSourceRequest(
                index_name="demo",
                semantic_configuration_name="semantic-config",
                search_fields=["content"],
                source_data_fields=["chunk_id", "content"],
            ),
        )
        request = self.transport.requests[0]
        self.assertEqual(request.method, "POST")
        self.assertIn("/knowledgesources", request.url)
        wire = json.loads(request.body)
        self.assertEqual(wire["kind"], "searchIndex")
        self.assertEqual(wire["searchIndexParameters"]["searchIndexName"], "demo")
        self.assertEqual(
            wire["searchIndexParameters"]["sourceDataFields"],
            [{"name": "chunk_id"}, {"name": "content"}],
        )
        self.assertEqual(
            wire["searchIndexParameters"]["semanticConfigurationName"], "semantic-config"
        )
        self.assertEqual(result["azure"]["name"], "demo-source")
        await self.repository.get_knowledge_source("demo-source")
        self.assertEqual(self.transport.requests[1].method, "GET")
        self.assert_agentic_version()

    async def test_base_creation_planning_and_redaction(self):
        result = await self.repository.create_knowledge_base(
            "demo-base", KnowledgeBaseRequest(knowledge_source_names=["demo-source"])
        )
        wire = json.loads(self.transport.requests[0].body)
        self.assertEqual(wire["outputMode"], "extractiveData")
        self.assertEqual(wire["retrievalReasoningEffort"], {"kind": "low"})
        parameters = wire["models"][0]["azureOpenAIParameters"]
        self.assertEqual(parameters["deploymentId"], "chat-deployment")
        self.assertEqual(parameters["modelName"], "gpt-4.1-mini")
        self.assertEqual(parameters["apiKey"], "test-openai-key")
        self.assertNotIn("apiKey", result["azure"]["models"][0]["azureOpenAIParameters"])
        result = await self.repository.get_knowledge_base("demo-base")
        self.assertNotIn("hidden", json.dumps(result))
        self.assert_agentic_version()

    async def test_managed_identity_omits_model_api_key(self):
        self.settings.azure_openai_api_key = None
        await self.repository.create_knowledge_base(
            "demo-base", KnowledgeBaseRequest(knowledge_source_names=["demo-source"])
        )
        parameters = json.loads(self.transport.requests[0].body)["models"][0][
            "azureOpenAIParameters"
        ]
        self.assertNotIn("apiKey", parameters)
        self.assertNotIn("authIdentity", parameters)

    async def test_missing_chat_configuration_fails_before_azure(self):
        self.settings.azure_openai_chat_deployment = None
        with self.assertRaises(ApplicationError) as error:
            await self.repository.create_knowledge_base(
                "demo-base", KnowledgeBaseRequest(knowledge_source_names=["demo-source"])
            )
        self.assertEqual(error.exception.status_code, 503)
        self.assertEqual(self.transport.requests, [])

    async def test_retrieve_filters_references_token_and_no_local_chat(self):
        self.settings.azure_openai_chat_deployment = None
        result = await self.repository.retrieve(
            "demo-base",
            AgenticRetrievalRequest(
                messages=[{"role": "user", "content": "Annual leave?"}],
                knowledge_source_params=[
                    {"knowledge_source_name": "demo-source", "filter_add_on": "is_public eq true"}
                ],
            ),
            "source-token",
        )
        request = self.transport.requests[-1]
        self.assertIn("/knowledgebases('demo-base')/retrieve", request.url)
        self.assertEqual(request.headers["x-ms-query-source-authorization"], "source-token")
        wire = json.loads(request.body)
        self.assertEqual(
            wire["messages"][0]["content"], [{"type": "text", "text": "Annual leave?"}]
        )
        self.assertEqual(wire["outputMode"], "extractiveData")
        self.assertEqual(wire["retrievalReasoningEffort"], {"kind": "low"})
        self.assertTrue(wire["includeActivity"])
        params = wire["knowledgeSourceParams"]
        self.assertEqual(params[0]["filterAddOn"], "is_public eq true")
        self.assertNotIn("filterAddOn", params[1])
        for source in params:
            self.assertTrue(source["includeReferences"])
            self.assertTrue(source["includeReferenceSourceData"])
        self.assertEqual(result["azure"]["references"][0]["sourceData"]["content"], "Annual leave")
        self.assertEqual(len(result["azure"]["activity"]), 1)
        self.assert_agentic_version()

    async def test_retrieve_without_filters_includes_all_source_data(self):
        await self.repository.retrieve(
            "demo-base", AgenticRetrievalRequest(messages=[{"role": "user", "content": "Question"}])
        )
        wire = json.loads(self.transport.requests[-1].body)
        self.assertNotIn("maxOutputDocuments", wire)
        self.assertEqual(len(wire["knowledgeSourceParams"]), 2)
        self.assertTrue(all(p["includeReferenceSourceData"] for p in wire["knowledgeSourceParams"]))

    async def test_output_limit_serializes_and_validates(self):
        from pydantic import ValidationError

        await self.repository.retrieve(
            "demo-base", AgenticRetrievalRequest(
                messages=[{"role": "user", "content": "Question"}], max_output_documents=50)
        )
        wire = json.loads(self.transport.requests[-1].body)
        self.assertEqual(wire["maxOutputDocuments"], 50)
        for invalid in (0, -1, 1001):
            with self.assertRaises(ValidationError):
                AgenticRetrievalRequest(
                    messages=[{"role": "user", "content": "Question"}],
                    max_output_documents=invalid)

    async def test_unknown_filter_source_does_not_send_retrieve(self):
        with self.assertRaises(ApplicationError) as error:
            await self.repository.retrieve(
                "demo-base",
                AgenticRetrievalRequest(
                    messages=[{"role": "user", "content": "Question"}],
                    knowledge_source_params=[{"knowledge_source_name": "missing-source"}],
                ),
            )
        self.assertEqual(error.exception.status_code, 422)
        self.assertEqual(len(self.transport.requests), 1)

    async def test_empty_upstream_base_returns_controlled_error(self):
        self.transport.empty_sources = True
        with self.assertRaises(ApplicationError) as error:
            await self.repository.retrieve(
                "demo-base",
                AgenticRetrievalRequest(messages=[{"role": "user", "content": "Question"}]),
            )
        self.assertEqual(error.exception.status_code, 502)
        self.assertEqual(len(self.transport.requests), 1)

    async def test_create_conflict_is_propagated_without_overwrite(self):
        self.transport.status = 409
        with self.assertRaises(HttpResponseError) as error:
            await self.repository.create_knowledge_source(
                "demo-source", KnowledgeSourceRequest(index_name="demo")
            )
        self.assertEqual(error.exception.status_code, 409)
        self.assertEqual(len(self.transport.requests), 1)
        self.assertEqual(self.transport.requests[0].method, "POST")

    def test_recursive_credential_redaction(self):
        self.assertEqual(
            redact_credentials(
                {
                    "models": [
                        {"api_key": "key", "connection": {"apiKey": "key", "modelName": "model"}}
                    ]
                }
            ),
            {"models": [{"connection": {"modelName": "model"}}]},
        )
