"""Exercise real Azure SDK serialization with a fake async transport."""

import json
import unittest
from unittest.mock import patch

import aiohttp
from azure.core.credentials import AzureKeyCredential
from azure.core.pipeline.transport import AsyncHttpResponse, AsyncHttpTransport

from app.core.config import Settings
from app.domain.index_schema import schema
from app.infrastructure.azure_search import AzureSearchRepository
from app.schemas.search import SearchOptions, VectorQuery


class FakeResponse(AsyncHttpResponse):
    def __init__(self, request, payload):
        super().__init__(request, None)
        self.status_code = 200
        self.headers = {"content-type": "application/json"}
        self.reason = "OK"
        self.content_type = "application/json"
        self._body = json.dumps(payload).encode()

    def body(self):
        return self._body

    def text(self, encoding=None):
        return self._body.decode(encoding or "utf-8")

    def json(self):
        return json.loads(self._body)

    async def close(self):
        pass


class FakeTransport(AsyncHttpTransport):
    def __init__(self):
        self.requests = []

    async def open(self):
        pass

    async def close(self):
        pass

    async def __aexit__(self, *args):
        pass

    async def send(self, request, **kwargs):
        self.requests.append(request)
        body = json.loads(request.body) if request.body else {}
        if "/docs/index" in request.url or "search.index" in request.url:
            payload = {
                "value": [
                    {"key": "a", "status": False, "statusCode": 400, "errorMessage": "wrong field"}
                ]
            }
        elif "/docs/search" in request.url:
            payload = {"@odata.count": 1, "value": [{"document_id": "a", "@search.score": 1}]}
        else:
            payload = body
        response = FakeResponse(request, payload)
        if "/docs/" not in request.url:
            response.status_code = 201
        return response


class SdkTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.transport = FakeTransport()
        self.session = aiohttp.ClientSession()
        settings = Settings(
            _env_file=None, azure_search_endpoint="https://example.search.windows.net"
        )
        self.repository = AzureSearchRepository(
            settings, AzureKeyCredential("test-key"), self.session
        )
        self.patch = patch(
            "app.infrastructure.azure_search.AioHttpTransport", return_value=self.transport
        )
        self.patch.start()

    async def asyncTearDown(self):
        self.patch.stop()
        await self.session.close()

    async def test_search_sdk_serializes_vector_and_metadata(self):
        result = await self.repository.search(
            "demo",
            SearchOptions(
                search_text="hello",
                include_total_count=True,
                filter="is_public eq true",
                vector_queries=[VectorQuery(vector=[0.1, 0.2])],
            ),
        )
        wire = json.loads(self.transport.requests[0].body)
        self.assertEqual(wire["search"], "hello")
        self.assertEqual(wire["vectorQueries"][0]["kind"], "vector")
        self.assertEqual(wire["vectorQueries"][0]["k"], 50)
        self.assertEqual(wire["filter"], "is_public eq true")
        self.assertEqual(result["azure"]["@odata.count"], 1)
        self.assertEqual(self.transport.requests[0].headers["api-key"], "test-key")

    async def test_index_create_uses_sdk_schema(self):
        result = await self.repository.create_index(schema("demo", 1536, "test"))
        self.assertEqual(result["azure"]["name"], "demo")
        self.assertEqual(result["azure"]["fields"][-1]["dimensions"], 1536)

    async def test_batch_partial_failure_is_visible(self):
        result = await self.repository.index_documents(
            "demo",
            [{"@search.action": "upload", "id": "a"}, {"@search.action": "delete", "id": "b"}],
        )
        wire = json.loads(self.transport.requests[0].body)
        self.assertEqual([item["@search.action"] for item in wire["value"]], ["upload", "delete"])
        self.assertFalse(result["all_succeeded"])
        self.assertEqual(result["azure"]["value"][0]["statusCode"], 400)

    async def test_raw_sdk_pipeline_keeps_extra_fields(self):
        body = {"search": "hello", "hybridSearch": {"maxTextRecallSize": 20}}
        result = await self.repository.raw_search("demo", body)
        self.assertEqual(result["azure"]["value"][0]["document_id"], "a")
        request = self.transport.requests[0]
        self.assertEqual(json.loads(request.content), body)
        self.assertEqual(request.headers["api-key"], "test-key")
