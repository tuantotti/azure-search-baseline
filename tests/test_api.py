"""HTTP contract and application-service tests without Azure credentials."""

import unittest
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from app.container import compose_services
from app.core.config import Settings
from app.main import create_app


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.repository = AsyncMock()
        self.repository.search.return_value = {
            "elapsed_ms": 1,
            "azure": {
                "value": [{"document_id": "doc1", "@search.score": 1, "@search.rerankerScore": 3}]
            },
        }
        self.repository.raw_search.return_value = {"elapsed_ms": 1, "azure": {"value": []}}
        self.repository.create_index.return_value = {"elapsed_ms": 1, "azure": {"name": "demo"}}
        self.embeddings = AsyncMock()
        self.embeddings.embed.return_value = {
            "elapsed_ms": 2,
            "azure": {"data": [{"index": 0, "embedding": [0.1, 0.2]}]},
        }
        settings = Settings(
            _env_file=None,
            azure_search_endpoint="https://example.search.windows.net",
            log_level="ERROR",
        )
        self.client = TestClient(
            create_app(settings, compose_services(self.repository, self.embeddings))
        )
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def test_default_docs_and_no_custom_auth(self):
        self.assertEqual(self.client.get("/docs").status_code, 200)
        spec = self.client.get("/openapi.json").json()
        self.assertTrue(spec["openapi"].startswith("3."))
        self.assertNotIn("securitySchemes", spec.get("components", {}))
        response = self.client.post("/indexes/demo/search", json={"search": "hello"})
        self.assertEqual(response.status_code, 200)

    def test_raw_parameters_and_optional_source_token(self):
        body = {"search": "hello", "hybridSearch": {"maxTextRecallSize": 100}}
        response = self.client.post(
            "/indexes/demo/search?api_version=2025-09-01",
            json=body,
            headers={"X-MS-Query-Source-Authorization": "source-token"},
        )
        self.assertEqual(response.status_code, 200)
        self.repository.raw_search.assert_awaited_once_with(
            "demo", body, "2025-09-01", "source-token"
        )

    def test_vector_has_no_keyword_query(self):
        response = self.client.post(
            "/indexes/demo/search/text",
            json={
                "query": "hello",
                "mode": "vector",
                "parameters": {"filter": "is_public eq true"},
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        options = self.repository.search.call_args.args[1]
        self.assertIsNone(options.search_text)
        self.assertEqual(options.vector_queries[0].vector, [0.1, 0.2])
        self.assertEqual(options.filter, "is_public eq true")

    def test_bm25_does_not_require_embedding_configuration(self):
        response = self.client.post(
            "/indexes/demo/search/text", json={"query": "hello", "mode": "bm25"}
        )
        self.assertEqual(response.status_code, 200)
        self.embeddings.embed.assert_not_called()

    def test_comparison_reuses_embedding(self):
        response = self.client.post(
            "/indexes/demo/experiments/compare",
            json={
                "query": "hello",
                "modes": ["bm25", "vector", "hybrid", "hybrid_semantic"],
                "qrels": {"doc1": 3},
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.embeddings.embed.assert_awaited_once()
        self.assertEqual(len(response.json()["runs"]), 4)
        self.assertTrue(all(run["metrics"]["ndcg_at_10"] == 1 for run in response.json()["runs"]))

    def test_invalid_options_fail_before_embedding(self):
        response = self.client.post(
            "/indexes/demo/search/text",
            json={"query": "hello", "parameters": {"search_text": "override"}},
        )
        self.assertEqual(response.status_code, 422)
        self.embeddings.embed.assert_not_called()

    def test_no_silent_semantic_fallback(self):
        self.repository.search.return_value["azure"]["value"] = [{"document_id": "doc1"}]
        response = self.client.post(
            "/indexes/demo/search/text", json={"query": "hello", "mode": "hybrid_semantic"}
        )
        self.assertEqual(response.status_code, 502)

    def test_index_creation_no_custom_auth(self):
        response = self.client.post("/indexes/demo/baseline-schema", json={"dimensions": 1536})
        self.assertEqual(response.status_code, 201)
        self.assertEqual(self.repository.create_index.call_args.args[0]["name"], "demo")

    def test_unsupported_sdk_option_is_not_silently_ignored(self):
        response = self.client.post("/indexes/demo/search/sdk", json={"unknown_option": 1})
        self.assertEqual(response.status_code, 422)
