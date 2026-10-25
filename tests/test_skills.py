"""Preset and advanced JSON experiments through real SDK HTTP serialization."""

import json
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock

import test_enrichment as fixtures
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from app.container import compose_services
from app.core.errors import ApplicationError
from app.domain.skill_presets import OUTPUTS, PRESETS
from app.main import create_app
from app.schemas.enrichment import (
    ENTITY_OUTPUTS,
    AttachSkillsetRequest,
    SkillsRequest,
    SummarySkillRequest,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class SkillTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await fixtures.EnrichmentTests.asyncSetUp(self)

    async def asyncTearDown(self):
        await fixtures.EnrichmentTests.asyncTearDown(self)

    def writes(self):
        return [r for r in self.transport.requests if r.method != "GET"]

    async def configure(self, skills):
        return await self.service.configure_skills(
            "demo-indexer", SkillsRequest(skillset_name="experiment-skills", skills=skills)
        )

    async def test_six_presets_wire_fields_and_mappings(self):
        result = await self.configure(
            [
                {
                    "kind": "classification",
                    "categories": ["Finance", "HR", "Other"],
                    "instruction": "Prefer the most specific label.",
                },
                {
                    "kind": "entities",
                    "categories": list(ENTITY_OUTPUTS),
                    "minimum_precision": 0.7,
                    "input_path": "/document/body",
                    "language_code": "fr",
                },
                {"kind": "key_phrases"},
                {"kind": "language_detection"},
                {"kind": "sentiment", "language_code": "de"},
                {"kind": "summary"},
            ]
        )
        skillset, index, indexer = [json.loads(r.body) for r in self.writes()]
        skills = {s["name"]: s for s in skillset["skills"]}
        for kind, (name, odata) in PRESETS.items():
            self.assertEqual(skills[name]["@odata.type"], odata)
            self.assertEqual(
                {o["name"]: o["targetName"] for o in skills[name]["outputs"]}, OUTPUTS[kind]
            )
        self.assertEqual(skills["document-entities"]["categories"], list(ENTITY_OUTPUTS))
        self.assertEqual(skills["document-entities"]["minimumPrecision"], 0.7)
        self.assertEqual(skills["document-entities"]["inputs"][0]["source"], "/document/body")
        self.assertEqual(skills["document-entities"]["defaultLanguageCode"], "fr")
        self.assertEqual(skills["document-key-phrases"]["defaultLanguageCode"], "en")
        self.assertEqual(skills["document-sentiment"]["defaultLanguageCode"], "de")
        classification_prompt = json.loads(
            skills["document-classification"]["inputs"][1]["source"][1:]
        )
        self.assertIn('["Finance", "HR", "Other"]', classification_prompt)
        self.assertIn("Prefer the most specific label.", classification_prompt)
        fields = {f["name"]: f for f in index["fields"]}
        self.assertNotIn("classification", fields)
        for kind, outputs in OUTPUTS.items():
            for target in outputs.values():
                field = fields[target]
                self.assertTrue(field["searchable"])
                self.assertTrue(field["retrievable"])
                if kind != "summary":
                    self.assertTrue(field["filterable"])
                    self.assertTrue(field["facetable"])
                self.assertEqual(
                    field["type"],
                    "Collection(Edm.String)"
                    if kind in {"entities", "key_phrases"}
                    else "Edm.String",
                )
                self.assertIn(
                    {"sourceFieldName": "/document/" + target, "targetFieldName": target},
                    indexer["outputFieldMappings"],
                )
        self.assertFalse(result["azure"]["processing_started"])
        self.assertNotIn("secret-key", json.dumps(result))

    async def test_builtins_do_not_require_chat_and_billing_default_is_omitted(self):
        self.settings.azure_openai_endpoint = None
        await self.configure(
            [
                {"kind": "entities"},
                {"kind": "key_phrases"},
                {"kind": "language_detection"},
                {"kind": "sentiment"},
            ]
        )
        self.assertNotIn("cognitiveServices", json.loads(self.writes()[0].body))

    async def test_replace_only_selected_presets_and_keep_old_fields(self):
        await self.configure([{"kind": "entities"}, {"kind": "language_detection"}])
        skillset, index, indexer = [json.loads(r.body) for r in self.writes()]
        self.transport.skillset = skillset
        self.transport.skillset["name"] = "old-skills"
        self.transport.index = index
        self.transport.indexer = indexer
        self.transport.indexer["skillsetName"] = "old-skills"
        self.transport.requests.clear()
        await self.configure(
            [{"kind": "entities", "categories": ["Person"], "minimum_precision": 0.9}]
        )
        writes = [json.loads(r.body) for r in self.writes()]
        self.assertEqual(len(writes), 2)  # Existing fields need no index update.
        names = [s["name"] for s in writes[0]["skills"]]
        self.assertEqual(names.count("document-entities"), 1)
        self.assertIn("document-language", names)
        self.assertIn("split", names)
        entity = next(s for s in writes[0]["skills"] if s["name"] == "document-entities")
        self.assertEqual(entity["minimumPrecision"], 0.9)
        self.assertEqual(len(entity["outputs"]), 1)
        self.assertIn("entities_organizations", [f["name"] for f in self.transport.index["fields"]])
        self.assertNotIn(
            "entities_organizations",
            [m["targetFieldName"] for m in writes[1]["outputFieldMappings"]],
        )

    async def test_legacy_summary_can_be_replaced_with_masked_old_key(self):
        legacy = self.repository.summary_skill("Old prompt")
        legacy["apiKey"] = "***"
        self.transport.skillset["skills"].append(legacy)
        await self.service.configure_summary(
            "demo-indexer", SummarySkillRequest(skillset_name="next-skills", prompt="New prompt")
        )
        created = json.loads(self.writes()[0].body)
        summaries = [s for s in created["skills"] if s["name"] == "document-summary"]
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["apiKey"], "secret-key")
        self.assertEqual(json.loads(summaries[0]["inputs"][1]["source"][1:]), "New prompt")

    async def test_output_and_reserved_name_collisions_do_not_write(self):
        for existing in [
            {
                "name": "custom",
                "context": "/document",
                "outputs": [{"name": "out", "targetName": "document_category"}],
            },
            {
                "name": "document-classification",
                "@odata.type": "#Microsoft.Skills.Custom.ChatCompletionSkill",
                "outputs": [],
            },
        ]:
            self.transport.skillset["skills"] = [existing]
            with self.assertRaises(ApplicationError) as error:
                await self.configure([{"kind": "classification", "categories": ["A"]}])
            self.assertEqual(error.exception.status_code, 409)
            self.assertEqual(self.writes(), [])

    async def test_billing_identity_key_and_existing_connection(self):
        self.transport.skillset["cognitiveServices"] = {
            "@odata.type": "#Microsoft.Azure.Search.AIServicesByIdentity",
            "subdomainUrl": "https://old.services.ai.azure.com",
            "identity": None,
        }
        await self.configure([{"kind": "entities"}])
        self.assertEqual(
            json.loads(self.writes()[0].body)["cognitiveServices"]["subdomainUrl"],
            "https://old.services.ai.azure.com",
        )
        self.settings.azure_ai_services_endpoint = "https://billing.services.ai.azure.com"
        self.transport.skillset["cognitiveServices"] = {
            "@odata.type": "#Microsoft.Azure.Search.AIServicesByKey",
            "key": "***",
        }
        for key in (None, SecretStr("billing-secret")):
            self.transport.requests.clear()
            self.settings.azure_ai_services_key = key
            await self.configure([{"kind": "entities"}])
            connection = json.loads(self.writes()[0].body)["cognitiveServices"]
            self.assertEqual(connection["subdomainUrl"], self.settings.azure_ai_services_endpoint)
            if key:
                self.assertEqual(connection["key"], "billing-secret")
                self.assertTrue(connection["@odata.type"].endswith("AIServicesByKey"))
            else:
                self.assertTrue(connection["@odata.type"].endswith("AIServicesByIdentity"))
                self.assertIsNone(connection["identity"])
                self.assertNotIn("key", connection)

    async def test_billing_key_without_endpoint_fails_before_writes(self):
        self.settings.azure_ai_services_key = SecretStr("billing-secret")
        with self.assertRaises(ApplicationError) as error:
            await self.configure([{"kind": "entities"}])
        self.assertEqual(error.exception.status_code, 503)
        self.assertEqual(self.transport.requests, [])

    async def test_advanced_json_is_lossless_and_inspection_redacts_secrets(self):
        definition = {
            "skills": [
                {
                    "@odata.type": "#Microsoft.Skills.Custom.FutureSkill",
                    "name": "future",
                    "inputs": [],
                    "outputs": [],
                    "futureParameter": {"flag": True},
                    "apiKey": "submitted-secret",
                    "httpHeaders": {"X-Custom-Credential": "header-secret"},
                }
            ],
            "futureProperty": {"nested": [1, 2]},
            "cognitiveServices": {"key": "billing-secret"},
        }
        result = await self.service.create_skillset("future-skills", definition)
        wire = json.loads(self.writes()[0].body)
        self.assertEqual(wire, {**definition, "name": "future-skills"})
        self.assertNotIn("secret", json.dumps(result))
        self.transport.skillset = wire
        self.transport.requests.clear()
        inspected = await self.service.inspect_skillset("old-skills")
        self.assertEqual(inspected["azure"]["futureProperty"], {"nested": [1, 2]})
        self.assertNotIn("secret", json.dumps(inspected))
        self.assertEqual(definition["skills"][0]["apiKey"], "submitted-secret")

    async def test_advanced_create_conflict_and_error_redaction(self):
        definition = {"skills": [{"@odata.type": "#Microsoft.Skills.Custom.FutureSkill"}]}
        self.transport.exists = True
        with self.assertRaises(ApplicationError) as error:
            await self.service.create_skillset("future-skills", definition)
        self.assertEqual(error.exception.status_code, 409)
        self.assertEqual(self.writes(), [])
        self.transport.exists = False
        self.transport.fail = ("/skillsets", 400)
        with self.assertRaises(ApplicationError) as error:
            await self.service.create_skillset("future-skills", definition)
        self.assertNotIn("secret-key", error.exception.message)

    async def test_attach_explicit_fields_and_mappings_and_preserve_indexer(self):
        request = AttachSkillsetRequest(
            skillset_name="old-skills",
            fields=[
                {"name": "custom_entities", "type": "Collection(Edm.String)", "searchable": True}
            ],
            output_field_mappings=[
                {
                    "sourceFieldName": "/document/custom_entities/*/name",
                    "targetFieldName": "custom_entities",
                }
            ],
        )
        result = await self.service.attach_skillset("demo-indexer", request)
        index, indexer = [json.loads(r.body) for r in self.writes()]
        self.assertEqual(index["fields"][-1]["name"], "custom_entities")
        self.assertEqual(indexer["dataSourceName"], "source")
        self.assertEqual(indexer["schedule"], self.transport.indexer["schedule"])
        self.assertEqual(indexer["skillsetName"], "old-skills")
        self.assertEqual(self.writes()[0].headers["If-Match"], '"index-v1"')
        self.assertEqual(self.writes()[1].headers["If-Match"], '"indexer-v1"')
        self.assertFalse(result["azure"]["processing_started"])

    async def test_attach_rejects_conflicts_before_writes(self):
        for options in [
            {"fields": [{"name": "content", "type": "Edm.Int32"}]},
            {
                "output_field_mappings": [
                    {"sourceFieldName": "/document/out", "targetFieldName": "missing"}
                ]
            },
            {"fields": [{"name": "other"}]},
            {"fields": [{"name": "other", "type": "Edm.String"}] * 2},
        ]:
            with self.assertRaises(ApplicationError):
                await self.service.attach_skillset(
                    "demo-indexer", AttachSkillsetRequest(skillset_name="old-skills", **options)
                )
            self.assertEqual(self.writes(), [])

    async def test_attach_partial_failure_reports_completed_update(self):
        self.transport.fail = ("/indexers/", 412)
        with self.assertRaises(ApplicationError) as error:
            await self.service.attach_skillset(
                "demo-indexer",
                AttachSkillsetRequest(
                    skillset_name="old-skills", fields=[{"name": "new", "type": "Edm.String"}]
                ),
            )
        self.assertEqual(error.exception.status_code, 412)
        self.assertIn("index demo updated", error.exception.message)
        self.assertNotIn("secret-key", error.exception.message)

    async def test_advanced_projections_attach_only_without_automatic_additions(self):
        self.transport.skillset["indexProjections"] = {"selectors": []}
        # Nonempty projection object signals caller-managed mapping.
        with self.assertRaises(ApplicationError):
            await self.service.attach_skillset(
                "demo-indexer",
                AttachSkillsetRequest(
                    skillset_name="old-skills", fields=[{"name": "new", "type": "Edm.String"}]
                ),
            )
        await self.service.attach_skillset(
            "demo-indexer", AttachSkillsetRequest(skillset_name="old-skills")
        )
        self.assertEqual(len(self.writes()), 1)

    async def test_shipped_examples_configure_through_real_sdk_pipeline(self):
        examples = EXAMPLES
        for number in range(12, 19):
            path = next(examples.glob(f"{number:02d}-*.json"))
            with self.subTest(example=path.name):
                self.transport.requests.clear()
                request = SkillsRequest.model_validate_json(path.read_text())
                await self.service.configure_skills("demo-indexer", request)
                definition = json.loads(self.writes()[0].body)
                self.assertEqual(definition["name"], request.skillset_name)
                self.assertEqual(len(definition["skills"]), len(request.skills) + 1)
        self.transport.requests.clear()
        custom = json.loads((examples / "19-custom-entity-skillset.json").read_text())
        await self.service.create_skillset("uat-custom-entities-v1", custom)
        self.assertEqual(json.loads(self.writes()[0].body)["skills"], custom["skills"])
        self.transport.skillset = custom
        self.transport.exists = True
        attachment = AttachSkillsetRequest.model_validate_json(
            (examples / "20-attach-custom-entity-skillset.json").read_text()
        )
        self.transport.requests.clear()
        await self.service.attach_skillset("demo-indexer", attachment)
        self.assertEqual(len(self.writes()), 2)

    async def test_advanced_empty_mappings_from_azure_and_existing_mapping_reuse(self):
        self.transport.indexer["outputFieldMappings"] = None
        self.transport.indexer["fieldMappings"] = []
        request = AttachSkillsetRequest(
            skillset_name="old-skills",
            output_field_mappings=[
                {"sourceFieldName": "/document/content", "targetFieldName": "content"}
            ],
        )
        await self.service.attach_skillset("demo-indexer", request)
        self.assertEqual(len(json.loads(self.writes()[0].body)["outputFieldMappings"]), 1)
        self.transport.indexer["outputFieldMappings"] = [
            {**request.output_field_mappings[0], "mappingFunction": None}
        ]
        self.transport.requests.clear()
        await self.service.attach_skillset("demo-indexer", request)
        self.assertEqual(len(json.loads(self.writes()[0].body)["outputFieldMappings"]), 1)

    async def test_advanced_invalid_definition_never_writes(self):
        for definition in (
            {"skills": []},
            {"skills": [{}]},
            {"name": "different", "skills": [{"@odata.type": "test"}]},
        ):
            with self.assertRaises(ApplicationError) as error:
                await self.service.create_skillset("future-skills", definition)
            self.assertEqual(error.exception.status_code, 422)
            self.assertEqual(self.transport.requests, [])

    async def test_missing_chat_for_classification_fails_before_any_write(self):
        self.settings.azure_openai_endpoint = None
        with self.assertRaises(ApplicationError) as error:
            await self.configure([{"kind": "classification", "categories": ["Finance"]}])
        self.assertEqual(error.exception.status_code, 503)
        self.assertEqual(self.transport.requests, [])

    async def test_duplicate_managed_names_are_rejected(self):
        await self.configure([{"kind": "entities"}])
        generated = json.loads(self.writes()[0].body)["skills"][-1]
        self.transport.skillset["skills"] = [generated, deepcopy(generated)]
        self.transport.requests.clear()
        with self.assertRaises(ApplicationError) as error:
            await self.configure([{"kind": "entities"}])
        self.assertEqual(error.exception.status_code, 409)
        self.assertEqual(self.writes(), [])


class SkillValidationTests(unittest.TestCase):
    def test_invalid_presets(self):
        for skills in [
            [{"kind": "classification", "categories": ["A", " A "]}],
            [{"kind": "classification", "categories": [" "]}],
            [{"kind": "classification", "categories": []}],
            [{"kind": "entities", "categories": ["Unsupported"]}],
            [{"kind": "entities", "categories": ["Person", "Person"]}],
            [{"kind": "entities", "minimum_precision": 2}],
            [{"kind": "sentiment", "input_path": "/document/pages/*"}],
            [{"kind": "summary"}, {"kind": "summary"}],
        ]:
            with self.subTest(skills=skills), self.assertRaises(ValidationError):
                SkillsRequest(skillset_name="experiment-skills", skills=skills)

    def test_routes_swagger_and_validation(self):
        repository = AsyncMock()
        services = compose_services(AsyncMock(), AsyncMock(), indexer_repository=repository)
        services.enrichment.configure_skills = AsyncMock(
            return_value={"elapsed_ms": 1, "azure": {"processing_started": False}}
        )
        services.enrichment.attach_skillset = AsyncMock(return_value={"elapsed_ms": 1, "azure": {}})
        services.enrichment.create_skillset = AsyncMock(
            return_value={"elapsed_ms": 1, "azure": {"name": "test-skills"}}
        )
        services.enrichment.inspect_skillset = AsyncMock(
            return_value={"elapsed_ms": 1, "azure": {}}
        )
        configured = fixtures.Settings(
            _env_file=None,
            azure_search_endpoint="https://example.search.windows.net",
            log_level="ERROR",
        )
        with TestClient(create_app(configured, services)) as client:
            self.assertEqual(
                client.post(
                    "/indexers/demo-indexer/skills",
                    json={
                        "skillset_name": "test-skills",
                        "skills": [{"kind": "classification", "categories": []}],
                    },
                ).status_code,
                422,
            )
            self.assertEqual(
                client.post(
                    "/indexers/demo-indexer/skills",
                    json={"skillset_name": "test-skills", "skills": [{"kind": "entities"}]},
                ).status_code,
                200,
            )
            self.assertEqual(
                client.post(
                    "/indexers/demo-indexer/skillset", json={"skillset_name": "test-skills"}
                ).status_code,
                200,
            )
            self.assertEqual(
                client.post("/skillsets/test-skills", json={"skills": []}).status_code, 201
            )
            self.assertEqual(client.get("/skillsets/test-skills").status_code, 200)
            spec = client.get("/openapi.json").json()
            self.assertIn("SkillsRequest", spec["components"]["schemas"])
            self.assertIn(
                "discriminator",
                spec["components"]["schemas"]["SkillsRequest"]["properties"]["skills"]["items"],
            )
            repository.indexer_operation.assert_not_called()
