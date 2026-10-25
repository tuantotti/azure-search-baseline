"""Preset wire definitions, fields, and ownership checks without Azure I/O."""

from typing import Any

from app.schemas.enrichment import ENTITY_OUTPUTS, SkillPreset

PREFIX = "azure-search-baseline preset:"
# Keep the original summary name so existing pipelines can be upgraded.
PRESETS = {
    "classification": ("document-classification", "#Microsoft.Skills.Custom.ChatCompletionSkill"),
    "entities": ("document-entities", "#Microsoft.Skills.Text.V3.EntityRecognitionSkill"),
    "key_phrases": ("document-key-phrases", "#Microsoft.Skills.Text.KeyPhraseExtractionSkill"),
    "language_detection": ("document-language", "#Microsoft.Skills.Text.LanguageDetectionSkill"),
    "sentiment": ("document-sentiment", "#Microsoft.Skills.Text.V3.SentimentSkill"),
    "summary": ("document-summary", "#Microsoft.Skills.Custom.ChatCompletionSkill"),
}
OUTPUTS = {
    "classification": {"response": "document_category"},
    "entities": {output: f"entities_{output}" for output in ENTITY_OUTPUTS.values()},
    "key_phrases": {"keyPhrases": "key_phrases"},
    "language_detection": {"languageCode": "detected_language"},
    "sentiment": {"sentiment": "document_sentiment"},
    "summary": {"response": "summary"},
}


def managed_skill(skill: dict[str, Any], kind: str) -> bool:
    name, odata = PRESETS[kind]
    outputs = skill.get("outputs") or []
    expected = OUTPUTS[kind]
    shape = (
        skill.get("name") == name
        and skill.get("@odata.type") == odata
        and skill.get("context", "/document") == "/document"
        and bool(outputs)
        and all(
            o.get("name") in expected and o.get("targetName") == expected[o["name"]]
            for o in outputs
        )
    )
    if not shape:
        return False
    if skill.get("description") == f"{PREFIX}{kind}:v1":
        return True
    # Recognize only the precise shape of the previously shipped summary preset.
    inputs = {i.get("name"): i.get("source") for i in skill.get("inputs", [])}
    return (
        kind == "summary"
        and not skill.get("description")
        and inputs.get("systemMessage") == "='You summarize documents faithfully.'"
        and inputs.get("text") == "/document/content"
        and isinstance(inputs.get("userMessage"), str)
        and inputs["userMessage"].startswith("=")
        and bool(skill.get("uri"))
    )


def build_preset(config: SkillPreset, chat_skill: dict[str, Any] | None = None):
    """Build one skill with predictable output fields and mappings."""
    kind = config.kind
    name, odata = PRESETS[kind]
    skill = dict(chat_skill or {})
    skill.update(
        {
            "name": name,
            "@odata.type": odata,
            "context": "/document",
            "description": f"{PREFIX}{kind}:v1",
        }
    )
    outputs = OUTPUTS[kind]
    if kind == "entities":
        outputs = {ENTITY_OUTPUTS[c]: OUTPUTS[kind][ENTITY_OUTPUTS[c]] for c in config.categories}
        skill.update(categories=config.categories, minimumPrecision=config.minimum_precision)
    if kind not in {"summary", "classification"}:
        skill["inputs"] = [{"name": "text", "source": config.input_path}]
    else:
        for item in skill["inputs"]:
            if item["name"] == "text":
                item["source"] = config.input_path
    if hasattr(config, "language_code"):
        skill["defaultLanguageCode"] = config.language_code
    skill["outputs"] = [
        {"name": output, "targetName": target} for output, target in outputs.items()
    ]
    fields = []
    mappings = []
    for target in outputs.values():
        field = {
            "name": target,
            "type": "Collection(Edm.String)"
            if kind in {"entities", "key_phrases"}
            else "Edm.String",
            "searchable": True,
            "retrievable": True,
        }
        if kind != "summary":
            field.update(filterable=True, facetable=True)
        fields.append(field)
        mappings.append({"sourceFieldName": f"/document/{target}", "targetFieldName": target})
    return skill, fields, mappings
