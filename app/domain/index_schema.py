from typing import Any


def schema(name: str, dimensions: int, description: str) -> dict[str, Any]:
    """Build a chunk index with BM25, HNSW, metadata, ACL and semantic fields."""
    fields = []
    for field in (
        "chunk_id",
        "document_id",
        "title",
        "content",
        "domain",
        "source",
        "document_type",
        "classification",
        "source_uri",
    ):
        spec = {"name": field, "type": "Edm.String", "retrievable": True}
        if field == "chunk_id":
            spec["key"] = True
        if field in ("title", "content"):
            spec.update(searchable=True, analyzer="standard.lucene")
        else:
            spec["filterable"] = True
        fields.append(spec)
    fields.extend(
        [
            {"name": "page_start", "type": "Edm.Int32", "retrievable": True},
            {"name": "is_public", "type": "Edm.Boolean", "filterable": True},
            {"name": "allowed_group_ids", "type": "Collection(Edm.String)", "filterable": True},
            {
                "name": "content_vector",
                "type": "Collection(Edm.Single)",
                "searchable": True,
                "retrievable": False,
                "dimensions": dimensions,
                "vectorSearchProfile": "vector-profile",
            },
        ]
    )
    return {
        "name": name,
        "description": description,
        "fields": fields,
        "vectorSearch": {
            "algorithms": [
                {"name": "hnsw", "kind": "hnsw", "hnswParameters": {"metric": "cosine"}}
            ],
            "profiles": [{"name": "vector-profile", "algorithm": "hnsw"}],
        },
        "semantic": {
            "configurations": [
                {
                    "name": "semantic-config",
                    "prioritizedFields": {
                        "titleField": {"fieldName": "title"},
                        "prioritizedContentFields": [{"fieldName": "content"}],
                    },
                }
            ]
        },
    }
