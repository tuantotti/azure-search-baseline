"""Single-query, reproducible comparisons using one shared embedding."""

import random
import statistics
from typing import Any

from app.core.errors import ApplicationError
from app.domain.metrics import metrics
from app.schemas.search import CompareRequest

from .retrieval import RetrievalService


class ExperimentService:
    def __init__(self, retrieval: RetrievalService) -> None:
        self._retrieval = retrieval

    async def compare(
        self,
        index: str,
        request: CompareRequest,
        version: str | None = None,
        source_token: str | None = None,
    ) -> dict[str, Any]:
        self._retrieval.build_options(request, request.modes[0], None)
        vector, embedding_ms = None, 0.0
        if any(mode != "bm25" for mode in request.modes):
            embedded = await self._retrieval.embeddings.embed([request.query])
            vector = embedded["azure"]["data"][0]["embedding"]
            embedding_ms = embedded["elapsed_ms"]
        jobs = [(mode, repeat) for repeat in range(request.repeats) for mode in request.modes]
        random.Random(request.seed).shuffle(jobs)
        runs = []
        for mode, repeat in jobs:
            options = self._retrieval.build_options(request, mode, vector)
            if request.qrels is not None and options.select and "*" not in options.select:
                if request.document_id_field not in options.select:
                    options.select.append(request.document_id_field)
            result = await self._retrieval.repository.search(index, options, version, source_token)
            self._retrieval.verify_semantic(mode, result)
            scores = None
            if request.qrels is not None:
                hits = result["azure"]["value"]
                if any(request.document_id_field not in hit for hit in hits):
                    raise ApplicationError("document_id_field must be retrievable", 422)
                normalized = [{"document_id": hit[request.document_id_field]} for hit in hits]
                scores = {
                    "recall_at_top": metrics(normalized, request.qrels, options.top)["recall"],
                    "mrr_at_10": metrics(normalized, request.qrels, 10)["mrr"],
                    "ndcg_at_10": metrics(normalized, request.qrels, 10)["ndcg"],
                }
            runs.append({"mode": mode, "repeat": repeat, "metrics": scores, **result})
        return {
            "query_embedding_ms": embedding_ms,
            "runs": runs,
            "mean_search_ms": {
                mode: statistics.mean(run["elapsed_ms"] for run in runs if run["mode"] == mode)
                for mode in request.modes
            },
        }
