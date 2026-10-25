import math
from typing import Any


def metrics(
    hits: list[dict[str, Any]], qrels: dict[str, int], cutoff: int
) -> dict[str, float | None]:
    """Score distinct documents present within a fixed chunk retrieval budget."""
    # Distinct documents among the first cutoff CHUNKS, preserving rank.
    ranked = list(dict.fromkeys(h["document_id"] for h in hits[:cutoff]))
    relevant = {doc for doc, grade in qrels.items() if grade > 0}
    if not relevant:
        return {"recall": None, "mrr": None, "ndcg": None}
    grades = [qrels.get(doc, 0) for doc in ranked]
    dcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(grades))
    ideal = sorted(qrels.values(), reverse=True)[:cutoff]
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return {
        "recall": len(set(ranked) & relevant) / len(relevant),
        "mrr": next((1 / (i + 1) for i, g in enumerate(grades) if g > 0), 0),
        "ndcg": dcg / idcg if idcg else 0,
    }
