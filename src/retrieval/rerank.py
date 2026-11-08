"""Cross-encoder reranking and the compound router."""

from __future__ import annotations

import math
import re
from typing import Any, Protocol

from src.retrieval.index import Hit

# Generic and brand names in the corpus. A question naming none of them is searched across all labels.
COMPOUND_ALIASES: dict[str, tuple[str, ...]] = {
    "semaglutide": ("semaglutide", "ozempic"),
    "metformin": ("metformin", "glucophage"),
    "atorvastatin": ("atorvastatin", "lipitor"),
    "apixaban": ("apixaban", "eliquis"),
    "sertraline": ("sertraline", "zoloft"),
}


def route_compounds(question: str) -> list[str]:
    q = question.lower()
    return [c for c, names in COMPOUND_ALIASES.items() if any(re.search(rf"\b{n}\b", q) for n in names)]


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """ms-marco MiniLM cross-encoder; logits are squashed to (0, 1) so gate thresholds are readable."""

    def __init__(self, model: str) -> None:
        from sentence_transformers import CrossEncoder

        self._model: Any = CrossEncoder(model, device="cpu")

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        logits = self._model.predict([(query, t) for t in texts])
        return [1 / (1 + math.exp(-float(x))) for x in logits]


def rerank(reranker: Reranker | None, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
    if reranker is None:
        return hits[:top_k]
    scores = reranker.score(query, [h.text for h in hits])
    ranked = sorted(zip(hits, scores, strict=True), key=lambda x: x[1], reverse=True)[:top_k]
    return [Hit(h.chunk_id, s, h.payload) for h, s in ranked]
