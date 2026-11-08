"""Retrieval-only ablation (no LLM): evidence Hit@5 and MRR@5 over the answerable questions.

A hit is a passage from a *current* label that contains one of the question's gold fact strings.
Variants: dense-only nomic-embed-text (the rejected baseline), dense-only bge-m3, sparse BM25,
hybrid RRF, hybrid + cross-encoder rerank, and hybrid + rerank with the freshness filter off
(which also reports how many top-5 passages come from superseded labels).

    python -m scripts.build_index --embed-model nomic-embed-text   # once, for the nomic baseline
    python -m evals.retrieval_ablation
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from datetime import date
from typing import Any

from qdrant_client import QdrantClient

from evals.common import ANSWERABLE, RESULTS, load_questions, split_of
from src import log
from src.config import Settings
from src.retrieval.embeddings import Bm25Embedder, OllamaEmbedder
from src.retrieval.index import Hit, HybridIndex, Mode, collection_name
from src.retrieval.rerank import CrossEncoderReranker, Reranker, rerank, route_compounds

K = 5
POOL = 20


def evidence_rank(facts: list[str], hits: list[Hit]) -> int | None:
    fl = [f.lower() for f in facts]
    for i, h in enumerate(hits, 1):
        if h.payload.get("is_current") and any(f in " ".join(h.text.split()).lower() for f in fl):
            return i
    return None


def run_variant(index: HybridIndex, qs: list[dict[str, Any]], mode: Mode, reranker: Reranker | None,
                fresh: bool) -> dict[str, Any]:
    ranks: list[int | None] = []
    stale = total = 0
    t0 = time.perf_counter()
    for q in qs:
        hits = index.search(q["question"], limit=POOL, mode=mode, fresh_only=fresh,
                            compounds=route_compounds(q["question"]) or None)
        top = rerank(reranker, q["question"], hits, K)
        ranks.append(evidence_rank(q["must_contain"], top))
        stale += sum(not h.payload.get("is_current") for h in top)
        total += len(top)
    by_split: dict[str, float] = {}
    for sp in ("train", "dev", "test"):
        idx = [i for i, q in enumerate(qs) if split_of(q["id"]) == sp]
        if idx:
            by_split[sp] = round(sum(ranks[i] is not None for i in idx) / len(idx), 3)
    return {"hit_at_5": round(sum(r is not None for r in ranks) / len(qs), 3),
            "mrr_at_5": round(sum(1 / r for r in ranks if r) / len(qs), 3),
            "hit_at_5_by_split": by_split,
            "superseded_share_top5": round(stale / total, 3) if total else 0.0,
            "sec_per_query": round((time.perf_counter() - t0) / len(qs), 3)}


def main() -> None:
    log.setup()
    s = Settings.load()
    qs = [q for q in load_questions() if q["category"] in ANSWERABLE]
    client = QdrantClient(path=s.qdrant_path)
    sparse = Bm25Embedder()
    bge = HybridIndex(client, collection_name(s.collection, s.embed_model),
                      OllamaEmbedder(s.ollama_host, s.embed_model), sparse)
    nomic_s = replace(s, embed_model="nomic-embed-text")
    nomic = HybridIndex(client, collection_name(s.collection, nomic_s.embed_model),
                        OllamaEmbedder(s.ollama_host, nomic_s.embed_model), sparse)
    rr = CrossEncoderReranker(s.reranker_model)
    variants: list[tuple[str, HybridIndex, Mode, Reranker | None, bool]] = [
        ("dense nomic-embed-text", nomic, "dense", None, True),
        ("dense bge-m3", bge, "dense", None, True),
        ("sparse BM25", bge, "sparse", None, True),
        ("hybrid bge-m3 + BM25 (RRF)", bge, "hybrid", None, True),
        (f"hybrid + rerank ({s.reranker_model.split('/')[-1]})", bge, "hybrid", rr, True),
        ("hybrid + rerank, freshness off", bge, "hybrid", rr, False),
    ]
    out: dict[str, Any] = {"date": date.today().isoformat(), "questions": len(qs), "k": K, "pool": POOL,
                           "variants": {}}
    for name, index, mode, reranker, fresh in variants:
        if not index.exists():
            out["variants"][name] = {"skipped": f"collection {index.collection} not built"}
            continue
        out["variants"][name] = run_variant(index, qs, mode, reranker, fresh)
        print(name, json.dumps(out["variants"][name]))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "retrieval_ablation.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
