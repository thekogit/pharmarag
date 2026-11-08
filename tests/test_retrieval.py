from __future__ import annotations

from collections.abc import Callable

import pytest

from src.ingestion.chunking import Chunk
from src.retrieval.gates import completeness, detect_contradictions, reorder
from src.retrieval.index import Hit, HybridIndex, collection_name, mark_current
from src.retrieval.rerank import rerank, route_compounds


def hit(cid: str, score: float, text: str = "x", **payload: str) -> Hit:
    base = {"text": text, "compound": "apixaban", "set_id": "new", "effective_date": "2025-01-01"}
    return Hit(cid, score, {**base, **payload})


def test_collection_name_encodes_embedding_model() -> None:
    assert collection_name("fda_labels", "bge-m3") == "fda_labels__bge-m3"
    assert collection_name("fda_labels", "nomic-embed-text:latest") == "fda_labels__nomic-embed-text-latest"


def test_mark_current_picks_latest_label_per_compound(chunk_factory: Callable[..., Chunk]) -> None:
    old = chunk_factory("old", set_id="o", date="2021-01-14", compound="semaglutide")
    new = chunk_factory("new", set_id="n", date="2026-06-01", compound="semaglutide")
    other = chunk_factory("only", set_id="m", date="2010-01-01", compound="metformin")
    cur = mark_current([old, new, other])
    assert (cur[old.chunk_id], cur[new.chunk_id], cur[other.chunk_id]) == (False, True, True)


def test_freshness_and_compound_filters(memory_index: HybridIndex, chunk_factory: Callable[..., Chunk]) -> None:
    chunks = [
        chunk_factory("maximum dose is 1 mg once weekly", set_id="o", date="2021-01-14", compound="semaglutide"),
        chunk_factory("maximum dose is 2 mg once weekly", set_id="n", date="2026-06-01", compound="semaglutide", n=1),
        chunk_factory("starting dose is 500 mg twice daily", set_id="m", compound="metformin", n=2),
    ]
    memory_index.upsert(chunks)
    fresh = memory_index.search("maximum dose once weekly", compounds=["semaglutide"])
    assert [h.payload["set_id"] for h in fresh] == ["n"]
    stale = memory_index.search("maximum dose once weekly", compounds=["semaglutide"], fresh_only=False)
    assert {h.payload["set_id"] for h in stale} == {"n", "o"}
    for mode in ("dense", "sparse", "hybrid"):
        assert memory_index.search("starting dose", mode=mode, compounds=["metformin"])  # type: ignore[arg-type]


def test_search_on_missing_collection_explains_rebuild(memory_index: HybridIndex) -> None:
    memory_index.collection = "absent__other-model"
    with pytest.raises(RuntimeError, match="build_index"):
        memory_index.search("q")


def test_route_compounds_uses_brand_names() -> None:
    assert route_compounds("What is the max dose of Ozempic?") == ["semaglutide"]
    assert route_compounds("Compare Eliquis and metformin") == ["metformin", "apixaban"]
    assert route_compounds("What is the price of tirzepatide?") == []


class FixedReranker:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [self.scores[t] for t in texts]


def test_rerank_orders_and_replaces_scores() -> None:
    hits = [hit("a", 0.1, "A"), hit("b", 0.2, "B"), hit("c", 0.3, "C")]
    out = rerank(FixedReranker({"A": 0.9, "B": 0.2, "C": 0.5}), "q", hits, top_k=2)
    assert [(h.chunk_id, h.score) for h in out] == [("a", 0.9), ("c", 0.5)]


def test_completeness_scores_each_sub_question() -> None:
    comp = completeness({"dose?": [hit("a", 0.9)], "price?": [hit("b", 0.2)], "none?": []})
    assert comp.score == pytest.approx(1 / 3)
    assert comp.missing == ["price?", "none?"]


def test_reorder_puts_best_at_both_ends() -> None:
    hits = [hit(str(i), s) for i, s in enumerate([0.1, 0.9, 0.5, 0.7, 0.3])]
    assert [h.score for h in reorder(hits)] == [0.9, 0.5, 0.1, 0.3, 0.7]


class KeywordNli:
    """Contradiction iff the two sentences disagree on '1 mg' vs '2 mg'."""

    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []

    def contradiction_probs(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.pairs += pairs
        return [0.95 if ("1 mg" in a) != ("1 mg" in b) else 0.01 for a, b in pairs]


def test_contradictions_only_across_versions_of_same_compound() -> None:
    new = hit("n", 0.9, "The maximum recommended dosage of semaglutide is 2 mg once weekly.",
              compound="semaglutide", set_id="n", effective_date="2026-06-01")
    old = hit("o", 0.8, "The maximum recommended dosage of semaglutide is 1 mg once weekly.",
              compound="semaglutide", set_id="o", effective_date="2021-01-14")
    same_label = hit("s", 0.7, "The maximum recommended dosage of semaglutide is 1 mg once weekly here.",
                     compound="semaglutide", set_id="n", effective_date="2026-06-01")
    other_drug = hit("x", 0.6, "The maximum recommended dosage of metformin is 1 mg once weekly.",
                     compound="metformin", set_id="m", effective_date="2026-01-01")
    nli = KeywordNli()
    rep = detect_contradictions([new, old, same_label, other_drug], nli)
    assert len(rep.conflicts) == 1
    c = rep.conflicts[0]
    assert (c.newer["chunk_id"], c.older["chunk_id"]) == ("n", "o")
    assert c.newer["effective_date"] > c.older["effective_date"]
    assert all(("metformin" in a) == ("metformin" in b) for a, b in nli.pairs)


def test_revision_filter_needs_a_changed_number_or_negation() -> None:
    from src.retrieval.gates import _candidate, _differs

    assert _differs("Use is not recommended in pregnancy.", "Use is recommended in pregnancy.")
    assert not _differs("Studied in monkeys and rabbits.", "Studied in rabbits and monkeys.")
    assert _candidate("The maximum recommended dosage is 2 mg once weekly.",
                      "The maximum recommended dosage is 1 mg once weekly.")
    assert not _candidate("Combined P-gp and strong CYP3A4 inhibitors increase blood levels of apixaban.",
                          "Patients had a mean age of 59 years and 56% were men.")
