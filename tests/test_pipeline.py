from __future__ import annotations

from collections.abc import Callable

from src.generation.generator import REFUSAL
from src.ingestion.chunking import Chunk
from src.pipeline import BASELINE, DIAGNOSTIC, Pipeline
from src.retrieval.index import HybridIndex
from tests.conftest import ScriptedLLM


class OverlapReranker:
    """Score = share of query words found in the passage."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        q = set(query.lower().replace("?", "").split())
        return [len(q & set(t.lower().split())) / len(q) for t in texts]


class NoConflicts:
    def contradiction_probs(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [0.0] * len(pairs)


def seeded(index: HybridIndex, chunk_factory: Callable[..., Chunk]) -> HybridIndex:
    index.upsert([chunk_factory("the recommended dose of apixaban is 5 mg taken orally twice daily")])
    return index


def test_gate_refuses_before_generation_when_nothing_is_covered(
        memory_index: HybridIndex, chunk_factory: Callable[..., Chunk]) -> None:
    llm = ScriptedLLM([{"sub_questions": ["wholesale price of apixaban in Poland"]}])
    p = Pipeline(DIAGNOSTIC, seeded(memory_index, chunk_factory), llm, OverlapReranker(), NoConflicts())  # type: ignore[arg-type]
    out = p.run("What is the wholesale price of apixaban in Poland?")
    assert out["outcome"] == "gate_refused" and out["answer"] == REFUSAL
    assert len(llm.calls) == 1  # decomposition only: the generator was never called


def test_diagnostic_answers_with_verified_quotes(memory_index: HybridIndex,
                                                 chunk_factory: Callable[..., Chunk]) -> None:
    llm = ScriptedLLM([
        {"sub_questions": ["what is the recommended dose of apixaban"]},
        {"status": "answered", "claims": [{"text": "5 mg twice daily", "citations": [
            {"passage": 1, "quote": "5 mg taken orally twice daily"}]}]},
    ])
    p = Pipeline(DIAGNOSTIC, seeded(memory_index, chunk_factory), llm, OverlapReranker(), NoConflicts())  # type: ignore[arg-type]
    out = p.run("What is the recommended dose of apixaban?")
    assert out["outcome"] == "answered"
    assert out["gate"]["completeness"] == 1.0
    assert out["grounding"] is not None and out["grounding"]["first_attempt_valid"]
    assert "[1]" in out["answer"]


def test_baseline_skips_gates_and_grounding(memory_index: HybridIndex, chunk_factory: Callable[..., Chunk]) -> None:
    llm = ScriptedLLM(["The dose is 5 mg twice daily [1]."])
    p = Pipeline(BASELINE, seeded(memory_index, chunk_factory), llm, OverlapReranker(), None)  # type: ignore[arg-type]
    out = p.run("What is the recommended dose of apixaban?")
    assert out["gate"] == {"enabled": False}
    assert out["outcome"] == "answered" and out["structured"] is None
    assert len(llm.calls) == 1
