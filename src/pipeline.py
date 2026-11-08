"""The answering pipeline as a LangGraph graph.

    decompose -> retrieve -> gates --(no evidence)--> refuse
                                   \\--------------> generate

Every diagnostic component can be switched off, which is how the baseline and the ablations run.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from src.config import Settings
from src.generation.generator import (
    REFUSAL,
    decompose,
    generate_baseline,
    generate_grounded,
    render,
)
from src.generation.llm import LLM
from src.retrieval.gates import NliModel, completeness, detect_contradictions, reorder
from src.retrieval.index import Hit, HybridIndex, Mode
from src.retrieval.rerank import Reranker, rerank, route_compounds

CANDIDATES = 20
PER_SUB_QUESTION = 4
MAX_PASSAGES = 8
BASELINE_PASSAGES = 6


@dataclass(frozen=True)
class PipelineConfig:
    name: str = "diagnostic"
    mode: Mode = "hybrid"
    rerank: bool = True
    freshness: bool = True
    gates: bool = True
    grounding: bool = True

    @classmethod
    def from_settings(cls, s: Settings) -> PipelineConfig:
        m: Mode = s.retrieval_mode if s.retrieval_mode in ("dense", "sparse", "hybrid") else "hybrid"  # type: ignore[assignment]
        return cls("custom", m, s.rerank, s.freshness, s.gates, s.grounding)


BASELINE = PipelineConfig("baseline", "hybrid", rerank=True, freshness=False, gates=False, grounding=False)
DIAGNOSTIC = PipelineConfig()
DIAGNOSTIC_NO_FRESHNESS = replace(DIAGNOSTIC, name="diagnostic_no_freshness", freshness=False)
CONFIGS = {c.name: c for c in (BASELINE, DIAGNOSTIC, DIAGNOSTIC_NO_FRESHNESS)}


class State(TypedDict, total=False):
    question: str
    sub_questions: list[str]
    per_sub: dict[str, list[Hit]]
    hits: list[Hit]
    gate: dict[str, Any]
    answer: str
    structured: dict[str, Any] | None
    grounding: dict[str, Any] | None
    outcome: Literal["answered", "partial", "refused", "gate_refused"]
    timings: dict[str, float]


class Pipeline:
    def __init__(self, cfg: PipelineConfig, index: HybridIndex, llm: LLM,
                 reranker: Reranker | None, nli: NliModel | None) -> None:
        self.cfg = cfg
        self.index = index
        self.llm = llm
        self.reranker = reranker if cfg.rerank else None
        self.nli = nli
        self.graph = self._build()

    # --- nodes ------------------------------------------------------------------------------
    def _decompose(self, s: State) -> State:
        t0 = time.perf_counter()
        subs = decompose(self.llm, s["question"]) if self.cfg.gates else [s["question"]]
        return {"sub_questions": subs, "timings": {**s.get("timings", {}), "decompose": time.perf_counter() - t0}}

    def _retrieve(self, s: State) -> State:
        t0 = time.perf_counter()
        per_sub: dict[str, list[Hit]] = {}
        fallback = route_compounds(s["question"])
        k = PER_SUB_QUESTION if self.cfg.gates else BASELINE_PASSAGES
        for sq in s["sub_questions"]:
            compounds = route_compounds(sq) or fallback
            cands = self.index.search(sq, limit=CANDIDATES, mode=self.cfg.mode,
                                      fresh_only=self.cfg.freshness, compounds=compounds or None)
            per_sub[sq] = rerank(self.reranker, sq, cands, k)
        best: dict[str, Hit] = {}
        for hits in per_sub.values():
            for h in hits:
                if h.chunk_id not in best or h.score > best[h.chunk_id].score:
                    best[h.chunk_id] = h
        merged = sorted(best.values(), key=lambda h: h.score, reverse=True)[:MAX_PASSAGES]
        return {"per_sub": per_sub, "hits": merged,
                "timings": {**s["timings"], "retrieve": time.perf_counter() - t0}}

    def _gates(self, s: State) -> State:
        t0 = time.perf_counter()
        if not self.cfg.gates:
            return {"gate": {"enabled": False}, "timings": {**s["timings"], "gates": 0.0}}
        comp = completeness(s["per_sub"]) if self.reranker else None
        hits = reorder(s["hits"])
        contra = detect_contradictions(hits, self.nli) if self.nli else None
        gate = {
            "enabled": True,
            "completeness": comp.score if comp else None,
            "coverage": [asdict(c) for c in comp.coverage] if comp else [],
            "missing": comp.missing if comp else [],
            "pairs_checked": contra.pairs_checked if contra else 0,
            "conflicts": [asdict(c) for c in contra.conflicts] if contra else [],
        }
        return {"hits": hits, "gate": gate, "timings": {**s["timings"], "gates": time.perf_counter() - t0}}

    def _route_after_gates(self, s: State) -> str:
        g = s["gate"]
        if not s["hits"] or (g.get("enabled") and g.get("completeness") == 0.0):
            return "refuse"
        return "generate"

    def _refuse(self, s: State) -> State:
        return {"answer": REFUSAL, "outcome": "gate_refused", "structured": None, "grounding": None}

    def _generate(self, s: State) -> State:
        t0 = time.perf_counter()
        hits = s["hits"]
        if not self.cfg.grounding:
            answer = generate_baseline(self.llm, s["question"], hits)
            outcome: Literal["answered", "refused"] = "refused" if REFUSAL in answer else "answered"
            return {"answer": answer, "outcome": outcome, "structured": None, "grounding": None,
                    "timings": {**s["timings"], "generate": time.perf_counter() - t0}}
        from src.retrieval.gates import Conflict
        conflicts = [Conflict(**c) for c in s["gate"].get("conflicts", [])]
        res = generate_grounded(self.llm, s["question"], hits, missing=s["gate"].get("missing", []),
                                conflicts=conflicts)
        return {"answer": render(res.answer, hits), "outcome": res.answer.status,
                "structured": res.answer.model_dump(),
                "grounding": {"attempts": res.attempts, "first_attempt_valid": res.first_attempt_valid,
                              "salvaged": res.salvaged, "errors": res.validation_errors},
                "timings": {**s["timings"], "generate": time.perf_counter() - t0}}

    def _build(self) -> Any:
        g: Any = StateGraph(State)  # LangGraph overloads reject bound methods as node actions
        g.add_node("decompose", self._decompose)
        g.add_node("retrieve", self._retrieve)
        g.add_node("gates", self._gates)
        g.add_node("refuse", self._refuse)
        g.add_node("generate", self._generate)
        g.set_entry_point("decompose")
        g.add_edge("decompose", "retrieve")
        g.add_edge("retrieve", "gates")
        g.add_conditional_edges("gates", self._route_after_gates, {"refuse": "refuse", "generate": "generate"})
        g.add_edge("refuse", END)
        g.add_edge("generate", END)
        return g.compile()

    def run(self, question: str) -> State:
        out: State = self.graph.invoke({"question": question, "timings": {}})
        return out


@dataclass
class Components:
    """Heavy, shareable parts. Embedded Qdrant locks its folder, so one process holds one client."""

    index: HybridIndex
    llm: LLM
    reranker: Reranker
    nli: NliModel

    @classmethod
    def load(cls, s: Settings) -> Components:
        from src.retrieval.embeddings import Bm25Embedder, OllamaEmbedder
        from src.retrieval.gates import CrossEncoderNli
        from src.retrieval.rerank import CrossEncoderReranker

        index = HybridIndex.from_settings(s, OllamaEmbedder(s.ollama_host, s.embed_model), Bm25Embedder())
        return cls(index, LLM.from_settings(s), CrossEncoderReranker(s.reranker_model), CrossEncoderNli(s.nli_model))


def build_pipeline(cfg: PipelineConfig | None = None, s: Settings | None = None,
                   components: Components | None = None) -> Pipeline:
    """Wire the real components (Ollama embeddings, BM25, Qdrant, cross-encoders, cloud LLM)."""
    s = s or Settings.load()
    cfg = cfg or PipelineConfig.from_settings(s)
    c = components or Components.load(s)
    return Pipeline(cfg, c.index, c.llm, c.reranker, c.nli if cfg.gates else None)
