"""Pre-generation gates. They run after retrieval and before any answer text is generated.

1. Completeness: every sub-question must have at least one passage above a relevance threshold.
   None covered -> refuse without calling the generator. Some covered -> the generator is told
   which parts are not covered and must say so.
2. Reordering: passages are placed by relevance at both ends of the context, where long-context
   models attend best, instead of in retrieval order.
3. Contradictions: sentence pairs from *different label versions* of the same compound are checked
   with an NLI cross-encoder. A detected conflict is passed to the generator, which must disclose
   it, and is recorded in the trace.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from src.retrieval.index import Hit

COVERAGE_THRESHOLD = 0.5  # sigmoid(cross-encoder logit) > 0.5  <=>  logit > 0
CONTRADICTION_THRESHOLD = 0.9
MIN_SHARED_TERMS = 4
MIN_JACCARD = 0.4  # pair only near-paraphrases: the same statement, possibly revised
MAX_PAIRS = 200
_NEGATION = re.compile(r"\b(not|no|never|without|contraindicated|avoid)\b", re.I)
_NUM = re.compile(r"\d+(?:\.\d+)?")

_STOP = frozenset({
    "a", "an", "the", "of", "to", "in", "for", "on", "with", "and", "or", "is", "are", "was", "were", "be",
    "been", "by", "as", "at", "from", "that", "this", "these", "those", "it", "its", "not", "no", "may", "can",
    "should", "if", "than", "then", "which", "who", "whom", "whose", "when", "while", "into", "over", "under",
    "per", "patients", "patient", "use", "used", "using", "dose", "doses", "dosage", "mg",
})


@dataclass
class SubQuestionCoverage:
    sub_question: str
    best_score: float
    covered: bool
    hit_ids: list[str]


@dataclass
class Completeness:
    coverage: list[SubQuestionCoverage]

    @property
    def score(self) -> float:
        return sum(c.covered for c in self.coverage) / len(self.coverage) if self.coverage else 0.0

    @property
    def missing(self) -> list[str]:
        return [c.sub_question for c in self.coverage if not c.covered]


def completeness(per_sub: dict[str, list[Hit]], threshold: float = COVERAGE_THRESHOLD) -> Completeness:
    cov = []
    for sq, hits in per_sub.items():
        best = max((h.score for h in hits), default=0.0)
        cov.append(SubQuestionCoverage(sq, best, best > threshold,
                                       [h.chunk_id for h in hits if h.score > threshold]))
    return Completeness(cov)


def reorder(hits: list[Hit]) -> list[Hit]:
    """Most relevant first and last, least relevant in the middle (counters lost-in-the-middle)."""
    ranked = sorted(hits, key=lambda h: h.score, reverse=True)
    front: list[Hit] = []
    back: list[Hit] = []
    for i, h in enumerate(ranked):
        (front if i % 2 == 0 else back).append(h)
    return front + back[::-1]


class NliModel(Protocol):
    def contradiction_probs(self, pairs: list[tuple[str, str]]) -> list[float]: ...


class CrossEncoderNli:
    def __init__(self, model: str) -> None:
        from sentence_transformers import CrossEncoder

        self._model: Any = CrossEncoder(model, device="cpu")
        labels = {v.lower(): int(k) for k, v in self._model.config.id2label.items()}
        self._contra = labels["contradiction"]

    def contradiction_probs(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        probs = self._model.predict(pairs, apply_softmax=True)
        return [float(p[self._contra]) for p in probs]


@dataclass
class Conflict:
    compound: str
    newer: dict[str, str]  # chunk_id, set_id, effective_date, sentence
    older: dict[str, str]
    probability: float


@dataclass
class ContradictionReport:
    pairs_checked: int = 0
    conflicts: list[Conflict] = field(default_factory=list)


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.;])\s+|\n+", text)
    return [p.strip(" |-") for p in parts if 25 <= len(p.strip()) <= 400]


def _terms(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z0-9\-]+|\d+(?:\.\d+)?", s.lower()) if w not in _STOP}


def _differs(a: str, b: str) -> bool:
    """A revision worth checking changes a number or flips a negation; rewording alone does not."""
    return set(_NUM.findall(a)) != set(_NUM.findall(b)) or bool(_NEGATION.search(a)) != bool(_NEGATION.search(b))


def _candidate(s_new: str, s_old: str) -> bool:
    tn, to = _terms(s_new), _terms(s_old)
    shared = len(tn & to)
    return (shared >= MIN_SHARED_TERMS and shared / len(tn | to) >= MIN_JACCARD
            and s_new != s_old and _differs(s_new, s_old))


def detect_contradictions(hits: list[Hit], nli: NliModel,
                          threshold: float = CONTRADICTION_THRESHOLD) -> ContradictionReport:
    """Check near-paraphrase sentence pairs from different label versions of the same compound.

    A conflict needs a contradiction probability >= threshold in both directions. Pairs are limited to
    near-paraphrases that differ in a number or a negation, because an NLI model calls almost any two
    unrelated sentences about one drug "contradictory".
    """
    report = ContradictionReport()
    candidates: list[tuple[Hit, str, Hit, str]] = []
    for a, b in itertools.combinations(hits, 2):
        pa, pb = a.payload, b.payload
        if pa["compound"] != pb["compound"] or pa["set_id"] == pb["set_id"]:
            continue
        new, old = (a, b) if pa["effective_date"] >= pb["effective_date"] else (b, a)
        for s_new in _sentences(new.text):
            for s_old in _sentences(old.text):
                if _candidate(s_new, s_old):
                    candidates.append((new, s_new, old, s_old))
    candidates = candidates[:MAX_PAIRS]
    report.pairs_checked = len(candidates)
    fwd = nli.contradiction_probs([(s_old, s_new) for _, s_new, _, s_old in candidates])
    bwd = nli.contradiction_probs([(s_new, s_old) for _, s_new, _, s_old in candidates])
    probs = [min(f, b) for f, b in zip(fwd, bwd, strict=True)]
    seen: set[tuple[str, str]] = set()
    for (new, s_new, old, s_old), p in zip(candidates, probs, strict=True):
        key = (new.chunk_id, old.chunk_id)
        if p >= threshold and key not in seen:
            seen.add(key)
            report.conflicts.append(Conflict(
                compound=str(new.payload["compound"]),
                newer={"chunk_id": new.chunk_id, "set_id": str(new.payload["set_id"]),
                       "effective_date": str(new.payload["effective_date"]), "sentence": s_new},
                older={"chunk_id": old.chunk_id, "set_id": str(old.payload["set_id"]),
                       "effective_date": str(old.payload["effective_date"]), "sentence": s_old},
                probability=round(p, 3)))
    return report
