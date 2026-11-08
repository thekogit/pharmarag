from __future__ import annotations

from collections import Counter
from typing import Any

from evals.calibrate import cohen_kappa, prf
from evals.common import load_questions, split_of
from evals.judge import _RESULT, ABSTENTION, CONFLICT, FAITHFULNESS, applicable
from evals.level1 import check, numbers, regressions, summarise


def trace(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "trace_id": "diagnostic:q1", "qid": "q1", "category": "single", "compound": "apixaban",
        "gold_section": "DOSAGE", "must_contain": ["5 mg"], "outcome": "answered",
        "answer": "The dose is 5 mg twice daily [1].", "structured": None, "grounding": None,
        "observed_conflicts": [], "latency_s": 1.0, "tokens": {"calls": 1, "prompt_tokens": 10, "completion_tokens": 5},
        "passages": [{"n": 1, "score": 0.9, "is_current": True, "compound": "apixaban", "section": "DOSAGE",
                      "text": "The recommended dose is 5 mg taken orally twice daily."},
                     {"n": 2, "score": 0.5, "is_current": False, "compound": "apixaban", "section": "DOSAGE",
                      "text": "Older label: 2.5 mg twice daily."}],
    }
    return {**base, **kw}


def test_numbers_normalise_units() -> None:
    assert numbers("Take 2.5 mg for 35 days, 1,000 mg max") == {"2.5 mg", "35 day", "1000 mg"}


def test_level1_hit_stale_and_ungrounded() -> None:
    r = check(trace())
    assert r["hit"] and r["rr"] == 1.0 and r["fact_recall"] == 1.0 and r["failure_modes"] == []
    stale = check(trace(answer="The dose is 2.5 mg [2] or 10 mg."))
    assert stale["stale_citations"] == [2]
    assert stale["unsupported_numbers"] == ["10 mg"]
    assert set(stale["failure_modes"]) == {"stale_evidence", "ungrounded_number"}


def test_level1_silent_contradiction_and_abstention() -> None:
    conflict = [{"compound": "apixaban"}]
    silent = check(trace(observed_conflicts=conflict))
    disclosed = check(trace(observed_conflicts=conflict, answer="5 mg [1]; an older label version differs."))
    assert silent["silent_contradiction"] and not disclosed["silent_contradiction"]
    oos = check(trace(category="out_of_scope", must_contain=[]))
    assert "abstention_error" in oos["failure_modes"]
    assert check(trace(category="out_of_scope", must_contain=[], outcome="gate_refused"))["failure_modes"] == []


def test_summary_and_regression_gate() -> None:
    rows = [check(trace()), check(trace(category="out_of_scope", must_contain=[], outcome="refused"))]
    s = summarise(rows)
    assert s["hit_at_k"] == 1.0 and s["correct_refusals"] == {"n": 1, "of": 1, "rate": 1.0}
    good = {**s, "verbatim_grounding": {"n": 3, "of": 3, "rate": 1.0}}
    assert regressions({"baseline": s, "diagnostic": good}) == []
    bad = {**good, "verbatim_grounding": {"n": 2, "of": 3, "rate": 0.667}}
    assert regressions({"baseline": s, "diagnostic": bad})


def test_split_is_deterministic_and_roughly_15_42_42() -> None:
    qs = load_questions()
    c = Counter(split_of(q["id"]) for q in qs)
    assert c["train"] + c["dev"] + c["test"] == len(qs)
    assert 0.08 < c["train"] / len(qs) < 0.25 and split_of("q001") == split_of("q001")


def test_kappa_and_prf() -> None:
    h = ["FAIL", "FAIL", "PASS", "PASS"]
    assert cohen_kappa(h, h) == 1.0
    assert cohen_kappa(h, ["PASS", "PASS", "FAIL", "FAIL"]) == -1.0
    assert prf(h, ["FAIL", "PASS", "FAIL", "PASS"]) == {"precision": 0.5, "recall": 0.5, "positives": 2, "n": 4}


def test_judge_result_parsing_and_criteria() -> None:
    assert _RESULT.search("good answer [RESULT] 5").group(1) == "5"  # type: ignore[union-attr]
    assert _RESULT.search('Feedback: ... [RESULT] "B"').group(1) == "B"  # type: ignore[union-attr]
    assert "Score 3: Not used" in FAITHFULNESS.rubric()
    assert applicable(trace(outcome="refused")) == [ABSTENTION]
    assert applicable(trace(observed_conflicts=[{}])) == [ABSTENTION, FAITHFULNESS, CONFLICT]
