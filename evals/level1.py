"""Level 1: deterministic checks over recorded traces. No model calls; same input, same numbers.

The five structural failure modes measured here:
  1. retrieval_miss       no passage from a current label contains a gold fact string
  2. stale_evidence       the answer cites a passage from a superseded label version
  3. silent_contradiction the passages contain a detected cross-version conflict and the answer
                          neither refuses nor discloses it (the "context collapse" rate)
  4. ungrounded_number    the answer states a dose/percentage/duration found in no passage
  5. abstention_error     an out-of-scope question is answered, or an answerable one is refused

    python -m evals.level1 baseline diagnostic
"""

from __future__ import annotations

import argparse
import json
import re
from statistics import median
from typing import Any

from evals.common import ANSWERABLE, RESULTS, TRACES, percentile, read_jsonl
from src.generation.generator import cited_passages
from src.generation.schema import quote_in_passage

REFUSED = {"refused", "gate_refused"}
_NUMBER = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)\s?(mg|mcg|µg|mL|g|%|kg|hours?|days?|weeks?|months?|"
                     r"years?|mmol/L|mL/min(?:/1\.73 ?m2)?)\b", re.I)
_DISCLOSURE = re.compile(r"(conflict|discrepan|differ|inconsisten|superseded|outdated|older|earlier|previous|"
                         r"newer|updated|revised)\b.{0,60}\b(label|version|labeling)|label version conflict",
                         re.I | re.S)


def _norm_num(n: str, unit: str) -> str:
    return f"{n.replace(',', '')} {unit.lower().rstrip('s')}"


def numbers(text: str) -> set[str]:
    return {_norm_num(n, u) for n, u in _NUMBER.findall(text)}


def answer_body(t: dict[str, Any]) -> str:
    """The claims only: rendered source lists repeat label dates and would count as numbers."""
    if t.get("structured"):
        return " ".join(c["text"] for c in t["structured"]["claims"])
    return str(t["answer"])


def cited(t: dict[str, Any]) -> list[int]:
    if t.get("structured"):
        return sorted({c["passage"] for cl in t["structured"]["claims"] for c in cl["citations"]})
    return cited_passages(t["answer"])


def disclosed_conflict(t: dict[str, Any]) -> bool:
    if t.get("structured"):
        return bool(t["structured"].get("conflict_note"))
    return bool(_DISCLOSURE.search(t["answer"]))


def check(t: dict[str, Any]) -> dict[str, Any]:
    cat = t["category"]
    refused = t["outcome"] in REFUSED
    passages = t["passages"]
    ranked = sorted(passages, key=lambda p: p["score"], reverse=True)
    facts_lc = [f.lower() for f in t["must_contain"]]
    gold = [i for i, p in enumerate(ranked, 1)  # evidence hit: a current passage holding a gold fact
            if p["is_current"] and any(f in " ".join(p["text"].split()).lower() for f in facts_lc)]
    body = answer_body(t)
    supported = " ".join(p["text"] for p in passages)
    unsupported = sorted(numbers(body) - numbers(supported)) if not refused else []
    cites = [n for n in cited(t) if 1 <= n <= len(passages)]
    stale = [n for n in cites if passages[n - 1]["is_current"] is False]
    conflicts = bool(t["observed_conflicts"])
    facts = [f.lower() in body.lower() for f in t["must_contain"]]
    r: dict[str, Any] = {
        "trace_id": t["trace_id"], "category": cat, "outcome": t["outcome"], "refused": refused,
        "has_gold": bool(t["must_contain"]) and cat in ANSWERABLE,
        "hit": bool(gold), "rr": 1 / gold[0] if gold else 0.0,
        "fact_recall": sum(facts) / len(facts) if facts else None,
        "unsupported_numbers": unsupported, "stale_citations": stale,
        "conflict_in_context": conflicts,
        "silent_contradiction": conflicts and not refused and not disclosed_conflict(t),
        "partial_flagged": bool(t.get("structured") and t["structured"].get("not_covered")),
        "latency_s": t["latency_s"], "tokens": sum(t["tokens"].values()) if t.get("tokens") else 0,
    }
    if t.get("structured"):
        quotes = [(c["passage"], c["quote"]) for cl in t["structured"]["claims"] for c in cl["citations"]]
        r["quotes"] = len(quotes)
        r["quotes_verified"] = sum(1 <= n <= len(passages) and quote_in_passage(q, passages[n - 1]["text"])
                                   for n, q in quotes)
        r["first_attempt_valid"] = t["grounding"]["first_attempt_valid"] if t.get("grounding") else None
        r["attempts"] = t["grounding"]["attempts"] if t.get("grounding") else 0
        r["salvaged"] = t["grounding"]["salvaged"] if t.get("grounding") else False
    r["failure_modes"] = [m for m, bad in (
        ("retrieval_miss", r["has_gold"] and not r["hit"]),
        ("stale_evidence", bool(stale) and not refused),
        ("silent_contradiction", r["silent_contradiction"]),
        ("ungrounded_number", bool(unsupported)),
        ("abstention_error", (cat == "out_of_scope" and not refused) or (cat in ANSWERABLE and refused)),
    ) if bad]
    return r


def _rate(num: int, den: int) -> dict[str, Any]:
    return {"n": num, "of": den, "rate": round(num / den, 3) if den else None}


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ans = [r for r in rows if r["category"] in ANSWERABLE]
    oos = [r for r in rows if r["category"] == "out_of_scope"]
    part = [r for r in rows if r["category"] == "partial"]
    gold = [r for r in rows if r["has_gold"]]
    answered = [r for r in rows if not r["refused"]]
    conflict = [r for r in rows if r["conflict_in_context"] and not r["refused"]]
    facts = [r["fact_recall"] for r in ans if r["fact_recall"] is not None]
    lat = [r["latency_s"] for r in rows]
    s: dict[str, Any] = {
        "traces": len(rows),
        "hit_at_k": round(sum(r["hit"] for r in gold) / len(gold), 3) if gold else None,
        "mrr": round(sum(r["rr"] for r in gold) / len(gold), 3) if gold else None,
        "fact_recall": round(sum(facts) / len(facts), 3) if facts else None,
        "correct_refusals": _rate(sum(r["refused"] for r in oos), len(oos)),
        "false_refusals": _rate(sum(r["refused"] for r in ans), len(ans)),
        "partial_refused_entirely": _rate(sum(r["refused"] for r in part), len(part)),
        "partial_flagged_missing_part": _rate(sum(r["partial_flagged"] for r in part), len(part)),
        "silent_contradiction": _rate(sum(r["silent_contradiction"] for r in conflict), len(conflict)),
        "stale_evidence": _rate(sum(bool(r["stale_citations"]) for r in answered), len(answered)),
        "ungrounded_number": _rate(sum(bool(r["unsupported_numbers"]) for r in answered), len(answered)),
        "latency_median_s": round(median(lat), 2) if lat else None,
        "latency_p95_s": round(percentile(lat, 95), 2),
        "tokens_mean": round(sum(r["tokens"] for r in rows) / len(rows)) if rows else 0,
        "failure_modes": {m: sum(m in r["failure_modes"] for r in rows) for m in (
            "retrieval_miss", "stale_evidence", "silent_contradiction", "ungrounded_number", "abstention_error")},
    }
    quoted = [r for r in rows if "quotes" in r and not r["refused"]]
    if quoted:
        s["verbatim_grounding"] = _rate(sum(r["quotes_verified"] for r in quoted), sum(r["quotes"] for r in quoted))
        s["first_attempt_valid"] = _rate(sum(bool(r["first_attempt_valid"]) for r in quoted), len(quoted))
        s["salvaged"] = _rate(sum(bool(r["salvaged"]) for r in quoted), len(quoted))
        s["mean_attempts"] = round(sum(r["attempts"] for r in quoted) / len(quoted), 2)
    return s


def regressions(summaries: dict[str, dict[str, Any]]) -> list[str]:
    """CI gate: relative checks of the diagnostic pipeline against the baseline on the same questions."""
    d, b = summaries.get("diagnostic"), summaries.get("baseline")
    if d is None or b is None:
        return ["both 'baseline' and 'diagnostic' summaries are required"]
    errs = []
    vg = d.get("verbatim_grounding", {})
    if vg.get("rate") != 1.0:
        errs.append(f"diagnostic verbatim grounding is {vg} (every final quote must verify)")
    if (d["silent_contradiction"]["rate"] or 0) > (b["silent_contradiction"]["rate"] or 0):
        errs.append("diagnostic has a higher silent-contradiction rate than the baseline")
    if d["correct_refusals"]["n"] < b["correct_refusals"]["n"]:
        errs.append("diagnostic refuses fewer out-of-scope questions than the baseline")
    if d["stale_evidence"]["n"] > 0:
        errs.append(f"diagnostic cited superseded labels: {d['stale_evidence']}")
    return errs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--assert", dest="check", action="store_true", help="exit 1 on a regression (CI)")
    args = ap.parse_args()
    out: dict[str, Any] = {}
    for name in args.configs:
        rows = [check(t) for t in read_jsonl(TRACES / f"{name}.jsonl")]
        out[name] = {"summary": summarise(rows), "rows": rows}
        print(name, json.dumps(out[name]["summary"], indent=1))
    if args.check:
        errs = regressions({k: v["summary"] for k, v in out.items()})
        for e in errs:
            print("REGRESSION:", e)
        raise SystemExit(1 if errs else 0)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "level1.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
