"""Judge calibration against human labels: label sheet out, Cohen's kappa / precision / recall in.

    python -m evals.calibrate sheet    # writes evals/labels/label_sheet.csv (judge verdicts hidden)
    python -m evals.calibrate score    # reads the filled sheet, writes evals/results/calibration.json

Fill `human_label` with PASS or FAIL (and optionally `failure_notes` for open coding). The sheet does
not show the judge's verdict, so labelling is blind. FAIL is the positive class: the judge's job is to
catch failures. Splits are by question (15% train / 42.5% dev / 42.5% test); report the test split.
"""

from __future__ import annotations

import argparse
import csv
import json
from typing import Any

from evals.common import LABELS, RESULTS, TRACES, read_jsonl, split_of

SHEET = LABELS / "label_sheet.csv"
FIELDS = ["item_id", "split", "criterion", "question", "answer", "passages", "human_label", "failure_notes"]


def make_sheet() -> None:
    judged = list(read_jsonl(RESULTS / "judge_absolute.jsonl"))
    traces = {t["trace_id"]: t for f in TRACES.glob("*.jsonl") if ".limit" not in f.name for t in read_jsonl(f)}
    existing: dict[str, dict[str, str]] = {}
    if SHEET.exists():
        with SHEET.open(encoding="utf-8", newline="") as f:
            existing = {r["item_id"]: r for r in csv.DictReader(f)}
    LABELS.mkdir(parents=True, exist_ok=True)
    with SHEET.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for j in judged:
            t = traces[j["trace_id"]]
            item = f"{j['trace_id']}|{j['criterion']}"
            prev = existing.get(item, {})
            w.writerow({
                "item_id": item, "split": split_of(t["qid"]), "criterion": j["criterion"],
                "question": t["question"], "answer": t["answer"],
                "passages": "\n\n".join(f"[{p['n']}] ({p['effective_date']}, {p['section']}) {p['text']}"
                                        for p in t["passages"]),
                "human_label": prev.get("human_label", ""), "failure_notes": prev.get("failure_notes", ""),
            })
    print(f"wrote {len(judged)} items to {SHEET}")


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    labels = set(a) | set(b)
    pe = sum((a.count(lab) / n) * (b.count(lab) / n) for lab in labels)
    return None if pe == 1 else round((po - pe) / (1 - pe), 3)


def prf(human: list[str], judge: list[str], positive: str = "FAIL") -> dict[str, Any]:
    tp = sum(h == positive and j == positive for h, j in zip(human, judge, strict=True))
    fp = sum(h != positive and j == positive for h, j in zip(human, judge, strict=True))
    fn = sum(h == positive and j != positive for h, j in zip(human, judge, strict=True))
    return {"precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None,
            "positives": sum(h == positive for h in human), "n": len(human)}


def score() -> None:
    judged = {f"{j['trace_id']}|{j['criterion']}": j["verdict"] for j in read_jsonl(RESULTS / "judge_absolute.jsonl")}
    with SHEET.open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f)]
    labelled = [r for r in rows if r["human_label"].strip().upper() in {"PASS", "FAIL"}
                and judged.get(r["item_id"]) in {"PASS", "FAIL"}]
    out: dict[str, Any] = {"items": len(rows), "labelled": len(labelled)}
    if not labelled:
        out["status"] = "pending: no human labels yet"
    for split in ("train", "dev", "test"):
        for crit in sorted({r["criterion"] for r in labelled}):
            sub = [r for r in labelled if r["split"] == split and r["criterion"] == crit]
            h = [r["human_label"].strip().upper() for r in sub]
            j = [str(judged[r["item_id"]]) for r in sub]
            out.setdefault(split, {})[crit] = {"kappa": cohen_kappa(h, j), **prf(h, j)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "calibration.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["sheet", "score"])
    args = ap.parse_args()
    make_sheet() if args.action == "sheet" else score()


if __name__ == "__main__":
    main()
