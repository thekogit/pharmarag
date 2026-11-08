"""Shared helpers for the evaluation harness."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
TRACES = EVALS / "traces"
RESULTS = EVALS / "results"
LABELS = EVALS / "labels"
QUESTIONS = EVALS / "questions.jsonl"

# Question categories. "answerable" questions must be answered; "out_of_scope" must be refused;
# "partial" questions mix an answerable part with one the labels cannot answer.
ANSWERABLE = {"single", "table", "version", "multi"}
CATEGORIES = ANSWERABLE | {"partial", "out_of_scope"}


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_questions(path: Path = QUESTIONS) -> list[dict[str, Any]]:
    qs = list(read_jsonl(path))
    for q in qs:
        if q["category"] not in CATEGORIES:
            raise ValueError(f"{q['id']}: unknown category {q['category']!r}")
    return qs


def split_of(qid: str) -> str:
    """Deterministic 15 / 42.5 / 42.5 train / dev / test split by question id.

    Splitting by question (not by trace) keeps both pipelines' traces of one question in the same
    split, so nothing about a test question leaks into choices made on train or dev.
    """
    import hashlib

    b = int(hashlib.sha256(qid.encode()).hexdigest(), 16) % 1000
    return "train" if b < 150 else "dev" if b < 575 else "test"


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = min(len(s) - 1, max(0, round(p / 100 * (len(s) - 1))))
    return s[k]
