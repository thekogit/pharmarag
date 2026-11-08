"""Level 2: a local Prometheus-2-7B judge (Q5_K_M GGUF on Ollama). No data leaves the machine.

Absolute mode: one binary criterion per call, using Prometheus-2's absolute-grading prompt format.
The rubric only defines scores 1 (FAIL) and 5 (PASS); a score of 4 or 5 counts as PASS.

Pairwise mode: baseline vs diagnostic for the same question, using the relative-grading format.
Each pair is judged twice with the responses swapped between <output_a> and <output_b>; a winner
only counts when both orders agree, and the agreement rate is reported as position consistency.

    python -m evals.judge --configs baseline diagnostic
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx

from evals.common import RESULTS, TRACES, read_jsonl, write_jsonl
from src import log
from src.config import Settings

logger = logging.getLogger("judge")

ABS_SYSTEM = ("You are a fair judge assistant tasked with providing clear, objective feedback based on a "
              "specific criteria, ensuring each assessment reflects the absolute standards set for performance.")
REL_SYSTEM = ("You are a fair judge assistant assigned to deliver insightful feedback that compares individual "
              "performances, highlighting how each stands relative to others within the same corpus.")

ABSOLUTE = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "(write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Score Rubrics:
{rubric}

###Feedback: """

RELATIVE = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of two responses strictly based on the given score rubric, not evaluating in general.
2. Make comparisons between Response A and Response B. Instead of examining Response A and Response B separately, go straight to the point and mention about the commonalities and differences between them.
3. After writing the feedback, indicate the better response, either "A" or "B".
4. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (Either "A" or "B")"
5. Please do not generate any other opening, closing, and explanations.

###Instruction:
{instruction}

###Response A:
<output_a>
{response_a}
</output_a>

###Response B:
<output_b>
{response_b}
</output_b>

###Score Rubric:
{rubric}

###Feedback: """

_UNUSED = "Not used: this criterion is binary. Choose 1 or 5."


@dataclass(frozen=True)
class Criterion:
    name: str
    question: str
    fail: str
    passes: str

    def rubric(self) -> str:
        return (f"[{self.question}]\nScore 1: {self.fail}\nScore 2: {_UNUSED}\nScore 3: {_UNUSED}\n"
                f"Score 4: {_UNUSED}\nScore 5: {self.passes}")


FAITHFULNESS = Criterion(
    "faithfulness",
    "Is every factual statement in the response supported by the passages given in the instruction?",
    "FAIL: at least one statement is not stated in, or contradicts, the passages.",
    "PASS: every statement is directly supported by the passages, or the response declines to answer.")
ABSTENTION = Criterion(
    "abstention",
    "Does the response answer what the passages cover and clearly decline any part of the question they do not cover?",
    "FAIL: it answers something the passages do not cover, or it declines although the passages answer the question.",
    "PASS: it answers exactly the covered parts and explicitly declines or flags the uncovered parts.")
CONFLICT = Criterion(
    "conflict_disclosure",
    "The passages come from label versions with different effective dates that disagree. Does the response rely on "
    "the newer label and point out the disagreement?",
    "FAIL: it merges the versions silently, or presents the older label's statement as current without comment.",
    "PASS: it relies on the newer label and explicitly mentions that an older label version said something different.")
PAIRWISE_RUBRIC = ("Which response is more reliable for a clinician checking the current FDA label: it makes only "
                   "claims supported by its passages, relies on the newer label when versions differ, and is "
                   "explicit about anything the passages do not cover?")


def _passages_block(t: dict[str, Any], limit: int = 1500) -> str:
    return "\n\n".join(f"[{p['n']}] ({p['compound']} label effective {p['effective_date']}, {p['section']})\n"
                       f"{p['text'][:limit]}" for p in t["passages"]) or "(no passages were retrieved)"


def _instruction(t: dict[str, Any]) -> str:
    return f"Question: {t['question']}\n\nPassages available to the assistant:\n{_passages_block(t)}"


_RESULT = re.compile(r"\[RESULT\]\s*\(?\s*\"?([1-5AB])\b")


class Prometheus:
    def __init__(self, host: str, model: str) -> None:
        self.model = model
        self._url = host.rstrip("/") + "/api/generate"
        self._client = httpx.Client(timeout=600)

    def _generate(self, system: str, user: str) -> str:
        # Prometheus-2 is Mistral-instruct based: the system prompt goes inside the first [INST] block.
        r = self._client.post(self._url, json={
            "model": self.model, "raw": True, "stream": False,
            "prompt": f"[INST] {system}\n\n{user} [/INST]",
            "options": {"temperature": 0, "seed": 0, "num_ctx": 16384, "num_predict": 700},
        })
        r.raise_for_status()
        return str(r.json()["response"])

    def absolute(self, t: dict[str, Any], c: Criterion) -> dict[str, Any]:
        out = self._generate(ABS_SYSTEM, ABSOLUTE.format(instruction=_instruction(t), response=t["answer"],
                                                         rubric=c.rubric()))
        m = _RESULT.search(out)
        score = int(m.group(1)) if m and m.group(1).isdigit() else None
        return {"trace_id": t["trace_id"], "qid": t["qid"], "config": t["config"], "criterion": c.name,
                "score": score, "verdict": None if score is None else ("PASS" if score >= 4 else "FAIL"),
                "feedback": out.split("[RESULT]")[0].strip()[:1500], "judge": self.model}

    def pairwise(self, a: dict[str, Any], b: dict[str, Any]) -> str | None:
        instruction = (f"Question: {a['question']}\n\nPassages available to Response A:\n<passages_a>\n"
                       f"{_passages_block(a, 900)}\n</passages_a>\n\nPassages available to Response B:\n"
                       f"<passages_b>\n{_passages_block(b, 900)}\n</passages_b>")
        out = self._generate(REL_SYSTEM, RELATIVE.format(instruction=instruction, response_a=a["answer"],
                                                         response_b=b["answer"], rubric=PAIRWISE_RUBRIC))
        m = _RESULT.search(out)
        return m.group(1) if m and m.group(1) in "AB" else None


def applicable(t: dict[str, Any]) -> list[Criterion]:
    crit = [ABSTENTION]
    if t["outcome"] not in {"refused", "gate_refused"}:
        crit.append(FAITHFULNESS)
        if t["observed_conflicts"]:
            crit.append(CONFLICT)
    return crit


def summarise(rows: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"absolute": {}, "pairwise": {}}
    for r in rows:
        d = out["absolute"].setdefault(r["config"], {}).setdefault(r["criterion"], {"PASS": 0, "FAIL": 0, "unparsed": 0})
        d[r["verdict"] or "unparsed"] += 1
    for cfg in out["absolute"].values():
        for d in cfg.values():
            n = d["PASS"] + d["FAIL"]
            d["pass_rate"] = round(d["PASS"] / n, 3) if n else None
    if pairs:
        consistent = [p for p in pairs if p["consistent"]]
        wins: dict[str, int] = {}
        for p in consistent:
            wins[p["winner"]] = wins.get(p["winner"], 0) + 1
        out["pairwise"] = {"pairs": len(pairs), "position_consistent": len(consistent),
                           "position_consistency": round(len(consistent) / len(pairs), 3), "wins": wins}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", nargs=2, default=["baseline", "diagnostic"], metavar=("A", "B"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    log.setup()
    s = Settings.load()
    judge = Prometheus(s.ollama_host, s.judge_model)
    traces = {c: list(read_jsonl(TRACES / f"{c}.jsonl"))[: args.limit] for c in args.configs}
    rows: list[dict[str, Any]] = []
    for c in args.configs:
        for t in traces[c]:
            for crit in applicable(t):
                rows.append(judge.absolute(t, crit))
                logger.info("%s %s -> %s", t["trace_id"], crit.name, rows[-1]["verdict"])
    write_jsonl(RESULTS / "judge_absolute.jsonl", rows)
    a_name, b_name = args.configs
    by_q = {t["qid"]: t for t in traces[b_name]}
    pairs = []
    for ta in traces[a_name]:
        tb = by_q.get(ta["qid"])
        if tb is None:
            continue
        first = judge.pairwise(ta, tb)   # a in slot A
        second = judge.pairwise(tb, ta)  # a in slot B
        w1 = {"A": a_name, "B": b_name}.get(first or "")
        w2 = {"A": b_name, "B": a_name}.get(second or "")
        pairs.append({"qid": ta["qid"], "order_1": w1, "order_2": w2,
                      "consistent": w1 is not None and w1 == w2, "winner": w1 if w1 == w2 else None})
        logger.info("pair %s: %s / %s", ta["qid"], w1, w2)
    write_jsonl(RESULTS / "judge_pairwise.jsonl", pairs)
    summary = {"judge": s.judge_model, **summarise(rows, pairs)}
    (RESULTS / "judge_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
