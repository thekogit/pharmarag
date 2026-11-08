"""Run every benchmark question through one or more pipeline configs and record full traces.

    python -m evals.run_traces --configs baseline diagnostic

Each trace holds the question, sub-questions, retrieved passages (with label version metadata),
gate decisions, the answer, grounding retries, timings and token usage. Independently of the config,
the contradiction detector also runs on the retrieved passages as an *observer*, so silent
contradiction synthesis can be measured for pipelines that do not use the detector themselves.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import time
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from evals.common import TRACES, load_questions, write_jsonl
from src import log
from src.config import Settings
from src.pipeline import CONFIGS, Components, build_pipeline
from src.retrieval.gates import detect_contradictions
from src.retrieval.index import Hit

logger = logging.getLogger("run_traces")


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _passages(hits: list[Hit]) -> list[dict[str, Any]]:
    keys = ("chunk_id", "set_id", "compound", "effective_date", "publisher", "is_current", "section",
            "subsection", "kind", "text")
    return [{"n": i, "score": round(h.score, 4), **{k: h.payload.get(k) for k in keys}}
            for i, h in enumerate(hits, 1)]


def run_config(name: str, questions: list[dict[str, Any]], comps: Components,
               s: Settings, limit: int | None) -> list[dict[str, Any]]:
    pipe = build_pipeline(CONFIGS[name], s, comps)
    observer = comps.nli
    meta = {"config": name, "commit": _commit(), "llm": s.llm_model, "embed": s.embed_model,
            "run_at": datetime.now(UTC).isoformat(timespec="seconds")}
    rows = []
    for q in questions[:limit]:
        before = asdict(pipe.llm.usage)
        t0 = time.perf_counter()
        try:
            out = pipe.run(q["question"])
            err = None
        except Exception as e:  # a failed trace is recorded, not dropped
            logger.exception("trace %s/%s failed", name, q["id"])
            out, err = {}, f"{type(e).__name__}: {e}"
        latency = time.perf_counter() - t0
        after = asdict(pipe.llm.usage)
        hits: list[Hit] = out.get("hits", [])
        observed = detect_contradictions(hits, observer)
        rows.append({
            "trace_id": f"{name}:{q['id']}", **meta, "qid": q["id"], "question": q["question"],
            "category": q["category"], "compound": q.get("compound"), "gold_section": q.get("section"),
            "must_contain": q.get("must_contain", []), "sub_questions": out.get("sub_questions", []),
            "passages": _passages(hits), "gate": out.get("gate"), "answer": out.get("answer", ""),
            "outcome": out.get("outcome", "error"), "structured": out.get("structured"),
            "grounding": out.get("grounding"),
            "observed_conflicts": [asdict(c) for c in observed.conflicts],
            "latency_s": round(latency, 2), "timings": {k: round(v, 3) for k, v in out.get("timings", {}).items()},
            "tokens": {k: after[k] - before[k] for k in after}, "error": err,
        })
        logger.info("%s %s -> %s (%.1fs)", name, q["id"], rows[-1]["outcome"], latency)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", nargs="+", default=["baseline", "diagnostic"], choices=sorted(CONFIGS))
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions (smoke runs)")
    args = ap.parse_args()
    log.setup()
    s = Settings.load()
    questions = load_questions()
    comps = Components.load(s)
    for name in args.configs:
        rows = run_config(name, questions, comps, s, args.limit)
        suffix = f".limit{args.limit}" if args.limit else ""
        out = TRACES / f"{name}{suffix}.jsonl"
        write_jsonl(out, rows)
        print(f"wrote {len(rows)} traces to {out}")


if __name__ == "__main__":
    main()
