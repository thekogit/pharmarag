# Evaluation harness

| File | What it is |
|---|---|
| `questions.jsonl` | 104 benchmark questions: `category`, gold `must_contain` strings (each verified to appear in a current label) |
| `run_traces.py` | runs the questions through pipeline configs and writes `traces/<config>.jsonl` |
| `level1.py` | deterministic checks and the five failure modes; `--assert` is the CI regression gate |
| `retrieval_ablation.py` | retrieval-only comparison (no LLM) |
| `judge.py` | local Prometheus-2-7B judge: binary single-criterion absolute grading plus position-swapped pairwise grading |
| `calibrate.py` | blind label sheet out, Cohen's κ / precision / recall in |
| `results/` | `level1.json`, `retrieval_ablation.json`, `corpus_report.json`, `judge_*.json(l)` |
| `traces/` | full traces of the 2026-11-08 run: passages, gate decisions, answers, retries, tokens |
| `labels/label_sheet.csv` | items for human labelling (`human_label` is empty: no human labels exist yet) |

Splits: `common.split_of(qid)` hashes the question id into 15% train, 42.5% dev and 42.5% test. Component
choices (ADR 0003) were made on train only.
