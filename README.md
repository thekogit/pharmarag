# PharmaRAG-Diagnostic

RAG over FDA drug labels that either quotes the label verbatim or says it can't answer. A benchmark
counts the ways RAG goes wrong instead of reporting a single accuracy number.

```text
Q: What is the maximum dose of Ozempic, and what is its US list price?

- The highest recommended maintenance dosage listed is 2 mg injected subcutaneously once weekly. [1]

Not covered by the labels: What is the US list price of Ozempic?

Sources:
[1] semaglutide - Novo Nordisk (Ozempic), effective 2026-06-01, DOSAGE
```

That is a real output (trace `diagnostic:q086` in `evals/traces/`). On the same question the baseline
pipeline refused the whole question, including the part the label does answer.

## Why

Drug labels change. The corpus holds the current label of five drugs plus four older versions of the
same products that are still published by repackagers: Ozempic's maximum dose was 1 mg in 2021 and is
2 mg now, and Lipitor's liver contraindication changed between 2009 and 2024. A RAG system that
retrieves both versions and blends them gives a confident, cited, wrong answer. This project tries to
make that failure, and four others, measurable.

## Architecture

```
DailyMed PDF ─► Docling (layout + TableFormer) ─► section-aware chunks ─► coherence gate ─┐
 (9 labels, SPL effective dates)                   tables keep titles + footnotes          │
                                                                                          ▼
                              Qdrant (embedded): dense bge-m3 + sparse BM25, is_current flag
                                                                                          │
question ─► decompose ─► hybrid search (RRF) per sub-question ─► MiniLM rerank ◄──────────┘
                                │
                                ▼   pre-generation gates
                 completeness ─► reorder ─► cross-version contradiction check (NLI)
                     │ nothing covered                     │
                     ▼                                     ▼
                  refuse                generate JSON (beam:cloud) ─► Pydantic validator:
                                        every quote must be an exact substring of its passage
                                        (≤ 2 retries, then keep only verified claims or refuse)
```

| Stage | Code | Decision record |
|---|---|---|
| Ingestion: Docling, canonical sections, table + footnote binding, coherence gate | `src/ingestion/` | [ADR 0001](docs/adr/0001-docling-table-aware-ingestion.md) |
| Hybrid index, freshness metadata, collection named after the embedding model | `src/retrieval/index.py` | [ADR 0002](docs/adr/0002-hybrid-index-and-freshness.md) |
| Reranker choice (made on the train split only) | `src/retrieval/rerank.py` | [ADR 0003](docs/adr/0003-reranker-selection.md) |
| Pre-generation gates | `src/retrieval/gates.py` | [ADR 0004](docs/adr/0004-pre-generation-gates.md) |
| Verbatim grounding | `src/generation/` | [ADR 0005](docs/adr/0005-verbatim-grounding.md) |
| Options rejected without an experiment | | [ADR 0006](docs/adr/0006-considered-not-tested.md) |

Models: the generator is Ollama Cloud `beam:cloud` (OpenAI-compatible API). Locally: `bge-m3`
embeddings and the Prometheus-2-7B judge on Ollama, plus the MiniLM reranker and DeBERTa NLI
cross-encoders on CPU. Questions and passages are sent to the cloud generator, so this is **not** an
offline system.

## Failure taxonomy

Level 1 checks are deterministic: no model calls, same traces in, same numbers out (`evals/level1.py`).

| Failure mode | Detected when |
|---|---|
| Retrieval miss | no passage from a current label contains the question's gold fact |
| Stale evidence | the answer cites a passage from a superseded label version |
| Silent contradiction | the detector found a cross-version conflict in the context and the answer neither refused nor disclosed it |
| Ungrounded number | the answer states a dose, percentage or duration that appears in no passage |
| Abstention error | an out-of-scope question is answered, or an answerable one is refused |

## Results

Benchmark: 104 questions written against the label text (`evals/questions.jsonl`). 60 are single-fact
questions, 6 depend on tables, 8 are version-sensitive, 10 are multi-part, 5 are partially answerable
and 15 are out of scope. Every gold fact string was checked to appear in a current label.

### Retrieval only (no LLM)

`python -m evals.retrieval_ablation`, 2026-11-08, 84 answerable questions. A hit is a passage from a
current label containing a gold fact.

| Variant | Hit@5 | MRR@5 |
|---|---|---|
| dense `nomic-embed-text` | 0.929 | 0.842 |
| dense `bge-m3` | 0.917 | 0.818 |
| sparse BM25 | 0.940 | 0.823 |
| hybrid RRF | 0.964 | 0.856 |
| **hybrid + MiniLM rerank** | **0.976** | **0.888** |
| hybrid + rerank, freshness filter off | 0.940 | 0.723 (36.4% of top-5 passages superseded) |

### End to end (Level 1)

`python -m evals.run_traces --configs baseline diagnostic diagnostic_no_freshness` then
`python -m evals.level1 baseline diagnostic diagnostic_no_freshness`. Run on 2026-11-08 with
`beam:cloud` as the generator; 104 traces per config, 0 errors.

*Baseline* = hybrid + rerank with no freshness filter, no gates and a free-text cited answer.
*Diagnostic* = everything on.

| | baseline | diagnostic | diagnostic, freshness off |
|---|---|---|---|
| Evidence Hit@k / MRR | 0.976 / 0.729 | 0.976 / **0.888** | 0.940 / 0.735 |
| Fact recall (answerable) | **0.958** | 0.911 | 0.952 |
| Out-of-scope refused | 15/15 | 15/15 | 15/15 |
| Answerable refused (false refusals) | **2/84** | 4/84 | **1/84** |
| Partial questions: answered part kept, missing part flagged | 0/5 | **5/5** | **5/5** |
| Answers citing a superseded label | 60/85 | **0/85** | 31/88 |
| Silent contradiction | 8/9 | 0/0 | **0/5** |
| Ungrounded numbers | 0/85 | 0/85 | 0/88 |
| Final quotes verified verbatim | n/a | 177/177 | 222/222 |
| Valid on first attempt | n/a | 81/85 | 82/88 |
| Median / p95 latency (s) | 10.3 / 24.7 | 5.8 / 14.1 | 6.9 / 19.7 |
| Mean tokens per question | 1,886 | 2,218 | 2,436 |

What this shows, and what it doesn't:

- The freshness filter is what removes stale evidence (60 → 0 citations of superseded labels). The
  contradiction gate matters when superseded labels do reach the context: with freshness off, all
  5 detected conflicts were disclosed.
- The diagnostic pipeline is stricter. It costs recall: 0.911 vs 0.958 fact recall and 4 vs 2 false
  refusals. Two of those four come from the model failing, three times, to quote a markdown dosing
  table verbatim.
- "Silent contradiction" depends on the contradiction detector, which is imperfect (ADR 0004): about
  half its flags are real label revisions in an unvalidated spot-check, and it misses
  low-overlap rewordings. Treat those rows as indicative.
- 15/15 refusals and 0 ungrounded numbers mean these checks found no failures on this set. They don't
  prove the system can't fail.

### Local judge (Level 2)

`python -m evals.judge --configs baseline diagnostic`, 2026-11-08: Prometheus-2-7B (Q5_K_M) on local
Ollama, binary single-criterion rubrics, 387 verdicts per run, 0 unparseable. The judge ran twice.
**v1** used the original abstention rubric. **v2** adds to that rubric that declining is the correct
answer when the passages lack the information. v2 was written *after* seeing v1's results on this
same set, so it is a post-hoc fix, not a calibrated judge. v1 output is kept in
`evals/results/judge_v1_*`.

| Criterion, PASS rate | baseline v1 | baseline v2 | diagnostic v1 | diagnostic v2 |
|---|---|---|---|---|
| Faithfulness to the passages | 59/85 | 58/85 | 58/85 | 57/85 |
| Abstention | 69/104 | 72/104 | 59/104 | 67/104 |
| Conflict disclosure (traces with a detected conflict) | 1/9 | 1/9 | none reached the context | none reached the context |
| Correct out-of-scope refusals judged PASS | 0/15 | 8/15 | 0/15 | 9/15 |

Pairwise, baseline vs diagnostic, judged twice with A/B swapped: the same winner in both orders in
61/104 pairs for v1 (0.587) and 59/104 for v2 (0.567). Among consistent pairs the baseline won 49–12
(v1) and 50–9 (v2).

**Don't trust the judge numbers yet.** Every one of those 30 out-of-scope questions was correctly
refused, and the judge still failed all 30 refusals under v1 and 13 under v2. For example, v1 failed
the refusal of "In what year was Novo Nordisk founded?" because the answer "makes a general statement
about the passages not containing sufficient information" (trace `diagnostic:q100`). A judge that
penalises correct refusals also penalises the stricter pipeline. The pairwise preference for the
baseline comes with more than 40% position flips, and is consistent with a bias toward longer
free-text answers. The deterministic Level 1 checks above remain the primary result, and the judge
gets a verdict on its reliability once human labels exist (below).

### Human calibration

**Pending.** `python -m evals.calibrate sheet` writes `evals/labels/label_sheet.csv` (judge verdicts
hidden). Once `human_label` is filled with PASS/FAIL, `python -m evals.calibrate score` reports Cohen's
κ, precision and recall per criterion on the 15/42.5/42.5 train/dev/test splits. No agreement number
is claimed until that is done.

## Quickstart

Needs Python 3.12, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com) running locally
and an Ollama Cloud API key. A GPU helps the judge but isn't required.

```bash
git clone https://github.com/thekogit/pharmarag.git && cd pharmarag
uv sync --all-extras                         # locked dependencies (uv.lock)
cp .env.example .env                         # then set LLM_API_KEY
ollama pull bge-m3
ollama pull hf.co/mradermacher/prometheus-7b-v2.0-GGUF:Q5_K_M   # only for the judge

uv run python -m scripts.fetch_corpus        # 9 label PDFs from DailyMed into data/raw/
uv run python -m scripts.build_index         # Docling parse (cached), chunk, embed, index

uv run python chat.py                        # CLI
uv run chainlit run app.py                   # web UI
uv run uvicorn src.api:app                   # HTTP API: POST /query {"question": "..."}
```

Every component can be switched off for ablations (`RETRIEVAL_MODE`, `RERANK`, `FRESHNESS`, `GATES`,
`GROUNDING` in `.env`).

Evaluation:

```bash
uv run python -m scripts.build_index --embed-model nomic-embed-text   # for the dense baseline row
uv run python -m evals.retrieval_ablation
uv run python -m evals.run_traces --configs baseline diagnostic diagnostic_no_freshness
uv run python -m evals.level1 baseline diagnostic diagnostic_no_freshness
uv run python -m evals.judge --configs baseline diagnostic
uv run python -m evals.calibrate sheet
```

Optional Qdrant server instead of embedded mode: `docker compose up -d` (capped at 1.6 CPUs and 6 GB),
then set `QDRANT_URL`. The compose file has not been run on the author's machine.

## Tests and CI

`uv run pytest -q` runs 40 offline tests (fake embedders and LLM, Qdrant in memory). CI
(`.github/workflows/ci-evals.yml`) runs ruff, `mypy --strict`, the tests, and the Level 1 regression
gate over the committed traces. That gate fails if any diagnostic quote doesn't verify, if the
diagnostic pipeline cites a superseded label, or if it does worse than the baseline on out-of-scope
refusals or silent contradictions.

## Limitations

- 9 labels, 5 drugs, English only. The questions were written by the project, not sampled from users.
- Gold facts are string matches. A correct paraphrase counts as a miss.
- Verbatim quoting fails more often on markdown tables than on prose.
- The contradiction detector has unmeasured precision and misses low-overlap rewordings.
- Results depend on a cloud model (`beam:cloud`) that can change without notice. The traces record
  what it returned on 2026-11-08.
- Research demonstration. Not medical advice.

## License

MIT, see [LICENSE](LICENSE). FDA label content comes from DailyMed (U.S. National Library of
Medicine) and is not covered by this licence. The Prometheus-2 weights belong to their authors
(Apache-2.0).
