# ADR 0003: Keep the MiniLM-L-6 cross-encoder; choose it on the train split only

Status: accepted, 2026-11-08

## Context

During a smoke run, "What is the starting dose of Ozempic?" retrieved four CLINICAL_STUDIES passages,
which MiniLM scored at about 0.99, and the grounded generator correctly refused. A stronger reranker
or section-aware reranker input could fix that question. But that question sits in the test split,
and choosing components by looking at test questions would inflate the reported numbers.

## Decision

Reranker options were compared **only on the train split** (16 answerable questions; splits are a
fixed hash of the question id, see `evals/common.py:split_of`). Retrieval used hybrid search, with
evidence Hit@4 and MRR@4 as the metrics:

| Variant | Hit@4 | MRR@4 | s / query (CPU) |
|---|---|---|---|
| no rerank (RRF order) | 1.0 | 0.833 | 0 |
| MiniLM-L-6, pool 20 | 1.0 | **0.875** | 0.8 |
| MiniLM-L-6, pool 20, section header prepended | 1.0 | 0.865 | 0.7 |
| MiniLM-L-6, pool 40 | 1.0 | 0.875 | 1.5 |
| `BAAI/bge-reranker-base`, pool 20 | 1.0 | 0.833 | 4.0 |

`BAAI/bge-reranker-v2-m3` was also started. It was stopped after more than an hour of CPU time
without finishing the train split, and that cost alone rules it out on CPU.

MiniLM-L-6 with a pool of 20 stays. The train split is too easy to separate the options well
(everything reaches Hit@4 = 1.0), so this is a weak "no evidence to change" decision, not a win.

The scratch comparison script was not kept. These numbers come from one run on 2026-11-08 and are
recorded here as such.
