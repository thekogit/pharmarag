# ADR 0002: Hybrid bge-m3 + BM25 index with label-freshness metadata

Status: accepted, 2026-11-08

## Decision

- Dense: `bge-m3` (1024-d) served by local Ollama. Sparse: `Qdrant/bm25` with Qdrant's IDF modifier.
  The two are fused with reciprocal rank fusion, then reranked with `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- The collection name contains the embedding model (`fda_labels__bge-m3`), so querying with a
  different model fails loudly instead of mixing vector spaces.
- Every chunk stores `effective_date` (from the SPL `effectiveTime`) and `is_current`, which is true
  only for the latest label of each compound. Retrieval filters on `is_current` by default.

## Negative experiment: dense-only retrieval, including `nomic-embed-text`

`python -m evals.retrieval_ablation`, 2026-11-08, 84 answerable questions, evidence Hit@5: a passage
from a current label that contains a gold fact string. No LLM is involved.

| Variant | Hit@5 | MRR@5 |
|---|---|---|
| dense `nomic-embed-text` (with its `search_query:` / `search_document:` prefixes) | 0.929 | 0.842 |
| dense `bge-m3` | 0.917 | 0.818 |
| sparse BM25 | 0.940 | 0.823 |
| hybrid RRF | 0.964 | 0.856 |
| hybrid RRF + MiniLM rerank | **0.976** | **0.888** |
| hybrid + rerank, freshness filter off | 0.940 | 0.723 |

Read honestly: dense-only `nomic-embed-text` is *not* worse than dense-only `bge-m3` on this corpus.
What the numbers support is rejecting **dense-only** retrieval. Exact terms such as drug names, doses
and trial names are where BM25 helps, and fusion plus reranking is the best variant measured.
`bge-m3` is kept for its 8k-token context and multilingual coverage, not because it won the dense row.

With the freshness filter off, **36.4%** of the top-5 passages come from superseded labels, and MRR
drops from 0.888 to 0.723. That is the retrieval-side reason for ADR 0003's freshness and
contradiction gates.
