# ADR 0004: Pre-generation gates (completeness, reordering, contradictions)

Status: accepted, 2026-11-08

## Decision

After retrieval and before any answer text is generated (`src/retrieval/gates.py`):

1. **Completeness.** The LLM splits the question into at most 4 sub-questions. Each one needs a
   reranked passage scoring above 0.5 (sigmoid of the cross-encoder logit). If no sub-question is
   covered, the pipeline refuses without calling the generator. If only some are covered, the
   generator is told which parts are missing and must list them in `not_covered`.
2. **Relevance-weighted reordering.** Passages are placed best-first and best-last, with the weakest
   in the middle.
3. **Contradiction detection.** Sentence pairs from *different label versions of the same compound*
   are paired only when they are near-paraphrases (at least 4 shared content terms, Jaccard ≥ 0.4) that
   differ in a number or a negation. They are flagged when `cross-encoder/nli-deberta-v3-base` gives a
   contradiction probability ≥ 0.9 in both directions. A flagged conflict is passed to the generator,
   and the answer schema then requires a `conflict_note`.

## What went wrong first (recorded because it changed the design)

The first detector paired any sentences sharing 3 terms and used `nli-deberta-v3-xsmall` at 0.8. On
the 104 baseline traces it flagged 160 conflicts in 63 traces. All 6 that were sampled were false
positives, for example "an agent to reverse apixaban is available" against "PT/INR monitoring is not
useful". Its silent-contradiction rate (51 of 53) was therefore meaningless. Those runs were discarded.

The tightened detector flags 13 unique pairs on the same passages. In an unvalidated spot-check
(not a human-labelled metric), about half are real label revisions, including
"maximum recommended dosage is 2 mg" vs "1 mg", new pen strengths, and revised trial demographics. The
rest still pair unrelated statements, for example CYP3A4 *inhibitors* vs *inducers*. It also misses
low-overlap rewordings: Lipitor's "active liver disease" becoming "acute liver failure or
decompensated cirrhosis" is not caught. The freshness filter, not the detector, is what keeps that
superseded statement out of answers.

## Measured (`python -m evals.run_traces` + `python -m evals.level1`, 2026-11-08, 104 questions)

| | baseline | diagnostic | diagnostic, freshness off |
|---|---|---|---|
| Out-of-scope refused | 15/15 | 15/15 | 15/15 |
| Partial questions answered with the missing part flagged | 0/5 (2 refused entirely) | 5/5 | 5/5 |
| Answers citing a superseded label | 60/85 | 0/85 | 31/88 |
| Silent contradiction (detected conflict in context, not disclosed) | 8/9 | 0/0 | 0/5 |
| Answerable questions refused | 2/84 | 4/84 | 1/84 |

The gates cost recall: two more false refusals than the baseline. With freshness on, no conflicts
reach the diagnostic context, so the contradiction gate itself is exercised by the freshness-off run
(5 of 5 conflicts disclosed).
