# Retrieval & Synthesis Ablation Results

Benchmark evaluation performed across 35 curated pharmaceutical regulatory queries (30 answerable queries across semaglutide, metformin, atorvastatin, apixaban, sertraline labels and 5 unanswerable out-of-scope queries) using DailyMed / OpenFDA regulatory ground truth.

| Config | Hit@5 | MRR | Fact recall | Correct refusals | False refusals | Median latency (s) |
|---|---|---|---|---|---|---|
| dense, rerank=0 | 0.667 | 0.485 | 0.533 | 5/5 | 4 | 0.8 |
| hybrid, rerank=0 | 0.833 | 0.682 | 0.767 | 5/5 | 2 | 1.1 |
| hybrid, rerank=1 | 0.900 | 0.814 | 0.867 | 5/5 | 1 | 1.6 |

### Observations
- **Dense vs Hybrid:** Incorporating SPLADE sparse lexical embeddings alongside dense vectors via Reciprocal Rank Fusion (RRF) improved Hit@5 from 0.667 to 0.833 and MRR from 0.485 to 0.682, critically capturing exact chemical nomenclature and dosage numbers (e.g. "500 mg", "0.25 mg").
- **Cross-Encoder Reranking:** Adding `cross-encoder/ms-marco-MiniLM-L-6-v2` re-ranks candidate chunks by deep semantic interaction, pushing the exact regulatory section chunk into top ranks (MRR 0.814) and reducing false refusals to 1 while adding only ~0.5s CPU latency.
- **Grounded Refusal Compliance:** Across all 3 configurations, out-of-scope unanswerable questions achieved 100% correct refusal rate (5/5), adhering to zero-hallucination regulatory compliance.
