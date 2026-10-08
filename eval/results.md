# Retrieval & Synthesis Evaluation Framework

This directory contains the automated evaluation harness for the regulatory RAG pipeline across 35 curated queries (30 answerable regulatory queries across semaglutide, metformin, atorvastatin, apixaban, and sertraline FDA labels, plus 5 unanswerable out-of-scope queries for refusal testing).

## Evaluation Harness

The benchmark is executed via `eval/run_eval.py`:

```bash
# Requires running local model server (OpenAI-compatible, e.g. llama-server) and populated Qdrant collection
python eval/run_eval.py eval/questions.jsonl
```

### Metrics Tracked

- **Hit@5**: Top-5 document retrieval accuracy matching the target compound and section.
- **MRR (Mean Reciprocal Rank)**: Reciprocal rank of the first relevant section chunk.
- **Fact recall**: Exact presence of mandatory ground truth facts (from `must_contain` in `questions.jsonl`) in the synthesized response.
- **Correct refusals**: Proportion of unanswerable queries correctly refused (`do not contain sufficient information`).
- **False refusals**: Answerable queries erroneously refused.
- **Latency (s)**: End-to-end wall-clock latency per query (including multi-query expansion, retrieval, reranking, and generation).

## Benchmark Status

*Ablation benchmark results are pending execution against a dedicated local model server. Synthetic latency and metric values have been removed to adhere to strict empirical audit standards: no benchmark numbers are published without recorded command run output.*

Query datasets are fully defined in [`questions.jsonl`](questions.jsonl).
