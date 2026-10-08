# pharmarag

Local retrieval-augmented generation pipeline over pharmaceutical regulatory documents and drug labels with cited answers and explicit refusal.

```text
Query: What is the starting dose of semaglutide for type 2 diabetes?

Executive Summary: The recommended starting dosage of semaglutide is 0.25 mg once weekly.

Clinical/Regulatory Evidence:
- Initial dosing begins at 0.25 mg subcutaneously once weekly for 4 weeks [FDA Drug Label].
- The 0.25 mg dosage is intended for treatment initiation and is not effective for glycemic control [FDA Drug Label].
- Following 4 weeks at 0.25 mg, the dosage increases to 0.5 mg once weekly [FDA Drug Label].

Regulatory Implications:
- Indicated as an adjunct to diet and exercise to improve glycemic control in adults with type 2 diabetes [FDA Drug Label].
```

## Why
Answering queries across lengthy, heavily regulated technical documents—with strict attribution, traceability, and verifiable refusal when context is missing—is essential in healthcare and financial compliance. Standard LLM generations risk ungrounded extrapolation; regulated domains require exact grounding, provenance citations, and zero tolerance for hallucinated claims.

## How it works
- Section-aware chunking preserving document structure (`INDICATIONS`, `DOSAGE`, `WARNINGS`, `CONTRAINDICATIONS`, `CLINICAL_STUDIES`).
- Dense semantic embeddings combined with SPLADE sparse lexical embeddings fused via Reciprocal Rank Fusion (RRF) in Qdrant.
- Multi-query expansion generating targeted regulatory search angles.
- Cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) scoring full query-document interactions on CPU.
- LangGraph stateful execution DAG coordinating expansion, retrieval, and synthesis.
- Cited answers with inline source attribution and deterministic refusal when evidence is absent.

## Results
The evaluation harness tests retrieval accuracy (Hit@5, MRR), fact recall, and refusal precision across 35 curated regulatory queries (30 answerable queries across semaglutide, metformin, atorvastatin, apixaban, and sertraline labels, plus 5 unanswerable out-of-scope controls) via `eval/run_eval.py`:

```bash
python eval/run_eval.py eval/questions.jsonl
```

| Config | Status | Metrics Evaluated |
|---|---|---|
| Dense, Rerank=0 | Harness configured | Hit@5, MRR, Fact recall, Refusal precision, Latency |
| Hybrid, Rerank=0 | Harness configured | Hit@5, MRR, Fact recall, Refusal precision, Latency |
| Hybrid, Rerank=1 | Harness configured | Hit@5, MRR, Fact recall, Refusal precision, Latency |

*Ablation benchmark results pending local LLM server execution. Full benchmark queries and evaluation code are committed in `eval/questions.jsonl` and `eval/run_eval.py`.*

## Quickstart
Compatible with Python 3.10–3.12 and any OpenAI-compatible local model server (tested with Qwen 2.5 7B Instruct via llama-server):

```bash
git clone https://github.com/thekogit/pharmarag.git
cd pharmarag
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env

# Ingestion:
# 1. Ingest local drug-label PDF into Qdrant
python ingest_cli.py label.pdf --compound semaglutide
# 2. Or fetch directly from openFDA REST API
python ingest_drug.py semaglutide

# Query:
# Interactive CLI query session
python chat.py
# Or launch the Chainlit web interface
chainlit run app.py
```

## Tests
Run the offline unit test suite:

```bash
pytest -q
```

## Limitations
- Benchmark evaluation is conducted on a 35-question test set curated across 5 drug labels.
- String-match fact recall evaluates literal substring presence and can penalize valid syntactic paraphrases.
- Ingestion pipeline currently targets English FDA drug labels and structured regulatory PDF sections.
- Research demonstration only; not clinical advice.

## How I used AI
I used Antigravity and Gemini CLI Agent to draft parts of the code. I chose the architecture, reviewed every change, replaced the reranking pipeline with a cross-encoder, fixed vector indexing and collection dimensions, and wrote the test suite in `tests/` to verify retrieval behavior. Agent-made commits are visible in the git history.

## License
MIT License. See [LICENSE](LICENSE) for details.
