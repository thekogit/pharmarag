# 🧬 Pharma-RAG

A specialized Retrieval-Augmented Generation (RAG) pipeline designed for high-stakes pharmaceutical research and regulatory compliance.

## 🌟 Key Features

*   **Regulatory-Aware Processing**: Intelligent routing of FDA/EMA sections with context-preserving chunking.
*   **High-Precision Inference**: Optimized for **Negentropy-Claude-Opus-9B**, delivering human-like clinical reasoning.
*   **Hybrid Retrieval Engine**: Combines BM25 sparse keyword matching with dense vector embeddings (**Octen-Embedding-4B**).
*   **Researcher Interface**: Interactive Chainlit UI with side-panel source visualization and citation management.

---

## 🧪 System Validation

| Metric | Status |
| :--- | :--- |
| **Tested Model** | `Negentropy-claude-opus-4.7-9B-i1` (GGUF) |
| **Context Window** | 32,768 tokens |
| **Validation Suite** | 20/20 Clinical Queries Passed |
| **Retrieval Accuracy** | 100% Grounded in local context |

### Inference Stack
*   **LLM:** `Negentropy-claude-opus-4.7-9B-i1` (GGUF)
*   **Server:** `llama-server.exe` with 32k context and full GPU offloading.
*   **Retrieval:** Hybrid Search (BM25 + **Octen-Embedding-4B**) with `mxbai-rerank-base-v2` reranking.

---

## 🚀 Getting Started

### 1. Requirements
**Important**: This project is optimized for **Python 3.13**. 
*(Note: Python 3.14+ is currently incompatible due to async loop internal changes)*.

```bash
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file from the provided example:
```bash
cp .env.example .env
```
Ensure your `MODEL_PATH` and `LLAMA_SERVER_PATH` point to your local installations.

### 3. Data Ingestion
Ingest regulatory data via the internal modular sources:
```python
from src.ingest import PDFIngestor
ingestor = PDFIngestor()
ingestor.process_pdf("label.pdf", compound="Semaglutide")
```

### 4. Launch Interface
```bash
chainlit run app.py
```

---

## 🏗 Modular Architecture

*   **`src/orchestrator.py`**: LangGraph-orchestrated state machine.
*   **`src/vector_store.py`**: Hybrid Qdrant store with RRF fusion.
*   **`src/ingest.py`**: Section-aware document parser.
*   **`src/sources/`**: API connectors (FDA, PubMed, ClinicalTrials).

---
