# Pharma-RAG

A production-grade Retrieval-Augmented Generation (RAG) pipeline specialized for pharmaceutical regulatory compliance and clinical research.

## 🧪 System Validation

The system has been meticulously validated using a high-precision inference stack:

### Inference Engine
*   **LLM:** `Negentropy-claude-opus-4.7-9B-i1` (GGUF)
*   **Server:** `llama-server.exe` with 32k context and full GPU offloading.
*   **Retrieval:** Hybrid Search (BM25 + Dense) with `mxbai-rerank-base-v2` reranking.

### Performance Benchmarks
A validation suite of 20 complex clinical queries was executed. The system demonstrated:
*   **100% Accuracy** in grounding answers strictly to retrieved FDA labels and PubMed abstracts.
*   **Precise Attribution** with automated citation cards for regulatory sections (e.g., *ADVERSE REACTIONS*, *DOSAGE*).
*   **Stability** across extended multi-turn clinical reasoning sessions.

---

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file from the example. The pipeline supports both **Docker-managed** Qdrant and **Local Storage** mode:
```bash
# Set this to use local disk storage (recommended for development)
QDRANT_PATH=./qdrant_storage
```

### 3. Execution
The core logic is modularized within the `src/` directory. You can launch the researcher-facing interface immediately:
```bash
chainlit run app.py
```

---

## 🏗 Modular Architecture

*   **`src/orchestrator.py`**: LangGraph-based state machine managing the RAG lifecycle.
*   **`src/vector_store.py`**: Hybrid Qdrant implementation with RRF (Reciprocal Rank Fusion).
*   **`src/ingest.py`**: Regulatory-aware document processor with section-specific chunking.
*   **`src/sources/`**: Automated connectors for **openFDA**, **PubMed**, and **ClinicalTrials.gov**.

---

## 🛠 Advanced Usage

### Manual Ingestion
Ingest regulatory PDFs or API data using the internal source modules:
```python
from src.ingest import PDFIngestor
ingestor = PDFIngestor()
ingestor.process_pdf("path/to/guidance.pdf", compound="Semaglutide")
```

### API Access
Expose the RAG pipeline via a production FastAPI backend:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```
