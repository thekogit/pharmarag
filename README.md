# Pharma-RAG

A local-first Retrieval-Augmented Generation (RAG) pipeline designed for life science and regulatory data.

## System Architecture & Data Sources

The system is engineered to process clinical and regulatory data and synthesize answers grounded strictly in retrieved context.

### Data Sources
The ingestion pipeline is capable of handling unstructured and semi-structured documents typically found in the pharmaceutical industry:
*   **ClinicalTrials.gov (API):** Automated ingestion of trial protocols, outcomes, and phase data.
*   **openFDA (API):** Real-time drug label data.
*   **PubMed (API):** Automated search and fetch of peer-reviewed biomedical abstracts.
*   **Local PDFs:** Regulatory documents, guidance, and ICH guidelines.

### Architectural Strategies & Choices

1. **Regulatory-Aware Ingestion**
   *   **Section Routing:** A `RegulatoryRouter` identifies standard FDA/EMA section headers (e.g., INDICATIONS, ADVERSE REACTIONS) using regex patterns.
   *   **Specialized Chunking:** Narrative sections (Clinical Studies) are chunked at 1000 characters with 10% overlap, while dense sections (Dosage) are chunked at 500 characters with 0% overlap to maintain data integrity.
   *   **Contextual Prepend:** Every text chunk is prepended with its section name (e.g., `[Section: DOSAGE]`) to preserve semantic context during retrieval.

2. **Two-Stage Retrieval & Reranking**
   *   **Query Expansion:** The system uses the local LLM to generate 3 semantic variations of the user's query (clinical, regulatory, and safety-focused) to maximize recall.
   *   **Reranking:** Top-20 candidates from vector search are reranked using a cross-encoder (`BAAI/bge-reranker-v2-m3`) to select the most relevant Top-5 chunks for synthesis.

3. **Inference & VRAM Optimization**
   *   **Model:** `Qwen3.5-9B-DeepSeek-V4-Flash` served via `llama.cpp`.
   *   **Quantization:** RotorQuant `iso3` KV cache compression allows full context usage within a 12GB VRAM envelope (e.g., RTX 4070 Super).
   *   **CPU-Bound Embeddings:** Embedding and reranking models are pinned to the CPU to prevent GPU OOM crashes.

4. **Modern Interface**
   *   **Chainlit UI:** A professional researcher-focused chat interface with clickable citation cards and side-panel source visualization.
   *   **FastAPI Backend:** A production-ready API layer exposing `/query` and `/health` endpoints.

## 🛠 Configuration

### 1. Create your `.env` file
```bash
cp .env.example .env
```

### 2. Configure Variable Definitions
| Variable | Description | Default |
| :--- | :--- | :--- |
| `LLAMA_SERVER_PATH` | Path to `llama-server.exe`. | `C:\tools\llama-server.exe` |
| `MODEL_PATH` | Path to GGUF model file. | `./models/model.gguf` |
| `LLAMA_PORT` | Local LLM server port. | `8080` |
| `QDRANT_HOST` | Qdrant hostname. | `localhost` |
| `QDRANT_PORT` | Qdrant port. | `6333` |

---

## 🚀 Usage

### 1. Boot the Vector DB
```bash
docker-compose up -d
```

### 2. Ingest Data
**Manual PDF Ingestion:**
```bash
python ingest_pdf.py path/to/document.pdf "FDA" "Label" --compound "Semaglutide" --date "2024-01-01"
```

**API Ingestion:**
```bash
python ingest_api.py --source clinical_trials --query "NCT06014450" --compound "Semaglutide"
python ingest_api.py --source fda --query "Ozempic" --compound "Semaglutide" --limit 1
python ingest_api.py --source pubmed --query "Semaglutide weight loss" --limit 5
```

### 3. Start the Interface
**Chainlit UI (Recommended):**
```bash
chainlit run app.py
```

**FastAPI Backend:**
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

**CLI Chat:**
```bash
python chat.py
```
