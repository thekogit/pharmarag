# Pharma-RAG

A local-first Retrieval-Augmented Generation (RAG) pipeline designed for life science and regulatory data.

## System Architecture & Data Sources

The system is engineered to process clinical and regulatory data and synthesize answers grounded strictly in retrieved context.

### Data Sources
The ingestion pipeline is capable of handling unstructured and semi-structured documents typically found in the pharmaceutical industry:
*   **ClinicalTrials.gov:** Trial protocols, outcomes, and phase data.
*   **openFDA:** Drug labels, adverse event reports, and enforcement actions.
*   **PubMed:** Peer-reviewed biomedical literature.
*   **EMA EPARs:** European Public Assessment Reports.
*   **ICH Guidelines:** International Council for Harmonisation PDFs.

### Architectural Strategies & Choices

1. **Master Prompt Synthesis**
   *   Compliance is handled at runtime using a Master Prompt within a LangGraph state machine (`src/orchestrator.py`). The model is utilized strictly as a synthesis engine over the retrieved context and is instructed to cite sources or refuse to answer if the context is insufficient, thereby avoiding the catastrophic forgetting often associated with fine-tuning.

2. **RotorQuant `iso3` KV Cache (12GB VRAM Envelope)**
   *   `Qwen3.5-9B-DeepSeek-V4-Flash` is served via `llama.cpp` using RotorQuant `iso3` compression. This results in a 10.3x reduction in memory footprint, allowing the inference engine to run on a single 12GB RTX 4070 Super. 

3. **CPU-Bound Embeddings (`src/vector_store.py`)**
   *   To prevent Out Of Memory (OOM) crashes during concurrent inference and retrieval operations, the embedding model (`Octen-Embedding-4B`) is pinned strictly to the CPU.

4. **Hybrid Search Vector DB (Qdrant)**
   *   Qdrant is utilized for its native support of Hybrid Search. Dense semantic vectors (from Octen-Embedding-4B) are combined with sparse keyword vectors (BM25 via fastembed) to support both conceptual and exact-match queries.

5. **PDF Parsing (`src/ingest.py`)**
   *   `PyMuPDF` is implemented for fast text extraction, and `pdfplumber` is utilized for the structural extraction of complex regulatory tables.

## 🛠 Configuration

Pharma-RAG uses environment variables for all local paths and connection settings. 

### 1. Create your `.env` file
Copy the provided template to start:
```bash
cp .env.example .env
```

### 2. Configure Variable Definitions
Open `.env` and fill in the following keys:

| Variable | Description | Example / Default |
| :--- | :--- | :--- |
| `LLAMA_SERVER_PATH` | **REQUIRED**. Absolute path to your compiled `llama-server.exe`. | `C:\tools\llama-server.exe` |
| `MODEL_PATH` | **REQUIRED**. Path to your GGUF model file. | `./models/qwen3.5-9b.gguf` |
| `LLAMA_PORT` | The port the local LLM server will listen on. | `8080` |
| `NGL` | Number of layers to offload to GPU (set 0 for CPU-only). | `99` |
| `QDRANT_HOST` | Hostname for your Qdrant instance. | `localhost` |
| `QDRANT_PORT` | Port for your Qdrant instance. | `6333` |
| `OPENAI_API_BASE` | Base URL for LLM requests (matches local server). | `http://localhost:8080/v1` |

### 3. Troubleshooting Setup
- **Server Not Starting:** Ensure `LLAMA_SERVER_PATH` is the absolute path to the `.exe`, not just the folder.
- **Connection Refused:** If Qdrant is in Docker, ensure `QDRANT_PORT` matches the port mapped in your `docker-compose.yml`.
- **Empty Answers:** Check the `DEBUG` logs in the terminal. If "Retrieved 0 documents" appears, verify that you have run the ingestion script first.

---

## 🚀 Usage

The vector database runs in Docker, while the `llama.cpp` inference engine is managed automatically by the application.

1. **Boot the Vector DB**:
   ```bash
   docker-compose up -d
   ```

2. **Ingest Documents**:
   Parse a PDF, chunk it, and push it to the vector store:
   ```bash
   python ingest_pdf.py path/to/your/document.pdf "FDA" "Guidance Document"
   ```

3. **Query the System (Chat)**:
   The interactive CLI will automatically start the `llama-server` if it is not running:
   ```bash
   python chat.py
   ```