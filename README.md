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

## Quick Start (Hybrid Docker/Local)

The vector database runs in Docker, while the `llama.cpp` inference engine runs natively on the host to utilize a custom Windows `turboquant` build that supports `iso3` KV cache compression.

1. **Configuration**:
   Copy the example environment file to create your local configuration:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` to configure your environment. You must set `LLAMA_SERVER_PATH` to the absolute path of your `llama-server.exe` and ensure `MODEL_PATH` points to your downloaded model. (The `.env` file is ignored by git).
   
2. **Download Weights**: 
   The model file `Qwen3.5-9B-DeepSeek-V4-Flash.i1-Q5_K_S.gguf` must be downloaded and placed directly into the `./models/` folder. 

3. **Boot the Vector DB**:
   ```bash
   docker-compose up -d
   ```
   The `pharmarag-qdrant` vector database will be started.

4. **Start the Inference Engine**:
   The local inference script is run via:
   ```bash
   python src/engine.py
   ```

5. **Ingest Documents**:
   The ingestion script is used to parse a PDF, chunk it, embed it on the CPU, and push it to Qdrant:
   ```bash
   python ingest_pdf.py path/to/your/document.pdf "FDA" "Guidance Document"
   ```

6. **Query the System (Chat)**:
   The interactive CLI can be launched to query the local model against the ingested documents:
   ```bash
   python chat.py
   ```