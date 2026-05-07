# 2026-05-07 Environment Refactor & Lifecycle Management Design

## Goal
Remove hardcoded sensitive data (paths, user names) from the codebase by moving them to a `.env` file, and implement an auto-managed `llama-server` lifecycle to ensure the RAG app is "alive" before user interaction.

## Architecture
- **Environment Management**: Centralized configuration via `python-dotenv`. All scripts will load settings from `.env`.
- **Server Lifecycle**: `chat.py` acts as the entry point and orchestrates the `InferenceEngine`. It will check for server availability and start it if missing.
- **Diagnostic Layer**: Temporary print statements in `orchestrator.py` to trace the data flow from Qdrant retrieval to LLM generation.

## Component Changes

### 1. Environment Configuration (`.env.example`)
Define a template for:
- `LLAMA_SERVER_PATH`: Absolute path to `llama-server.exe`.
- `MODEL_PATH`: Absolute or relative path to the GGUF model.
- `LLAMA_PORT`: Default 8080.
- `QDRANT_HOST`: Default localhost.
- `QDRANT_PORT`: Default 6333.

### 2. Inference Engine (`src/engine.py`)
- Refactor `InferenceEngine` to accept parameters from environment variables.
- Add a `start_non_blocking()` method using `subprocess.Popen` without `.wait()` to allow `chat.py` to continue execution.

### 3. Orchestrator (`src/orchestrator.py`)
- Standardize environment variable usage for `ChatOpenAI` and `VectorStore`.
- Add diagnostic logging in `synth_node` to print the count of retrieved docs and the first 100 characters of the prompt context.

### 4. Vector Store (`src/vector_store.py`)
- Remove hardcoded "localhost" for Qdrant client.

### 5. CLI Entrance (`chat.py`)
- Implement a `wait_for_server(url, timeout)` function.
- On startup:
  1. Load `.env`.
  2. Check if `llama-server` is running.
  3. If not, trigger `engine.start_non_blocking()`.
  4. Wait for health check success.
  5. Start interactive loop.

## Success Criteria
- [ ] No hardcoded absolute user paths remain in the repo.
- [ ] `.env.example` provides clear instructions for new users.
- [ ] `chat.py` automatically starts the server if it's down.
- [ ] Answers are no longer empty (verified via diagnostic logs).
