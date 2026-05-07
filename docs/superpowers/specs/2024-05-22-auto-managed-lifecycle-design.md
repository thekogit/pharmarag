# Design Doc: 2024-05-22-auto-managed-lifecycle-design

## Problem Statement
The current `chat.py` assumes the backend `llama-server` is already running. If it's not, the RAG application fails when the user attempts to query. We need to automate the detection and startup of the `llama-server` to improve user experience.

## Proposed Solution
Modify `chat.py` to use `InferenceEngine` for checking server status and starting it if necessary before entering the interactive loop.

## Architecture & Components

### `is_server_ready(url: str) -> bool`
- **Location:** `chat.py`
- **Purpose:** Checks if the llama-server is responding correctly on its health endpoint.
- **Implementation:** Uses `urllib.request.urlopen` to GET `{url}/health`. Returns `True` if status is 200, `False` otherwise.

### `main()` update
- **Location:** `chat.py`
- **Logic:**
    1. Instantiate `InferenceEngine`.
    2. Check `is_server_ready`.
    3. Start server if not ready.
    4. Wait up to 30 seconds for initialization.
    5. Proceed to query loop or exit on failure.

## Data Flow
1. User runs `python chat.py`.
2. Lifecycle management logic checks `http://localhost:{engine.port}`.
3. If down, `engine.start(wait=False)` is called.
4. Polling ensures server is ready before `rag_app.invoke` can be called.

## Error Handling
- Connection errors during polling are caught and treated as "not ready".
- 30-second timeout for server startup.
- Fatal exit if server fails to start.

## Testing Strategy
- **Reproduction:** Run `chat.py` without `llama-server` running.
- **Verification:** Observe "Llama server not responding. Starting it..." and subsequent "Server is READY."
- **Regression:** Run `chat.py` with `llama-server` already running. Observe it starting immediately without re-launching the server.
