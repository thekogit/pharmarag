# Environment Refactor & Lifecycle Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove hardcoded paths, manage sensitive data via `.env`, and implement auto-managed `llama-server` lifecycle in `chat.py`.

**Architecture:** Centralized configuration via `python-dotenv`. `chat.py` orchestrates `InferenceEngine` for non-blocking server startup and health checks.

**Tech Stack:** Python, LangChain, LangGraph, python-dotenv, qdrant-client.

---

### Task 1: Environment Template

**Files:**
- Create: `.env.example`

- [ ] **Step 1: Create .env.example with placeholders**

```text
# Llama Server Configuration
LLAMA_SERVER_PATH=C:\path\to\llama-server.exe
MODEL_PATH=./models/model_name.gguf
LLAMA_PORT=8080
NGL=99
CONTEXT_SIZE=32768

# Vector Store Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333

# OpenAI Compatibility (for local llama-server)
OPENAI_API_BASE=http://localhost:8080/v1
OPENAI_API_KEY=none
```

- [ ] **Step 2: Commit**
```bash
git add .env.example
git commit -m "chore: add .env.example template"
```

---

### Task 2: Refactor Inference Engine

**Files:**
- Modify: `src/engine.py`

- [ ] **Step 1: Update InferenceEngine to use env vars and return process**

```python
import subprocess
import sys
import os
from dotenv import load_dotenv

load_dotenv()

class InferenceEngine:
    def __init__(self):
        # Load from .env with fallbacks
        self.server_path = os.getenv("LLAMA_SERVER_PATH")
        self.model_path = os.getenv("MODEL_PATH", "./models/model.gguf")
        self.ngl = os.getenv("NGL", "99")
        self.cache_type_k = os.getenv("CACHE_TYPE_K", "iso3")
        self.cache_type_v = os.getenv("CACHE_TYPE_V", "iso3")
        self.context_size = os.getenv("CONTEXT_SIZE", "32768")
        self.port = os.getenv("LLAMA_PORT", "8080")

    def start(self, wait=True):
        if not self.server_path or not os.path.exists(self.server_path):
            print(f"FATAL: llama-server not found at {self.server_path}. Check .env")
            sys.exit(1)
            
        cmd = [
            self.server_path,
            "-m", self.model_path,
            "-ngl", self.ngl,
            "--flash-attn", "on",
            "--cache-type-k", self.cache_type_k,
            "--cache-type-v", self.cache_type_v,
            "-c", self.context_size,
            "--jinja",
            "--host", "0.0.0.0",
            "--port", self.port
        ]
        
        print(f"Executing: {" ".join(cmd)}")
        process = subprocess.Popen(cmd)
        if wait:
            process.wait()
        return process
```

- [ ] **Step 2: Commit**
```bash
git add src/engine.py
git commit -m "refactor: use env vars in InferenceEngine and support non-blocking start"
```

---

### Task 3: Refactor Vector Store & Orchestrator

**Files:**
- Modify: `src/vector_store.py`
- Modify: `src/orchestrator.py`

- [ ] **Step 1: Remove hardcoded host in VectorStore**
Change `host="localhost"` to use `os.getenv("QDRANT_HOST", "localhost")`.

- [ ] **Step 2: Add diagnostic logging to Orchestrator**
In `synth_node`, add:
```python
print(f"DEBUG: Retrieved {len(s['docs'])} documents.")
# ... existing code ...
print(f"DEBUG: Context length: {len(ctx)} chars.")
```

- [ ] **Step 3: Commit**
```bash
git add src/vector_store.py src/orchestrator.py
git commit -m "refactor: use env vars for Qdrant and add diagnostics to orchestrator"
```

---

### Task 4: Auto-Managed Lifecycle in chat.py

**Files:**
- Modify: `chat.py`

- [ ] **Step 1: Implement server check and auto-start**

```python
import sys
import time
import urllib.request
from src.orchestrator import rag_app
from src.engine import InferenceEngine

def is_server_ready(url):
    try:
        with urllib.request.urlopen(f"{url}/health") as response:
            return response.getcode() == 200
    except:
        return False

def main():
    engine = InferenceEngine()
    base_url = f"http://localhost:{engine.port}"
    
    if not is_server_ready(base_url):
        print("Llama server not responding. Starting it...")
        engine.start(wait=False)
        
        print("Waiting for server to initialize...")
        for _ in range(30): # 30 seconds timeout
            if is_server_ready(base_url):
                print("Server is READY.")
                break
            time.sleep(1)
        else:
            print("FATAL: Server failed to start in time.")
            sys.exit(1)

    # ... existing interactive loop ...
```

- [ ] **Step 2: Commit**
```bash
git add chat.py
git commit -m "feat: auto-manage llama-server lifecycle in chat.py"
```

---

### Task 5: Documentation Update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add Setup Instructions**
Explain how to copy `.env.example` to `.env` and configure paths.

- [ ] **Step 2: Commit**
```bash
git add README.md
git commit -m "docs: add environment setup instructions"
```
