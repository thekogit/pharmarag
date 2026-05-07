# Refactor Inference Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `InferenceEngine` to use environment variables for all configurations and support non-blocking startup by returning the `subprocess.Popen` object.

**Architecture:** Use `os.getenv` with sensible defaults in `__init__`. Update `start` to take a `wait` parameter and return the process.

**Tech Stack:** Python, `subprocess`, `os`, `dotenv`, `pytest`.

---

### Task 1: Research and Setup

**Files:**
- Read: `src/engine.py`
- Read: `.env.example`

- [x] **Step 1: Verify current state of engine.py**
- [x] **Step 2: Verify .env.example**

### Task 2: Write Failing Tests for InferenceEngine

**Files:**
- Create: `tests/test_engine.py`

- [ ] **Step 1: Write tests for InferenceEngine initialization and start**

```python
import os
import unittest
from unittest.mock import patch, MagicMock
from src.engine import InferenceEngine

class TestInferenceEngine(unittest.TestCase):
    def setUp(self):
        # Clear relevant env vars
        self.env_patcher = patch.dict(os.environ, {
            "LLAMA_SERVER_PATH": "/mock/path/llama-server",
            "MODEL_PATH": "/mock/path/model.gguf",
            "NGL": "42",
            "CACHE_TYPE_K": "f16",
            "CACHE_TYPE_V": "f16",
            "CONTEXT_SIZE": "2048",
            "LLAMA_PORT": "9090"
        })
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()

    def test_init_loads_env_vars(self):
        engine = InferenceEngine()
        assert engine.server_path == "/mock/path/llama-server"
        assert engine.model_path == "/mock/path/model.gguf"
        assert engine.ngl == "42"
        assert engine.cache_type_k == "f16"
        assert engine.cache_type_v == "f16"
        assert engine.context_size == "2048"
        assert engine.port == "9090"

    @patch("os.path.exists")
    @patch("subprocess.Popen")
    def test_start_non_blocking(self, mock_popen, mock_exists):
        mock_exists.return_value = True
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        
        engine = InferenceEngine()
        process = engine.start(wait=False)
        
        assert process == mock_process
        mock_popen.assert_called_once()
        mock_process.wait.assert_not_called()

    @patch("os.path.exists")
    @patch("subprocess.Popen")
    def test_start_blocking(self, mock_popen, mock_exists):
        mock_exists.return_value = True
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        
        engine = InferenceEngine()
        process = engine.start(wait=True)
        
        assert process == mock_process
        mock_popen.assert_called_once()
        mock_process.wait.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_engine.py`
Expected: FAIL (AttributeError or mismatch in parameters)

### Task 3: Refactor InferenceEngine

**Files:**
- Modify: `src/engine.py`

- [ ] **Step 1: Implement the refactored InferenceEngine**

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
        
        print(f"Executing: {' '.join(cmd)}")
        process = subprocess.Popen(cmd)
        if wait:
            process.wait()
        return process

if __name__ == "__main__":
    engine = InferenceEngine()
    engine.start()
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/test_engine.py`
Expected: PASS

- [ ] **Step 3: Commit changes**

```bash
git add src/engine.py tests/test_engine.py
git commit -m "refactor: use env vars in InferenceEngine and support non-blocking start"
```
