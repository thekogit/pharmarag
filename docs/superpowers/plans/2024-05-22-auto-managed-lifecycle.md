# Auto-Managed Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically detect if the llama-server is running and start it if missing when `chat.py` is launched.

**Architecture:** Use `urllib.request` to poll the server's health endpoint. Leverage the `InferenceEngine` to launch the server as a background process.

**Tech Stack:** Python, `urllib`, `subprocess`, `unittest.mock`.

---

### Task 1: Implement `is_server_ready` with TDD

**Files:**
- Modify: `chat.py`
- Create: `tests/test_chat_lifecycle.py`

- [ ] **Step 1: Write the failing test for `is_server_ready`**

```python
import unittest
from unittest.mock import patch, MagicMock
from chat import is_server_ready

class TestChatLifecycle(unittest.TestCase):
    @patch("urllib.request.urlopen")
    def test_is_server_ready_success(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.getcode.return_value = 200
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        assert is_server_ready("http://localhost:8080") is True

    @patch("urllib.request.urlopen")
    def test_is_server_ready_failure(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("Connection refused")
        
        assert is_server_ready("http://localhost:8080") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chat_lifecycle.py`
Expected: FAIL (ImportError: cannot import name 'is_server_ready' from 'chat')

- [ ] **Step 3: Implement `is_server_ready` in `chat.py`**

```python
import urllib.request

def is_server_ready(url):
    try:
        with urllib.request.urlopen(f"{url}/health") as response:
            return response.getcode() == 200
    except:
        return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_chat_lifecycle.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add chat.py tests/test_chat_lifecycle.py
git commit -m "test: add is_server_ready and its tests"
```

### Task 2: Implement Auto-Start Logic in `main()`

**Files:**
- Modify: `chat.py`

- [ ] **Step 1: Update imports and modify `main()`**

```python
import sys
import time
import urllib.request
from src.orchestrator import rag_app
from src.engine import InferenceEngine

# ... is_server_ready ...

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

    print("==================================================")
    # ... rest of existing main ...
```

- [ ] **Step 2: Manual Verification**
1. Stop any running llama-server.
2. Run `python chat.py`.
3. Verify it prints "Starting it...", waits, and then shows "Server is READY."

- [ ] **Step 3: Commit**

```bash
git add chat.py
git commit -m "feat: auto-manage llama-server lifecycle in chat.py"
```
