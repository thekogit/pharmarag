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
        # The current implementation uses different variable names or hardcoded values
        # This is expected to fail initially.
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
        # The current start() method doesn't take wait parameter and doesn't return process
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
