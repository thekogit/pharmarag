import subprocess
import sys
import os
from dotenv import load_dotenv
from src.logger import logger

# Find the project root (one level up from src/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(project_root, '.env')

if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    # Fallback to default load_dotenv behavior
    load_dotenv()

class InferenceEngine:
    def __init__(self):
        # Load from .env with fallbacks
        self.server_path = os.getenv("LLAMA_SERVER_PATH")
        # Support legacy env keys if present
        legacy_model_dir = os.getenv("MODELS_DIR", "./models")
        legacy_model_name = os.getenv("MODEL_NAME", "model.gguf")
        legacy_model_path = os.path.join(legacy_model_dir, legacy_model_name)
        
        self.model_path = os.getenv("MODEL_PATH", legacy_model_path)
        self.ngl = os.getenv("NGL", "99")
        self.cache_type_k = os.getenv("CACHE_TYPE_K", "iso3")
        self.cache_type_v = os.getenv("CACHE_TYPE_V", "iso3")
        self.context_size = os.getenv("CONTEXT_SIZE", "32768")
        self.port = os.getenv("LLAMA_PORT", "8080")

    def is_running(self):
        """
        Checks if the llama-server is already running on the configured port.
        """
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', int(self.port))) == 0

    def start(self, wait=True):
        """
        Starts the llama-server if it's not already running.
        """
        if self.is_running():
            logger.info(f"InferenceEngine: llama-server already running on port {self.port}.")
            return None

        if not self.server_path:
            logger.error("FATAL: LLAMA_SERVER_PATH not found in environment.")
            logger.error("Please add 'LLAMA_SERVER_PATH=C:\\path\\to\\llama-server.exe' to your .env file.")
            sys.exit(1)
            
        if not os.path.exists(self.server_path):
            logger.error(f"FATAL: llama-server not found at {self.server_path}. Check your .env path.")
            sys.exit(1)
            
        if not os.path.exists(self.model_path):
            logger.error(f"FATAL: Model file not found at {self.model_path}. Run download_models.py first.")
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
        
        logger.info(f"Executing: {' '.join(cmd)}")
        process = subprocess.Popen(cmd)
        if wait:
            process.wait()
        return process

if __name__ == "__main__":
    engine = InferenceEngine()
    engine.start()
