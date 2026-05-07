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
        """
        Starts the llama-server. We enforce RotorQuant iso3 compression.
        """
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
