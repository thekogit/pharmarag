# Qdrant for hybrid search (BM25 sparse + Dense)
# CPU-side Octen-Embedding-4B restricted to CPU to avoid GPU OOM.

import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, host=None, port=6333):
        if host is None:
            host = os.getenv("QDRANT_HOST", "localhost")
        # Pinning the 4B model to CPU. Put it on the GPU during inference and watch the node crash.
        self.embedder = SentenceTransformer("Octen/Octen-Embedding-4B", device="cpu")
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = "pharma_docs"
        self._initialized = False

    def _ensure_initialized(self):
        if self._initialized:
            return
        try:
            if not self.client.collection_exists(collection_name=self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=2560, distance=Distance.COSINE),
                )
            self._initialized = True
        except Exception as e:
            print(f"CRITICAL: Failed to connect to Qdrant: {e}")
            raise

    def ingest(self, text: str, payload: dict):
        self._ensure_initialized()
        vector = self.embedder.encode(text).tolist()
        
        # Ensure payload has the text and meta structure expected by orchestrator
        formatted_payload = {
            "text": text,
            "meta": payload
        }
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[{
                "id": payload.get("id", str(hash(text))), # Lazy ID generation if not provided
                "vector": vector,
                "payload": formatted_payload
            }]
        )
