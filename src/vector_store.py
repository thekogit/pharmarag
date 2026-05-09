# Qdrant for hybrid search (BM25 sparse + Dense)
# CPU-side Octen-Embedding-4B restricted to CPU to avoid GPU OOM.

import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseVector, PointStruct, Prefetch, FusionQuery, Fusion
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from src.logger import logger

class VectorStore:
    def __init__(self, host=None, port=6333, path=None):
        self.path = path or os.getenv("QDRANT_PATH")
        self.host = host or os.getenv("QDRANT_HOST", "localhost")
        self.port = int(port or os.getenv("QDRANT_PORT", 6333))
        self.collection_name = os.getenv("COLLECTION_NAME", "pharma_docs")
        
        if self.path:
            logger.info(f"Initializing Qdrant in local storage mode at: {self.path}")
            self.client = QdrantClient(path=self.path)
        else:
            self.client = QdrantClient(host=self.host, port=self.port)
            
        self._embedder = None
        self._sparse_embedder = None
        self._initialized = False

    @property
    def embedder(self):
        if self._embedder is None:
            logger.info("Loading dense embedding model (Octen-Embedding-4B)...")
            model_path = os.getenv("EMBEDDING_MODEL_PATH", "./models/Octen-Embedding-4B")
            if not os.path.exists(model_path):
                # Fallback to HF if local path doesn't exist
                model_path = "Octen/Octen-Embedding-4B"
            self._embedder = SentenceTransformer(model_path, device="cpu")
        return self._embedder

    @property
    def sparse_embedder(self):
        if self._sparse_embedder is None:
            logger.info("Loading sparse embedding model (SPLADE/BM25)...")
            # fastembed handles downloading/caching automatically
            self._sparse_embedder = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
        return self._sparse_embedder

    def _ensure_initialized(self):
        if self._initialized:
            return
        try:
            try:
                exists = self.client.collection_exists(collection_name=self.collection_name)
            except AttributeError:
                exists = self.client.exists_collection(collection_name=self.collection_name)
                
            if not exists:
                logger.info(f"Creating collection: {self.collection_name} with hybrid support")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": VectorParams(size=2560, distance=Distance.COSINE)
                    },
                    sparse_vectors_config={
                        "sparse": SparseVectorParams()
                    }
                )
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant at {self.host}:{self.port}: {e}")
            raise

    def ingest(self, text: str, payload: dict):
        try:
            self._ensure_initialized()
            dense_vector = self.embedder.encode(text).tolist()
            
            # Generate sparse vector
            sparse_result = list(self.sparse_embedder.embed([text]))[0]
            
            # Debugging sparse_result
            if isinstance(sparse_result, dict):
                indices = sparse_result["indices"]
                values = sparse_result["values"]
            else:
                indices = sparse_result.indices
                values = sparse_result.values

            sparse_vector = SparseVector(
                indices=indices.tolist() if hasattr(indices, "tolist") else list(indices),
                values=values.tolist() if hasattr(values, "tolist") else list(values)
            )
            
            # Ensure payload has the text and meta structure expected by orchestrator
            formatted_payload = {
                "text": text,
                "meta": payload
            }
            
            point_id = payload.get("id") or str(hash(text))
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector={
                            "dense": dense_vector,
                            "sparse": sparse_vector
                        },
                        payload=formatted_payload
                    )
                ]
            )
            logger.debug(f"Ingested point {point_id}")
        except Exception as e:
            import traceback
            logger.error(f"Failed to ingest chunk: {e}\n{traceback.format_exc()}")
            raise

    def search(self, query: str, limit: int = 10):
        self._ensure_initialized()
        dense_vector = self.embedder.encode(query).tolist()
        
        sparse_result = list(self.sparse_embedder.embed([query]))[0]
        sparse_vector = SparseVector(
            indices=sparse_result.indices.tolist(),
            values=sparse_result.values.tolist()
        )
        
        # Native Hybrid Search (Fusion)
        # Note: Depending on Qdrant version, you might use Prefetch or separate calls
        # Here we use the query_points API which is cleaner in newer versions
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=limit
                ),
                Prefetch(
                    query=sparse_vector,
                    using="sparse",
                    limit=limit
                )
            ],
            query=FusionQuery(
                fusion=Fusion.RRF
            ),
            limit=limit
        )
        return results.points
