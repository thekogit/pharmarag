import os
import unittest
from unittest.mock import patch
from src.vector_store import VectorStore

class TestVectorStore(unittest.TestCase):
    def setUp(self):
        self._orig_qdrant_path = os.environ.get("QDRANT_PATH")
        self._orig_qdrant_host = os.environ.get("QDRANT_HOST")
        if "QDRANT_PATH" in os.environ:
            del os.environ["QDRANT_PATH"]

    def tearDown(self):
        if self._orig_qdrant_path is not None:
            os.environ["QDRANT_PATH"] = self._orig_qdrant_path
        elif "QDRANT_PATH" in os.environ:
            del os.environ["QDRANT_PATH"]

        if self._orig_qdrant_host is not None:
            os.environ["QDRANT_HOST"] = self._orig_qdrant_host
        elif "QDRANT_HOST" in os.environ:
            del os.environ["QDRANT_HOST"]

    @patch('src.vector_store.SentenceTransformer')
    @patch('src.vector_store.QdrantClient')
    def test_init_with_env_var(self, mock_qdrant, mock_transformer):
        os.environ["QDRANT_HOST"] = "remote-host"
        vs = VectorStore()
        mock_qdrant.assert_called_with(host="remote-host", port=6333)
        
    @patch('src.vector_store.SentenceTransformer')
    @patch('src.vector_store.QdrantClient')
    def test_init_default(self, mock_qdrant, mock_transformer):
        if "QDRANT_HOST" in os.environ:
            del os.environ["QDRANT_HOST"]
        vs = VectorStore()
        mock_qdrant.assert_called_with(host="localhost", port=6333)

    @patch('src.vector_store.SentenceTransformer')
    @patch('src.vector_store.QdrantClient')
    def test_init_explicit_host(self, mock_qdrant, mock_transformer):
        vs = VectorStore(host="explicit-host")
        mock_qdrant.assert_called_with(host="explicit-host", port=6333)

    @patch('src.vector_store.SentenceTransformer')
    @patch('src.vector_store.QdrantClient')
    def test_init_with_path(self, mock_qdrant, mock_transformer):
        vs = VectorStore(path="./custom_storage")
        mock_qdrant.assert_called_with(path="./custom_storage")

if __name__ == '__main__':
    unittest.main()
