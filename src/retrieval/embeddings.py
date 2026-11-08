"""Dense embeddings from local Ollama (bge-m3) and sparse BM25 from fastembed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import httpx


class DenseEmbedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@dataclass
class SparseVec:
    indices: list[int]
    values: list[float]


class SparseEmbedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[SparseVec]: ...

    def embed_query(self, text: str) -> SparseVec: ...


# Models trained with task prefixes lose accuracy without them (nomic-embed-text documents this).
_PREFIXES: dict[str, tuple[str, str]] = {"nomic-embed-text": ("search_document: ", "search_query: ")}


class OllamaEmbedder:
    """Dense embeddings from the local Ollama daemon (bge-m3 by default, 1024-d) via /api/embed."""

    def __init__(self, host: str, model: str, batch_size: int = 16, timeout: float = 300) -> None:
        self.name = model
        self._url = host.rstrip("/") + "/api/embed"
        self._batch = batch_size
        self._client = httpx.Client(timeout=timeout)
        self._doc_prefix, self._query_prefix = _PREFIXES.get(model.split(":")[0], ("", ""))

    def _post(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            r = self._client.post(self._url, json={"model": self.name, "input": texts[i : i + self._batch]})
            r.raise_for_status()
            out.extend(r.json()["embeddings"])
        return out

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._post([self._doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._post([self._query_prefix + text])[0]


class Bm25Embedder:
    """Qdrant/bm25 term weights; IDF is applied server-side by the collection's IDF modifier."""

    def __init__(self) -> None:
        from fastembed import SparseTextEmbedding

        self._model: Any = SparseTextEmbedding(model_name="Qdrant/bm25")

    @staticmethod
    def _conv(e: Any) -> SparseVec:
        return SparseVec(indices=[int(i) for i in e.indices], values=[float(v) for v in e.values])

    def embed_documents(self, texts: list[str]) -> list[SparseVec]:
        return [self._conv(e) for e in self._model.embed(texts)]

    def embed_query(self, text: str) -> SparseVec:
        return self._conv(next(iter(self._model.query_embed(text))))
