from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

import pytest
from qdrant_client import QdrantClient

from src.ingestion.chunking import Chunk
from src.retrieval.embeddings import SparseVec
from src.retrieval.index import HybridIndex


class FakeDense:
    """Deterministic bag-of-words hashing into 64 dims: similar texts get similar vectors."""

    name = "fake-dense"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * 64
            for w in t.lower().split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
            out.append(v if any(v) else [1.0] + [0.0] * 63)
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class FakeSparse:
    def _vec(self, t: str) -> SparseVec:
        idx = sorted({int(hashlib.md5(w.encode()).hexdigest(), 16) % 10_000 for w in t.lower().split()})
        return SparseVec(indices=idx, values=[1.0] * len(idx))

    def embed_documents(self, texts: list[str]) -> list[SparseVec]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> SparseVec:
        return self._vec(text)


class ScriptedLLM:
    """Returns queued responses; records every message list it was sent."""

    def __init__(self, responses: list[str | dict[str, Any]]) -> None:
        self.responses = [r if isinstance(r, str) else json.dumps(r) for r in responses]
        self.calls: list[list[dict[str, str]]] = []
        self.model = "scripted"

    def chat(self, messages: list[dict[str, str]], *, json_mode: bool = False, max_tokens: int = 2048) -> str:
        self.calls.append(messages)
        return self.responses.pop(0)


def make_chunk(text: str, *, compound: str = "apixaban", set_id: str = "new", date: str = "2025-01-01",
               section: str = "DOSAGE", n: int = 0) -> Chunk:
    return Chunk(chunk_id=f"00000000-0000-0000-0000-{abs(hash((set_id, n, text))) % 10**12:012d}",
                 set_id=set_id, compound=compound, effective_date=date, publisher="P", section=section,
                 subsection="", part="full", kind="text", text=text)


@pytest.fixture
def chunk_factory() -> Callable[..., Chunk]:
    return make_chunk


@pytest.fixture
def memory_index() -> HybridIndex:
    idx = HybridIndex(QdrantClient(":memory:"), "test__fake", FakeDense(), FakeSparse())
    idx.recreate(dim=64)
    return idx
