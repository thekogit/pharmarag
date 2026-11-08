"""Hybrid Qdrant index: dense bge-m3 + sparse BM25, fused with RRF, with label-freshness metadata."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal

from qdrant_client import QdrantClient, models

from src.config import Settings
from src.ingestion.chunking import Chunk
from src.retrieval.embeddings import DenseEmbedder, SparseEmbedder

log = logging.getLogger(__name__)

Mode = Literal["dense", "sparse", "hybrid"]


@dataclass
class Hit:
    chunk_id: str
    score: float
    payload: dict[str, Any]

    @property
    def text(self) -> str:
        return str(self.payload["text"])


def collection_name(base: str, embed_model: str) -> str:
    """The embedding model is part of the name, so a model change can never mix vector spaces."""
    return f"{base}__{re.sub(r'[^a-z0-9]+', '-', embed_model.lower()).strip('-')}"


def mark_current(chunks: list[Chunk]) -> dict[str, bool]:
    """Per compound, only the label with the latest effective_date is current."""
    latest: dict[str, str] = defaultdict(str)
    for c in chunks:
        latest[c.compound] = max(latest[c.compound], c.effective_date)
    return {c.chunk_id: c.effective_date == latest[c.compound] for c in chunks}


def embed_text(c: Chunk) -> str:
    """What gets embedded: the label context plus the chunk. The stored text stays verbatim."""
    sub = f" / {c.subsection}" if c.subsection else ""
    return f"{c.compound} | {c.section}{sub}\n{c.text}"


class HybridIndex:
    def __init__(self, client: QdrantClient, collection: str, dense: DenseEmbedder,
                 sparse: SparseEmbedder) -> None:
        self.client = client
        self.collection = collection
        self.dense = dense
        self.sparse = sparse

    @classmethod
    def from_settings(cls, s: Settings, dense: DenseEmbedder, sparse: SparseEmbedder) -> HybridIndex:
        client = QdrantClient(url=s.qdrant_url) if s.qdrant_url else QdrantClient(path=s.qdrant_path)
        return cls(client, collection_name(s.collection, dense.name), dense, sparse)

    def exists(self) -> bool:
        return bool(self.client.collection_exists(self.collection))

    def recreate(self, dim: int, *, payload_indexes: bool = False) -> None:
        """Drop and create the collection. Payload indexes only matter (and only work) in server mode."""
        if self.exists():
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config={"dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        if not payload_indexes:
            return
        for field, schema in (("compound", models.PayloadSchemaType.KEYWORD),
                              ("section", models.PayloadSchemaType.KEYWORD),
                              ("is_current", models.PayloadSchemaType.BOOL)):
            self.client.create_payload_index(self.collection, field, schema)

    def upsert(self, chunks: list[Chunk], batch: int = 32) -> None:
        current = mark_current(chunks)
        for i in range(0, len(chunks), batch):
            part = chunks[i : i + batch]
            texts = [embed_text(c) for c in part]
            dense = self.dense.embed(texts)
            sparse = self.sparse.embed_documents(texts)
            points = [
                models.PointStruct(
                    id=c.chunk_id,
                    vector={"dense": d, "bm25": models.SparseVector(indices=s.indices, values=s.values)},
                    payload={**c.payload(), "is_current": current[c.chunk_id]},
                )
                for c, d, s in zip(part, dense, sparse, strict=True)
            ]
            self.client.upsert(self.collection, points=points)
            log.info("indexed %d/%d", min(i + batch, len(chunks)), len(chunks))

    def search(self, query: str, *, limit: int = 20, mode: Mode = "hybrid", fresh_only: bool = True,
               compounds: list[str] | None = None) -> list[Hit]:
        if not self.exists():
            raise RuntimeError(
                f"Collection {self.collection!r} not found. Build it with `python -m scripts.build_index` "
                "(the name includes the embedding model, so a changed EMBED_MODEL needs a rebuild).")
        must: list[models.Condition] = []
        if fresh_only:
            must.append(models.FieldCondition(key="is_current", match=models.MatchValue(value=True)))
        if compounds:
            must.append(models.FieldCondition(key="compound", match=models.MatchAny(any=compounds)))
        flt = models.Filter(must=must) if must else None
        dense_q = self.dense.embed_query(query) if mode != "sparse" else None
        sparse_q = self.sparse.embed_query(query) if mode != "dense" else None
        sv = models.SparseVector(indices=sparse_q.indices, values=sparse_q.values) if sparse_q else None
        if mode == "dense":
            res = self.client.query_points(self.collection, query=dense_q, using="dense", limit=limit,
                                           query_filter=flt, with_payload=True)
        elif mode == "sparse":
            res = self.client.query_points(self.collection, query=sv, using="bm25", limit=limit,
                                           query_filter=flt, with_payload=True)
        else:
            res = self.client.query_points(
                self.collection,
                prefetch=[models.Prefetch(query=dense_q, using="dense", limit=limit * 2, filter=flt),
                          models.Prefetch(query=sv, using="bm25", limit=limit * 2, filter=flt)],
                query=models.FusionQuery(fusion=models.Fusion.RRF), limit=limit, with_payload=True)
        return [Hit(str(p.id), float(p.score), dict(p.payload or {})) for p in res.points]
