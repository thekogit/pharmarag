"""HTTP API: `uvicorn src.api:app`."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from src.pipeline import Pipeline, build_pipeline

app = FastAPI(title="PharmaRAG-Diagnostic")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    outcome: str
    structured: dict[str, Any] | None
    gate: dict[str, Any] | None
    passages: list[dict[str, Any]]


@lru_cache(maxsize=1)
def pipeline() -> Pipeline:
    return build_pipeline()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    out = pipeline().run(req.question)
    return QueryResponse(
        answer=out["answer"], outcome=out["outcome"], structured=out.get("structured"), gate=out.get("gate"),
        passages=[{"n": i, **{k: h.payload.get(k) for k in ("compound", "publisher", "effective_date",
                                                            "section", "text")}}
                  for i, h in enumerate(out.get("hits", []), 1)])
