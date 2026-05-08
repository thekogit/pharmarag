from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from src.orchestrator import rag_app

app = FastAPI(title="PharmaRAG API")

class QueryRequest(BaseModel):
    question: str

class DocumentModel(BaseModel):
    text: str
    meta: Dict[str, Any]

class QueryResponse(BaseModel):
    answer: str
    docs: List[DocumentModel]

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    result = rag_app.invoke({"question": request.question})
    return {
        "answer": result.get("answer", ""),
        "docs": result.get("docs", [])
    }
