"""Settings. This is the only module that reads the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # Generator LLM (OpenAI-compatible; Ollama Cloud beam:cloud by default)
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    # Local Ollama: dense embeddings and the judge
    ollama_host: str
    embed_model: str
    judge_model: str
    # Vector store
    qdrant_path: str
    qdrant_url: str
    collection: str
    # Retrieval
    reranker_model: str
    nli_model: str
    retrieval_mode: str
    rerank: bool
    freshness: bool
    gates: bool
    grounding: bool
    # Paths
    raw_dir: Path
    parsed_dir: Path

    @classmethod
    def load(cls) -> Settings:
        return cls(
            llm_base_url=os.getenv("LLM_BASE_URL", "https://ollama.com/v1"),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_model=os.getenv("LLM_MODEL", "beam:cloud"),
            ollama_host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
            embed_model=os.getenv("EMBED_MODEL", "bge-m3"),
            judge_model=os.getenv("JUDGE_MODEL", "hf.co/mradermacher/prometheus-7b-v2.0-GGUF:Q5_K_M"),
            qdrant_path=os.getenv("QDRANT_PATH", str(ROOT / "qdrant_storage")),
            qdrant_url=os.getenv("QDRANT_URL", ""),
            collection=os.getenv("COLLECTION_NAME", "fda_labels"),
            reranker_model=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            nli_model=os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-base"),
            retrieval_mode=os.getenv("RETRIEVAL_MODE", "hybrid"),
            rerank=_flag("RERANK", "1"),
            freshness=_flag("FRESHNESS", "1"),
            gates=_flag("GATES", "1"),
            grounding=_flag("GROUNDING", "1"),
            raw_dir=ROOT / "data" / "raw",
            parsed_dir=ROOT / "data" / "parsed",
        )
