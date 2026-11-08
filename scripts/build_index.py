"""Build the hybrid index: Docling parse (cached) -> chunks + coherence gate -> bge-m3 + BM25 -> Qdrant.

Writes a corpus report (chunk counts, coherence-gate results, current vs superseded labels) to
evals/results/corpus_report.json. Re-running rebuilds the collection from the cached parses.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import asdict, replace
from datetime import date

from src import log
from src.config import ROOT, Settings
from src.ingestion.chunking import Chunk, blocks_from_docling, chunk_label
from src.ingestion.dailymed import LabelMeta, fetch, load_manifest
from src.ingestion.parse import parse_pdf
from src.retrieval.embeddings import Bm25Embedder, OllamaEmbedder
from src.retrieval.index import HybridIndex, mark_current

logger = logging.getLogger("build_index")


def build_chunks(s: Settings) -> tuple[list[Chunk], list[LabelMeta]]:
    metas = fetch(load_manifest(ROOT / "corpus" / "manifest.toml"), s.raw_dir)
    chunks: list[Chunk] = []
    for m in metas:
        doc = parse_pdf(s.raw_dir / m.pdf_name, s.parsed_dir)
        cs = chunk_label(blocks_from_docling(doc), asdict(m))
        logger.info("%s %s: %d chunks", m.compound, m.effective_date, len(cs))
        chunks.extend(cs)
    return chunks, metas


def report(chunks: list[Chunk], metas: list[LabelMeta]) -> dict[str, object]:
    current = mark_current(chunks)
    tables = [c for c in chunks if c.kind == "table"]
    flagged = [c for c in tables if c.coherence_issues]
    return {
        "built": date.today().isoformat(),
        "labels": [{**asdict(m), "current": any(current[c.chunk_id] for c in chunks if c.set_id == m.set_id),
                    "chunks": sum(c.set_id == m.set_id for c in chunks)} for m in metas],
        "chunks": len(chunks),
        "chunks_current": sum(current.values()),
        "by_kind": dict(Counter(c.kind for c in chunks)),
        "coherence_gate": {
            "tables": len(tables),
            "tables_with_footnote_markers": sum(bool(set("*†‡§¶#Þßàèðø") & set(c.text)) for c in tables),
            "tables_with_unbound_footnotes": len(flagged),
            "issues": dict(Counter(i for c in flagged for i in c.coherence_issues)),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunks-only", action="store_true", help="parse and chunk, but do not embed or index")
    ap.add_argument("--embed-model", help="Ollama embedding model (default: EMBED_MODEL); builds its own collection")
    args = ap.parse_args()
    log.setup()
    s = Settings.load()
    if args.embed_model:
        s = replace(s, embed_model=args.embed_model)
    chunks, metas = build_chunks(s)
    rep = report(chunks, metas)
    out = ROOT / "evals" / "results" / "corpus_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("chunks", "chunks_current", "by_kind", "coherence_gate")}, indent=1))
    if args.chunks_only:
        return
    dense = OllamaEmbedder(s.ollama_host, s.embed_model)
    index = HybridIndex.from_settings(s, dense, Bm25Embedder())
    index.recreate(dim=len(dense.embed(["dimension probe"])[0]), payload_indexes=bool(s.qdrant_url))
    index.upsert(chunks)
    print(f"indexed {len(chunks)} chunks into {index.collection}")


if __name__ == "__main__":
    main()
