import argparse
from src.ingest import PDFIngestor
from src.vector_store import VectorStore

p = argparse.ArgumentParser(description="Ingest one drug-label PDF into Qdrant")
p.add_argument("pdf")
p.add_argument("--compound", required=True)
p.add_argument("--source", default="FDA Drug Label")
p.add_argument("--doc-type", default="Label")
a = p.parse_args()

chunks = PDFIngestor().process(a.pdf, {"compound": a.compound, "source": a.source, "doc_type": a.doc_type})
vs = VectorStore()
for c in chunks:
    vs.ingest(c["text"], c["metadata"])
print(f"Ingested {len(chunks)} chunks from {a.pdf}")
