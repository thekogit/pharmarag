import sys
import os
from dotenv import load_dotenv
from src.ingest import PDFIngestor
from src.vector_store import VectorStore
import uuid

load_dotenv()

def main():
    if len(sys.argv) < 2:
        print("Usage: python ingest_pdf.py <path_to_pdf> [source_name] [doc_type]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    source_name = sys.argv[2] if len(sys.argv) > 2 else "Unknown Source"
    doc_type = sys.argv[3] if len(sys.argv) > 3 else "Document"

    if not os.path.exists(pdf_path):
        print(f"Error: File '{pdf_path}' not found.")
        sys.exit(1)

    print(f"Parsing PDF: {pdf_path}")
    ingestor = PDFIngestor()
    content = ingestor.process(pdf_path)

    # Very naive chunking for demonstration (split by paragraphs/newlines)
    chunks = [c.strip() for c in content.split('\n\n') if len(c.strip()) > 50]
    print(f"Split into {len(chunks)} chunks.")

    print("Connecting to Qdrant Vector Store...")
    vs = VectorStore(
        host=os.getenv("QDRANT_HOST", "localhost"), 
        port=int(os.getenv("QDRANT_PORT", 6333))
    )

    print("Embedding and storing chunks. This will hit your CPU hard. Wait...")
    for i, chunk in enumerate(chunks):
        payload = {
            "id": str(uuid.uuid4()),
            "source": source_name,
            "doc_type": doc_type,
            "chunk_index": i
        }
        vs.ingest(chunk, payload)
        if i % 10 == 0:
            print(f"  Ingested {i}/{len(chunks)} chunks...")

    print("Ingestion complete!")

if __name__ == "__main__":
    main()
