import sys
import os
import argparse
from dotenv import load_dotenv
from src.ingest import PDFIngestor
from src.vector_store import VectorStore
import uuid

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Ingest a PDF into the PharmaRAG system.")
    parser.add_argument("pdf_path", help="Path to the PDF file to ingest.")
    parser.add_argument("source_name", nargs="?", default="Unknown Source", help="Source name of the document.")
    parser.add_argument("doc_type", nargs="?", default="Document", help="Type of the document.")
    parser.add_argument("--compound", default="Unknown Compound", help="Name of the drug/compound.")
    parser.add_argument("--date", default="2024-01-01", help="Date of the document in ISO format.")
    
    args = parser.parse_args()

    pdf_path = args.pdf_path
    source_name = args.source_name
    doc_type = args.doc_type

    if not os.path.exists(pdf_path):
        print(f"Error: File '{pdf_path}' not found.")
        sys.exit(1)

    print(f"Parsing PDF: {pdf_path}")
    ingestor = PDFIngestor()
    context = {
        "source": source_name,
        "doc_type": doc_type,
        "compound": args.compound,
        "date": args.date
    }
    chunks = ingestor.process(pdf_path, context)
    print(f"Split into {len(chunks)} chunks using hierarchical routing.")

    print("Connecting to Qdrant Vector Store...")
    vs = VectorStore(
        host=os.getenv("QDRANT_HOST", "localhost"), 
        port=int(os.getenv("QDRANT_PORT", 6333))
    )

    print("Embedding and storing chunks. This will hit your CPU hard. Wait...")
    for i, chunk in enumerate(chunks):
        payload = chunk['metadata'].copy()
        payload['id'] = str(uuid.uuid4())
        vs.ingest(chunk['text'], payload)
        if i % 10 == 0:
            print(f"  Ingested {i}/{len(chunks)} chunks...")

    print("Ingestion complete!")

if __name__ == "__main__":
    main()
