import sys
import os
import argparse
import uuid
from dotenv import load_dotenv
from src.ingest import PDFIngestor
from src.vector_store import VectorStore
from src.logger import logger

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
        logger.error(f"File '{pdf_path}' not found.")
        sys.exit(1)

    logger.info(f"Parsing PDF: {pdf_path}")
    ingestor = PDFIngestor()
    context = {
        "source": source_name,
        "doc_type": doc_type,
        "compound": args.compound,
        "date": args.date
    }
    
    try:
        chunks = ingestor.process(pdf_path, context)
        logger.info(f"Split into {len(chunks)} chunks using hierarchical routing.")
    except Exception as e:
        logger.error(f"Failed to process PDF: {e}")
        sys.exit(1)

    logger.info("Connecting to Qdrant Vector Store...")
    vs = VectorStore(
        host=os.getenv("QDRANT_HOST", "localhost"), 
        port=int(os.getenv("QDRANT_PORT", 6333))
    )

    logger.info("Embedding and storing chunks. This will hit your CPU hard. Wait...")
    for i, chunk in enumerate(chunks):
        payload = chunk['metadata'].copy()
        payload['id'] = str(uuid.uuid4())
        try:
            vs.ingest(chunk['text'], payload)
            if i > 0 and i % 10 == 0:
                logger.info(f"  Ingested {i}/{len(chunks)} chunks...")
        except Exception as e:
            logger.error(f"Failed to ingest chunk {i}: {e}")

    logger.info("Ingestion complete!")

if __name__ == "__main__":
    main()
