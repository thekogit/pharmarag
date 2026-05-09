import argparse
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
from src.sources.clinical_trials import ClinicalTrialsFetcher
from src.sources.fda import OpenFDAFetcher
from src.sources.pubmed import PubMedFetcher
from src.ingest import PDFIngestor
from src.vector_store import VectorStore
from src.logger import logger

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Ingest pharmaceutical data from public APIs.")
    parser.add_argument("--source", choices=["clinical_trials", "fda", "pubmed"], required=True, help="API source to query")
    parser.add_argument("--query", required=True, help="Search query (NCT ID, Drug Name, or Keywords)")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of results to fetch")
    parser.add_argument("--compound", default="Unknown Compound", help="Drug/Compound name for metadata")
    parser.add_argument("--date", help="Document date in ISO format (default: current date)")

    args = parser.parse_args()
    date_val = args.date or datetime.now().strftime("%Y-%m-%d")

    ingestor = PDFIngestor()
    vs = VectorStore(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333))
    )

    logger.info(f"Fetching from {args.source} for query: {args.query}...")

    chunks_to_ingest = []
    
    common_context = {
        "source": args.source.upper(),
        "compound": args.compound,
        "date": date_val
    }

    def process_fetcher_result(sections, base_context):
        # Extract internal metadata if present
        internal_meta = sections.pop("_metadata", {})
        context = base_context.copy()
        context.update(internal_meta)
        return ingestor.chunk_sections(sections, context)

    if args.source == "clinical_trials":
        fetcher = ClinicalTrialsFetcher()
        try:
            data = fetcher.fetch_by_id(args.query)
            sections = fetcher.transform(data)
            chunks = process_fetcher_result(sections, common_context)
            chunks_to_ingest.extend(chunks)
        except Exception as e:
            logger.error(f"Error fetching from ClinicalTrials.gov: {e}")

    elif args.source == "fda":
        fetcher = OpenFDAFetcher()
        try:
            results = fetcher.fetch_by_name(args.query, limit=args.limit)
            for res in results:
                sections = fetcher.transform(res)
                chunks = process_fetcher_result(sections, common_context)
                chunks_to_ingest.extend(chunks)
        except Exception as e:
            logger.error(f"Error fetching from FDA: {e}")

    elif args.source == "pubmed":
        fetcher = PubMedFetcher()
        try:
            results = fetcher.fetch_abstracts(args.query, limit=args.limit)
            for res in results:
                sections = fetcher.transform(res)
                chunks = process_fetcher_result(sections, common_context)
                chunks_to_ingest.extend(chunks)
        except Exception as e:
            logger.error(f"Error fetching from PubMed: {e}")

    logger.info(f"Ingesting {len(chunks_to_ingest)} chunks into Vector Store...")
    for i, chunk in enumerate(chunks_to_ingest):
        payload = chunk["metadata"].copy()
        payload["id"] = str(uuid.uuid4())
        try:
            vs.ingest(chunk["text"], payload)
            if i > 0 and i % 10 == 0:
                logger.info(f"  Ingested {i}/{len(chunks_to_ingest)} chunks...")
        except Exception as e:
            logger.error(f"Failed to ingest chunk {i}: {e}")

    logger.info("Ingestion complete!")

if __name__ == "__main__":
    main()
