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
    vs = VectorStore()

    print(f"Fetching from {args.source} for query: {args.query}...")

    chunks_to_ingest = []

    if args.source == "clinical_trials":
        fetcher = ClinicalTrialsFetcher()
        # Handle both single ID and potential future search
        try:
            data = fetcher.fetch_by_id(args.query)
            sections = fetcher.transform(data)
            chunks = ingestor.chunk_sections(sections)
            for c in chunks:
                c["doc_type"] = "Clinical Trial"
            chunks_to_ingest.extend(chunks)
        except Exception as e:
            print(f"Error fetching from ClinicalTrials.gov: {e}")

    elif args.source == "fda":
        fetcher = OpenFDAFetcher()
        results = fetcher.fetch_by_name(args.query, limit=args.limit)
        for res in results:
            sections = fetcher.transform(res)
            chunks = ingestor.chunk_sections(sections)
            for c in chunks:
                c["doc_type"] = "Drug Label"
            chunks_to_ingest.extend(chunks)

    elif args.source == "pubmed":
        fetcher = PubMedFetcher()
        results = fetcher.fetch_abstracts(args.query, limit=args.limit)
        for res in results:
            sections = fetcher.transform(res)
            chunks = ingestor.chunk_sections(sections)
            for c in chunks:
                # PubMed records might have their own date
                c["doc_type"] = "Journal Article"
            chunks_to_ingest.extend(chunks)

    print(f"Ingesting {len(chunks_to_ingest)} chunks into Vector Store...")
    for i, chunk in enumerate(chunks_to_ingest):
        payload = {
            "id": str(uuid.uuid4()),
            "source": args.source.upper(),
            "doc_type": chunk.get("doc_type", "API Data"),
            "section_name": chunk["metadata"]["section"],
            "chunk_index": i,
            "compound": args.compound,
            "date": date_val
        }
        vs.ingest(chunk["text"], payload)
        if i % 10 == 0:
            print(f"  Ingested {i}/{len(chunks_to_ingest)} chunks...")

    print("Ingestion complete!")

if __name__ == "__main__":
    main()
