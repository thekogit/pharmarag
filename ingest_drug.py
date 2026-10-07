import os
import argparse
from src.sources.fda import OpenFDAFetcher
from src.sources.pubmed import PubMedFetcher
from src.sources.clinical_trials import ClinicalTrialsFetcher
from src.orchestrator import get_vs
from src.ingest import RecursiveCharacterTextSplitter
from src.logger import logger
from dotenv import load_dotenv

load_dotenv()

def ingest_drug(drug_name: str):
    vs = get_vs()
    
    # 1. Fetch from FDA
    logger.info(f"--- Fetching FDA labels for {drug_name} ---")
    fda = OpenFDAFetcher()
    try:
        results = fda.fetch_by_name(drug_name, limit=1)
        for res in results:
            transformed = fda.transform(res)
            meta = transformed.pop("_metadata")
            for section, text in transformed.items():
                if section.startswith("_"): continue
                vs.ingest(f"[Section: {section}] {text}", meta)
    except Exception as e:
        logger.error(f"FDA Fetch failed: {e}")

    # 2. Fetch from PubMed
    logger.info(f"--- Fetching PubMed abstracts for {drug_name} ---")
    pubmed = PubMedFetcher()
    try:
        results = pubmed.fetch_abstracts(drug_name, limit=5)
        for res in results:
            transformed = pubmed.transform(res)
            meta = transformed.pop("_metadata")
            vs.ingest(transformed["CLINICAL_STUDIES"], meta)
    except Exception as e:
        logger.error(f"PubMed Fetch failed: {e}")

    logger.info(f"Ingestion complete for {drug_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest data for a specific drug")
    parser.add_argument("drug", help="Name of the drug to ingest")
    args = parser.parse_args()
    
    ingest_drug(args.drug)
