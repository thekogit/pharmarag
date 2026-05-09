from Bio import Entrez
from typing import List
import os
from src.logger import logger

class PubMedFetcher:
    """
    Fetcher for PubMed abstracts using Bio.Entrez.
    """
    def __init__(self, email: str = None):
        Entrez.email = email or os.getenv("PUBMED_EMAIL", "your.email@example.com")

    def fetch_abstracts(self, query: str, limit: int = 10) -> List[dict]:
        """
        Searches PubMed and fetches abstracts for the given query.
        """
        logger.info(f"Searching PubMed for query: {query}")
        try:
            # 1. Search for PMIDs
            handle = Entrez.esearch(db="pubmed", term=query, retmax=limit)
            record = Entrez.read(handle)
            handle.close()
            
            id_list = record.get("IdList", [])
            if not id_list:
                logger.info("No results found in PubMed.")
                return []
                
            # 2. Fetch details for these PMIDs
            handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="xml")
            records = Entrez.read(handle)
            handle.close()
            
            results = []
            for article in records.get("PubmedArticle", []):
                citation = article.get("MedlineCitation", {})
                pmid = citation.get("PMID", "Unknown")
                
                article_data = citation.get("Article", {})
                title = article_data.get("ArticleTitle", "No Title")
                
                abstract_data = article_data.get("Abstract", {})
                abstract_texts = abstract_data.get("AbstractText", [])
                # AbstractText can be a list of strings (for structured abstracts)
                abstract = " ".join(abstract_texts) if abstract_texts else "No Abstract"
                
                # Extract date
                journal = article_data.get("Journal", {})
                pub_date = journal.get("JournalIssue", {}).get("PubDate", {})
                year = pub_date.get("Year", "Unknown")
                month = pub_date.get("Month", "Unknown")
                day = pub_date.get("Day", "Unknown")
                date_str = f"{year}-{month}-{day}"
                
                results.append({
                    "title": title,
                    "abstract": abstract,
                    "date": date_str,
                    "pmid": pmid
                })
                
            return results
        except Exception as e:
            logger.error(f"Error fetching from PubMed: {e}")
            return []

    def transform(self, record: dict) -> dict:
        """
        Transforms a PubMed record into sections.
        """
        text = f"Title: {record['title']}\n\nAbstract:\n{record['abstract']}"
        
        transformed = {
            "CLINICAL_STUDIES": text,
            "_metadata": {
                "source": "PUBMED",
                "doc_type": "Journal Article",
                "external_id": record["pmid"],
                "date": record["date"]
            }
        }
        return transformed
