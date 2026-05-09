import requests
from src.logger import logger

class ClinicalTrialsFetcher:
    """
    Fetcher for ClinicalTrials.gov V2 API data.
    """
    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

    def fetch_by_id(self, nct_id: str) -> dict:
        """
        Fetches full study JSON from ClinicalTrials.gov V2 API by NCT ID.
        """
        logger.info(f"Fetching Clinical Trial data for ID: {nct_id}")
        url = f"{self.BASE_URL}/{nct_id}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch trial {nct_id}: {e}")
            raise

    def transform(self, data: dict) -> dict:
        """
        Transforms complex ClinicalTrials.gov JSON into a sectioned dictionary
        compatible with the RAG ingestion pipeline.
        """
        protocol = data.get("protocolSection", {})
        
        # Identification Module
        ident = protocol.get("identificationModule", {})
        nct_id = ident.get("nctId", "Unknown NCT ID")
        brief_title = ident.get("briefTitle", "")
        official_title = ident.get("officialTitle", "")
        
        # Description Module
        desc = protocol.get("descriptionModule", {})
        brief_summary = desc.get("briefSummary", "")
        detailed_description = desc.get("detailedDescription", "")
        
        # Eligibility Module
        eligibility = protocol.get("eligibilityModule", {})
        criteria = eligibility.get("eligibilityCriteria", "")
        
        # Construct CLINICAL_STUDIES section
        clinical_studies_text = f"Title: {brief_title}\n"
        if official_title and official_title != brief_title:
            clinical_studies_text += f"Official Title: {official_title}\n"
        
        clinical_studies_text += f"\nSummary:\n{brief_summary}\n"
        
        if detailed_description:
            clinical_studies_text += f"\nDetailed Description:\n{detailed_description}\n"
            
        transformed = {
            "CLINICAL_STUDIES": clinical_studies_text,
            "ELIGIBILITY": criteria,
            "_metadata": {
                "source": "ClinicalTrials.gov",
                "doc_type": "Trial Protocol",
                "external_id": nct_id,
                "title": brief_title
            }
        }
        
        return transformed
