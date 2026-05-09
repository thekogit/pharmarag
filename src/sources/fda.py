from src.logger import logger
import requests

class OpenFDAFetcher:
    """
    Fetcher for openFDA drug label data.
    """
    BASE_URL = "https://api.fda.gov/drug/label.json"

    def fetch_by_name(self, name: str, limit: int = 1) -> list:
        """
        Fetches drug label data from openFDA by brand or generic name.
        """
        logger.info(f"Fetching from openFDA for query: {name}")
        # openFDA uses '+' as AND by default. For 'either', use 'OR'.
        # However, a general search on openfda fields is often more reliable.
        search_query = f'openfda.brand_name:"{name}" OR openfda.generic_name:"{name}"'
        params = {
            "search": search_query,
            "limit": limit
        }
        response = requests.get(self.BASE_URL, params=params)
        
        if response.status_code == 404:
            # Try a broader search if the specific field search fails
            logger.info(f"  FDA Specific search failed, trying broad search for: {name}")
            params["search"] = f'"{name}"'
            response = requests.get(self.BASE_URL, params=params)
            
        response.raise_for_status()
            
        return response.json().get("results", [])

    def transform(self, result: dict) -> dict:
        """
        Transforms openFDA JSON into standardized sections.
        """
        mapping = {
            "indications_and_usage": "INDICATIONS",
            "dosage_and_administration": "DOSAGE",
            "contraindications": "CONTRAINDICATIONS",
            "warnings": "WARNINGS",
            "warnings_and_precautions": "WARNINGS",
            "adverse_reactions": "ADVERSE_REACTIONS",
            "clinical_studies": "CLINICAL_STUDIES"
        }
        
        transformed = {}
        for fda_field, section_name in mapping.items():
            if fda_field in result:
                value = result[fda_field]
                if isinstance(value, list):
                    text = "\n".join(value)
                else:
                    text = str(value)
                
                # If section already exists (e.g. warnings and warnings_and_precautions), append
                if section_name in transformed:
                    transformed[section_name] += "\n" + text
                else:
                    transformed[section_name] = text
        
        # Standardize metadata
        label_id = result.get("id", "Unknown")
        brand = result.get("openfda", {}).get("brand_name", ["Unknown"])[0]
        
        # We store common metadata in a separate dict to be merged during ingestion
        transformed["_metadata"] = {
            "source": "FDA",
            "doc_type": "Drug Label",
            "compound": brand,
            "external_id": label_id
        }
                    
        return transformed
