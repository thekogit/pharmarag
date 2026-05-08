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
        params = {
            "search": f'(openfda.brand_name:"{name}"+openfda.generic_name:"{name}")',
            "limit": limit
        }
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
                    
        return transformed
