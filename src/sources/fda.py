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
        query = f'?search=(openfda.brand_name:"{name}"+openfda.generic_name:"{name}")&limit={limit}'
        url = self.BASE_URL + query
        response = requests.get(url)
        
        if response.status_code != 200:
            return []
            
        return response.json().get("results", [])

    def transform(self, result: dict) -> dict:
        """
        Transforms openFDA JSON into standardized sections.
        """
        # Minimal implementation for now
        return {}
