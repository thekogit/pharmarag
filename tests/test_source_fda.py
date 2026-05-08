import unittest
from unittest.mock import patch, MagicMock
from src.sources.fda import OpenFDAFetcher

class TestOpenFDAFetcher(unittest.TestCase):
    def setUp(self):
        self.fetcher = OpenFDAFetcher()

    @patch("requests.get")
    def test_fetch_by_name_success(self, mock_get):
        # Configure mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [{"id": "test_id", "openfda": {"brand_name": ["Ibuprofen"]}}]
        }
        mock_get.return_value = mock_response

        # Execute
        result = self.fetcher.fetch_by_name("ibuprofen")

        # Verify
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "test_id")
        self.assertIn("https://api.fda.gov/drug/label.json", mock_get.call_args[0][0])
        self.assertIn('brand_name:"ibuprofen"', mock_get.call_args[0][0])

if __name__ == "__main__":
    unittest.main()
