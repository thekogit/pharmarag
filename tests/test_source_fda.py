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

    def test_transform(self):
        sample_result = {
            "indications_and_usage": ["For relief of pain."],
            "dosage_and_administration": ["Take 1 tablet."],
            "contraindications": ["None."],
            "warnings": ["May cause drowsiness."],
            "adverse_reactions": ["Nausea."],
            "clinical_studies": ["Study A showed X."]
        }
        transformed = self.fetcher.transform(sample_result)
        
        # Verify mapping
        self.assertEqual(transformed["INDICATIONS"], "For relief of pain.")
        self.assertEqual(transformed["DOSAGE"], "Take 1 tablet.")
        self.assertEqual(transformed["CONTRAINDICATIONS"], "None.")
        self.assertEqual(transformed["WARNINGS"], "May cause drowsiness.")
        self.assertEqual(transformed["ADVERSE_REACTIONS"], "Nausea.")
        self.assertEqual(transformed["CLINICAL_STUDIES"], "Study A showed X.")

    def test_transform_with_list_joining(self):
        sample_result = {
            "indications_and_usage": ["Indication 1", "Indication 2"]
        }
        transformed = self.fetcher.transform(sample_result)
        self.assertEqual(transformed["INDICATIONS"], "Indication 1\nIndication 2")

if __name__ == "__main__":
    unittest.main()
