import unittest
from unittest.mock import patch, MagicMock
from src.sources.clinical_trials import ClinicalTrialsFetcher

class TestClinicalTrialsFetcher(unittest.TestCase):
    def setUp(self):
        self.fetcher = ClinicalTrialsFetcher()
        self.sample_nct_id = "NCT01234567"
        self.sample_api_response = {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT01234567",
                    "briefTitle": "Sample Clinical Trial Title"
                },
                "descriptionModule": {
                    "briefSummary": "This is a brief summary of the trial.",
                    "detailedDescription": "This is a more detailed description of the trial with more technical info."
                },
                "eligibilityModule": {
                    "eligibilityCriteria": "Inclusion Criteria:\n- Age > 18\n- Confirmed diagnosis\n\nExclusion Criteria:\n- Prior treatment with X\n- Active infection"
                }
            }
        }

    @patch("requests.get")
    def test_fetch_by_id_success(self, mock_get):
        # Configure mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.sample_api_response
        mock_get.return_value = mock_response

        # Execute
        result = self.fetcher.fetch_by_id(self.sample_nct_id)

        # Verify
        self.assertEqual(result, self.sample_api_response)
        mock_get.assert_called_once_with(f"https://clinicaltrials.gov/api/v2/studies/{self.sample_nct_id}")

    def test_transform(self):
        # Execute
        transformed = self.fetcher.transform(self.sample_api_response)

        # Verify structure
        self.assertIn("CLINICAL_STUDIES", transformed)
        self.assertIn("ELIGIBILITY", transformed)
        
        # Verify content
        self.assertIn("Sample Clinical Trial Title", transformed["CLINICAL_STUDIES"])
        self.assertIn("This is a brief summary", transformed["CLINICAL_STUDIES"])
        self.assertIn("Inclusion Criteria", transformed["ELIGIBILITY"])
        self.assertIn("Exclusion Criteria", transformed["ELIGIBILITY"])

    @patch("requests.get")
    def test_fetch_by_id_error(self, mock_get):
        # Configure mock for 404
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        # Execute & Verify
        with self.assertRaises(Exception):
            self.fetcher.fetch_by_id("INVALID_ID")

    def test_integration_with_ingestor(self):
        from src.ingest import PDFIngestor
        ingestor = PDFIngestor()
        
        transformed = self.fetcher.transform(self.sample_api_response)
        context = {"source": "CLINICALTRIALS", "doc_type": "Clinical Trial"}
        chunks = ingestor.chunk_sections(transformed, context)
        
        # Verify chunks
        self.assertTrue(len(chunks) >= 2) # At least one for CLINICAL_STUDIES and one for ELIGIBILITY
        
        sections_found = set(chunk["metadata"]["section"] for chunk in chunks)
        self.assertIn("CLINICAL_STUDIES", sections_found)
        self.assertIn("ELIGIBILITY", sections_found)
        
        # Check one CLINICAL_STUDIES chunk
        clinical_chunk = next(c for c in chunks if c["metadata"]["section"] == "CLINICAL_STUDIES")
        self.assertTrue(clinical_chunk["text"].startswith("[Section: CLINICAL_STUDIES]"))
        self.assertIn("Sample Clinical Trial Title", clinical_chunk["text"])

if __name__ == "__main__":
    unittest.main()
