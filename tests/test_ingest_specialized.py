import unittest
from src.ingest import PDFIngestor

class TestPDFIngestorSpecialized(unittest.TestCase):
    def setUp(self):
        self.ingestor = PDFIngestor()

    def test_chunk_sections_specialized_clinical(self):
        # CLINICAL_STUDIES should now use dense_splitter (500/0)
        long_text = "Clinical study sentence. " * 100 # Definitely more than 500 chars
        sections = {"CLINICAL_STUDIES": long_text}
        
        chunks = self.ingestor.chunk_sections(sections, {})
        
        self.assertTrue(len(chunks) > 1)
        for chunk in chunks:
            self.assertTrue(chunk["text"].startswith("[Section: CLINICAL_STUDIES]"))
            self.assertEqual(chunk["metadata"]["section"], "CLINICAL_STUDIES")
            # Max size should be around 500 + prefix length
            self.assertLessEqual(len(chunk["text"]), 600) 

    def test_chunk_sections_specialized_dense(self):
        # DOSAGE should use 500/0 (dense)
        long_text = "Dosage instruction detail. " * 50 # More than 500 chars
        sections = {"DOSAGE": long_text}
        
        chunks = self.ingestor.chunk_sections(sections, {})
        
        self.assertTrue(len(chunks) > 1)
        for chunk in chunks:
            self.assertTrue(chunk["text"].startswith("[Section: DOSAGE]"))
            self.assertEqual(chunk["metadata"]["section"], "DOSAGE")
            # Max size should be around 500 + prefix length
            self.assertLessEqual(len(chunk["text"]), 600)

    def test_chunk_sections_multiple_sections(self):
        sections = {
            "CLINICAL_STUDIES": "Study data " * 50,
            "DOSAGE": "Dosage data " * 30
        }
        
        chunks = self.ingestor.chunk_sections(sections, {})
        
        sections_found = set(chunk["metadata"]["section"] for chunk in chunks)
        self.assertEqual(sections_found, {"CLINICAL_STUDIES", "DOSAGE"})

if __name__ == "__main__":
    unittest.main()
