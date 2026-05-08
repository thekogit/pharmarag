import pytest
from unittest.mock import patch, MagicMock
from src.sources.pubmed import PubMedFetcher

def test_fetch_abstracts():
    fetcher = PubMedFetcher(email="test@example.com")
    
    # Mock return value for Entrez.read
    mock_id_list = {"IdList": ["12345"]}
    mock_article_data = {
        "PubmedArticle": [
            {
                "MedlineCitation": {
                    "PMID": "12345",
                    "Article": {
                        "ArticleTitle": "Test Title",
                        "Abstract": {
                            "AbstractText": ["Test Abstract Part 1", "Part 2"]
                        },
                        "Journal": {
                            "JournalIssue": {
                                "PubDate": {
                                    "Year": "2023",
                                    "Month": "05",
                                    "Day": "15"
                                }
                            }
                        }
                    }
                }
            }
        ]
    }
    
    with patch("Bio.Entrez.esearch") as mock_search, \
         patch("Bio.Entrez.efetch") as mock_fetch, \
         patch("Bio.Entrez.read", side_effect=[mock_id_list, mock_article_data]):
        
        mock_search.return_value = MagicMock()
        mock_fetch.return_value = MagicMock()
        
        results = fetcher.fetch_abstracts("aspirin", limit=1)
        
        assert len(results) == 1
        assert results[0]["title"] == "Test Title"
        assert "Test Abstract Part 1 Part 2" in results[0]["abstract"]
        assert results[0]["date"] == "2023-05-15"
        assert results[0]["pmid"] == "12345"

def test_transform():
    fetcher = PubMedFetcher()
    record = {
        "title": "Test Title",
        "abstract": "Test Abstract",
        "date": "2023-05-15",
        "pmid": "12345"
    }
    transformed = fetcher.transform(record)
    assert "CLINICAL_STUDIES" in transformed
    assert "Test Title" in transformed["CLINICAL_STUDIES"]
    assert "Test Abstract" in transformed["CLINICAL_STUDIES"]
