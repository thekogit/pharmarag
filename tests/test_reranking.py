import pytest
from unittest.mock import MagicMock, patch
from src.orchestrator import retrieve_node, RAGState

@pytest.fixture
def mock_vs():
    with patch("src.orchestrator.get_vs") as mock:
        vs = MagicMock()
        mock.return_value = vs
        yield vs

@pytest.fixture
def mock_cross_encoder():
    with patch("src.orchestrator.CrossEncoder") as mock:
        encoder = MagicMock()
        mock.return_value = encoder
        # Mock predict to return scores in descending order for simplicity in some cases
        # or just return what we expect.
        yield encoder

def test_retrieve_node_reranking(mock_vs, mock_cross_encoder):
    # Setup state
    state: RAGState = {
        "question": "What is the dosage of drug X?",
        "expanded": ["What is the dosage of drug X?", "variation 1", "variation 2"],
        "docs": [],
        "answer": ""
    }

    # Mock VectorStore search results
    # Each query returns some hits, some overlapping
    hit1 = MagicMock(id=1, payload={"text": "doc1", "meta": {"source": "S1"}})
    hit2 = MagicMock(id=2, payload={"text": "doc2", "meta": {"source": "S2"}})
    hit3 = MagicMock(id=3, payload={"text": "doc3", "meta": {"source": "S3"}})
    
    # Query 0 (original) returns 1, 2
    # Query 1 returns 2, 3
    # Query 2 returns 1, 3
    
    mock_vs.client.query_points.side_effect = [
        MagicMock(points=[hit1, hit2]),
        MagicMock(points=[hit2, hit3]),
        MagicMock(points=[hit1, hit3])
    ]
    
    # Mock reranker scores
    # We have 3 unique docs: 1, 2, 3
    # Let's say doc 3 is the best, then 1, then 2
    mock_cross_encoder.return_value.predict.return_value = [0.1, 0.5, 0.9] # scores for [doc1, doc2, doc3]

    # Run node
    with patch("src.orchestrator.get_reranker", return_value=mock_cross_encoder.return_value):
        result = retrieve_node(state)

    # Assertions
    # 1. Deduplication: only 3 unique docs should be passed to reranker
    # 2. Top-N: If we limit to Top-2 for this test (I'll use Top-5 in impl, but here it's 3 unique)
    # The result should be sorted by score: doc3, doc2, doc1
    
    assert len(result["docs"]) == 3
    assert result["docs"][0]["text"] == "doc3"
    assert result["docs"][1]["text"] == "doc2"
    assert result["docs"][2]["text"] == "doc1"
    
    # Verify mock calls
    assert mock_vs.client.query_points.call_count == 3
    # Verify reranker was called with (question, [doc1_text, doc2_text, doc3_text])
    # The actual implementation might pass list of pairs
    # mock_cross_encoder.return_value.predict.assert_called_once()
