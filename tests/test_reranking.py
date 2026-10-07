import pytest
from unittest.mock import MagicMock, patch
from src.orchestrator import retrieve_node, RAGState
import src.orchestrator as orch

@patch("src.orchestrator.get_vs")
def test_retrieve_node_reranking(mock_get_vs, monkeypatch):
    # Setup state
    state: RAGState = {
        "question": "What is the dosage of drug X?",
        "expanded": ["What is the dosage of drug X?", "variation 1", "variation 2"],
        "docs": [],
        "answer": ""
    }

    # Mock VectorStore
    mock_vs = MagicMock()
    mock_get_vs.return_value = mock_vs

    # Mock VectorStore search results
    hit1 = MagicMock(id=1, payload={"text": "doc1", "meta": {"source": "S1"}})
    hit2 = MagicMock(id=2, payload={"text": "doc2", "meta": {"source": "S2"}})
    hit3 = MagicMock(id=3, payload={"text": "doc3", "meta": {"source": "S3"}})

    mock_vs.search.side_effect = [
        [hit1, hit2],
        [hit2, hit3],
        [hit1, hit3]
    ]

    mock_reranker = MagicMock()
    # Unique hits: hit1, hit2, hit3 -> assign scores 0.1, 0.5, 0.9
    mock_reranker.predict.return_value = [0.1, 0.5, 0.9]
    monkeypatch.setattr(orch, "get_reranker", lambda: mock_reranker)

    result = retrieve_node(state)

    # Assertions: Result sorted by score descending: doc3, doc2, doc1
    assert len(result["docs"]) == 3
    assert result["docs"][0]["text"] == "doc3"
    assert result["docs"][1]["text"] == "doc2"
    assert result["docs"][2]["text"] == "doc1"
    
    assert mock_vs.search.call_count == 3
    assert mock_reranker.predict.call_count == 1
