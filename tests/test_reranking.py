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
def mock_llama():
    # Mocking llama_cpp.Llama instead of sentence_transformers.CrossEncoder
    with patch("src.orchestrator.Llama") as mock:
        llama_inst = MagicMock()
        mock.return_value = llama_inst
        # Mocking the call output for mxbai-rerank scoring heuristic
        llama_inst.return_value = {
            "choices": [
                {
                    "logprobs": {
                        "token_logprobs": [0.8] # Default mock score
                    }
                }
            ]
        }
        yield llama_inst

def test_retrieve_node_reranking(mock_vs, mock_llama):
    # Setup state
    state: RAGState = {
        "question": "What is the dosage of drug X?",
        "expanded": ["What is the dosage of drug X?", "variation 1", "variation 2"],
        "docs": [],
        "answer": ""
    }

    # Mock VectorStore search results
    hit1 = MagicMock(id=1, payload={"text": "doc1", "meta": {"source": "S1"}})
    hit2 = MagicMock(id=2, payload={"text": "doc2", "meta": {"source": "S2"}})
    hit3 = MagicMock(id=3, payload={"text": "doc3", "meta": {"source": "S3"}})
    
    mock_vs.client.query_points.side_effect = [
        MagicMock(points=[hit1, hit2]),
        MagicMock(points=[hit2, hit3]),
        MagicMock(points=[hit1, hit3])
    ]
    
    # Mock reranker scores for different docs
    # doc3 > doc2 > doc1
    mock_llama.side_effect = [
        {"choices": [{"logprobs": {"token_logprobs": [0.1]}}]}, # doc1
        {"choices": [{"logprobs": {"token_logprobs": [0.5]}}]}, # doc2
        {"choices": [{"logprobs": {"token_logprobs": [0.9]}}]}, # doc3
    ]

    # Run node
    with patch("src.orchestrator.get_reranker", return_value=mock_llama):
        result = retrieve_node(state)

    # Assertions
    # Result should be sorted by score: doc3, doc2, doc1
    assert len(result["docs"]) == 3
    assert result["docs"][0]["text"] == "doc3"
    assert result["docs"][1]["text"] == "doc2"
    assert result["docs"][2]["text"] == "doc1"
    
    # Verify mock calls
    assert mock_vs.client.query_points.call_count == 3
    # Llama (reranker) should be called 3 times (once per unique doc)
    assert mock_llama.call_count == 3
