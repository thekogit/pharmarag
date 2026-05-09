import unittest
from unittest.mock import patch, MagicMock

# Mock VectorStore before importing src.orchestrator to avoid Qdrant connection error
with patch('src.vector_store.VectorStore'):
    from src.orchestrator import synth_node, RAGState

class TestOrchestrator(unittest.TestCase):
    @patch('src.orchestrator.get_llm')
    @patch('src.orchestrator.synth_prompt')
    @patch('src.orchestrator.logger')
    def test_synth_node_logging(self, mock_logger, mock_prompt, mock_get_llm):
        # Mock LLM and Prompt
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Mock Answer"
        mock_prompt.__or__.return_value.__or__.return_value = mock_chain
        
        state: RAGState = {
            "question": "What is the dose?",
            "expanded": ["What is the dose?"],
            "docs": [
                {"text": "Dose is 10mg", "meta": {"source": "S1", "doc_type": "T1"}},
                {"text": "Dose is 20mg", "meta": {"source": "S2", "doc_type": "T2"}}
            ],
            "answer": ""
        }
        
        result = synth_node(state)
        
        # Verify logging
        mock_logger.info.assert_any_call("Retrieved 2 documents for synthesis.")
        self.assertEqual(result["answer"], "Mock Answer")

if __name__ == '__main__':
    unittest.main()
