import unittest
from unittest.mock import patch, MagicMock

# Mock VectorStore before importing src.orchestrator to avoid Qdrant connection error
with patch('src.vector_store.VectorStore'):
    from src.orchestrator import expand_node, RAGState

class TestExpansion(unittest.TestCase):
    @patch('src.orchestrator.get_vs')
    @patch('src.orchestrator.get_llm')
    def test_expand_node_generates_variations(self, mock_get_llm, mock_get_vs):
        # Setup mock Qdrant for fail-fast check
        mock_vs = MagicMock()
        mock_get_vs.return_value = mock_vs
        
        # Setup mock LLM
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        state: RAGState = {
            "question": "What is the dose?",
            "expanded": [],
            "docs": [],
            "answer": ""
        }
        
        # We need to mock the prompt and the chain
        with patch('src.orchestrator.expansion_prompt') as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "How much dose is recommended?\nWhat is the clinical dosage?\nCan you tell me the dose amount?"
            mock_prompt.__or__.return_value.__or__.return_value = mock_chain
            
            result = expand_node(state)
            
            expected_expanded = [
                "What is the dose?",
                "How much dose is recommended?",
                "What is the clinical dosage?",
                "Can you tell me the dose amount?"
            ]
            
            self.assertEqual(result["expanded"], expected_expanded)
            self.assertEqual(len(result["expanded"]), 4)
            self.assertEqual(result["expanded"][0], "What is the dose?")

if __name__ == '__main__':
    unittest.main()
