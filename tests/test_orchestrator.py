import unittest
from unittest.mock import patch, MagicMock

# Mock VectorStore before importing src.orchestrator to avoid Qdrant connection error
with patch('src.vector_store.VectorStore'):
    from src.orchestrator import synth_node, RAGState

class TestOrchestrator(unittest.TestCase):
    @patch('src.orchestrator.get_llm')
    @patch('src.orchestrator.synth_prompt')
    @patch('builtins.print')
    def test_synth_node_logging(self, mock_print, mock_prompt, mock_get_llm):
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
        mock_print.assert_any_call("DEBUG: Retrieved 2 documents.")
        # Context length calculation check:
        # doc1: SOURCE METADATA: [S1 | T1]\nCONTENT:\nDose is 10mg -> ~54 chars
        # separator: \n\n---\n\n -> 7 chars
        # doc2: SOURCE METADATA: [S2 | T2]\nCONTENT:\nDose is 20mg -> ~54 chars
        # Total approx 115 chars
        
        # Find the call with "DEBUG: Total Context length:"
        log_calls = [call.args[0] for call in mock_print.call_args_list if "DEBUG: Total Context length:" in call.args[0]]
        self.assertTrue(len(log_calls) > 0)
        self.assertIn("chars.", log_calls[0])
        
        self.assertEqual(result["answer"], "Mock Answer")

if __name__ == '__main__':
    unittest.main()
