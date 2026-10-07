from types import SimpleNamespace
import src.orchestrator as orch

class FakeVS:
    def search(self, q, limit=10, **kwargs):
        docs = ["cats sleep a lot", "aspirin dose is 81 mg daily", "weather is mild"]
        return [SimpleNamespace(id=i, payload={"text": t, "meta": {}}) for i, t in enumerate(docs)]

class FakeReranker:
    def predict(self, pairs):
        return [1.0 if "aspirin" in doc else 0.0 for _, doc in pairs]

def test_reranker_puts_relevant_doc_first(monkeypatch):
    monkeypatch.setattr(orch, "get_vs", lambda: FakeVS())
    monkeypatch.setattr(orch, "get_reranker", lambda: FakeReranker())
    out = orch.retrieve_node({"question": "aspirin dose", "expanded": ["aspirin dose"], "docs": [], "answer": ""})
    assert out["docs"][0]["text"] == "aspirin dose is 81 mg daily"
