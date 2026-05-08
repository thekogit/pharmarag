from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import os
from dotenv import load_dotenv
from src.vector_store import VectorStore
from sentence_transformers import CrossEncoder

# The Master Prompt. We aren't doing fine-tuning. We force compliance right here.
PHARMA_MASTER_PROMPT = """
You are a Senior Regulatory Affairs and Clinical Data Specialist. Your task is to answer the user's query based STRICTLY and ONLY on the provided context documents.

### STRICT COMPLIANCE RULES:
1. NO HALLUCINATION: If the context does not contain the information necessary to answer the prompt, you must explicitly state: "The provided regulatory and clinical documents do not contain sufficient information to answer this query."
2. TRACEABILITY: You must cite the specific source for every factual claim. Use inline brackets referencing the 'source' or 'doc_type' metadata provided in the context (e.g., [ClinicalTrials.gov], [FDA Drug Label], [EMA EPAR]).
3. PRECISION: Maintain a formal, objective, and scientific tone. Do not use speculative language.

### OUTPUT STRUCTURE:
Unless the user specifies otherwise, format your response strictly as follows:
- **Executive Summary:** A concise 1-2 sentence direct answer.
- **Clinical/Regulatory Evidence:** Bulleted points detailing the findings, mechanisms, or trial data, with inline citations.
- **Regulatory Implications (if applicable):** Note any phase statuses, adverse event warnings, or specific compliance notes found in the text.

### CONTEXT:
{ctx}
"""

EXPANSION_PROMPT = """
You are an expert at clinical and regulatory query expansion.
The user is asking a question about pharmaceutical regulations or clinical data.
Generate exactly 3 semantic variations of the following question.
Each variation should focus on a different aspect (e.g., dosage, clinical trials, regulatory approval).
Output ONLY the variations, one per line. Do not include numbering or extra text.

Original Question: {question}
"""

synth_prompt = ChatPromptTemplate.from_messages([
    ("system", PHARMA_MASTER_PROMPT),
    ("human", "Query: {q}")
])

expansion_prompt = ChatPromptTemplate.from_messages([
    ("system", EXPANSION_PROMPT),
    ("human", "{question}")
])

# Find the project root (one level up from src/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(project_root, '.env')

if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    load_dotenv()

# Lazy initialization for VectorStore, LLM, and Reranker
_vs = None
_llm = None
_reranker = None

def get_vs():
    global _vs
    if _vs is None:
        _vs = VectorStore(host=os.getenv("QDRANT_HOST", "localhost"), port=int(os.getenv("QDRANT_PORT", 6333)))
    return _vs

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            base_url=os.getenv("OPENAI_API_BASE", "http://localhost:8080/v1"), 
            api_key=os.getenv("OPENAI_API_KEY", "none"),
            model="qwen3.5-9b-deepseek-v4-flash",
            temperature=0.0,
            max_tokens=2048,
        )
    return _llm

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cpu")
    return _reranker

class RAGState(TypedDict):
    question: str
    expanded: List[str]
    docs: List[dict]
    answer: str

def expand_node(s: RAGState):
    variations_text = (expansion_prompt | get_llm() | StrOutputParser()).invoke({"question": s["question"]})
    variations = [v.strip() for v in variations_text.split("\n") if v.strip()]
    return {**s, "expanded": [s["question"]] + variations}

def retrieve_node(s: RAGState):
    try:
        vs_instance = get_vs()
        vs_instance._ensure_initialized()
        queries = s.get("expanded", [s["question"]])
        
        all_hits = {}
        for q in queries:
            vector = vs_instance.embedder.encode(q).tolist()
            search_result = vs_instance.client.query_points(
                collection_name=vs_instance.collection_name,
                query=vector,
                limit=10
            ).points
            for hit in search_result:
                if hit.id not in all_hits:
                    all_hits[hit.id] = hit

        unique_hits = list(all_hits.values())
        if not unique_hits:
            return {**s, "docs": [{"text": "No relevant documents found in the database.", "meta": {"source": "System", "doc_type": "Info"}}]}

        # Reranking
        reranker = get_reranker()
        pairs = [[s["question"], (hit.payload.get("text") or hit.payload.get("content") or "")] for hit in unique_hits]
        scores = reranker.predict(pairs)
        
        # Sort by score descending
        scored_hits = sorted(zip(unique_hits, scores), key=lambda x: x[1], reverse=True)
        top_hits = scored_hits[:5]
        
        docs = []
        for hit, score in top_hits:
            p = hit.payload or {}
            text = p.get("text") or p.get("content") or ""
            meta = p.get("meta")
            if not meta or not isinstance(meta, dict):
                meta = {k: v for k, v in p.items() if k not in ["text", "content"]}
            docs.append({"text": text, "meta": meta})

        return {**s, "docs": docs}
    except Exception as e:
        print(f"ERROR in retrieve_node: {e}")
        return {**s, "docs": [{"text": f"Connection Error: {str(e)}. Is Qdrant/Docker running?", "meta": {"source": "System", "doc_type": "Error"}}]}

def synth_node(s: RAGState):
    # If we have an error doc, just report it as the answer
    for d in s["docs"]:
        if d["meta"].get("doc_type") == "Error":
            return {**s, "answer": f"I cannot provide an answer because of a system error: {d['text']}"}

    print(f"DEBUG: Retrieved {len(s['docs'])} documents.")
    for i, d in enumerate(s['docs']):
        print(f"DEBUG: Doc {i} source: {d['meta'].get('source', 'Unknown')}")
        print(f"DEBUG: Doc {i} text snippet: {d['text'][:50]}...")

    ctx = "\n\n---\n\n".join(
        f"SOURCE METADATA: [{d['meta'].get('source','Unknown')} | {d['meta'].get('doc_type','Unknown')}]\nCONTENT:\n{d['text']}" 
        for d in s["docs"]
    )
    print(f"DEBUG: Total Context length: {len(ctx)} chars.")
    
    # Check if we are sending any actual content to the LLM
    has_content = any(len(d['text'].strip()) > 0 for d in s['docs'])
    if not has_content:
        print("DEBUG: No document content found. LLM might refuse to answer.")

    answer = (synth_prompt | get_llm() | StrOutputParser()).invoke({"ctx": ctx, "q": s["question"]})
    return {**s, "answer": answer}


def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("expand", expand_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("synthesize", synth_node)
    graph.set_entry_point("expand")
    graph.add_edge("expand", "retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()

rag_app = build_graph()
