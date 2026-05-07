from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

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

synth_prompt = ChatPromptTemplate.from_messages([
    ("system", PHARMA_MASTER_PROMPT),
    ("human", "Query: {q}")
])

import os
from dotenv import load_dotenv

# Find the project root (one level up from src/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(project_root, '.env')

if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    load_dotenv()

from src.vector_store import VectorStore

llm = ChatOpenAI(
    base_url=os.getenv("OPENAI_API_BASE", "http://localhost:8080/v1"), api_key=os.getenv("OPENAI_API_KEY", "none"),
    model="qwen3.5-9b-deepseek-v4-flash",
    temperature=0.0,
    max_tokens=2048,
)

# Connect to the local Qdrant instance
vs = VectorStore(host=os.getenv("QDRANT_HOST", "localhost"), port=int(os.getenv("QDRANT_PORT", 6333)))

class RAGState(TypedDict):
    question: str
    expanded: List[str]
    docs: List[dict]
    answer: str

def expand_node(s: RAGState):
    return {**s, "expanded": [s["question"]]}

def retrieve_node(s: RAGState):
    # Actually fetch from Qdrant
    vector = vs.embedder.encode(s["question"]).tolist()
    search_result = vs.client.query_points(
        collection_name=vs.collection_name,
        query=vector,
        limit=3
    ).points
    docs = [{"text": hit.payload.get("text", ""), "meta": hit.payload.get("meta", {})} for hit in search_result]

    # Fallback if DB is empty
    if not docs:
        docs = [{"text": "No relevant documents found in the database.", "meta": {"source": "System", "doc_type": "Error"}}]

    return {**s, "docs": docs} 

def synth_node(s: RAGState):
    print(f"DEBUG: Retrieved {len(s['docs'])} documents.")
    ctx = "\n\n---\n\n".join(
        f"SOURCE METADATA: [{d['meta'].get('source','Unknown')} | {d['meta'].get('doc_type','Unknown')}]\nCONTENT:\n{d['text']}" 
        for d in s["docs"]
    )
    print(f"DEBUG: Context length: {len(ctx)} chars.")
    answer = (synth_prompt | llm | StrOutputParser()).invoke({"ctx": ctx, "q": s["question"]})
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
