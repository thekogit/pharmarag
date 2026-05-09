from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import os
from dotenv import load_dotenv
from src.vector_store import VectorStore
from src.logger import logger
from llama_cpp import Llama

# The Master Prompt. We aren't doing fine-tuning. We force compliance right here.
PHARMA_MASTER_PROMPT = """
You are a Senior Regulatory Affairs and Clinical Data Specialist. Your task is to answer the user's query based STRICTLY and ONLY on the provided context documents.

### INSTRUCTIONS:
- Keep your internal thinking process extremely concise.
- Focus on generating the final structured response.

### STRICT COMPLIANCE RULES:
1. NO HALLUCINATION: If the context does not contain the information necessary to answer the prompt, you must explicitly state: "The provided regulatory and clinical documents do not contain sufficient information to answer this query."
2. TRACEABILITY: You must cite the specific source for every factual claim. Use inline brackets referencing the 'source' or 'doc_type' metadata provided in the context (e.g., [ClinicalTrials.gov], [FDA Drug Label], [PubMed]).
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
        _vs = VectorStore(
            host=os.getenv("QDRANT_HOST", "localhost"), 
            port=int(os.getenv("QDRANT_PORT", 6333))
        )
    return _vs

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            base_url=os.getenv("OPENAI_API_BASE", "http://localhost:8080/v1"), 
            api_key=os.getenv("OPENAI_API_KEY", "none"),
            model=os.getenv("LLM_MODEL_NAME", "qwen2.5-7b-instruct-q4_k_m"),
            temperature=0.0,
            max_tokens=4096,
            timeout=120, # Increase timeout to 2 minutes
        )
    return _llm

def get_reranker():
    global _reranker
    if _reranker is None:
        model_path = os.getenv("RERANKER_PATH", "./models/mxbai-rerank-base-v2.i1-Q4_K_M.gguf")
        if not os.path.exists(model_path):
            logger.warning(f"Reranker model not found at {model_path}. Proceeding without reranking.")
            return None
        # Load GGUF reranker via llama-cpp-python
        _reranker = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=0, # Keep on CPU to avoid OOM with the LLM
            logits_all=True,
            verbose=False
        )
    return _reranker

class RAGState(TypedDict):
    question: str
    expanded: List[str]
    docs: List[dict]
    answer: str

def expand_node(s: RAGState):
    # Fail-fast: Check Qdrant connection before doing expensive LLM expansion
    try:
        vs_instance = get_vs()
        # Simple health check
        vs_instance.client.get_collections()
    except Exception as e:
        logger.error(f"FAIL-FAST: Qdrant connection failed: {e}")
        return {**s, "expanded": [s["question"]]}

    try:
        variations_text = (expansion_prompt | get_llm() | StrOutputParser()).invoke({"question": s["question"]})
        variations = [v.strip() for v in variations_text.split("\n") if v.strip()]
        return {**s, "expanded": [s["question"]] + variations}
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}. Using original query.")
        return {**s, "expanded": [s["question"]]}

def retrieve_node(s: RAGState):
    try:
        vs_instance = get_vs()
        queries = s.get("expanded", [s["question"]])
        
        all_hits = {}
        for q in queries:
            search_result = vs_instance.search(q, limit=10)
            for hit in search_result:
                if hit.id not in all_hits:
                    all_hits[hit.id] = hit

        unique_hits = list(all_hits.values())
        if not unique_hits:
            return {**s, "docs": [{"text": "No relevant documents found in the database.", "meta": {"source": "System", "doc_type": "Info"}}]}

        # Reranking using GGUF model
        reranker = get_reranker()
        if reranker:
            scored_hits = []
            for hit in unique_hits:
                text = (hit.payload.get("text") or hit.payload.get("content") or "")
                # mxbai-rerank heuristic: logprob of the first token as a score
                prompt = f"query: {s['question']} document: {text}"
                output = reranker(prompt, max_tokens=1, logprobs=1)
                score = output["choices"][0]["logprobs"]["token_logprobs"][0] if output["choices"][0]["logprobs"] else 0
                scored_hits.append((hit, score))
            
            scored_hits = sorted(scored_hits, key=lambda x: x[1], reverse=True)
            top_hits = scored_hits[:5]
        else:
            top_hits = [(hit, 0) for hit in unique_hits[:5]]
        
        docs = []
        for hit, score in top_hits:
            p = hit.payload or {}
            text = p.get("text") or p.get("content") or ""
            meta = p.get("meta") or {}
            # Standardize source and doc_type for citations
            meta_clean = {
                "source": meta.get("source") or meta.get("source_name") or "Unknown Source",
                "doc_type": meta.get("doc_type") or "Document",
                "compound": meta.get("compound") or "N/A",
                "date": meta.get("date") or "Unknown Date"
            }
            docs.append({"text": text, "meta": meta_clean})

        return {**s, "docs": docs}
    except Exception as e:
        logger.error(f"Error in retrieve_node: {e}")
        return {**s, "docs": [{"text": f"Retrieval Error: {str(e)}", "meta": {"source": "System", "doc_type": "Error"}}]}

def synth_node(s: RAGState):
    for d in s["docs"]:
        if d["meta"].get("doc_type") == "Error":
            return {**s, "answer": f"I cannot provide an answer because of a system error: {d['text']}"}

    logger.info(f"Retrieved {len(s['docs'])} documents for synthesis.")
    
    ctx = "\n\n---\n\n".join(
        f"SOURCE: [{d['meta'].get('source', 'Unknown')} | {d['meta'].get('doc_type', 'Unknown')}]\n"
        f"COMPOUND: {d['meta'].get('compound', 'N/A')}\n"
        f"DATE: {d['meta'].get('date', 'Unknown')}\n"
        f"CONTENT:\n{d['text']}" 
        for d in s["docs"]
    )
    
    try:
        answer = (synth_prompt | get_llm() | StrOutputParser()).invoke({"ctx": ctx, "q": s["question"]})
        return {**s, "answer": answer}
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        return {**s, "answer": "An error occurred while generating the final response."}

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
