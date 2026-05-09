import asyncio
import os
from src.orchestrator import rag_app

async def test_query():
    print("Sending query to PharmaRAG...")
    question = "What are the cardiovascular outcomes associated with semaglutide according to the clinical trials and FDA data?"
    
    # Using ainvoke as it's a LangGraph app
    response = await rag_app.ainvoke({"question": question})
    
    print("\n" + "="*50)
    print("ANSWER:")
    print(response["answer"])
    print("="*50)
    
    print("\nSOURCES USED:")
    for i, doc in enumerate(response["docs"]):
        meta = doc.get("meta", {})
        source = meta.get("source", "Unknown")
        doc_type = meta.get("doc_type", "Unknown")
        section = meta.get("section_name") or meta.get("section") or "N/A"
        print(f"{i+1}. [{source} | {doc_type}] Section: {section}")

if __name__ == "__main__":
    asyncio.run(test_query())

