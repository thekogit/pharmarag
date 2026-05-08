import chainlit as cl
from src.orchestrator import rag_app, get_vs
from src.engine import InferenceEngine
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@cl.on_chat_start
async def start():
    """
    Initializes the session, starts the inference engine, and checks dependencies.
    """
    # 1. Start Inference Engine (llama-server) if not running
    engine = InferenceEngine()
    if not engine.is_running():
        await cl.Message(content="Starting Inference Engine (llama-server)...").send()
        engine.start(wait=False)
        # Give it a few seconds to initialize
        time.sleep(5)
    
    # 2. Check Qdrant Connection
    vs = get_vs()
    qdrant_status = "Connected"
    try:
        # Simple health check
        vs.client.get_collections()
    except Exception:
        qdrant_status = "Disconnected (Check Docker/Qdrant)"
        await cl.Message(
            content="⚠️ **Critical Error: Qdrant is not reachable.**\nPlease ensure your Qdrant container is running: `docker-compose up -d`"
        ).send()

    cl.user_session.set("rag_app", rag_app)
    
    await cl.Message(
        content=f"""# PharmaRAG Assistant Initialized
Status:
- Inference Engine: **{'Running' if engine.is_running() else 'Starting/Error'}**
- Vector DB (Qdrant): **{qdrant_status}**

Welcome! I can answer questions based on FDA Labels, ClinicalTrials, and PubMed.

**Ask me a question to get started.**"""
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """
    Main message handler that invokes the RAG pipeline.
    """
    rag_app = cl.user_session.get("rag_app")
    
    # Create an initial message to show progress/thinking
    msg = cl.Message(content="")
    await msg.send()

    try:
        # Run the RAG pipeline asynchronously
        # rag_app is a compiled LangGraph graph which supports ainvoke
        response = await rag_app.ainvoke({"question": message.content})
        
        answer = response.get("answer", "I'm sorry, I couldn't generate an answer.")
        docs = response.get("docs", [])
        
        elements = []
        
        # Create citation elements for each retrieved document
        for i, doc in enumerate(docs):
            meta = doc.get("meta", {})
            
            # Extract metadata for display
            source = meta.get("source") or meta.get("doc_type") or "Unknown Source"
            section = meta.get("section") or meta.get("section_name") or ""
            
            # Construct a descriptive name for the citation
            # We append the index to ensure uniqueness and help the user locate the reference
            citation_name = f"{source}"
            if section:
                citation_name += f" ({section})"
            citation_name += f" [{i+1}]"
            
            # Add the document text as a citation element
            elements.append(
                cl.Text(
                    name=citation_name,
                    content=doc.get("text", "No content available."),
                    display="side" # Shows in the side panel when clicked
                )
            )

        # Update the message with the final answer and citations
        msg.content = answer
        msg.elements = elements
        await msg.update()
        
    except Exception as e:
        # Handle errors gracefully in the UI
        msg.content = f"**Error:** An error occurred while processing your request.\n\nDetails: {str(e)}"
        await msg.update()

if __name__ == "__main__":
    # This block is just for local testing/reference; 
    # Chainlit is normally run via `chainlit run app.py`
    pass
