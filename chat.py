import sys
import time
import urllib.request
from src.orchestrator import rag_app
from src.engine import InferenceEngine

def is_server_ready(url):
    try:
        with urllib.request.urlopen(f"{url}/health") as response:
            return response.getcode() == 200
    except:
        return False

def main():
    engine = InferenceEngine()
    base_url = f"http://localhost:{engine.port}"
    
    if not is_server_ready(base_url):
        print("Llama server not responding. Starting it...")
        engine.start(wait=False)
        
        print("Waiting for server to initialize...")
        for _ in range(30): # 30 seconds timeout
            if is_server_ready(base_url):
                print("Server is READY.")
                break
            time.sleep(1)
        else:
            print("FATAL: Server failed to start in time.")
            sys.exit(1)

    print("==================================================")
    print(" Pharma-RAG Interactive CLI (Type 'exit' to quit) ")
    print("==================================================")
    
    while True:
        try:
            query = input("\nQuery: ")
            if query.lower() in ['exit', 'quit']:
                break
            if not query.strip():
                continue
                
            print("\nSearching Qdrant and consulting Master Prompt...")
            result = rag_app.invoke({'question': query})
            
            print('\n--- Answer ---')
            print(result['answer'])
            print('--------------')
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()

