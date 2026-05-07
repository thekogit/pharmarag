import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

def check_qdrant():
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", 6333))
    client = QdrantClient(host=host, port=port)
    collection_name = "pharma_docs"
    
    print(f"Checking collection: {collection_name} on {host}:{port}")
    
    if not client.collection_exists(collection_name):
        print(f"Collection {collection_name} does not exist.")
        return
        
    count = client.count(collection_name).count
    print(f"Point count: {count}")
    
    if count > 0:
        # Scroll to see some points
        points, _ = client.scroll(collection_name, limit=5, with_payload=True)
        for i, point in enumerate(points):
            print(f"\n--- Point {i} ---")
            print(f"ID: {point.id}")
            print(f"Payload: {point.payload}")

if __name__ == "__main__":
    check_qdrant()
