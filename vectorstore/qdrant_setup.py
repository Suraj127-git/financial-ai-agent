from qdrant_client import QdrantClient
from app.config.settings import settings

def initialize_qdrant():
    client = QdrantClient(url=settings.QDRANT_URL)
    
    # Create collection if not exists
    try:
        client.get_collection("chat_history")
    except:
        client.create_collection(
            collection_name="chat_history",
            vectors_config={"size": 384, "distance": "Cosine"}
        )