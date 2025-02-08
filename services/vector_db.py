from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from app.config.settings import settings

encoder = SentenceTransformer('all-MiniLM-L6-v2')
client = QdrantClient(url=settings.QDRANT_URL)

def store_interaction(question: str, answer: str):
    vector = encoder.encode(question + " " + answer).tolist()
    payload = {"question": question, "answer": answer}
    
    client.upsert(
        collection_name="chat_history",
        points=[{
            "id": hash(question),  # Simple hashing for demo
            "vector": vector,
            "payload": payload
        }]
    )