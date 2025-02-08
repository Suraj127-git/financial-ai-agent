# backend/app/services/qdrant_service.py
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import numpy as np
from app.core.config import QDRANT_HOST, QDRANT_PORT, EMBEDDING_MODEL_NAME

# Initialize the embedding model (384 dimensions for all-MiniLM-L6-v2)
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

class QdrantService:
    def __init__(self):
        # Connect to Qdrant (make sure you have a Qdrant instance running locally on port 6333)
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.collection_name = "chat_history"
        self._init_collection()

    def _init_collection(self):
        # Create collection if it does not exist
        collections = self.client.get_collections().collections
        if self.collection_name not in [col.name for col in collections]:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=384,  # Dimension for the chosen embedding model
                    distance=models.Distance.COSINE
                )
            )

    def insert(self, question: str, answer: str):
        # Encode the question to get its embedding vector
        vector = embedder.encode(question).tolist()
        # Use a simple hash as the point id
        point_id = abs(hash(question)) % (10 ** 8)
        payload = {"question": question, "answer": answer}
        self.client.upsert(
            collection_name=self.collection_name,
            points=[models.PointStruct(id=point_id, vector=vector, payload=payload)]
        )

    def search(self, question: str, top: int = 1, threshold: float = 0.7) -> str:
        # Encode the query
        query_vector = embedder.encode(question).tolist()
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top
        )
        if results:
            # Evaluate the cosine similarity (computed manually)
            best = results[0]
            stored_vector = best.vector
            sim = np.dot(query_vector, stored_vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(stored_vector) + 1e-10
            )
            if sim > threshold:
                return best.payload.get("answer")
        return None
