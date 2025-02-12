from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from app.config.settings import settings
from typing import Optional, List, Dict
import hashlib
import logging

# Initialize encoder and client
encoder = SentenceTransformer('all-MiniLM-L6-v2')
client = QdrantClient(url=settings.QDRANT_URL)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_id(question: str) -> int:
    """Generate consistent hash-based ID using SHA-256"""
    return int(hashlib.sha256(question.encode()).hexdigest()[:15], 16)

def store_interaction(question: str, answer: str, batch_size: int = 64) -> bool:
    """
    Store conversation interaction with enhanced error handling and batching
    """
    try:
        # Validate input types
        if not isinstance(question, str) or not isinstance(answer, str):
            logger.error("Invalid input types for question/answer")
            return False

        # Encode question and answer together for better context preservation
        combined_text = f"Q: {question} A: {answer}"
        vector = encoder.encode(combined_text).tolist()

        # Generate unique ID
        point_id = generate_id(combined_text)

        # Create payload with timestamp
        payload = {
            "question": question,
            "answer": answer,
            "metadata": {
                "length": len(combined_text),
                "source": "chat_interaction"
            }
        }

        # Upsert with batching configuration
        client.upsert(
            collection_name="chat_history",
            points=[models.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            )],
            batch_size=batch_size
        )

        logger.info(f"Stored interaction: {question[:50]}...")
        return True

    except Exception as e:
        logger.error(f"Error storing interaction: {str(e)}")
        return False

def search_chat_history(
    question: str,
    score_threshold: float = 0.82,
    limit: int = 3
) -> List[Dict]:
    """
    Enhanced semantic search with multiple results and scoring
    """
    try:
        # Validate input
        if not isinstance(question, str) or len(question.strip()) == 0:
            logger.error("Invalid search query")
            return []

        # Encode question with normalization
        vector = encoder.encode(question).tolist()

        # Perform search with filters
        results = client.search(
            collection_name="chat_history",
            query_vector=vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.source",
                        match=models.MatchValue(value="chat_interaction")
                    )
                ]
            ),
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False
        )

        # Process and format results
        formatted_results = []
        for hit in results:
            formatted_results.append({
                "question": hit.payload["question"],
                "answer": hit.payload["answer"],
                "score": hit.score,
                "id": hit.id
            })

        logger.info(f"Found {len(formatted_results)} matches for: {question[:50]}...")
        return formatted_results

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return []