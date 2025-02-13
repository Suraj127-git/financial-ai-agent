from qdrant_client import QdrantClient
from qdrant_client.http import models
from transformers import AutoTokenizer, AutoModel
import torch
from config.settings import settings
from typing import Optional, List, Dict
import hashlib
import logging

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('microsoft/mpnet-base')
model = AutoModel.from_pretrained('microsoft/mpnet-base')

# Initialize client
client = QdrantClient(url=settings.QDRANT_URL)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def mean_pooling(model_output, attention_mask):
    """Perform mean pooling on token embeddings"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encode_text(text: str) -> List[float]:
    """Encode text using MPNet model"""
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = mean_pooling(outputs, inputs['attention_mask'])
    return embeddings[0].numpy().tolist()

def generate_id(question: str) -> int:
    """Generate consistent hash-based ID using SHA-256"""
    return int(hashlib.sha256(question.encode()).hexdigest()[:15], 16)

def store_interaction(question: str, answer: str, batch_size: int = 64) -> bool:
    """Store conversation interaction with enhanced error handling and batching"""
    try:
        if not isinstance(question, str) or not isinstance(answer, str):
            logger.error("Invalid input types for question/answer")
            return False

        combined_text = f"Q: {question} A: {answer}"
        vector = encode_text(combined_text)
        point_id = generate_id(combined_text)

        payload = {
            "question": question,
            "answer": answer,
            "metadata": {
                "length": len(combined_text),
                "source": "chat_interaction"
            }
        }

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
    """Enhanced semantic search with multiple results and scoring"""
    try:
        if not isinstance(question, str) or len(question.strip()) == 0:
            logger.error("Invalid search query")
            return []

        vector = encode_text(question)

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

        formatted_results = [{
            "question": hit.payload["question"],
            "answer": hit.payload["answer"],
            "score": hit.score,
            "id": hit.id
        } for hit in results]

        logger.info(f"Found {len(formatted_results)} matches for: {question[:50]}...")
        return formatted_results

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return []