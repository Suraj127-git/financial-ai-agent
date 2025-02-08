# backend/app/core/agent.py
from app.services.llm_service import generate_response
from app.services.qdrant_service import QdrantService

class AIAgent:
    def __init__(self):
        self.qdrant = QdrantService()

    def get_response(self, question: str) -> str:
        # Check if a similar question already exists in Qdrant
        existing_response = self.qdrant.search(question)
        if existing_response:
            return existing_response

        # Generate a new response using the free LLM model (distilgpt2)
        response = generate_response(question)
        # Store the question and answer in Qdrant for future reference
        self.qdrant.insert(question, response)
        return response
