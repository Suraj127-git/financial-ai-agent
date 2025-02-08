# backend/app/core/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# Qdrant settings
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# Model settings
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "distilgpt2")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
