from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "HuggingFaceH4/zephyr-7b-beta")
    
settings = Settings()