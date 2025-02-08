from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    QDRANT_URL: str = "http://localhost:6333"
    LLM_MODEL: str = "HuggingFaceH4/zephyr-7b-beta"
    
settings = Settings()