from crewai import Agent, Task, Process
from langchain.llms import HuggingFaceHub
from app.services.vector_db import store_interaction
from app.config.settings import settings

def setup_llm():
    return HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )

def process_chat(message: str) -> str:
    llm = setup_llm()
    
    agent = Agent(
        role="Chat Assistant",
        goal="Provide helpful and accurate responses",
        backstory="You're an AI assistant trained to help users with various questions",
        llm=llm,
        verbose=True
    )
    
    task = Task(
        description=f"Respond to: {message}",
        agent=agent
    )
    
    result = Process(tasks=[task]).execute()
    
    # Store interaction in Qdrant
    store_interaction(message, result)
    
    return result