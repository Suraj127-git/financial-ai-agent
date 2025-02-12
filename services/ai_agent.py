from crewai import Agent, Task, Process
from langchain.llms import HuggingFaceHub
from services.vector_db import store_interaction, search_chat_history
from config.settings import settings
from typing import Optional
import yaml
import os

def setup_llm():
    return HuggingFaceHub(
        repo_id=settings.LLM_MODEL,
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )

def load_crewai_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'crewai_config.yml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def create_agents(llm):
    config = load_crewai_config()
    agents = {}
    
    for agent_key, agent_config in config['agents'].items():
        agents[agent_key] = Agent(
            role=agent_config['role'],
            goal=agent_config['goal'],
            backstory=agent_config['backstory'],
            llm=llm,
            verbose=agent_config['verbose']
        )
    return agents

def create_tasks(message, agents):
    config = load_crewai_config()
    tasks = []
    
    def get_agent_by_role(role):
        return next(agent for agent in agents.values() if agent.role == role)
    
    for task_key, task_config in config['tasks'].items():
        agent = get_agent_by_role(task_config['agent'])
        description = task_config['description'].format(message=message)
        
        tasks.append(Task(
            description=description,
            agent=agent,
            expected_output=task_config['expected_output']
        ))
    
    return tasks

def process_chat(message: str) -> str:
    llm = setup_llm()
    
    # Create agents and tasks from config
    agents = create_agents(llm)
    tasks = create_tasks(message, agents)
    
    # Check history first
    historical_answer = search_chat_history(message)
    if historical_answer:
        return historical_answer
    
    # Execute process
    process = Process(tasks=tasks)
    result = process.execute()
    
    # Store interaction after generating response
    store_interaction(message, result)
    
    return result