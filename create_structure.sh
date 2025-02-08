#!/bin/bash

# Create the directory structure
mkdir -p app/api
mkdir -p app/core
mkdir -p app/models
mkdir -p app/services

# Create the files
touch app/api/__init__.py
touch app/api/chat.py
touch app/core/__init__.py
touch app/core/agent.py
touch app/core/config.py
touch app/models/__init__.py
touch app/services/__init__.py
touch app/services/llm_service.py
touch app/services/qdrant_service.py
touch app/main.py
touch app/requirements.txt

echo "Directory structure and files created successfully."