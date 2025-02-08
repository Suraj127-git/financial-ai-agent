# backend/app/main.py
from fastapi import FastAPI
from app.api import chat

app = FastAPI(title="Chat App with AI Agent")

app.include_router(chat.router, prefix="/api/chat")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
