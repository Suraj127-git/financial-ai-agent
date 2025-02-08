# backend/app/api/chat.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.core.agent import AIAgent

router = APIRouter()

class ChatMessage(BaseModel):
    message: str

@router.post("/")
async def chat(message: ChatMessage):
    try:
        # Create an instance of the AI agent (in a production app, you might want to cache this)
        agent = AIAgent()
        response = agent.get_response(message.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
