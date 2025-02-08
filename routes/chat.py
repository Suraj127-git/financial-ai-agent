from fastapi import APIRouter, HTTPException
from app.models.schemas import ChatRequest, ChatResponse
from app.services.ai_agent import process_chat

chat_router = APIRouter()

@chat_router.post("/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        response = process_chat(request.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))