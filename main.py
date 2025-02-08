from fastapi import FastAPI
from app.routes.chat import chat_router

app = FastAPI()
app.include_router(chat_router, prefix="/api/chat")

@app.get("/")
def read_root():
    return {"status": "Chat API running"}