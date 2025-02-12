from fastapi import FastAPI
from routes.chat import chat_router

app = FastAPI()
app.include_router(chat_router, prefix="/api/chat")

@app.get("/")
def read_root():
    """
    Simple health check endpoint to verify that the API is running.
    """
    return {"status": "Chat API running"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)