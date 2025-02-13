from fastapi import FastAPI
from routes.chat import chat_router
import logging
from helper.log_helper import configure_logging
configure_logging()

app = FastAPI()
app.include_router(chat_router, prefix="/api/chat")

@app.get("/")
def read_root():
    """
    Simple health check endpoint to verify that the API is running.
    """
    logger = logging.getLogger()
    logger.info("Financial AI Chat API")
    return {"status": "Chat API running"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)