from fastapi import FastAPI
from pydantic import BaseModel

from chat_baseline import generate_answer
from config import HOST, PORT

# ------------------------------------------------------
# FastAPI App
# ------------------------------------------------------
app = FastAPI(
    title="Pharma RAG Service",
    description="Two-stage RAG powered by Qwen + Cross-Encoder reranking",
    version="1.0.0"
)

# ------------------------------------------------------
# Root endpoint — fixes your 404 issue
# ------------------------------------------------------
@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Pharma RAG API is running.",
        "endpoints": {
            "POST /chat": "Submit a drug-related question"
        }
    }

# ------------------------------------------------------
# Chat request schema
# ------------------------------------------------------
class Query(BaseModel):
    question: str

# ------------------------------------------------------
# RAG Chat Endpoint
# ------------------------------------------------------
@app.post("/chat")
async def chat_endpoint(payload: Query):
    return generate_answer(payload.question)

# ------------------------------------------------------
# Run server
# ------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=False
    )
