import os
import logging
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from s5_chat import generate_answer
from config import HOST, PORT

import gradio as gr
from ui import get_gradio_app


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)


app = FastAPI(
    title="Pharma RAG Service",
    description="Two-stage RAG powered by Qwen + Cross-Encoder reranking",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


class Query(BaseModel):
    question: str


@app.post("/chat")
async def chat_endpoint(payload: Query):
    return generate_answer(payload.question)


@app.get("/")
def root():
    return {
        "status": "ok",
        "ui": "/ui",
        "endpoint": "/chat"
    }


gradio_blocks = get_gradio_app()
gradio_app = gr.routes.App.create_app(gradio_blocks)
app.mount("/ui", gradio_app)


# Run Everything
if __name__ == "__main__":
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        reload=False
    )
