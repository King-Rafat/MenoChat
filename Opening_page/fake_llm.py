from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import time
import uuid

app = FastAPI(
    title="MenoChat LLM TEST API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = "test-model"
    messages: List[Message]
    temperature: Optional[float] = 0.7


@app.post("/v1/chat/completions")
def get_response(
    request: ChatRequest,
    authorization: Optional[str] = Header(None)
):
    if authorization is None:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")

    last_message = request.messages[-1].content

    reply = f"Echo: {last_message}"

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": reply
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(str(request.messages)),
            "completion_tokens": len(reply),
            "total_tokens": len(str(request.messages)) + len(reply)
        }
    }