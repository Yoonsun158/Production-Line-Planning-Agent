"""
FastAPI server exposing a Dify-compatible SSE endpoint.

The frontend's existing callDify() sends POST to /v1/chat-messages
and expects SSE events: agent_thought, message, message_end.
This server translates the ReAct agent's output into that exact format.

Usage:
    cd backend
    python server.py
"""

import json
import uuid
import logging
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn

from rag import KnowledgeBase
from agent_core import run_agent

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

API_TOKEN = "brd-local-key"

app = FastAPI(title="BRD-Agent Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

kb = KnowledgeBase()


def _sse_line(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_agent(query: str, conversation_id: str) -> AsyncGenerator[str, None]:
    """Run the ReAct agent and yield Dify-compatible SSE events."""
    msg_id = uuid.uuid4().hex[:16]
    conv_id = conversation_id or uuid.uuid4().hex[:16]

    for event in run_agent(query, kb):
        etype = event["type"]

        if etype == "thought":
            yield _sse_line({
                "event": "agent_thought",
                "id": msg_id,
                "thought": event["content"],
                "message_id": msg_id,
                "conversation_id": conv_id,
            })

        elif etype == "tool_call":
            yield _sse_line({
                "event": "agent_thought",
                "id": msg_id,
                "thought": f"🔧 调用工具 {event['tool']}({event['input']})",
                "tool": event["tool"],
                "tool_input": event["input"],
                "message_id": msg_id,
                "conversation_id": conv_id,
            })

        elif etype == "observation":
            yield _sse_line({
                "event": "agent_thought",
                "id": msg_id,
                "thought": f"📋 {event['content'][:200]}",
                "message_id": msg_id,
                "conversation_id": conv_id,
            })

        elif etype == "answer":
            yield _sse_line({
                "event": "message",
                "id": msg_id,
                "answer": event["content"],
                "message_id": msg_id,
                "conversation_id": conv_id,
            })

        elif etype == "error":
            yield _sse_line({
                "event": "agent_thought",
                "id": msg_id,
                "thought": f"❌ {event['content']}",
                "message_id": msg_id,
                "conversation_id": conv_id,
            })

    yield _sse_line({
        "event": "message_end",
        "id": msg_id,
        "conversation_id": conv_id,
    })


@app.post("/v1/chat-messages")
async def chat_messages(request: Request):
    auth = request.headers.get("Authorization", "")
    if API_TOKEN and not auth.endswith(API_TOKEN):
        return {"error": "Unauthorized"}, 401

    body = await request.json()
    query = body.get("query", "")
    conversation_id = body.get("conversation_id", "")
    response_mode = body.get("response_mode", "streaming")

    if not query:
        return {"error": "query is required"}, 400

    logger.info("Query: %s", query[:80])

    if response_mode == "streaming":
        return StreamingResponse(
            _stream_agent(query, conversation_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    full_answer = ""
    conv_id = conversation_id or uuid.uuid4().hex[:16]
    for event in run_agent(query, kb):
        if event["type"] == "answer":
            full_answer = event["content"]
    return {"answer": full_answer, "conversation_id": conv_id}


@app.get("/health")
async def health():
    return {"status": "ok", "kb_records": kb.count}


if __name__ == "__main__":
    if kb.count == 0:
        logger.warning(
            "Knowledge base is empty. Run 'python init_kb.py' first."
        )
    else:
        logger.info("Knowledge base loaded — %d records", kb.count)
    uvicorn.run(app, host="0.0.0.0", port=8000)
