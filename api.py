from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from crew_agent import CrewHarness, detect_prompt_injection, mask_pii
from governance import AutogenReviewTeam, GovernancePolicy, ResponseCache
from knowledge_base import DOCUMENTS
from rag_core import RetrievalSystem

app = FastAPI(title="Cred Banking Support Agent")
rag_system = RetrievalSystem(DOCUMENTS)
crew = CrewHarness(rag_system)
review_team = AutogenReviewTeam()
policy = GovernancePolicy()
cache = ResponseCache()


class AskRequest(BaseModel):
    query: str = Field(..., description="Loan-policy or account-detail question")
    record_id: str | None = Field(default=None)


class AskResponse(BaseModel):
    answer: str
    trace_id: str
    tool_used: str
    confidence: float
    status: str = "ok"


class AddDocumentRequest(BaseModel):
    title: str
    content: str


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "logger": record.name,
            "level": record.levelname,
            "trace_id": getattr(record, "trace_id", "n/a"),
            "message": record.getMessage(),
            "module": record.module,
        }
        return json.dumps(payload)


logger = logging.getLogger("cred_support_agent")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    trace_id = uuid.uuid4().hex
    masked_query = mask_pii(request.query)
    if detect_prompt_injection(masked_query):
        logger.warning("Prompt injection blocked", extra={"trace_id": trace_id})
        return AskResponse(answer="I cannot process this request because it appears to contain a prompt injection attempt.", trace_id=trace_id, tool_used="guardrail_block", confidence=0.0, status="rejected")

    cached = cache.get(masked_query)
    if cached:
        cached["trace_id"] = trace_id
        logger.info("cached_answer", extra={"trace_id": trace_id})
        return AskResponse(**cached)

    try:
        response = crew.kickoff(masked_query, record_id=request.record_id)
        verdict = review_team.review(response.answer, ["loan policy support context"])
        payload = {
            "answer": verdict.final_answer,
            "trace_id": trace_id,
            "tool_used": response.tool_used,
            "confidence": float(response.confidence),
            "status": "ok",
        }
        cache.set(masked_query, payload)
        logger.info("request_processed", extra={"trace_id": trace_id})
        return AskResponse(**payload)
    except Exception as exc:  # pragma: no cover
        logger.exception("agent_failed", extra={"trace_id": trace_id})
        return AskResponse(answer=f"The agent failed: {exc}", trace_id=trace_id, tool_used="error", confidence=0.0, status="error")


@app.post("/add-document")
def add_document(request: AddDocumentRequest) -> Dict[str, Any]:
    trace_id = uuid.uuid4().hex
    safe_doc = {
        "id": f"custom-{uuid.uuid4().hex[:6]}",
        "title": request.title,
        "content": request.content,
    }
    DOCUMENTS.append(safe_doc)
    rag_system.documents = DOCUMENTS
    rag_system.__init__(DOCUMENTS)
    logger.info("document-added", extra={"trace_id": trace_id})
    return {"status": "ok", "trace_id": trace_id, "document_id": safe_doc["id"]}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_text()
            msg = json.loads(message)
            payload = mask_pii(msg.get("query", ""))
            response = crew.kickoff(payload)
            await websocket.send_json({"answer": response.answer, "tool_used": response.tool_used, "confidence": response.confidence})
    except WebSocketDisconnect:
        await websocket.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
