"""FastAPI route handlers."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agentic_graph_rag.agent.tool_registry import ToolName
from agentic_graph_rag.trace_explain import explain_trace_payload
from api.deps import get_service

router = APIRouter(prefix="/api/v1")

VALID_MODES = Literal[
    "vector", "cypher", "hybrid",
    "agent_pattern", "agent_llm", "agent_mangle",
]
VALID_TOOLS = ToolName


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    mode: VALID_MODES = "agent_pattern"
    session_id: str = Field(default="", max_length=128)


class SearchRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    tool: VALID_TOOLS = "vector_search"


def _format_sse_event(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/health")
def health():
    svc = get_service()
    return svc.health()


@router.post("/query")
def query(req: QueryRequest):
    svc = get_service()
    qa = svc.query(req.text, mode=req.mode, session_id=req.session_id)
    return qa.model_dump()


@router.post("/query/stream")
def query_stream(req: QueryRequest):
    svc = get_service()

    def _event_stream() -> Iterator[str]:
        for item in svc.stream_query(req.text, mode=req.mode, session_id=req.session_id):
            yield _format_sse_event(item["event"], item["data"])

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/trace/{trace_id}")
def get_trace(trace_id: str):
    svc = get_service()
    trace = svc.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace.model_dump()


@router.get("/trace/{trace_id}/explain")
def get_trace_explain(trace_id: str):
    svc = get_service()
    trace = svc.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return explain_trace_payload(trace)


@router.post("/search")
def search(req: SearchRequest):
    svc = get_service()
    results = svc.search(req.text, tool=req.tool)
    return [r.model_dump() for r in results]


@router.get("/graph/stats")
def graph_stats():
    svc = get_service()
    return svc.graph_stats()


@router.get("/metrics")
def metrics():
    from api.middleware import get_metrics
    return get_metrics().snapshot()
