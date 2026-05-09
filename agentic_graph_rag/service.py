"""PipelineService — typed contract for Agentic Graph RAG pipeline.

All clients (FastAPI, MCP, Streamlit) use this service.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING

from rag_core.neo4j_utils import open_neo4j_session
from rag_core.models import PipelineTrace, QAResult, WorkflowMemoryEntry
from rag_core.models import (
    ReflectionStep,
)

from agentic_graph_rag.agent import tools as agent_tools
from agentic_graph_rag.agent.retrieval_agent import run as agent_run
from agentic_graph_rag.agent.tool_registry import TOOL_NAMES
from agentic_graph_rag.medical_aliases import expand_medical_aliases
from agentic_graph_rag.trace_store import TraceStore, create_trace_store

if TYPE_CHECKING:
    from neo4j import Driver
    from openai import OpenAI

    from agentic_graph_rag.reasoning.reasoning_engine import ReasoningEngine

logger = logging.getLogger(__name__)
_SESSION_CONTEXT_LIMIT = 5
_FOLLOW_UP_MAX_TURNS = 3
_TURN_QUERY_LIMIT = 180
_TURN_ANSWER_LIMIT = 240
_FOLLOW_UP_CUES = (
    "它",
    "它们",
    "这个",
    "这个病",
    "这种",
    "这个方案",
    "那它",
    "那个",
    "那些",
    "其",
    "该",
    "前者",
    "后者",
    "继续",
    "然后",
    "那么",
    "再说",
    "再讲",
    "what about",
    "how about",
    "and what about",
    "it",
    "they",
    "this",
    "that",
    "these",
    "those",
)


class PipelineService:
    """Typed contract for the Agentic Graph RAG pipeline."""

    def __init__(
        self,
        driver: Driver,
        openai_client: OpenAI,
        reasoning: ReasoningEngine | None = None,
        trace_store: TraceStore | None = None,
    ):
        self._driver = driver
        self._client = openai_client
        self._reasoning = reasoning
        self._trace_store = trace_store or create_trace_store()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _prepare_query_request(
        self,
        text: str,
        mode: str,
        session_id: str,
    ) -> tuple[str, bool, object | None, list[WorkflowMemoryEntry], list[ReflectionStep]]:
        use_llm = mode == "agent_llm"
        reasoning = self._reasoning if mode == "agent_mangle" else None
        (
            workflow_memory_seed,
            reflection_history_seed,
            recent_turns,
        ) = self._load_session_context(session_id)
        effective_text = expand_medical_aliases(self._contextualize_query(text, recent_turns))
        if effective_text != text:
            workflow_memory_seed = list(workflow_memory_seed)
            workflow_memory_seed.append(
                WorkflowMemoryEntry(
                    stage="session_context",
                    message="Applied recent conversation context to resolve follow-up references.",
                    metadata={
                        "original_query": text,
                        "expanded_query": effective_text,
                    },
                )
            )
        return (
            effective_text,
            use_llm,
            reasoning,
            workflow_memory_seed,
            reflection_history_seed,
        )

    def _run_agent_query(
        self,
        text: str,
        mode: str,
        session_id: str,
    ) -> tuple[QAResult, str]:
        (
            effective_text,
            use_llm,
            reasoning,
            workflow_memory_seed,
            reflection_history_seed,
        ) = self._prepare_query_request(text, mode, session_id)

        qa = agent_run(
            effective_text,
            self._driver,
            openai_client=self._client,
            use_llm_router=use_llm,
            reasoning=reasoning,
            session_id=session_id,
            workflow_memory_seed=workflow_memory_seed,
            reflection_history_seed=reflection_history_seed,
        )

        if qa.trace:
            qa.trace.query = text
            qa.trace.expanded_query = effective_text if effective_text != text else ""
            qa.trace.final_answer = qa.answer
            self._cache_trace(qa.trace)
        qa.query = text
        qa.expanded_query = effective_text if effective_text != text else ""
        return qa, effective_text

    @staticmethod
    def _iter_answer_tokens(answer: str) -> Iterator[str]:
        if not answer:
            return iter(())
        return iter(answer)

    @staticmethod
    def _adapt_qa_result_to_stream_events(
        qa: QAResult,
        *,
        text: str,
        session_id: str,
    ) -> Iterator[dict]:
        """Convert a completed QAResult into the public stream event contract."""
        trace = qa.trace or PipelineTrace(query=text)
        selected_tool = (
            trace.router_step.decision.suggested_tool
            if trace.router_step and trace.router_step.decision
            else ""
        )
        yield {
            "event": "status",
            "data": {
                "stage": "retrieval_started",
                "tool": selected_tool,
            },
        }
        yield {
            "event": "status",
            "data": {
                "stage": "generation_started",
                "sources": len(qa.sources),
            },
        }
        for token in PipelineService._iter_answer_tokens(qa.answer):
            yield {"event": "token", "data": {"text": token}}

        confidence = qa.confidence

        yield {
            "event": "done",
            "data": {
                "answer": qa.answer,
                "trace_id": trace.trace_id,
                "session_id": session_id,
                "retries": qa.retries,
                "sources": len(qa.sources),
                "confidence": confidence,
            },
        }

    def query(
        self,
        text: str,
        mode: str = "agent_pattern",
        session_id: str = "",
    ) -> QAResult:
        """Full pipeline: route -> retrieve -> generate -> trace."""
        qa, _effective_text = self._run_agent_query(text, mode, session_id)
        return qa

    def stream_query(
        self,
        text: str,
        mode: str = "agent_pattern",
        session_id: str = "",
    ) -> Iterator[dict]:
        """Stream pipeline progress and generation tokens over a stable event contract."""
        started = time.perf_counter()
        yield {"event": "status", "data": {"stage": "routing_started"}}
        qa, _effective_text = self._run_agent_query(text, mode, session_id)
        trace = qa.trace or PipelineTrace(query=text)
        trace.total_duration_ms = int((time.perf_counter() - started) * 1000)
        yield from self._adapt_qa_result_to_stream_events(
            qa,
            text=text,
            session_id=session_id,
        )

    def search(self, text: str, tool: str = "vector_search") -> list:
        """Run a specific retrieval tool directly (no agent routing)."""
        tool_map = {name: getattr(agent_tools, name) for name in TOOL_NAMES}
        fn = tool_map.get(tool)
        if fn is None:
            raise ValueError(f"Unknown tool: {tool}. Valid: {', '.join(TOOL_NAMES)}")
        return fn(text, self._driver, self._client)

    def get_trace(self, trace_id: str) -> PipelineTrace | None:
        """Retrieve trace from store."""
        return self._trace_store.get(trace_id)

    def health(self) -> dict:
        """Neo4j connectivity check."""
        try:
            with open_neo4j_session(self._driver) as session:
                session.run("RETURN 1").single()
            return {"status": "ok", "neo4j": "connected"}
        except Exception as e:
            return {"status": "degraded", "neo4j": str(e)}

    def graph_stats(self) -> dict:
        """Node and edge counts."""
        try:
            with open_neo4j_session(self._driver) as session:
                result = session.run(
                    "MATCH (n) RETURN count(n) AS nodes "
                    "UNION ALL "
                    "MATCH ()-[r]->() RETURN count(r) AS nodes"
                )
                counts = [r["nodes"] for r in result]
            return {"nodes": counts[0] if counts else 0, "edges": counts[1] if len(counts) > 1 else 0}
        except Exception as e:
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _cache_trace(self, trace: PipelineTrace) -> None:
        """Add trace to store."""
        self._trace_store.put(trace)

    def _load_session_context(
        self,
        session_id: str,
    ) -> tuple[list[WorkflowMemoryEntry], list[ReflectionStep], list[dict[str, str]]]:
        """Load recent session memory and reflection history for follow-up queries."""
        if not session_id:
            return [], [], []

        traces = self._trace_store.get_session_traces(session_id, limit=_SESSION_CONTEXT_LIMIT)
        workflow_memory = self._trace_store.get_session_memory(
            session_id,
            limit=_SESSION_CONTEXT_LIMIT,
        )
        reflection_history: list[ReflectionStep] = []
        for trace in traces:
            reflection_history.extend(trace.reflection_steps)
        return workflow_memory, reflection_history, self._recent_conversation_turns(traces)

    def _recent_conversation_turns(self, traces: list[PipelineTrace]) -> list[dict[str, str]]:
        turns: list[dict[str, str]] = []
        for trace in traces[-_FOLLOW_UP_MAX_TURNS:]:
            if not trace.query or not trace.final_answer:
                continue
            turns.append(
                {
                    "query": self._trim_text(trace.query, _TURN_QUERY_LIMIT),
                    "answer": self._trim_text(trace.final_answer, _TURN_ANSWER_LIMIT),
                }
            )
        return turns

    def _contextualize_query(self, text: str, recent_turns: list[dict[str, str]]) -> str:
        if not recent_turns or not self._looks_like_follow_up(text):
            return text

        history_lines: list[str] = []
        for idx, turn in enumerate(recent_turns, start=1):
            history_lines.append(f"Turn {idx} User: {turn['query']}")
            history_lines.append(f"Turn {idx} Assistant: {turn['answer']}")

        history = "\n".join(history_lines)
        return (
            "Conversation history:\n"
            f"{history}\n\n"
            "Current follow-up question:\n"
            f"{text}\n\n"
            "Resolve omitted references using the conversation history and focus on the current question."
        )

    def _looks_like_follow_up(self, text: str) -> bool:
        normalized = " ".join(text.strip().lower().split())
        if not normalized:
            return False
        if len(normalized) <= 12:
            return True
        return any(cue in normalized for cue in _FOLLOW_UP_CUES)

    @staticmethod
    def _trim_text(text: str, limit: int) -> str:
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return compact[: max(0, limit - 3)].rstrip() + "..."
