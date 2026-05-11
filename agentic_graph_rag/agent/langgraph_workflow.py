"""LangGraph-based orchestration for the retrieval self-correction loop."""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from rag_core.config import get_settings
from rag_core.models import (
    ClaimVerificationStep,
    EscalationStep,
    GeneratorStep,
    PipelineTrace,
    ProviderDiagnostic,
    QAResult,
    QueryType,
    ReflectionStep,
    RouterDecision,
    RouterStep,
    SearchResult,
    ToolStep,
    WorkflowMemoryEntry,
)
from rag_core.reflector import resolve_reflection_verdict

from agentic_graph_rag.agent.budget import BudgetTracker, LLMBudgetExceeded
from agentic_graph_rag.agent.correction_planner import (
    CorrectionGap,
    CorrectionPlan,
    build_gap_report,
)
from agentic_graph_rag.agent.query_signals import (
    extract_query_signals,
    has_strong_form_anchor,
)

_ANCHOR_PATTERN = re.compile(r"[A-Za-z0-9_.-]+|[\u4e00-\u9fff]{2,}")
_REFLECTION_MIN_REMAINING_BUDGET_MS = 1000
_REFLECTION_TRANSPORT_FAILURE_MARKERS = (
    "connection error",
    "connection timed out",
    "connection_timeout",
    "error code: 522",
    "eof occurred",
    "retryable",
    "timed out",
)
_ANCHOR_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "does",
    "how",
    "is",
    "of",
    "or",
    "the",
    "to",
    "what",
    "which",
    "with",
    "为什么",
    "什么",
    "哪些",
    "如何",
    "多少",
    "怎么",
    "是否",
    "有关",
}
_RELATION_QUERY_TERMS = {
    "compare",
    "difference",
    "impact",
    "relation",
    "relationship",
    "依赖",
    "关系",
    "关联",
    "区别",
    "影响",
    "差异",
}
_MAX_INITIAL_TOOLS = 2
InitialRetrievalOutput = tuple[
    list[SearchResult],
    int,
    list[WorkflowMemoryEntry],
    list[ReflectionStep],
    PipelineTrace | None,
]


@dataclass(frozen=True)
class SelfCorrectionOps:
    """Callbacks that bind workflow nodes to project-specific retrieval logic."""

    execution_sources: Callable[
        [str, str, dict[str, dict[str, list[SearchResult]]], list[str] | None],
        tuple[list[str], list[str]],
    ]
    run_tool: Callable[..., list[SearchResult]]
    cache_tool_results: Callable[
        [str, str, list[SearchResult], dict[str, dict[str, list[SearchResult]]]],
        None,
    ]
    cache_hybrid_provider_results: Callable[
        [str, dict[str, list[SearchResult]], dict[str, dict[str, list[SearchResult]]]],
        None,
    ]
    build_provider_diagnostics: Callable[
        [dict[str, list[SearchResult]] | None, list[str], list[str]],
        list[ProviderDiagnostic],
    ]
    evaluate_reflection: Callable[..., ReflectionStep]
    plan_incremental_retry: Callable[
        [
            str,
            str,
            ReflectionStep,
            dict[str, dict[str, list[SearchResult]]],
            dict[str, list[SearchResult]] | None,
        ],
        tuple[str, list[str], list[str]] | None,
    ]
    get_next_tool: Callable[[str, set[str], ReflectionStep, RouterDecision], str | None]
    should_rewrite_query: Callable[[str, ReflectionStep, list[str]], bool]
    generate_retry_query: Callable[..., str]
    rerank_results: Callable[..., list[SearchResult]]


class SelfCorrectionState(TypedDict, total=False):
    """State carried across LangGraph nodes for self-correction."""

    ops: SelfCorrectionOps
    driver: Any
    openai_client: Any
    decision: RouterDecision
    trace: PipelineTrace | None
    relevance_threshold: float
    max_retries: int
    base_query: str
    current_query: str
    current_tool: str
    attempt: int
    tried_tools: list[str]
    reflection_history: list[ReflectionStep]
    channel_cache: dict[str, dict[str, list[SearchResult]]]
    forced_hybrid_providers: list[str] | None
    results: list[SearchResult]
    best_results: list[SearchResult]
    best_score: float
    best_attempt: int
    best_rank: tuple[int, int, int, int]
    retries_used: int
    reused_sources: list[str]
    executed_sources: list[str]
    provider_results_sink: dict[str, list[SearchResult]] | None
    pending_reflection: ReflectionStep | None
    pending_reflection_signal: float | None
    pending_reflection_threshold: float | None
    last_reflection: ReflectionStep | None
    next_step: str
    last_elapsed_ms: int
    tool_step_logged_for_attempt: bool
    total_reranks: int
    max_reranks: int
    rewrite_attempted: bool
    max_query_rewrites: int
    query_history: list[str]
    started_at_monotonic: float
    time_budget_ms: int
    memory: list[WorkflowMemoryEntry]
    stop_requested: bool
    final_results: list[SearchResult]


def _select_best_results(
    state: SelfCorrectionState,
    reflection: ReflectionStep,
) -> tuple[list[SearchResult], float, int, tuple[int, int, int, int]]:
    """Track the strongest evidence seen so far by lexical anchor ranking.

    `best_score` is retained for trace compatibility but no longer carries
    decision power — the verdict/evidence_status enums drive control flow.
    """
    best_results = state.get("best_results", [])
    best_score = state.get("best_score", 0.0)
    best_attempt = state.get("best_attempt", 0)
    best_rank = state.get("best_rank", (-1, -1, -1, -10**9))
    candidate_rank = _reflection_rank(
        reflection,
        query=state["current_query"],
        results=state["results"],
        executed_sources=state.get("executed_sources", []),
        reused_sources=state.get("reused_sources", []),
    )
    if candidate_rank > best_rank:
        return (
            state["results"],
            _top_evidence_signal(state["results"]),
            state["attempt"],
            candidate_rank,
        )
    return best_results, best_score, best_attempt, best_rank


def _top_evidence_signal(results: list[SearchResult]) -> float:
    """Return the strongest per-provider normalized signal, or 0 if none."""
    if not results:
        return 0.0
    signals = [r.score_normalized for r in results if r.score_normalized is not None]
    if signals:
        return max(signals)
    fallback_scores = [r.score for r in results if 0.0 <= r.score <= 1.0]
    if fallback_scores:
        return max(fallback_scores)
    return 0.0


def _initial_tool_plan(query: str, decision: RouterDecision) -> list[str]:
    """Preserve router's primary tool and add at most one companion channel."""
    tools = [decision.suggested_tool]
    signals = extract_query_signals(query)

    if has_strong_form_anchor(signals):
        if "bm25_search" not in tools:
            tools.append("bm25_search")
    elif decision.suggested_tool in {"bm25_search", "cypher_traverse"}:
        tools.append("vector_search")

    planned: list[str] = []
    for tool in tools:
        if tool and tool not in planned:
            planned.append(tool)
        if len(planned) >= _MAX_INITIAL_TOOLS:
            break
    return planned or [decision.suggested_tool]


def _decision_for_initial_tool(decision: RouterDecision, tool: str) -> RouterDecision:
    if tool == decision.suggested_tool:
        return decision
    return decision.model_copy(update={"suggested_tool": tool})


def _merge_initial_retrieval_outputs(
    outputs: list[InitialRetrievalOutput],
    target_trace: PipelineTrace | None,
) -> tuple[list[SearchResult], int, list[WorkflowMemoryEntry], list[ReflectionStep]]:
    results: list[SearchResult] = []
    existing_ids: set[str] = set()
    retries = 0
    memory: list[WorkflowMemoryEntry] = []
    reflections: list[ReflectionStep] = []

    for tool_results, tool_retries, tool_memory, tool_reflections, tool_trace in outputs:
        retries += tool_retries
        memory.extend(tool_memory)
        reflections.extend(tool_reflections)
        for result in tool_results:
            chunk_id = result.chunk.id
            dedupe_key = chunk_id or result.chunk.enriched_content
            if dedupe_key in existing_ids:
                continue
            existing_ids.add(dedupe_key)
            results.append(result)
        if target_trace is not None and tool_trace is not None:
            target_trace.tool_steps.extend(tool_trace.tool_steps)
            target_trace.escalation_steps.extend(tool_trace.escalation_steps)

    return results, retries, memory, reflections


def _start_llm_call_or_skip(state: AgentWorkflowState, operation: str) -> tuple[bool, list[WorkflowMemoryEntry]]:
    budget = state.get("budget")
    if budget is None:
        return True, list(state.get("memory", []))
    try:
        budget.start_llm_call(operation)
    except LLMBudgetExceeded as exc:
        memory = _append_memory_entry(
            state,
            stage="budget",
            message=str(exc),
            metadata=budget.snapshot(),
        )
        return False, memory
    memory = _append_memory_entry(
        state,
        stage="budget",
        message=f"started LLM call: {operation}",
        metadata=budget.snapshot(),
    )
    return True, memory


def _record_reflection_trace(
    state: SelfCorrectionState,
    reflection: ReflectionStep,
) -> None:
    trace = state.get("trace")
    if trace is None:
        return
    ops = state["ops"]
    if not state.get("tool_step_logged_for_attempt", False):
        provider_results = state.get("provider_results_sink")
        executed_sources = state.get("executed_sources", [])
        if provider_results is None and len(executed_sources) == 1:
            provider_results = {executed_sources[0]: state["results"]}
        trace.tool_steps.append(
            ToolStep(
                tool_name=state["current_tool"],
                results_count=len(state["results"]),
                relevance_score=_top_evidence_signal(state["results"]),
                duration_ms=state.get("last_elapsed_ms", 0),
                query_used=state["current_query"],
                cache_hit=bool(state.get("reused_sources")),
                reused_sources=state.get("reused_sources", []),
                executed_sources=state.get("executed_sources", []),
                provider_diagnostics=ops.build_provider_diagnostics(
                    provider_results,
                    state.get("reused_sources", []),
                    state.get("executed_sources", []),
                ),
            )
        )
        trace.reflection_steps.append(reflection)


def _build_skip_reflection(
    state: SelfCorrectionState,
    *,
    top_signal: float,
    threshold: float,
) -> ReflectionStep:
    """Build a synthetic reflection step when retrieval signals are strong
    enough to skip the LLM judge call.

    No numeric score is filled — the `verdict == "answer"` enum alone carries
    the decision. `top_signal` is only stored for observability.
    """
    return ReflectionStep(
        attempt=state["attempt"],
        tool_name=state["current_tool"],
        query_used=state["current_query"],
        evidence_status="sufficient",
        gap_type="none",
        action="answer",
        required_tool="none",
        verdict="answer",
        missing_information=[],
        missing_entities=[],
        missing_relationships=[],
        coverage_gap_sources=[],
        candidate_fix_paths=["skip_reflection"],
        preferred_tools=[],
        preferred_providers=[],
        retry_scope="",
        reasoning=(
            f"Skipped LLM reflection because top normalized vector score "
            f"{top_signal:.3f} >= {threshold:.3f}."
        ),
        failure_type="",
        recommended_action="answer",
        should_retry=False,
        should_rewrite_query=False,
        should_rerank_again=False,
        comparison_to_previous="Auto-accepted due to strong retrieval evidence.",
    )


def _build_budget_exhausted_reflection(state: SelfCorrectionState) -> ReflectionStep:
    """Stop before an expensive judge call when the request budget is gone."""
    return ReflectionStep(
        attempt=state["attempt"],
        tool_name=state["current_tool"],
        query_used=state["current_query"],
        evidence_status="insufficient",
        gap_type="off_topic",
        action="stop",
        required_tool="none",
        verdict="retry",
        missing_information=["Request time budget is too low for reflection."],
        missing_entities=[],
        missing_relationships=[],
        coverage_gap_sources=[],
        candidate_fix_paths=["skip_reflection_due_to_budget"],
        preferred_tools=[],
        preferred_providers=[],
        retry_scope="tool_escalation",
        reasoning="Skipped LLM reflection because the remaining request time budget was too low.",
        failure_type="insufficient_context",
        recommended_action="stop_due_to_time_budget",
        should_retry=True,
        should_rewrite_query=False,
        should_rerank_again=False,
        comparison_to_previous="Budget guard stop.",
    )


def _finalize_reflection_update(
    state: SelfCorrectionState,
    reflection: ReflectionStep,
    *,
    memory_message: str,
    memory_metadata: dict[str, Any],
) -> dict[str, Any]:
    reflection_history = list(state.get("reflection_history", []))
    reflection_history.append(reflection)
    best_results, best_score, best_attempt, best_rank = _select_best_results(
        state,
        reflection,
    )
    _record_reflection_trace(state, reflection)
    memory = _append_memory_entry(
        state,
        stage="reflection",
        message=memory_message,
        metadata=memory_metadata,
    )
    return {
        "last_reflection": reflection,
        "reflection_history": reflection_history,
        "best_results": best_results,
        "best_score": best_score,
        "best_attempt": best_attempt,
        "best_rank": best_rank,
        "tool_step_logged_for_attempt": True,
        "memory": memory,
    }


def _should_retry_after_answer_verdict(
    state: SelfCorrectionState,
    reflection: ReflectionStep,
) -> bool:
    """Allow an additional retry when the answer verdict looks thin.

    With the verdict-based architecture we no longer have a numeric score to
    compare. We retry when the reflection reports partial/insufficient
    evidence despite emitting an `answer` verdict — this catches the case
    where the judge defaults to `answer` on weak evidence.
    """
    evidence_status = (reflection.evidence_status or "").strip().lower()
    if evidence_status not in {"partial", "insufficient", "empty"}:
        return False
    return state["attempt"] < state["max_retries"]


def _next_step_after_reflection(
    state: SelfCorrectionState,
    reflection: ReflectionStep,
) -> str:
    verdict = resolve_reflection_verdict(reflection)
    if verdict in {"answer", "stop"}:
        if verdict == "answer" and _should_retry_after_answer_verdict(state, reflection):
            return "prepare_retry"
        return "finish"
    if _budget_exhausted(state):
        return "prepare_retry"
    if (
        verdict == "rerank"
        and state.get("results")
        and state.get("total_reranks", 0) < state.get("max_reranks", 1)
    ):
        return "rerank_results"
    if state["attempt"] >= state["max_retries"]:
        return "finish"
    return "prepare_retry"


def _query_anchor_terms(query: str) -> list[str]:
    """Extract stable lexical anchors from a user query."""
    ordered: list[str] = []
    for match in _ANCHOR_PATTERN.findall(query.casefold()):
        token = match.strip()
        if not token or token in _ANCHOR_STOPWORDS:
            continue
        if token in _RELATION_QUERY_TERMS:
            continue
        if token.isalpha() and len(token) < 2:
            continue
        if token not in ordered:
            ordered.append(token)
    return ordered


def _relation_query_terms(query: str) -> list[str]:
    """Extract explicit relation operators from the query."""
    lowered = query.casefold()
    ordered: list[str] = []
    for term in _RELATION_QUERY_TERMS:
        if term in lowered and term not in ordered:
            ordered.append(term)
    return ordered


def _anchor_hit_profile(query: str, results: list[SearchResult]) -> tuple[int, int, int, int]:
    """Measure objective lexical evidence quality."""
    anchors = _query_anchor_terms(query)
    relation_terms = _relation_query_terms(query)
    if not anchors and not relation_terms:
        return (0, 0, 0, 0)

    distinct_hits: set[str] = set()
    exact_query_hit = 0
    relation_hits: set[str] = set()
    best_rank = 10**9
    normalized_query = query.casefold().strip()
    for result in results:
        text = (result.chunk.enriched_content or result.chunk.content or "").casefold()
        if not text:
            continue
        if normalized_query and normalized_query in text:
            exact_query_hit = 1
        hits = {anchor for anchor in anchors if anchor in text}
        distinct_hits.update(hits)
        relation_hits.update(term for term in relation_terms if term in text)
        candidate_rank = result.rank if result.rank > 0 else 10**9
        if candidate_rank < best_rank:
            best_rank = candidate_rank
    if best_rank == 10**9:
        best_rank = 0
    return (exact_query_hit, len(distinct_hits), len(relation_hits), -best_rank)


def _remaining_budget_ms(state: SelfCorrectionState) -> int:
    """Return remaining request budget in milliseconds."""
    budget_ms = max(0, int(state.get("time_budget_ms", 0)))
    started = float(state.get("started_at_monotonic", 0.0))
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return max(0, budget_ms - elapsed_ms)


def _budget_exhausted(state: SelfCorrectionState) -> bool:
    """Whether the request-level budget has been exhausted."""
    if "time_budget_ms" not in state:
        return False
    return _remaining_budget_ms(state) <= 0


def _reflection_budget_too_low(state: SelfCorrectionState) -> bool:
    """Avoid starting an LLM judge call that cannot fit the remaining budget."""
    if "time_budget_ms" not in state:
        return False
    return _remaining_budget_ms(state) < _REFLECTION_MIN_REMAINING_BUDGET_MS


def _is_reflection_transport_failure(reflection: ReflectionStep) -> bool:
    """Identify policy stops caused by retryable transport failures, not bad JSON."""
    if (reflection.recommended_action or "") != "stop_due_to_invalid_reflection":
        return False
    reason = (reflection.reasoning or "").casefold()
    return any(marker in reason for marker in _REFLECTION_TRANSPORT_FAILURE_MARKERS)


def _answer_guard_status(
    reason: str,
    *,
    has_answer: bool,
    retrieval_status: str,
) -> str:
    """Map guard outcome to answer status without exposing internal details.

    Reflection/verification budget failures should not invalidate an already
    generated answer grounded in retrieved evidence. They only mean the answer
    could not be fully policy-checked.
    """
    if has_answer and retrieval_status == "complete":
        return "partial"
    normalized = (reason or "").casefold()
    if (
        "time budget exhausted" in normalized
        or "request timed out" in normalized
        or "timed out" in normalized
        or "budget" in normalized
    ):
        return "skipped_timeout"
    if any(marker in normalized for marker in _REFLECTION_TRANSPORT_FAILURE_MARKERS):
        return "partial"
    return "partial"


def _relation_query_can_retry_graph(
    state: SelfCorrectionState,
    reflection: ReflectionStep,
) -> bool:
    """Use deterministic graph fallback when the reflection judge is unavailable."""
    decision = state.get("decision")
    if decision is None:
        return False
    return (
        _is_reflection_transport_failure(reflection)
        and decision.query_type in {QueryType.RELATION, QueryType.MULTI_HOP}
        and state["current_tool"] != "cypher_traverse"
        and "cypher_traverse" not in set(state.get("tried_tools", []))
        and state["attempt"] < state["max_retries"]
        and bool(state.get("results"))
    )


def _build_graph_retry_after_reflection_failure(
    state: SelfCorrectionState,
    reflection: ReflectionStep,
) -> ReflectionStep:
    """Preserve the transport failure while routing relation queries to graph."""
    candidate_fix_paths = list(reflection.candidate_fix_paths)
    if "reflection_transport_failure_graph_fallback" not in candidate_fix_paths:
        candidate_fix_paths.append("reflection_transport_failure_graph_fallback")
    return reflection.model_copy(
        update={
            "evidence_status": "partial",
            "gap_type": "missing_relation",
            "action": "retry_graph",
            "required_tool": "cypher_traverse",
            "verdict": "retry",
            "failure_type": "relation_missing",
            "recommended_action": "use_graph_traversal",
            "preferred_tools": ["cypher_traverse"],
            "preferred_providers": ["graph"],
            "retry_scope": "tool_escalation",
            "reasoning": (
                "Reflection LLM transport failed; relation query is falling back "
                f"deterministically to graph traversal. Original reason: {reflection.reasoning}"
            ),
            "should_retry": True,
            "should_rewrite_query": False,
            "should_rerank_again": False,
            "candidate_fix_paths": candidate_fix_paths,
        }
    )


def _reflection_rank(
    reflection: ReflectionStep,
    *,
    query: str,
    results: list[SearchResult],
    executed_sources: list[str],
    reused_sources: list[str],
) -> tuple[int, int, int, int]:
    """Rank evidence using deterministic lexical/objective signals only."""
    del reflection, executed_sources, reused_sources
    return _anchor_hit_profile(query, results)


def _current_missing_claims(reflection: ReflectionStep) -> list[str]:
    """Normalize the reflection's claimed gaps for loop-guard checks."""
    claims = [
        *reflection.missing_information,
        *reflection.missing_entities,
        *reflection.missing_relationships,
    ]
    normalized: list[str] = []
    for claim in claims:
        text = str(claim).strip().casefold()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _missing_claim_covered_by_results(claim: str, results: list[SearchResult]) -> bool:
    """Reject retries when the claimed gap already appears in retrieved evidence."""
    claim_terms = [
        token for token in _ANCHOR_PATTERN.findall(claim)
        if token and token not in _ANCHOR_STOPWORDS
    ]
    if not claim_terms:
        return False
    corpus = "\n".join(
        (result.chunk.enriched_content or result.chunk.content or "").casefold()
        for result in results
    )
    return all(term.casefold() in corpus for term in claim_terms)


def _should_block_reflection_retry(state: SelfCorrectionState) -> tuple[bool, str]:
    """Stop when reflection repeatedly hallucinates or re-asks covered gaps."""
    reflection = state.get("last_reflection")
    if reflection is None:
        return False, ""

    current_claims = _current_missing_claims(reflection)
    if not current_claims:
        return False, ""

    for claim in current_claims:
        if _missing_claim_covered_by_results(claim, state.get("results", [])):
            return True, f"reflection requested already-covered gap: {claim}"

    history = list(state.get("reflection_history", []))
    previous_claim_sets = [
        set(_current_missing_claims(step))
        for step in history[:-1]
        if _current_missing_claims(step)
    ]
    current_claim_set = set(current_claims)
    if previous_claim_sets and current_claim_set == previous_claim_sets[-1]:
        return True, "reflection repeated the same missing claims consecutively"
    if current_claim_set and sum(1 for item in previous_claim_sets if item == current_claim_set) >= 1:
        return True, "reflection repeated a previously requested missing claim set"
    return False, ""


def _stop_with_best_results(
    state: SelfCorrectionState,
    *,
    stage: str,
    message: str,
) -> dict[str, Any]:
    """Stop the loop and keep the strongest evidence collected so far."""
    memory = _append_memory_entry(
        state,
        stage=stage,
        message=message,
        metadata={
            "tool": state.get("current_tool", ""),
            "remaining_budget_ms": _remaining_budget_ms(state),
        },
    )
    return {
        "stop_requested": True,
        "retries_used": state.get("attempt", 0),
        "final_results": state.get("best_results") or state.get("results", []),
        "results": state.get("best_results") or state.get("results", []),
        "memory": memory,
    }


def _append_memory_entry(
    state: dict[str, Any],
    *,
    stage: str,
    message: str,
    metadata: dict[str, Any] | None = None,
) -> list[WorkflowMemoryEntry]:
    """Append one structured memory entry and sync it into the trace."""
    memory = list(state.get("memory", []))
    normalized_metadata = {
        key: value
        for key, value in (metadata or {}).items()
        if value not in (None, "", [], {}, ())
    }
    memory.append(
        WorkflowMemoryEntry(
            stage=stage,
            message=message,
            metadata=normalized_metadata,
        )
    )
    trace = state.get("trace")
    if trace is not None:
        trace.workflow_memory = list(memory)
    return memory


def _execute_attempt(state: SelfCorrectionState) -> dict[str, Any]:
    """Run the current tool once and cache any provider-level evidence."""
    ops = state["ops"]
    current_query = state["current_query"]
    current_tool = state["current_tool"]
    channel_cache = state["channel_cache"]
    forced_hybrid_providers = state.get("forced_hybrid_providers")

    reused_sources, executed_sources = ops.execution_sources(
        current_tool,
        current_query,
        channel_cache,
        forced_hybrid_providers,
    )
    provider_results_sink = {} if current_tool == "hybrid_search" else None

    started = time.perf_counter()
    results = ops.run_tool(
        current_tool,
        current_query,
        state["driver"],
        state["openai_client"],
        state["decision"],
        channel_cache=channel_cache,
        hybrid_enabled_providers=forced_hybrid_providers,
        provider_results_sink=provider_results_sink,
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    ops.cache_tool_results(current_query, current_tool, results, channel_cache)
    if current_tool == "hybrid_search" and provider_results_sink is not None:
        ops.cache_hybrid_provider_results(current_query, provider_results_sink, channel_cache)

    tried_tools = list(state.get("tried_tools", []))
    if current_tool not in tried_tools:
        tried_tools.append(current_tool)

    memory = _append_memory_entry(
        state,
        stage="retrieval",
        message=f"{current_tool} returned {len(results)} results",
        metadata={
            "query": current_query,
            "tool": current_tool,
            "reused_sources": reused_sources,
            "executed_sources": executed_sources,
        },
    )

    return {
        "results": results,
        "reused_sources": reused_sources,
        "executed_sources": executed_sources,
        "provider_results_sink": provider_results_sink,
        "last_elapsed_ms": elapsed_ms,
        "tried_tools": tried_tools,
        "forced_hybrid_providers": None,
        "tool_step_logged_for_attempt": False,
        "memory": memory,
    }


def _evaluate_reflection_node(state: SelfCorrectionState) -> dict[str, Any]:
    """Evaluate the current retrieval attempt without mutating workflow history.

    Skip-reflection optimization: if the top vector result has a strong
    normalized similarity, we trust it enough to skip the LLM judge. We
    intentionally restrict this to vector-source results because
    `score_normalized` from BM25 (score saturation) and graph (heuristic
    floors) is not directly comparable — mixing them would be the "same name,
    different semantics" anti-pattern.
    """
    ops = state["ops"]
    results = state.get("results", [])

    cfg = get_settings()
    threshold = cfg.agent.reflection_skip_score_threshold
    vector_signals = [
        result.score_normalized
        for result in results
        if result.source == "vector" and result.score_normalized is not None
    ]
    if vector_signals:
        top_signal = max(vector_signals)
        if top_signal >= threshold:
            reflection = _build_skip_reflection(
                state,
                top_signal=top_signal,
                threshold=threshold,
            )
            return {
                "pending_reflection": reflection,
                "pending_reflection_signal": top_signal,
                "pending_reflection_threshold": threshold,
            }

    if results and _reflection_budget_too_low(state):
        return {
            "pending_reflection": _build_budget_exhausted_reflection(state),
        }

    reflection = ops.evaluate_reflection(
        state["current_query"],
        results,
        openai_client=state["openai_client"],
        reflection_history=list(state.get("reflection_history", [])),
        workflow_memory=state.get("memory", []),
        tool_name=state["current_tool"],
        attempt=state["attempt"],
    )
    return {"pending_reflection": reflection}


def _interpret_verdict_node(state: SelfCorrectionState) -> dict[str, Any]:
    """Record reflection state and map its verdict to the next workflow step."""
    reflection = state.get("pending_reflection")
    if reflection is None:
        return {"next_step": "finish"}

    if _relation_query_can_retry_graph(state, reflection):
        reflection = _build_graph_retry_after_reflection_failure(state, reflection)

    reflection.verdict = resolve_reflection_verdict(reflection)

    pending_signal = state.get("pending_reflection_signal")
    if "skip_reflection" in reflection.candidate_fix_paths:
        memory_message = (
            f"{state['current_tool']} skipped LLM reflection with "
            f"top vector signal {pending_signal:.2f}"
            if pending_signal is not None
            else f"{state['current_tool']} skipped LLM reflection"
        )
        memory_metadata = {
            "skip_threshold": state.get(
                "pending_reflection_threshold",
                get_settings().agent.reflection_skip_score_threshold,
            ),
            "top_vector_signal": pending_signal,
            "retry_scope": reflection.retry_scope,
        }
    else:
        memory_message = (
            f"{state['current_tool']} classified evidence as "
            f"{reflection.evidence_status or 'unknown'} "
            f"with action {reflection.action or reflection.recommended_action or 'unknown'}"
        )
        memory_metadata = {
            "evidence_status": reflection.evidence_status,
            "gap_type": reflection.gap_type,
            "action": reflection.action,
            "verdict": reflection.verdict,
            "failure_type": reflection.failure_type,
            "missing_information": reflection.missing_information,
            "missing_entities": reflection.missing_entities,
            "retry_scope": reflection.retry_scope,
        }

    update = _finalize_reflection_update(
        state,
        reflection,
        memory_message=memory_message,
        memory_metadata=memory_metadata,
    )
    update["next_step"] = _next_step_after_reflection(
        {**state, **update},
        reflection,
    )
    return update


def _after_interpret_verdict(state: SelfCorrectionState) -> str:
    """Route using the decision computed by the verdict interpreter node."""
    return state.get("next_step", "finish")


def _rerank_results(state: SelfCorrectionState) -> dict[str, Any]:
    """Apply one local rerank pass before escalating to a broader tool."""
    started = time.perf_counter()
    reranked = state["ops"].rerank_results(
        state["current_query"],
        state["results"],
        openai_client=state["openai_client"],
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    trace = state.get("trace")
    if trace is not None:
        trace.tool_steps.append(
            ToolStep(
                tool_name="rerank_results",
                results_count=len(reranked),
                relevance_score=_top_evidence_signal(reranked),
                duration_ms=elapsed_ms,
                query_used=state["current_query"],
            )
        )

    memory = _append_memory_entry(
        state,
        stage="rerank",
        message=f"rerank_results reordered {len(reranked)} candidates",
        metadata={
            "query": state["current_query"],
            "previous_tool": state["current_tool"],
        },
    )
    return {
        "results": reranked,
        "last_elapsed_ms": elapsed_ms,
        "total_reranks": state.get("total_reranks", 0) + 1,
        "memory": memory,
    }


def _prepare_retry(state: SelfCorrectionState) -> dict[str, Any]:
    """Choose the next tool and optionally rewrite the query."""
    if _budget_exhausted(state):
        return _stop_with_best_results(
            state,
            stage="retry",
            message="request time budget exhausted; returning best known evidence",
        )
    should_block_retry, block_reason = _should_block_reflection_retry(state)
    if should_block_retry:
        return _stop_with_best_results(
            state,
            stage="retry",
            message=block_reason,
        )

    ops = state["ops"]
    current_query = state["current_query"]
    current_tool = state["current_tool"]
    reflection = state["last_reflection"]
    channel_cache = state["channel_cache"]
    provider_results_sink = state.get("provider_results_sink")

    incremental_plan = ops.plan_incremental_retry(
        current_query,
        current_tool,
        reflection,
        channel_cache,
        provider_results_sink,
    )
    if incremental_plan is not None:
        next_tool, forced_hybrid_providers, cached_sources_reused = incremental_plan
    else:
        next_tool = ops.get_next_tool(
            current_tool,
            set(state.get("tried_tools", [])),
            reflection,
            state["decision"],
        )
        forced_hybrid_providers = None
        cached_sources_reused = []

    if not next_tool:
        return _stop_with_best_results(
            state,
            stage="retry",
            message="retry plan exhausted; returning best known evidence",
        )

    next_query = current_query
    rewrite_attempted = state.get("rewrite_attempted", False)
    query_history = list(state.get("query_history", [state.get("base_query", current_query)]))
    if (
        not rewrite_attempted
        and state.get("max_query_rewrites", 1) > 0
        and ops.should_rewrite_query(next_tool, reflection, cached_sources_reused)
    ):
        candidate_query = ops.generate_retry_query(
            current_query,
            state["results"],
            openai_client=state["openai_client"],
            reflection=reflection,
            reflection_history=state.get("reflection_history", []),
            workflow_memory=state.get("memory", []),
        ).strip()
        rewrite_attempted = True
        normalized_history = {item.strip().casefold() for item in query_history if item.strip()}
        if candidate_query and candidate_query.casefold() not in normalized_history:
            next_query = candidate_query
            query_history.append(candidate_query)

    trace = state.get("trace")
    if trace is not None:
        fallback_reason = (
            f"reflection verdict={reflection.verdict or 'unknown'}, "
            f"evidence={reflection.evidence_status or 'unknown'}"
        )
        trace.escalation_steps.append(
            EscalationStep(
                from_tool=current_tool,
                to_tool=next_tool,
                reason=(
                    f"{reflection.failure_type or 'low_score'}: "
                    f"{reflection.reasoning or fallback_reason}"
                ),
                rephrased_query=next_query,
                cached_sources_reused=cached_sources_reused,
            )
        )

    memory = _append_memory_entry(
        state,
        stage="retry",
        message=f"retry planned from {current_tool} to {next_tool}",
        metadata={
            "current_query": current_query,
            "next_query": next_query,
            "cached_sources_reused": cached_sources_reused,
            "forced_hybrid_providers": forced_hybrid_providers or [],
        },
    )

    return {
        "current_query": next_query,
        "current_tool": next_tool,
        "attempt": state["attempt"] + 1,
        "forced_hybrid_providers": forced_hybrid_providers,
        "stop_requested": False,
        "rewrite_attempted": rewrite_attempted,
        "query_history": query_history,
        "memory": memory,
    }


def _after_prepare_retry(state: SelfCorrectionState) -> str:
    """Continue looping unless retry preparation requested a stop."""
    if state.get("stop_requested"):
        return "finish"
    return "execute_attempt"


@lru_cache(maxsize=1)
def _compile_self_correction_graph():
    """Compile the LangGraph workflow once and reuse it across requests."""
    graph = StateGraph(SelfCorrectionState)
    graph.add_node("execute_attempt", _execute_attempt)
    graph.add_node("evaluate_reflection", _evaluate_reflection_node)
    graph.add_node("interpret_verdict", _interpret_verdict_node)
    graph.add_node("rerank_results", _rerank_results)
    graph.add_node("prepare_retry", _prepare_retry)

    graph.add_edge(START, "execute_attempt")
    graph.add_edge("execute_attempt", "evaluate_reflection")
    graph.add_edge("evaluate_reflection", "interpret_verdict")
    graph.add_conditional_edges(
        "interpret_verdict",
        _after_interpret_verdict,
        {
            "rerank_results": "rerank_results",
            "prepare_retry": "prepare_retry",
            "finish": END,
        },
    )
    graph.add_edge("rerank_results", "evaluate_reflection")
    graph.add_conditional_edges(
        "prepare_retry",
        _after_prepare_retry,
        {
            "execute_attempt": "execute_attempt",
            "finish": END,
        },
    )
    return graph.compile()


def run_self_correction_workflow(
    *,
    query: str,
    driver: Any,
    openai_client: Any,
    decision: RouterDecision,
    max_retries: int,
    relevance_threshold: float,
    max_reranks: int = 1,
    max_query_rewrites: int = 0,
    request_time_budget_ms: int = 1500,
    trace: PipelineTrace | None,
    ops: SelfCorrectionOps,
    memory_seed: list[WorkflowMemoryEntry] | None = None,
    memory_sink: list[WorkflowMemoryEntry] | None = None,
    reflection_history_sink: list[ReflectionStep] | None = None,
) -> tuple[list[SearchResult], int]:
    """Execute the retrieval correction loop via LangGraph."""
    workflow = _compile_self_correction_graph()
    initial_state: SelfCorrectionState = {
        "ops": ops,
        "driver": driver,
        "openai_client": openai_client,
        "decision": decision,
        "trace": trace,
        "relevance_threshold": relevance_threshold,
        "max_retries": max_retries,
        "base_query": query,
        "current_query": query,
        "current_tool": decision.suggested_tool,
        "attempt": 0,
        "tried_tools": [],
        "reflection_history": [],
        "channel_cache": {},
        "forced_hybrid_providers": None,
        "results": [],
        "best_results": [],
        "best_score": 0.0,
        "best_attempt": 0,
        "best_rank": (-1, -1, -1, -10**9),
        "retries_used": max_retries,
        "reused_sources": [],
        "executed_sources": [],
        "provider_results_sink": None,
        "last_reflection": None,
        "last_elapsed_ms": 0,
        "tool_step_logged_for_attempt": False,
        "total_reranks": 0,
        "max_reranks": max(0, int(max_reranks)),
        "rewrite_attempted": False,
        "max_query_rewrites": max(0, int(max_query_rewrites)),
        "query_history": [query],
        "started_at_monotonic": time.perf_counter(),
        "time_budget_ms": max(0, int(request_time_budget_ms)),
        "memory": list(memory_seed or []),
        "stop_requested": False,
        "final_results": [],
    }
    final_state = workflow.invoke(initial_state)
    final_memory = list(final_state.get("memory", []))
    final_reflection_history = list(final_state.get("reflection_history", []))
    if trace is not None:
        trace.workflow_memory = final_memory
        trace.reflection_steps = final_reflection_history
    if memory_sink is not None:
        memory_sink.clear()
        memory_sink.extend(final_memory)
    if reflection_history_sink is not None:
        reflection_history_sink.clear()
        reflection_history_sink.extend(final_reflection_history)

    reflection = final_state.get("last_reflection")
    if reflection is not None and resolve_reflection_verdict(
        reflection,
    ) in {"answer", "stop"}:
        return final_state.get("results", []), final_state.get("attempt", 0)

    final_results = final_state.get("final_results") or final_state.get("best_results")
    if final_results:
        return final_results, final_state.get("retries_used", max_retries)
    return final_state.get("results", []), final_state.get("retries_used", max_retries)


@dataclass(frozen=True)
class AgentWorkflowOps:
    """Callbacks for the top-level route/retrieve/generate/completeness workflow."""

    classify_query: Callable[..., RouterDecision]
    is_cross_language_global: Callable[[str], bool]
    run_self_correction: Callable[..., tuple[list[SearchResult], int]]
    generate_answer: Callable[..., QAResult]
    evaluate_completeness: Callable[..., bool]
    comprehensive_search: Callable[..., list[SearchResult]]
    # CoVe-inspired claim verification (optional; skip when None).
    extract_claims: Callable[..., Any] | None = None
    verify_claims: Callable[..., Any] | None = None
    plan_correction: Callable[..., CorrectionPlan] | None = None
    run_correction_tool: Callable[..., list[SearchResult]] | None = None


class AgentWorkflowState(TypedDict, total=False):
    """State carried across the top-level agent workflow."""

    ops: AgentWorkflowOps
    query: str
    driver: Any
    openai_client: Any
    use_llm_router: bool
    trace: PipelineTrace
    settings: Any
    decision: RouterDecision
    results: list[SearchResult]
    retries: int
    qa_result: QAResult
    router_method: str
    router_duration_ms: int
    completeness_done: bool
    completeness_attempt: int
    completeness_complete: bool
    existing_ids: list[str]
    total_retries: int
    reflection_history: list[ReflectionStep]
    memory: list[WorkflowMemoryEntry]
    answer_guard_triggered: bool
    answer_guard_reason: str
    verification_retry_attempt: int
    correction_gaps: list[CorrectionGap]
    correction_plan: CorrectionPlan
    correction_added_results: int
    budget: BudgetTracker


def _route_query(state: AgentWorkflowState) -> dict[str, Any]:
    """Classify the query and record router metadata."""
    started = time.perf_counter()
    decision = state["ops"].classify_query(
        state["query"],
        use_llm=state["use_llm_router"],
        openai_client=state["openai_client"],
    )
    router_duration_ms = int((time.perf_counter() - started) * 1000)
    if state["ops"].is_cross_language_global(state["query"]) and decision.suggested_tool != "full_document_read":
        decision = RouterDecision(
            query_type=QueryType.GLOBAL,
            suggested_tool="full_document_read",
            confidence=decision.confidence,
            reasoning=decision.reasoning,
        )
    router_method = (
        "hard_rule"
        if decision.reasoning.startswith("Hard rule:")
        else ("llm" if state["use_llm_router"] else "pattern")
    )
    state["trace"].router_step = RouterStep(
        method=router_method,
        decision=decision,
        duration_ms=router_duration_ms,
    )
    memory = _append_memory_entry(
        state,
        stage="route",
        message=f"router selected {decision.suggested_tool}",
        metadata={
            "query_type": str(decision.query_type),
            "method": router_method,
            "confidence": decision.confidence,
        },
    )
    return {
        "decision": decision,
        "router_method": router_method,
        "router_duration_ms": router_duration_ms,
        "memory": memory,
    }


def _retrieve_evidence(state: AgentWorkflowState) -> dict[str, Any]:
    """Run the nested self-correction retrieval workflow."""
    initial_tools = _initial_tool_plan(state["query"], state["decision"])
    outputs: list[
        InitialRetrievalOutput
    ] = []

    def run_one(tool: str):
        tool_memory = list(state.get("memory", []))
        tool_reflections = list(state.get("reflection_history", []))
        tool_trace = PipelineTrace(
            trace_id=state["trace"].trace_id,
            timestamp=state["trace"].timestamp,
            query=state["query"],
            session_id=state["trace"].session_id,
            router_step=state["trace"].router_step,
        )
        tool_results, tool_retries = state["ops"].run_self_correction(
            query=state["query"],
            driver=state["driver"],
            openai_client=state["openai_client"],
            decision=_decision_for_initial_tool(state["decision"], tool),
            trace=tool_trace,
            memory_sink=tool_memory,
            reflection_history_sink=tool_reflections,
        )
        return tool_results, tool_retries, tool_memory, tool_reflections, tool_trace

    if len(initial_tools) == 1:
        outputs.append(run_one(initial_tools[0]))
    else:
        with ThreadPoolExecutor(max_workers=len(initial_tools)) as executor:
            futures = {
                executor.submit(run_one, tool): index
                for index, tool in enumerate(initial_tools)
            }
            completed: list[
                tuple[
                    int,
                    InitialRetrievalOutput,
                ]
            ] = []
            for future in as_completed(futures):
                completed.append((futures[future], future.result()))
            outputs.extend(result for _, result in sorted(completed, key=lambda item: item[0]))

    results, retries, memory, reflection_history = _merge_initial_retrieval_outputs(
        outputs,
        state["trace"],
    )
    existing_ids = [result.chunk.id for result in results if result.chunk.id]
    state["trace"].workflow_memory = list(memory)
    state["trace"].reflection_steps = list(reflection_history)
    answer_guard_triggered = False
    answer_guard_reason = ""
    for step in reversed(reflection_history):
        if (step.recommended_action or "") == "stop_due_to_invalid_reflection":
            answer_guard_triggered = True
            answer_guard_reason = step.reasoning or "Reflection policy guard stopped retry."
            break
    if not answer_guard_triggered:
        for entry in reversed(memory):
            if entry.stage != "retry":
                continue
            if (
                "requested already-covered gap" in entry.message
                or "repeated" in entry.message
                or "time budget exhausted" in entry.message
                or "retry plan exhausted" in entry.message
            ):
                answer_guard_triggered = True
                answer_guard_reason = entry.message
                break
    return {
        "results": results,
        "retries": retries,
        "total_retries": retries,
        "existing_ids": existing_ids,
        "memory": memory,
        "reflection_history": reflection_history,
        "answer_guard_triggered": answer_guard_triggered,
        "answer_guard_reason": answer_guard_reason,
    }


def _generate_answer_node(state: AgentWorkflowState) -> dict[str, Any]:
    """Generate the answer from the current evidence set.

    Wires the latest reflection verdict into `generate_answer` so the result
    carries discrete status fields instead of user-facing numeric confidence.
    """
    started = time.perf_counter()
    reflection_history = list(state.get("reflection_history", []))
    last_reflection = reflection_history[-1] if reflection_history else None
    reflection_verdict = (
        resolve_reflection_verdict(last_reflection) if last_reflection is not None else ""
    )
    can_start, budget_memory = _start_llm_call_or_skip(state, "generate_answer")
    if not can_start:
        qa_result = QAResult(
            answer="I don't have enough remaining LLM budget to generate an answer.",
            sources=state.get("results", []),
            answer_status="partial",
            retrieval_status="partial" if state.get("results") else "empty",
            verification_status="skipped",
            query=state["query"],
        )
        return {"qa_result": qa_result, "memory": budget_memory}
    state["memory"] = budget_memory
    qa_result = state["ops"].generate_answer(
        state["query"],
        state["results"],
        openai_client=state["openai_client"],
        reflection_verdict=reflection_verdict,
    )
    if state.get("answer_guard_triggered"):
        guard_reason = state.get("answer_guard_reason", "")
        update_payload: dict[str, Any] = {
            "answer_status": _answer_guard_status(
                guard_reason,
                has_answer=bool((qa_result.answer or "").strip()),
                retrieval_status=qa_result.retrieval_status,
            ),
        }
        qa_result = qa_result.model_copy(
            update=update_payload
        )
    settings = state.get("settings")
    model_name = ""
    if settings is not None:
        model_name = str(getattr(settings.openai, "llm_model", ""))
    state["trace"].generator_step = GeneratorStep(
        model=model_name,
        prompt_tokens=qa_result.prompt_tokens,
        completion_tokens=qa_result.completion_tokens,
        answer_status=qa_result.answer_status,
        retrieval_status=qa_result.retrieval_status,
        verification_status=qa_result.verification_status,
        duration_ms=int((time.perf_counter() - started) * 1000),
    )
    memory = _append_memory_entry(
        state,
        stage="generate",
        message="generated answer from current evidence",
        metadata={
            "sources": len(state["results"]),
            "answer_status": qa_result.answer_status,
            "retrieval_status": qa_result.retrieval_status,
            "verification_status": qa_result.verification_status,
            "answer_guard_triggered": state.get("answer_guard_triggered", False),
        },
    )
    return {"qa_result": qa_result, "memory": memory}


_MIN_COMPOSITE_ANSWER_CHARS = 80
_FACTUAL_SEPARATORS = ("，", "、", "；", ";", ",", " and ", " or ")
_NUMERIC_FACT_PATTERN = re.compile(
    r"(?:[<>]=?|≤|≥|=)\s*\d+(?:\.\d+)?"
    r"|\d+(?:\.\d+)?\s*(?:%|mg|ml|mmol|μg|ug|pg|g/l|iu|次|个月|天|年|分钟|小时)"
    r"|\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?",
    re.IGNORECASE,
)


def _answer_has_verifiable_claims(answer: str) -> bool:
    """Decide whether an answer is rich enough to warrant CoVe verification.

    CoVe is most valuable when the answer contains multiple composite facts
    (numeric thresholds, drug-dose pairs, entity relations). Short one-word
    or very terse answers rarely benefit from claim verification.
    """
    if not answer:
        return False
    text = answer.strip()
    has_numeric_fact = bool(_NUMERIC_FACT_PATTERN.search(text))
    if has_numeric_fact:
        return True
    if len(text) < _MIN_COMPOSITE_ANSWER_CHARS:
        return False
    separator_count = sum(text.count(sep) for sep in _FACTUAL_SEPARATORS)
    has_multiple_clauses = separator_count >= 2
    return has_multiple_clauses


def _after_generate(state: AgentWorkflowState) -> str:
    """Decide whether to run CoVe-style claim verification before finalizing.

    Triggering is decoupled from router classification (which is often wrong
    on domain-specific queries). We verify whenever the answer contains
    multiple facts or numeric thresholds — the conditions where claim-level
    support-check actually adds value.
    """
    qa_result = state.get("qa_result")
    if qa_result is None or not (qa_result.answer or "").strip():
        return "finish"
    # Skip verification when the generator returned its hard error fallback.
    if qa_result.answer.startswith("Error generating answer:"):
        return "finish"
    ops = state.get("ops")
    if ops is None or ops.extract_claims is None or ops.verify_claims is None:
        return _legacy_after_generate_branch(state)

    # Primary trigger: router thinks this needs cross-fact consistency.
    query_type = state["decision"].query_type
    if query_type in {QueryType.RELATION, QueryType.MULTI_HOP, QueryType.GLOBAL}:
        return "verify_answer"

    # Secondary trigger: the answer itself contains multiple verifiable facts,
    # regardless of how the router classified the query. This covers cases
    # where the router misclassified a compound / relation query as simple.
    if _answer_has_verifiable_claims(qa_result.answer):
        return "verify_answer"

    # Tertiary trigger: the agent actually invoked a graph/hybrid retrieval
    # tool, meaning the system itself judged cross-fact evidence relevant.
    trace = state.get("trace")
    graph_tools_used = {
        "cypher_traverse",
        "hybrid_search",
        "comprehensive_search",
        "full_document_read",
    }
    if trace is not None and any(
        step.tool_name in graph_tools_used for step in trace.tool_steps
    ):
        return "verify_answer"

    return _legacy_after_generate_branch(state)


def _legacy_after_generate_branch(state: AgentWorkflowState) -> str:
    """Legacy branching used when verification is not applicable.

    Triggers completeness check only for global queries with partial evidence.
    """
    qa_result = state.get("qa_result")
    if (
        state["decision"].query_type == QueryType.GLOBAL
        and qa_result is not None
        and bool((qa_result.answer or "").strip())
        and qa_result.retrieval_status in {"partial", "empty", "timeout"}
        and len(state.get("results", [])) < 8
    ):
        return "check_completeness"
    return "finish"


def _verify_answer_node(state: AgentWorkflowState) -> dict[str, Any]:
    """Run Chain-of-Verification claim checking against the knowledge graph.

    This node extracts atomic factual claims from the generated answer (one
    LLM call) and verifies each claim via `cypher_traverse` (no LLM calls).
    Possible-correct claims trigger a cautionary partial status. Incorrect
    claims trigger retry_required.
    """
    started = time.perf_counter()
    ops = state["ops"]
    qa_result = state["qa_result"]

    # Defensive: verification is only wired when both callbacks are provided.
    if ops.extract_claims is None or ops.verify_claims is None:
        return {}

    can_start, budget_memory = _start_llm_call_or_skip(state, "claim_extraction")
    if not can_start:
        verification = ClaimVerificationStep(
            claims_total=0,
            claims_supported=0,
            skipped_reason="llm_budget_exhausted",
            duration_ms=int((time.perf_counter() - started) * 1000),
            status="skipped",
        )
        state["trace"].verification_step = verification
        updated_qa = qa_result.model_copy(update={"verification_status": verification.status})
        if state["trace"].generator_step is not None:
            state["trace"].generator_step.verification_status = verification.status
        return {"qa_result": updated_qa, "memory": budget_memory}
    state["memory"] = budget_memory

    extraction = ops.extract_claims(
        qa_result.answer,
        query=state["query"],
        openai_client=state["openai_client"],
    )
    claims = getattr(extraction, "claims", [])

    if not claims:
        verification = ClaimVerificationStep(
            claims_total=0,
            claims_supported=0,
            skipped_reason="no_claims_extracted",
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        state["trace"].verification_step = verification
        memory = _append_memory_entry(
            state,
            stage="verify",
            message="claim extraction returned zero claims; verification skipped",
            metadata={"claims_total": 0},
        )
        return {"memory": memory}

    verification = ops.verify_claims(
        claims,
        driver=state["driver"],
        openai_client=state["openai_client"],
        existing_evidence=state.get("results", []),
    )
    verification.duration_ms = int((time.perf_counter() - started) * 1000)
    state["trace"].verification_step = verification

    updated_qa = qa_result
    if verification.skipped_reason:
        updated_qa = qa_result.model_copy(update={"verification_status": verification.status})
        if state["trace"].generator_step is not None:
            state["trace"].generator_step.verification_status = verification.status
        memory = _append_memory_entry(
            state,
            stage="verify",
            message=f"claim verification skipped: {verification.skipped_reason}",
            metadata={
                "claims_total": verification.claims_total,
                "claims_supported": verification.claims_supported,
                "claims_possible": verification.claims_possible,
                "claims_incorrect": verification.claims_incorrect,
                "skipped_reason": verification.skipped_reason,
            },
        )
        return {"qa_result": updated_qa, "memory": memory}

    if verification.unsupported_claims:
        answer_status = (
            "retry_required" if verification.status == "retry_required" else "partial"
        )
        updated_qa = qa_result.model_copy(update={
            "answer_status": answer_status,
            "verification_status": verification.status,
        })
    else:
        updated_qa = qa_result.model_copy(update={
            "answer_status": "verified",
            "verification_status": verification.status,
        })

    if state["trace"].generator_step is not None:
        state["trace"].generator_step.answer_status = updated_qa.answer_status
        state["trace"].generator_step.verification_status = updated_qa.verification_status

    memory = _append_memory_entry(
        state,
        stage="verify",
        message=(
            f"claim verification: correct={verification.claims_supported}, "
            f"possible={verification.claims_possible}, "
            f"incorrect={verification.claims_incorrect} ({verification.status})"
        ),
        metadata={
            "claims_total": verification.claims_total,
            "claims_supported": verification.claims_supported,
            "claims_possible": verification.claims_possible,
            "claims_incorrect": verification.claims_incorrect,
            "verification_status": verification.status,
            "answer_status_after": updated_qa.answer_status,
        },
    )
    update: dict[str, Any] = {"qa_result": updated_qa, "memory": memory}
    if verification.unsupported_claims:
        update["correction_gaps"] = build_gap_report(verification)
    return update


def _after_verify(state: AgentWorkflowState) -> str:
    """Route failed verification into the planner-only correction branch."""
    verification = state.get("trace").verification_step if state.get("trace") else None
    ops = state.get("ops")
    gaps = list(state.get("correction_gaps", []))
    retryable_partial = (
        verification is not None
        and verification.status == "partial"
        and any(
            gap.claim_role == "core"
            and gap.gap_type in {"missing_numeric_fact", "missing_entity"}
            for gap in gaps
        )
    )
    if (
        verification is not None
        and (verification.status == "retry_required" or retryable_partial)
        and any(gap.claim_role == "core" for gap in gaps)
        and state.get("verification_retry_attempt", 0) == 0
        and ops is not None
        and ops.plan_correction is not None
        and ops.run_correction_tool is not None
    ):
        return "plan_correction"
    return _legacy_after_generate_branch(state)


def _claim_focus_query(claim: Any) -> str:
    parts: list[str] = []
    for attr in ("text", "entities", "numeric_constraints", "relation_actions", "key_terms"):
        value = getattr(claim, attr, None)
        if isinstance(value, str):
            parts.append(value)
        elif value:
            parts.extend(str(item) for item in value if str(item).strip())
    seen: set[str] = set()
    unique_parts: list[str] = []
    for part in parts:
        normalized = " ".join(str(part).split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_parts.append(normalized)
    return " ".join(unique_parts)


def _plan_correction_node(state: AgentWorkflowState) -> dict[str, Any]:
    """Ask the planner which allowlisted retrieval tool should fill verification gaps."""
    ops = state["ops"]
    qa_result = state["qa_result"]
    verification = state["trace"].verification_step
    gaps = list(state.get("correction_gaps", []))
    if verification is None or not gaps or ops.plan_correction is None:
        return {}

    can_start, budget_memory = _start_llm_call_or_skip(state, "correction_planner")
    if not can_start:
        return {"memory": budget_memory}
    state["memory"] = budget_memory

    plan = ops.plan_correction(
        query=state["query"],
        answer=qa_result.answer,
        verification_status=verification.status,
        gaps=gaps,
        openai_client=state["openai_client"],
    )
    memory = _append_memory_entry(
        state,
        stage="verify_plan",
        message=f"correction planner selected {plan.action}",
        metadata={
            "action": plan.action,
            "tool": plan.tool or "",
            "focus_query": plan.focus_query,
            "gap_types": [gap.gap_type for gap in gaps],
            "reason": plan.reason,
        },
    )
    return {"correction_plan": plan, "memory": memory}


def _after_plan_correction(state: AgentWorkflowState) -> str:
    plan = state.get("correction_plan")
    if plan is None or plan.action != "retry_with_tool" or not plan.tool:
        return "finish"
    if state["ops"].run_correction_tool is None:
        return "finish"
    return "execute_correction_tool"


def _after_execute_correction_tool(state: AgentWorkflowState) -> str:
    if state.get("correction_added_results", 0) <= 0:
        return "finish"
    return "generate_answer"


def _execute_correction_tool_node(state: AgentWorkflowState) -> dict[str, Any]:
    """Execute the selected correction tool once and append unique evidence."""
    ops = state["ops"]
    plan = state.get("correction_plan")
    if plan is None or plan.action != "retry_with_tool" or not plan.tool:
        return {"verification_retry_attempt": 1}
    if ops.run_correction_tool is None:
        return {"verification_retry_attempt": 1}

    started = time.perf_counter()
    extra_results = ops.run_correction_tool(
        plan.tool,
        plan.focus_query or state["query"],
        state["driver"],
        state["openai_client"],
        state["decision"],
    )
    duration_ms = int((time.perf_counter() - started) * 1000)
    previous_count = len(state["results"])
    combined, existing_ids = _merge_unique_results(
        state["results"],
        extra_results,
        state.get("existing_ids", []),
    )
    added_results = len(combined) - previous_count
    trace = state.get("trace")
    if trace is not None:
        trace.tool_steps.append(
            ToolStep(
                tool_name=plan.tool,
                results_count=len(extra_results),
                duration_ms=duration_ms,
                query_used=plan.focus_query,
            )
        )
        trace.escalation_steps.append(
            EscalationStep(
                from_tool="verification",
                to_tool=plan.tool,
                reason=plan.reason,
                rephrased_query=plan.focus_query,
                duration_ms=duration_ms,
            )
        )
    memory = _append_memory_entry(
        state,
        stage="verify_retry",
        message=f"verification retry appended planner-selected {plan.tool} evidence",
        metadata={
            "tool": plan.tool,
            "focus_query": plan.focus_query,
            "added_results": added_results,
            "duration_ms": duration_ms,
        },
    )
    return {
        "results": combined,
        "existing_ids": existing_ids,
        "verification_retry_attempt": 1,
        "correction_added_results": added_results,
        "total_retries": state.get("total_retries", state.get("retries", 0)) + (1 if extra_results else 0),
        "memory": memory,
    }


def _check_completeness_node(state: AgentWorkflowState) -> dict[str, Any]:
    """Evaluate whether the current answer is complete for global queries.

    Forced incomplete when evidence is weak OR when the last reflection
    emitted a non-answer verdict. This no longer relies on any numeric
    reflection score.
    """
    qa_result = state["qa_result"]
    reflection_history = list(state.get("reflection_history", []))
    last_reflection = reflection_history[-1] if reflection_history else None
    forced_incomplete = False
    if qa_result.retrieval_status in {"empty", "timeout"}:
        forced_incomplete = True
    if last_reflection is not None:
        verdict = (last_reflection.verdict or "").strip().lower()
        evidence_status = (last_reflection.evidence_status or "").strip().lower()
        if (
            verdict in {"retry", "rerank"}
            or last_reflection.should_retry
            or evidence_status in {"partial", "insufficient", "empty"}
        ):
            forced_incomplete = True
    is_complete = False
    if not forced_incomplete:
        is_complete = state["ops"].evaluate_completeness(
            state["query"],
            qa_result.answer,
            openai_client=state["openai_client"],
        )
    if state["trace"].generator_step is not None:
        state["trace"].generator_step.completeness_check = is_complete
    memory = _append_memory_entry(
        state,
        stage="completeness",
        message="answer completeness evaluated",
        metadata={
            "complete": is_complete,
            "attempt": state.get("completeness_attempt", 0),
        },
    )
    return {
        "completeness_done": True,
        "completeness_complete": is_complete,
        "memory": memory,
    }


def _after_completeness(state: AgentWorkflowState) -> str:
    """Choose the next completeness branch based on current progress."""
    if state.get("completeness_complete", True):
        return "finish"
    if state.get("completeness_attempt", 0) == 0:
        return "augment_with_comprehensive"
    return "finish"


def _merge_unique_results(
    base_results: list[SearchResult],
    extra_results: list[SearchResult],
    existing_ids: list[str],
) -> tuple[list[SearchResult], list[str]]:
    seen_ids = set(existing_ids)
    combined = list(base_results)
    for result in extra_results:
        chunk_id = result.chunk.id
        if chunk_id and chunk_id in seen_ids:
            continue
        combined.append(result)
        if chunk_id:
            seen_ids.add(chunk_id)
    return combined, [chunk_id for chunk_id in seen_ids]


def _augment_with_comprehensive_node(state: AgentWorkflowState) -> dict[str, Any]:
    """Retry a global answer with comprehensive retrieval."""
    extra_results = state["ops"].comprehensive_search(
        state["query"],
        state["driver"],
        state["openai_client"],
    )
    combined, existing_ids = _merge_unique_results(
        state["results"],
        extra_results,
        state.get("existing_ids", []),
    )
    memory = _append_memory_entry(
        state,
        stage="augment",
        message="comprehensive search appended evidence",
        metadata={"added_results": len(extra_results)},
    )
    return {
        "results": combined,
        "existing_ids": existing_ids,
        "completeness_attempt": 1,
        "total_retries": state.get("total_retries", state.get("retries", 0)) + (1 if extra_results else 0),
        "memory": memory,
    }


@lru_cache(maxsize=1)
def _compile_agent_workflow_graph():
    """Compile the top-level agent workflow once and reuse it across requests."""
    graph = StateGraph(AgentWorkflowState)
    graph.add_node("route_query", _route_query)
    graph.add_node("retrieve_evidence", _retrieve_evidence)
    graph.add_node("generate_answer", _generate_answer_node)
    graph.add_node("verify_answer", _verify_answer_node)
    graph.add_node("check_completeness", _check_completeness_node)
    graph.add_node("plan_correction", _plan_correction_node)
    graph.add_node("execute_correction_tool", _execute_correction_tool_node)
    graph.add_node("augment_with_comprehensive", _augment_with_comprehensive_node)

    graph.add_edge(START, "route_query")
    graph.add_edge("route_query", "retrieve_evidence")
    graph.add_edge("retrieve_evidence", "generate_answer")
    graph.add_conditional_edges(
        "generate_answer",
        _after_generate,
        {
            "verify_answer": "verify_answer",
            "check_completeness": "check_completeness",
            "finish": END,
        },
    )
    graph.add_conditional_edges(
        "verify_answer",
        _after_verify,
        {
            "plan_correction": "plan_correction",
            "check_completeness": "check_completeness",
            "finish": END,
        },
    )
    graph.add_conditional_edges(
        "plan_correction",
        _after_plan_correction,
        {
            "execute_correction_tool": "execute_correction_tool",
            "finish": END,
        },
    )
    graph.add_conditional_edges(
        "execute_correction_tool",
        _after_execute_correction_tool,
        {
            "generate_answer": "generate_answer",
            "finish": END,
        },
    )
    graph.add_conditional_edges(
        "check_completeness",
        _after_completeness,
        {
            "augment_with_comprehensive": "augment_with_comprehensive",
            "finish": END,
        },
    )
    graph.add_edge("augment_with_comprehensive", "generate_answer")
    return graph.compile()


def run_agent_workflow(
    *,
    query: str,
    driver: Any,
    openai_client: Any,
    use_llm_router: bool,
    trace: PipelineTrace,
    settings: Any | None = None,
    ops: AgentWorkflowOps,
    workflow_memory_seed: list[WorkflowMemoryEntry] | None = None,
    reflection_history_seed: list[ReflectionStep] | None = None,
) -> QAResult:
    """Execute the top-level route/retrieve/generate/completeness workflow."""
    workflow = _compile_agent_workflow_graph()
    initial_state: AgentWorkflowState = {
        "ops": ops,
        "query": query,
        "driver": driver,
        "openai_client": openai_client,
        "use_llm_router": use_llm_router,
        "trace": trace,
        "settings": settings,
        "results": [],
        "retries": 0,
        "completeness_done": False,
        "completeness_attempt": 0,
        "completeness_complete": True,
        "existing_ids": [],
        "total_retries": 0,
        "verification_retry_attempt": 0,
        "correction_gaps": [],
        "correction_added_results": 0,
        "budget": BudgetTracker(max_llm_calls=4),
        "reflection_history": list(reflection_history_seed or []),
        "memory": list(workflow_memory_seed or []),
    }
    final_state = workflow.invoke(initial_state)
    qa_result = final_state["qa_result"]
    qa_result.sources = final_state.get("results", qa_result.sources)
    qa_result.retries = final_state.get("total_retries", final_state.get("retries", 0))
    qa_result.router_decision = final_state["decision"]
    trace.workflow_memory = list(final_state.get("memory", []))
    trace.reflection_steps = list(final_state.get("reflection_history", trace.reflection_steps))
    qa_result.trace = trace
    return qa_result
