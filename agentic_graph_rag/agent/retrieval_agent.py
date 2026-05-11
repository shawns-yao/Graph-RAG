"""Agentic Retrieval Agent — query routing + self-correction loop.

Main entry point for the Agentic Graph RAG system.
Routes queries to appropriate tools, evaluates results,
and retries with different strategies when quality is low.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from rag_core.config import get_settings
from rag_core.generator import generate_answer
from rag_core.models import (
    PipelineTrace,
    ProviderDiagnostic,
    QAResult,
    QueryType,
    ReflectionStep,
    RouterDecision,
    SearchResult,
    WorkflowMemoryEntry,
)
from rag_core.reflector import (
    evaluate_completeness,
    evaluate_reflection,
    generate_retry_query,
    resolve_reflection_verdict,
)
from rag_core.reranker import rerank

from agentic_graph_rag.agent.correction_planner import plan_correction
from agentic_graph_rag.agent.langgraph_workflow import (
    AgentWorkflowOps,
    SelfCorrectionOps,
    run_agent_workflow,
    run_self_correction_workflow,
)
from agentic_graph_rag.agent.need_resolver import resolve_retrieval_needs
from agentic_graph_rag.agent.router import classify_query
from agentic_graph_rag.agent.tool_registry import TOOL_NAMES
from agentic_graph_rag.agent.tools import (
    bm25_search,
    community_search,
    comprehensive_search,
    cypher_traverse,
    full_document_read,
    hybrid_search,
    temporal_query,
    vector_search,
)

if TYPE_CHECKING:
    from neo4j import Driver
    from openai import OpenAI

logger = logging.getLogger(__name__)

# Tool registry: query_type → tool function
_TOOL_REGISTRY = {
    "vector_search": vector_search,
    "bm25_search": bm25_search,
    "cypher_traverse": cypher_traverse,
    "community_search": community_search,
    "hybrid_search": hybrid_search,
    "temporal_query": temporal_query,
    "comprehensive_search": comprehensive_search,
    "full_document_read": full_document_read,
}

_GRAPH_FIRST_TOOLS = ["cypher_traverse", "hybrid_search"]
_HYBRID_RECALL_TOOLS = ["hybrid_search", "comprehensive_search"]
_HYBRID_MISSING_ENTITY_TOOLS = ["bm25_search", "vector_search", "cypher_traverse", "hybrid_search"]


@dataclass(frozen=True, slots=True)
class RetryPlan:
    tools: list[str]
    reason: str

_TOOL_CHANNEL_MAP = {
    "vector_search": "vector",
    "bm25_search": "bm25",
    "cypher_traverse": "graph",
}
_HYBRID_CHANNEL_ORDER = ["vector", "bm25", "graph"]
_VALID_PROVIDER_NAMES = set(_HYBRID_CHANNEL_ORDER)
_VALID_TOOL_NAMES = set(TOOL_NAMES)


def _extend_unique(target: list[str], values: list[str], *, valid: set[str] | None = None) -> None:
    """Append values in order once, with optional allow-list validation."""
    for value in values:
        if valid is not None and value not in valid:
            continue
        if value not in target:
            target.append(value)


# ---------------------------------------------------------------------------
# Tool selection
# ---------------------------------------------------------------------------

def select_tool(decision: RouterDecision) -> str:
    """Select the retrieval tool name based on router decision."""
    tool_name = decision.suggested_tool
    if tool_name in _TOOL_REGISTRY:
        return tool_name
    logger.warning("Unknown tool '%s', defaulting to vector_search", tool_name)
    return "vector_search"


def _hybrid_cache_plan(
    query: str,
    channel_cache: dict[str, dict[str, list[SearchResult]]],
    enabled_providers: list[str] | None = None,
) -> tuple[dict[str, list[SearchResult]], list[str], list[str]]:
    """Resolve cached and missing channels for a hybrid retrieval pass."""
    cached_for_query = channel_cache.get(query, {})
    forced_enabled = None if enabled_providers is None else set(enabled_providers)
    seed_results = {
        source: cached_for_query[source]
        for source in _HYBRID_CHANNEL_ORDER
        if source in cached_for_query and (forced_enabled is None or source not in forced_enabled)
    }
    resolved_enabled_providers = [
        source for source in _HYBRID_CHANNEL_ORDER
        if (forced_enabled is None and source not in seed_results)
        or (forced_enabled is not None and source in forced_enabled)
    ]
    return seed_results, resolved_enabled_providers, list(seed_results)


def _execution_sources(
    tool_name: str,
    query: str,
    channel_cache: dict[str, dict[str, list[SearchResult]]],
    hybrid_enabled_providers: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Describe which retrieval channels are reused vs freshly executed."""
    if tool_name == "hybrid_search":
        seed_results, enabled_providers, reused_sources = _hybrid_cache_plan(
            query,
            channel_cache,
            hybrid_enabled_providers,
        )
        _ = seed_results
        return reused_sources, enabled_providers

    channel = _TOOL_CHANNEL_MAP.get(tool_name)
    if channel is None:
        return [], []
    return [], [channel]


def _cache_tool_results(
    query: str,
    tool_name: str,
    results: list[SearchResult],
    channel_cache: dict[str, dict[str, list[SearchResult]]],
) -> None:
    """Persist single-channel retrieval results for later hybrid reuse."""
    channel = _TOOL_CHANNEL_MAP.get(tool_name)
    if channel is None:
        return
    channel_cache.setdefault(query, {})[channel] = results


def _cache_hybrid_provider_results(
    query: str,
    provider_results: dict[str, list[SearchResult]],
    channel_cache: dict[str, dict[str, list[SearchResult]]],
) -> None:
    """Persist provider-level hybrid outputs for future incremental retries."""
    if not provider_results:
        return

    query_cache = channel_cache.setdefault(query, {})
    for source in _HYBRID_CHANNEL_ORDER:
        if source in provider_results:
            query_cache[source] = provider_results[source]


def _build_provider_diagnostics(
    provider_results: dict[str, list[SearchResult]] | None,
    reused_sources: list[str],
    executed_sources: list[str],
) -> list[ProviderDiagnostic]:
    """Summarize provider-level retrieval evidence for trace inspection."""
    if not provider_results and not reused_sources and not executed_sources:
        return []

    diagnostics: list[ProviderDiagnostic] = []
    source_order = list(dict.fromkeys([
        *reused_sources,
        *executed_sources,
        *((provider_results or {}).keys()),
    ]))
    for source in source_order:
        results = (provider_results or {}).get(source, [])
        scores = [result.score for result in results]
        top_chunk_ids = [result.chunk.id for result in results[:3] if result.chunk.id]
        diagnostics.append(ProviderDiagnostic(
            source=source,
            results_count=len(results),
            top_score=max(scores, default=0.0),
            average_score=(sum(scores) / len(scores)) if scores else 0.0,
            reused=source in reused_sources,
            executed=source in executed_sources,
            top_chunk_ids=top_chunk_ids,
        ))
    return diagnostics


def _diagnostic_gap_sources(
    provider_results: dict[str, list[SearchResult]] | None,
) -> list[str]:
    """Return providers that produced no evidence in the current pass."""
    if not provider_results:
        return []
    return [
        source for source in _HYBRID_CHANNEL_ORDER
        if source in provider_results and not provider_results[source]
    ]


def _should_rewrite_query(
    next_tool: str,
    reflection: ReflectionStep,
    cached_sources_reused: list[str],
) -> bool:
    """Rewrite only when it materially helps broader fallback retrieval."""
    verdict = resolve_reflection_verdict(reflection)
    if verdict != "retry":
        return False
    if reflection.should_rewrite_query:
        return True
    if next_tool == "hybrid_search" and cached_sources_reused:
        return False
    if next_tool in {"comprehensive_search", "full_document_read"}:
        return True
    return (reflection.recommended_action or "").strip().lower() in {
        "target_missing_entity",
        "use_comprehensive_search",
    }


def _tool_kwargs(
    tool_name: str,
    decision: RouterDecision,
    *,
    query: str = "",
    channel_cache: dict[str, dict[str, list[SearchResult]]] | None = None,
    hybrid_enabled_providers: list[str] | None = None,
    provider_results_sink: dict[str, list[SearchResult]] | None = None,
) -> dict:
    """Build optional tool kwargs from router context."""
    if tool_name == "hybrid_search":
        kwargs: dict = {"query_type": decision.query_type}
        if channel_cache is not None:
            seed_results, enabled_providers, _ = _hybrid_cache_plan(
                query,
                channel_cache,
                hybrid_enabled_providers,
            )
            if seed_results:
                kwargs["seed_results"] = seed_results
            kwargs["enabled_providers"] = enabled_providers
        if provider_results_sink is not None:
            kwargs["provider_results"] = provider_results_sink
        return kwargs
    return {}


def _run_tool(
    tool_name: str,
    query: str,
    driver: Driver,
    openai_client: OpenAI,
    decision: RouterDecision,
    *,
    channel_cache: dict[str, dict[str, list[SearchResult]]] | None = None,
    hybrid_enabled_providers: list[str] | None = None,
    provider_results_sink: dict[str, list[SearchResult]] | None = None,
) -> list[SearchResult]:
    """Execute a retrieval tool with router-aware optional kwargs."""
    tool_fn = _TOOL_REGISTRY[tool_name]
    return tool_fn(
        query,
        driver,
        openai_client,
        **_tool_kwargs(
            tool_name,
            decision,
            query=query,
            channel_cache=channel_cache,
            hybrid_enabled_providers=hybrid_enabled_providers,
            provider_results_sink=provider_results_sink,
        ),
    )


def _preferred_providers_for_reflection(reflection: ReflectionStep) -> list[str]:
    """Return provider names explicitly provided by reflection."""
    ordered: list[str] = []
    for provider in reflection.preferred_providers:
        provider_name = provider.strip().lower()
        _extend_unique(ordered, [provider_name], valid=_VALID_PROVIDER_NAMES)
    return ordered


def _plan_incremental_retry(
    query: str,
    current_tool: str,
    reflection: ReflectionStep,
    channel_cache: dict[str, dict[str, list[SearchResult]]],
    provider_results: dict[str, list[SearchResult]] | None = None,
) -> tuple[str, list[str], list[str]] | None:
    """Return an incremental hybrid retry plan when one route needs refresh."""
    verdict = resolve_reflection_verdict(reflection)
    if verdict != "retry" or reflection.retry_scope == "stop" or not reflection.should_retry:
        return None
    if current_tool in {"comprehensive_search", "full_document_read", "community_search"}:
        return None
    if (
        (reflection.action or "").strip().lower() == "retry_graph"
        or (reflection.required_tool or "").strip().lower() == "cypher_traverse"
    ):
        return None

    refresh_sources = _preferred_providers_for_reflection(reflection)
    if reflection.failure_type in {"insufficient_recall", "insufficient_context", "no_results"}:
        for source in _diagnostic_gap_sources(provider_results):
            if source not in refresh_sources:
                refresh_sources.append(source)
    if not refresh_sources:
        return None

    cached_for_query = channel_cache.get(query, {})
    reusable_sources = [
        source for source in _HYBRID_CHANNEL_ORDER
        if source in cached_for_query and source not in refresh_sources
    ]
    if not reusable_sources:
        return None

    for source in refresh_sources:
        cached_for_query.pop(source, None)
    return "hybrid_search", refresh_sources, reusable_sources


def _rerank_results(
    query: str,
    results: list[SearchResult],
    *,
    openai_client: OpenAI,
) -> list[SearchResult]:
    """Apply a local rerank pass without triggering a fresh retrieval fan-out."""
    del openai_client
    if not results:
        return []
    return rerank(query, results, top_k=len(results))


# ---------------------------------------------------------------------------
# Self-correction loop
# ---------------------------------------------------------------------------

def self_correction_loop(
    query: str,
    driver: Driver,
    openai_client: OpenAI,
    decision: RouterDecision,
    max_retries: int | None = None,
    relevance_threshold: float | None = None,
    trace: PipelineTrace | None = None,
    memory_sink: list[WorkflowMemoryEntry] | None = None,
    reflection_history_sink: list[ReflectionStep] | None = None,
) -> tuple[list[SearchResult], int]:
    """Execute retrieval with self-correction.

    Tries the suggested tool first, evaluates relevance,
    and escalates to more powerful tools if quality is low.

    Returns (results, retries_used).
    """
    cfg = get_settings()
    if max_retries is None:
        max_retries = cfg.agent.max_retries
    if relevance_threshold is None:
        relevance_threshold = cfg.agent.relevance_threshold
    resolved_tool = select_tool(decision)
    resolved_decision = decision.model_copy(update={"suggested_tool": resolved_tool})

    ops = SelfCorrectionOps(
        execution_sources=_execution_sources,
        run_tool=_run_tool,
        cache_tool_results=_cache_tool_results,
        cache_hybrid_provider_results=_cache_hybrid_provider_results,
        build_provider_diagnostics=_build_provider_diagnostics,
        evaluate_reflection=evaluate_reflection,
        plan_incremental_retry=_plan_incremental_retry,
        get_next_tool=_get_next_tool,
        should_rewrite_query=_should_rewrite_query,
        generate_retry_query=generate_retry_query,
        rerank_results=_rerank_results,
    )
    return run_self_correction_workflow(
        query=query,
        driver=driver,
        openai_client=openai_client,
        decision=resolved_decision,
        max_retries=max_retries,
        relevance_threshold=relevance_threshold,
        max_reranks=cfg.agent.max_reranks,
        max_query_rewrites=cfg.agent.max_query_rewrites,
        request_time_budget_ms=cfg.agent.request_time_budget_ms,
        trace=trace,
        ops=ops,
        memory_seed=memory_sink,
        memory_sink=memory_sink,
        reflection_history_sink=reflection_history_sink,
    )


def _preferred_tools_for_reflection(reflection: ReflectionStep) -> list[str]:
    """Return tool names explicitly provided by reflection."""
    ordered: list[str] = []
    for tool in reflection.preferred_tools:
        tool_name = tool.strip().lower()
        _extend_unique(ordered, [tool_name], valid=_VALID_TOOL_NAMES)
    return ordered


def _retry_plan_for_reflection(
    reflection: ReflectionStep | None,
    decision: RouterDecision | None,
) -> RetryPlan:
    if reflection is None:
        return RetryPlan([], "no reflection")

    gap_type = (reflection.gap_type or "").strip().lower()
    failure_type = (reflection.failure_type or "").strip().lower()
    required_tool = (reflection.required_tool or "").strip().lower()
    tools: list[str] = []

    if required_tool and required_tool != "none":
        _extend_unique(tools, [required_tool], valid=_VALID_TOOL_NAMES)
    if gap_type in {"missing_numeric_threshold", "missing_diagnostic_criterion"}:
        _extend_unique(tools, ["bm25_search", "vector_search"], valid=_VALID_TOOL_NAMES)
    elif gap_type in {"missing_relation", "missing_treatment_option"}:
        _extend_unique(tools, _GRAPH_FIRST_TOOLS, valid=_VALID_TOOL_NAMES)
    elif gap_type == "missing_entity":
        _extend_unique(tools, _HYBRID_MISSING_ENTITY_TOOLS, valid=_VALID_TOOL_NAMES)
    elif gap_type in {"missing_comparison_target", "conflicting_evidence"}:
        _extend_unique(tools, _HYBRID_RECALL_TOOLS, valid=_VALID_TOOL_NAMES)

    return RetryPlan(tools, f"gap={gap_type or 'none'} failure={failure_type or 'none'}")


def _get_next_tool(
    current: str,
    tried: set[str],
    reflection: ReflectionStep | None = None,
    decision: RouterDecision | None = None,
) -> str | None:
    """Get next retry tool from gap-based planning and explicit reflection hints."""
    candidate_tools: list[str] = []
    if reflection is not None:
        candidate_tools.extend(_retry_plan_for_reflection(reflection, decision).tools)
        candidate_tools.extend(_preferred_tools_for_reflection(reflection))

    for tool in candidate_tools:
        if tool not in tried:
            return tool
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    query: str,
    driver: Driver,
    openai_client: OpenAI | None = None,
    use_llm_router: bool = False,
    session_id: str = "",
    workflow_memory_seed: list[WorkflowMemoryEntry] | None = None,
    reflection_history_seed: list[ReflectionStep] | None = None,
    forced_tool: str = "",
) -> QAResult:
    """Run the agentic retrieval pipeline.

    1. Classify query → select tool
    2. Execute with self-correction loop
    3. Generate answer from results

    Returns QAResult with answer, sources, discrete status fields, and metadata.
    """
    cfg = get_settings()
    if openai_client is None:
        from rag_core.config import make_openai_client
        openai_client = make_openai_client(cfg)

    trace = PipelineTrace(
        trace_id=f"tr_{uuid.uuid4().hex[:12]}",
        timestamp=datetime.now(timezone.utc).isoformat(),
        query=query,
        session_id=session_id,
    )
    t_start = time.perf_counter()
    # CoVe-inspired claim verification: extract atomic claims from the answer,
    # then check each against cypher_traverse. Wired into AgentWorkflowOps as
    # optional callbacks; the workflow node skips itself when they are absent.
    from agentic_graph_rag.generation.claim_verifier import (
        extract_claims,
        verify_claims,
    )

    def _verify_claims_wrapper(claims, *, driver, openai_client, existing_evidence=None):
        return verify_claims(
            claims,
            cypher_traverse=cypher_traverse,
            driver=driver,
            openai_client=openai_client,
            existing_evidence=existing_evidence,
        )

    ops = AgentWorkflowOps(
        classify_query=classify_query,
        run_self_correction=self_correction_loop,
        generate_answer=generate_answer,
        evaluate_completeness=evaluate_completeness,
        comprehensive_search=comprehensive_search,
        resolve_retrieval_needs=resolve_retrieval_needs,
        extract_claims=extract_claims,
        verify_claims=_verify_claims_wrapper,
        plan_correction=plan_correction,
        run_correction_tool=_run_tool,
    )
    if forced_tool:
        routed_type = QueryType.SIMPLE
        if forced_tool == "cypher_traverse":
            routed_type = QueryType.RELATION
        elif forced_tool == "full_document_read":
            routed_type = QueryType.GLOBAL
        elif forced_tool == "temporal_query":
            routed_type = QueryType.TEMPORAL

        forced_decision = RouterDecision(
            query_type=routed_type,
            reasoning=f"Benchmark forced initial tool: {forced_tool}.",
            suggested_tool=forced_tool,
        )
        ops = AgentWorkflowOps(
            classify_query=lambda query, *, use_llm, openai_client: forced_decision,
            run_self_correction=self_correction_loop,
            generate_answer=generate_answer,
            evaluate_completeness=evaluate_completeness,
            comprehensive_search=comprehensive_search,
            resolve_retrieval_needs=resolve_retrieval_needs,
            extract_claims=extract_claims,
            verify_claims=_verify_claims_wrapper,
            plan_correction=plan_correction,
            run_correction_tool=_run_tool,
        )
    qa_result = run_agent_workflow(
        query=query,
        driver=driver,
        openai_client=openai_client,
        use_llm_router=use_llm_router,
        trace=trace,
        settings=cfg,
        ops=ops,
        workflow_memory_seed=workflow_memory_seed,
        reflection_history_seed=reflection_history_seed,
    )
    trace.total_duration_ms = int((time.perf_counter() - t_start) * 1000)

    logger.info(
        "Agent result: %d sources, %d retries, answer_status=%s, retrieval_status=%s",
        len(qa_result.sources),
        qa_result.retries,
        qa_result.answer_status,
        qa_result.retrieval_status,
    )
    return qa_result
