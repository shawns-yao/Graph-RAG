"""Structured trace explain helpers for API and MCP consumers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from rag_core.models import PipelineTrace, ProviderDiagnostic, ToolStep


def explain_trace(trace: PipelineTrace) -> dict[str, Any]:
    """Build a stable consumer-facing explain view for a pipeline trace."""
    tool_views = [_explain_tool_step(step) for step in trace.tool_steps]
    provider_summary = _summarize_providers(trace.tool_steps)
    latest_reflection = trace.reflection_steps[-1] if trace.reflection_steps else None

    return {
        "trace_id": trace.trace_id,
        "timestamp": trace.timestamp,
        "query": trace.query,
        "session_id": trace.session_id,
        "total_duration_ms": trace.total_duration_ms,
        "router": _explain_router(trace),
        "retrieval": {
            "steps": tool_views,
            "provider_summary": provider_summary,
            "likely_gaps": [
                item["source"]
                for item in provider_summary
                if item["empty_steps"] > 0 and item["total_results"] == 0
            ],
        },
        "reflection": {
            "attempts": len(trace.reflection_steps),
            "latest_failure_type": latest_reflection.failure_type if latest_reflection else "",
            "latest_recommended_action": (
                latest_reflection.recommended_action if latest_reflection else ""
            ),
            "latest_missing_information": (
                latest_reflection.missing_information if latest_reflection else []
            ),
        },
        "memory": [
            {
                "stage": entry.stage,
                "message": entry.message,
                "metadata": entry.metadata,
            }
            for entry in trace.workflow_memory
        ],
        "escalations": [
            {
                "from_tool": step.from_tool,
                "to_tool": step.to_tool,
                "reason": step.reason,
                "rephrased_query": step.rephrased_query,
                "duration_ms": step.duration_ms,
                "cached_sources_reused": step.cached_sources_reused,
            }
            for step in trace.escalation_steps
        ],
        "generation": _explain_generation(trace),
        "verification": _explain_verification(trace),
    }


def explain_trace_payload(trace: PipelineTrace) -> dict[str, Any]:
    """Return raw trace plus structured explain view for external consumers."""
    return {
        "trace": trace.model_dump(),
        "explain": explain_trace(trace),
    }


def _explain_router(trace: PipelineTrace) -> dict[str, Any] | None:
    """Serialize router decision into a stable consumer shape."""
    if trace.router_step is None:
        return None

    decision = trace.router_step.decision
    return {
        "method": trace.router_step.method,
        "duration_ms": trace.router_step.duration_ms,
        "query_type": decision.query_type,
        "confidence": decision.confidence,
        "suggested_tool": decision.suggested_tool,
        "reasoning": decision.reasoning,
        "rules_fired": trace.router_step.rules_fired,
    }


def _explain_tool_step(step: ToolStep) -> dict[str, Any]:
    """Serialize provider-aware retrieval step diagnostics."""
    providers = [_explain_provider(diagnostic) for diagnostic in step.provider_diagnostics]
    empty_sources = [
        diagnostic["source"]
        for diagnostic in providers
        if diagnostic["results_count"] == 0
    ]

    return {
        "tool_name": step.tool_name,
        "query_used": step.query_used,
        "results_count": step.results_count,
        "relevance_score": step.relevance_score,
        "duration_ms": step.duration_ms,
        "cache_hit": step.cache_hit,
        "reused_sources": step.reused_sources,
        "executed_sources": step.executed_sources,
        "empty_sources": empty_sources,
        "providers": providers,
    }


def _explain_provider(diagnostic: ProviderDiagnostic) -> dict[str, Any]:
    """Serialize a provider diagnostic into a consumer-friendly shape."""
    status = "idle"
    if diagnostic.reused:
        status = "reused"
    elif diagnostic.executed:
        status = "executed"

    return {
        "source": diagnostic.source,
        "status": status,
        "results_count": diagnostic.results_count,
        "top_score": diagnostic.top_score,
        "average_score": diagnostic.average_score,
        "top_chunk_ids": diagnostic.top_chunk_ids,
    }


def _summarize_providers(tool_steps: list[ToolStep]) -> list[dict[str, Any]]:
    """Aggregate provider diagnostics across retrieval steps."""
    summary: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "source": "",
        "executed_steps": 0,
        "reused_steps": 0,
        "empty_steps": 0,
        "total_results": 0,
        "max_top_score": 0.0,
        "average_top_score": 0.0,
        "top_chunk_ids": [],
    })

    for step in tool_steps:
        for diagnostic in step.provider_diagnostics:
            item = summary[diagnostic.source]
            item["source"] = diagnostic.source
            item["executed_steps"] += int(diagnostic.executed)
            item["reused_steps"] += int(diagnostic.reused)
            item["empty_steps"] += int(diagnostic.results_count == 0)
            item["total_results"] += diagnostic.results_count
            item["max_top_score"] = max(item["max_top_score"], diagnostic.top_score)

            if item["average_top_score"] == 0.0:
                item["average_top_score"] = diagnostic.average_score
            else:
                item["average_top_score"] = round(
                    (item["average_top_score"] + diagnostic.average_score) / 2,
                    4,
                )

            for chunk_id in diagnostic.top_chunk_ids:
                if chunk_id and chunk_id not in item["top_chunk_ids"]:
                    item["top_chunk_ids"].append(chunk_id)

    return sorted(summary.values(), key=lambda item: item["source"])


def _explain_generation(trace: PipelineTrace) -> dict[str, Any] | None:
    """Serialize generation metadata for explain consumers."""
    if trace.generator_step is None:
        return None

    return {
        "model": trace.generator_step.model,
        "prompt_tokens": trace.generator_step.prompt_tokens,
        "completion_tokens": trace.generator_step.completion_tokens,
        "evidence_score": trace.generator_step.evidence_score,
        "confidence_level": trace.generator_step.confidence_level,
        "completeness_check": trace.generator_step.completeness_check,
        "duration_ms": trace.generator_step.duration_ms,
    }


def _explain_verification(trace: PipelineTrace) -> dict[str, Any] | None:
    """Serialize Chain-of-Verification results for explain consumers."""
    step = trace.verification_step
    if step is None:
        return None

    return {
        "claims_total": step.claims_total,
        "claims_supported": step.claims_supported,
        "support_rate": round(step.support_rate, 3),
        "unsupported_claims": [
            {"text": vc.text, "key_terms": vc.key_terms}
            for vc in step.unsupported_claims
        ],
        "verified_claims": [
            {"text": vc.text, "top_chunk_id": vc.top_chunk_id}
            for vc in step.verified_claims
        ],
        "skipped_reason": step.skipped_reason,
        "duration_ms": step.duration_ms,
    }
