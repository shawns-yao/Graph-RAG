"""Structured self-reflection and retry-query generation."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from rag_core.config import get_settings, make_openai_client
from rag_core.models import ReflectionStep, SearchResult, WorkflowMemoryEntry

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)

_TOP_RESULTS = 5
_ENUMERATION_CUES = (
    "list all",
    "summarize all",
    "overview",
    "需要哪些",
    "有哪些",
    "分类列出",
    "汇总",
    "完整",
    "列出",
    "整体",
)
_INCOMPLETE_ANSWER_CUES = (
    "i don't have enough context",
    "not enough context",
    "available evidence covers part of the question",
    "无法回答",
    "信息不足",
    "证据不足",
    "不完整",
    "未提及",
)
_VALID_VERDICTS = {"answer", "rerank", "retry", "stop"}
_VALID_EVIDENCE_STATUSES = {"empty", "insufficient", "partial", "sufficient"}
_VALID_GAP_TYPES = {
    "conflicting_evidence",
    "missing_comparison_target",
    "missing_diagnostic_criterion",
    "missing_entity",
    "missing_numeric_threshold",
    "missing_relation",
    "missing_treatment_option",
    "none",
    "off_topic",
}
_VALID_REFLECTION_ACTIONS = {
    "answer",
    "answer_with_caveat",
    "rerank",
    "retry_bm25",
    "retry_graph",
    "retry_hybrid",
    "retry_vector",
    "stop",
}
_VALID_RETRY_SCOPES = {"stop", "provider_refresh", "tool_escalation", "rerank_only"}
_VALID_FAILURE_TYPES = {
    "acceptable",
    "inconsistent_evidence",
    "insufficient_context",
    "insufficient_recall",
    "missing_entity",
    "no_results",
    "relation_missing",
}
_VALID_PROVIDER_NAMES = {"vector", "bm25", "graph"}
_VALID_TOOL_NAMES = {
    "bm25_search",
    "comprehensive_search",
    "cypher_traverse",
    "full_document_read",
    "hybrid_search",
    "vector_search",
}

_REFLECTION_SCORE_WEIGHTS = {
    "relevance": 0.35,
    "entity_completeness": 0.30,
    "logical_consistency": 0.15,
    "context_sufficiency": 0.20,
}
_REFLECTION_SCHEMA_FIELDS = {
    "evidence_status",
    "gap_type",
    "action",
    "required_tool",
    "missing_information",
    "missing_entities",
    "missing_relationships",
    "coverage_gap_sources",
    "candidate_fix_paths",
    "preferred_tools",
    "preferred_providers",
    "reasoning",
    "failure_type",
    "comparison_to_previous",
}
_REFLECTION_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": sorted(_REFLECTION_SCHEMA_FIELDS),
    "properties": {
        "evidence_status": {"type": "string", "enum": sorted(_VALID_EVIDENCE_STATUSES)},
        "gap_type": {"type": "string", "enum": sorted(_VALID_GAP_TYPES)},
        "action": {"type": "string", "enum": sorted(_VALID_REFLECTION_ACTIONS)},
        "required_tool": {"type": "string", "enum": ["none", *sorted(_VALID_TOOL_NAMES)]},
        "missing_information": {"type": "array", "items": {"type": "string"}},
        "missing_entities": {"type": "array", "items": {"type": "string"}},
        "missing_relationships": {"type": "array", "items": {"type": "string"}},
        "coverage_gap_sources": {
            "type": "array",
            "items": {"type": "string", "enum": sorted(_VALID_PROVIDER_NAMES)},
        },
        "candidate_fix_paths": {"type": "array", "items": {"type": "string"}},
        "preferred_tools": {
            "type": "array",
            "items": {"type": "string", "enum": sorted(_VALID_TOOL_NAMES)},
        },
        "preferred_providers": {
            "type": "array",
            "items": {"type": "string", "enum": sorted(_VALID_PROVIDER_NAMES)},
        },
        "reasoning": {"type": "string"},
        "failure_type": {"type": "string", "enum": sorted(_VALID_FAILURE_TYPES)},
        "comparison_to_previous": {"type": "string"},
    },
}

REFLECTION_PROMPT = """You are a strict retrieval policy classifier for a Graph RAG system.
Do not assign numeric scores. Choose only from the fixed enums below.

Evidence status:
- sufficient: evidence directly answers the query.
- partial: evidence answers part of the query but has a specific gap.
- insufficient: evidence is relevant but not enough for a reliable answer.
- empty: no useful evidence was retrieved.

Gap type:
- none
- missing_entity
- missing_relation
- missing_numeric_threshold
- missing_comparison_target
- missing_treatment_option
- missing_diagnostic_criterion
- conflicting_evidence
- off_topic

Action:
- answer
- answer_with_caveat
- rerank
- retry_vector
- retry_bm25
- retry_graph
- retry_hybrid
- stop

Return ONLY valid JSON with this exact schema:
{
  "evidence_status": "one of: sufficient, partial, insufficient, empty",
  "gap_type": "one of the fixed gap types",
  "action": "one of the fixed actions",
  "required_tool": "none|vector_search|bm25_search|cypher_traverse|hybrid_search|comprehensive_search|full_document_read",
  "missing_information": ["..."],
  "missing_entities": ["..."],
  "missing_relationships": ["..."],
  "coverage_gap_sources": ["vector|bm25|graph"],
  "candidate_fix_paths": ["short action sequence"],
  "preferred_tools": [
    "vector_search|bm25_search|cypher_traverse|hybrid_search|comprehensive_search|full_document_read"
  ],
  "preferred_providers": ["vector|bm25|graph"],
  "reasoning": "short explanation",
  "failure_type": "one of: no_results, insufficient_recall, missing_entity,
                   inconsistent_evidence, insufficient_context, acceptable",
  "comparison_to_previous": "short note"
}

Example 1:
Query: Which projects were jointly handled by Alice and Bob?
Missing information: Bob's project membership is absent.
Choose: evidence_status=partial, gap_type=missing_entity, action=retry_bm25

Example 2:
Query: What is the relationship between service A and service B?
Missing information: direct edge or shared passage connecting A and B.
Choose: evidence_status=partial, gap_type=missing_relation, action=retry_graph
"""

RETRY_QUERY_PROMPT = """You are rewriting a retrieval query for a Graph RAG system.
Use the reflection result and prior failed attempts to produce a better, more targeted query.

Rules:
- Return ONLY the rewritten query on one line.
- If reflection says a specific entity or relation is missing, explicitly include it.
- Avoid repeating a failed query wording from history.

Example:
Original query: What projects were jointly handled by Alice and Bob?
Missing information: Bob's project membership is absent.
Output: Bob projects and overlap with Alice projects
"""


def _clamp_score(value: object, default: float = 2.5) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(5.0, score))


def _coerce_str_list(value: object) -> list[str]:
    if isinstance(value, list):
        items = value
    elif value in (None, ""):
        items = []
    else:
        items = [value]
    return [str(item).strip() for item in items if str(item).strip()]


def _coerce_bool(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return default


def _normalize_verdict(value: object) -> str:
    verdict = str(value or "").strip().lower()
    if verdict in _VALID_VERDICTS:
        return verdict
    return ""


def resolve_reflection_verdict(
    reflection: ReflectionStep,
    *,
    relevance_threshold: float | None = None,
) -> str:
    """Resolve one stable control verdict from a structured reflection step."""
    verdict = _normalize_verdict(getattr(reflection, "verdict", ""))
    if verdict:
        return verdict

    action = (reflection.action or "").strip().lower()
    evidence_status = (reflection.evidence_status or "").strip().lower()
    if action:
        return _verdict_for_action(action, evidence_status)

    failure_type = (reflection.failure_type or "").strip().lower()
    recommended_action = (reflection.recommended_action or "").strip().lower()
    retry_scope = (reflection.retry_scope or "").strip().lower()

    if retry_scope == "rerank_only":
        return "rerank"
    if retry_scope == "stop" or not reflection.should_retry:
        return "stop"
    if failure_type == "acceptable":
        return "answer"
    if (
        recommended_action == "answer_ready"
        and failure_type in {"", "acceptable"}
    ):
        return "answer"
    if retry_scope in {"provider_refresh", "tool_escalation"}:
        return "retry"
    if reflection.should_rerank_again and retry_scope in {"", "rerank_only"}:
        return "rerank"
    if (
        relevance_threshold is not None
        and relevance_threshold > 0
        and reflection.overall_score >= relevance_threshold
    ):
        return "answer"
    return "retry"


def _apply_verdict_defaults(step: ReflectionStep) -> ReflectionStep:
    """Normalize control flags so workflow routing is verdict-driven."""
    verdict = resolve_reflection_verdict(step)
    step.verdict = verdict

    if verdict == "answer":
        step.should_retry = False
        step.should_rerank_again = False
        step.retry_scope = "stop"
        if not step.recommended_action:
            step.recommended_action = "answer_ready"
        if not step.failure_type:
            step.failure_type = "acceptable"
        return step

    if verdict == "rerank":
        step.should_retry = True
        step.should_rerank_again = True
        step.retry_scope = "rerank_only"
        if not step.recommended_action:
            step.recommended_action = "rerank_results"
        return step

    if verdict == "stop":
        step.should_retry = False
        step.should_rerank_again = False
        step.retry_scope = "stop"
        return step

    step.should_retry = True
    if not step.retry_scope:
        step.retry_scope = "tool_escalation"
    return step


def _sanitize_reflection_step(
    step: ReflectionStep,
    *,
    has_results: bool,
    invalid_verdict: bool = False,
) -> ReflectionStep:
    """Treat model output as untrusted and coerce unsafe states to safe defaults."""
    if invalid_verdict:
        step.verdict = "stop"

    step.preferred_tools = [
        tool
        for tool in step.preferred_tools
        if tool in _VALID_TOOL_NAMES
    ]
    step.preferred_providers = [
        provider
        for provider in step.preferred_providers
        if provider in _VALID_PROVIDER_NAMES
    ]

    if step.retry_scope not in _VALID_RETRY_SCOPES:
        step.retry_scope = "stop" if step.verdict == "stop" else ""
    if step.failure_type not in _VALID_FAILURE_TYPES:
        step.failure_type = ""

    has_missing_signals = bool(
        step.missing_information
        or step.missing_entities
        or step.missing_relationships
        or step.coverage_gap_sources
        or step.failure_type not in {"", "acceptable"}
    )
    if step.verdict == "answer" and has_missing_signals:
        step.verdict = "stop"
    if step.verdict == "rerank" and not has_results:
        step.verdict = "stop"

    return _apply_verdict_defaults(step)


def _policy_stop_reflection(
    query: str,
    *,
    tool_name: str,
    attempt: int,
    error_message: str,
) -> ReflectionStep:
    """Abort downstream retries when the judge output is untrusted."""
    return ReflectionStep(
        attempt=attempt,
        tool_name=tool_name,
        query_used=query,
        evidence_status="insufficient",
        gap_type="off_topic",
        action="stop",
        required_tool="none",
        verdict="stop",
        overall_score=0.0,
        relevance=0.0,
        entity_completeness=0.0,
        logical_consistency=0.0,
        context_sufficiency=0.0,
        missing_information=["Reflection policy guard rejected the model output."],
        missing_entities=[],
        missing_relationships=[],
        coverage_gap_sources=[],
        candidate_fix_paths=[],
        preferred_tools=[],
        preferred_providers=[],
        retry_scope="stop",
        reasoning=error_message,
        failure_type="insufficient_context",
        recommended_action="stop_due_to_invalid_reflection",
        should_retry=False,
        should_rewrite_query=False,
        should_rerank_again=False,
        comparison_to_previous="Policy guard stop.",
    )


def _verdict_for_action(action: str, evidence_status: str) -> str:
    action = action.strip().lower()
    evidence_status = evidence_status.strip().lower()
    if action == "answer":
        return "answer"
    if action == "rerank":
        return "rerank"
    if action == "stop":
        return "stop"
    if action == "answer_with_caveat":
        return "stop" if evidence_status in {"partial", "insufficient"} else "answer"
    return "retry"


def _failure_type_for_gap(gap_type: str, evidence_status: str) -> str:
    if evidence_status == "empty":
        return "no_results"
    return {
        "conflicting_evidence": "inconsistent_evidence",
        "missing_entity": "missing_entity",
        "missing_relation": "relation_missing",
        "none": "acceptable",
        "off_topic": "insufficient_recall",
    }.get(gap_type, "insufficient_context")


def _retry_scope_for_action(action: str) -> str:
    if action in {"answer", "answer_with_caveat", "stop"}:
        return "stop"
    if action == "rerank":
        return "rerank_only"
    return "tool_escalation"


def _tool_for_action(action: str, required_tool: str) -> str:
    if required_tool and required_tool != "none":
        return required_tool
    return {
        "retry_bm25": "bm25_search",
        "retry_graph": "cypher_traverse",
        "retry_hybrid": "hybrid_search",
        "retry_vector": "vector_search",
    }.get(action, "")


def _recommended_action_for_choice(action: str, gap_type: str) -> str:
    if action == "answer":
        return "answer_ready"
    if action == "answer_with_caveat":
        return "answer_with_caveat"
    if action == "rerank":
        return "rerank_results"
    if action == "retry_graph" or gap_type == "missing_relation":
        return "use_graph_traversal"
    if action == "retry_bm25":
        return "target_missing_entity"
    if action == "retry_hybrid":
        return "use_comprehensive_search"
    if action == "retry_vector":
        return "expand_recall"
    return action


def _legacy_score_for_status(evidence_status: str) -> float:
    return {
        "sufficient": 5.0,
        "partial": 3.0,
        "insufficient": 1.5,
        "empty": 0.0,
    }.get(evidence_status, 0.0)


def _validate_reflection_payload(data: dict[str, object]) -> str:
    """Return empty string when payload is safe, otherwise a policy error."""
    if set(data) != _REFLECTION_SCHEMA_FIELDS:
        unexpected = sorted(set(data) - _REFLECTION_SCHEMA_FIELDS)
        missing = sorted(_REFLECTION_SCHEMA_FIELDS - set(data))
        details: list[str] = []
        if unexpected:
            details.append(f"unexpected fields: {', '.join(unexpected)}")
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        return "; ".join(details) or "schema mismatch"

    evidence_status = str(data.get("evidence_status", "")).strip().lower()
    if evidence_status not in _VALID_EVIDENCE_STATUSES:
        return "invalid evidence_status"
    gap_type = str(data.get("gap_type", "")).strip().lower()
    if gap_type not in _VALID_GAP_TYPES:
        return "invalid gap_type"
    action = str(data.get("action", "")).strip().lower()
    if action not in _VALID_REFLECTION_ACTIONS:
        return "invalid action"
    required_tool = str(data.get("required_tool", "")).strip().lower()
    if required_tool not in {"none", *_VALID_TOOL_NAMES}:
        return "invalid required_tool"
    failure_type = str(data.get("failure_type", "")).strip().lower()
    if failure_type not in _VALID_FAILURE_TYPES:
        return "invalid failure_type"

    tool_values = _coerce_str_list(data.get("preferred_tools"))
    if any(tool not in _VALID_TOOL_NAMES for tool in tool_values):
        return "invalid preferred_tools"
    provider_values = _coerce_str_list(data.get("preferred_providers"))
    if any(provider not in _VALID_PROVIDER_NAMES for provider in provider_values):
        return "invalid preferred_providers"

    return ""


def _reflection_response_format() -> dict[str, object]:
    """Build the strict JSON schema response_format for supported providers."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "reflection_step",
            "strict": True,
            "schema": _REFLECTION_JSON_SCHEMA,
        },
    }


def _is_response_format_unsupported(exc: Exception) -> bool:
    """Detect providers that reject structured output parameters."""
    text = str(exc).casefold()
    return (
        ("response_format" in text or "json_schema" in text or "schema" in text)
        and any(
            marker in text
            for marker in ("unsupported", "not support", "unknown", "invalid", "extra inputs")
        )
    )


def _extract_json(text: str) -> dict[str, object]:
    text = text.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = -1
    depth = 0
    in_string = False
    escape = False
    for idx, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = idx
            depth += 1
            continue
        if char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    payload = json.loads(text[start:idx + 1])
                except json.JSONDecodeError:
                    start = -1
                    continue
                if isinstance(payload, dict):
                    return payload
    return {}


def _results_summary(results: list[SearchResult], limit: int = _TOP_RESULTS) -> str:
    if not results:
        return "No relevant content found."

    summary_parts = []
    for i, result in enumerate(results[:limit], start=1):
        text = result.chunk.enriched_content or result.chunk.content
        summary_parts.append(f"[Chunk {i} | score={result.score:.2f}] {text[:400]}")
    return "\n\n".join(summary_parts)


def _history_summary(reflection_history: list[ReflectionStep] | None) -> str:
    if not reflection_history:
        return "No previous reflection history."

    lines = []
    for step in reflection_history[-3:]:
        missing = ", ".join(step.missing_information) if step.missing_information else "none"
        lines.append(
            f"- attempt={step.attempt + 1}, tool={step.tool_name}, "
            f"score={step.overall_score:.2f}, failure={step.failure_type or 'unknown'}, "
            f"action={step.recommended_action or 'unknown'}, missing={missing}"
        )
    return "\n".join(lines)


def _memory_summary(workflow_memory: list[WorkflowMemoryEntry] | None) -> str:
    if not workflow_memory:
        return "No workflow memory."

    lines = []
    for entry in workflow_memory[-5:]:
        metadata = ", ".join(
            f"{key}={value}"
            for key, value in entry.metadata.items()
        )
        if metadata:
            lines.append(f"- stage={entry.stage}, note={entry.message}, meta={metadata}")
        else:
            lines.append(f"- stage={entry.stage}, note={entry.message}")
    return "\n".join(lines)


def _build_reflection_step(
    data: dict[str, object],
    *,
    query: str,
    tool_name: str,
    attempt: int,
) -> ReflectionStep:
    evidence_status = str(data.get("evidence_status", "")).strip().lower()
    gap_type = str(data.get("gap_type", "")).strip().lower()
    action = str(data.get("action", "")).strip().lower()
    required_tool = str(data.get("required_tool", "")).strip().lower()
    verdict = _verdict_for_action(action, evidence_status)
    score = _legacy_score_for_status(evidence_status)
    failure_type = str(data.get("failure_type", "")).strip().lower() or _failure_type_for_gap(
        gap_type,
        evidence_status,
    )
    recommended_action = str(data.get("recommended_action", "")).strip().lower() or action
    recommended_action = recommended_action or _recommended_action_for_choice(action, gap_type)
    if recommended_action == action:
        recommended_action = _recommended_action_for_choice(action, gap_type)
    retry_scope = str(data.get("retry_scope", "")).strip().lower() or _retry_scope_for_action(action)
    preferred_tools = _coerce_str_list(data.get("preferred_tools"))
    action_tool = _tool_for_action(action, required_tool)
    if action_tool and action_tool not in preferred_tools:
        preferred_tools.insert(0, action_tool)
    step = ReflectionStep(
        attempt=attempt,
        tool_name=tool_name,
        query_used=query,
        evidence_status=evidence_status,
        gap_type=gap_type,
        action=action,
        required_tool=required_tool,
        verdict=verdict,
        overall_score=score,
        relevance=score,
        entity_completeness=score,
        logical_consistency=score,
        context_sufficiency=score,
        missing_information=_coerce_str_list(data.get("missing_information")),
        missing_entities=_coerce_str_list(data.get("missing_entities")),
        missing_relationships=_coerce_str_list(data.get("missing_relationships")),
        coverage_gap_sources=_coerce_str_list(data.get("coverage_gap_sources")),
        candidate_fix_paths=_coerce_str_list(data.get("candidate_fix_paths")),
        preferred_tools=preferred_tools,
        preferred_providers=_coerce_str_list(data.get("preferred_providers")),
        retry_scope=retry_scope,
        reasoning=str(data.get("reasoning", "")).strip(),
        failure_type=failure_type,
        recommended_action=recommended_action,
        should_retry=_coerce_bool(data.get("should_retry"), default=verdict in {"retry", "rerank"}),
        should_rewrite_query=_coerce_bool(
            data.get("should_rewrite_query"),
            default=recommended_action in {"target_missing_entity", "use_comprehensive_search"},
        ),
        should_rerank_again=_coerce_bool(data.get("should_rerank_again"), default=verdict == "rerank"),
        comparison_to_previous=str(data.get("comparison_to_previous", "")).strip(),
    )
    return _sanitize_reflection_step(
        step,
        has_results=True,
        invalid_verdict=False,
    )


def reflection_to_confidence(reflection: ReflectionStep) -> float:
    """Map reflection scores from the 0-5 space into 0-1 confidence."""
    if reflection.overall_score <= 0:
        return 0.0
    raw_score = (
        _REFLECTION_SCORE_WEIGHTS["relevance"] * reflection.relevance
        + _REFLECTION_SCORE_WEIGHTS["entity_completeness"] * reflection.entity_completeness
        + _REFLECTION_SCORE_WEIGHTS["logical_consistency"] * reflection.logical_consistency
        + _REFLECTION_SCORE_WEIGHTS["context_sufficiency"] * reflection.context_sufficiency
    )
    confidence = min(1.0, max(0.1, raw_score / 5.0))
    return round(confidence, 3)


def _fallback_reflection(
    query: str,
    results: list[SearchResult],
    *,
    tool_name: str,
    attempt: int,
    error_message: str = "",
) -> ReflectionStep:
    if not results:
        return ReflectionStep(
            attempt=attempt,
            tool_name=tool_name,
            query_used=query,
            evidence_status="empty",
            gap_type="missing_entity",
            action="retry_hybrid",
            required_tool="hybrid_search",
            verdict="retry",
            overall_score=0.0,
            relevance=0.0,
            entity_completeness=0.0,
            logical_consistency=0.0,
            context_sufficiency=0.0,
            missing_information=["No evidence retrieved."],
            missing_entities=[],
            missing_relationships=[],
            coverage_gap_sources=[],
            candidate_fix_paths=["expand_recall"],
            preferred_tools=[],
            preferred_providers=[],
            retry_scope="tool_escalation",
            reasoning="No results were returned by the retrieval tool.",
            failure_type="no_results",
            recommended_action="expand_recall",
            should_retry=True,
            should_rewrite_query=True,
            should_rerank_again=False,
            comparison_to_previous="Initial failure." if attempt == 0 else "Still no usable evidence.",
        )

    reasoning = "Reflection fallback used."
    if error_message:
        reasoning = f"{reasoning} {error_message}"

    return ReflectionStep(
        attempt=attempt,
        tool_name=tool_name,
        query_used=query,
        evidence_status="insufficient",
        gap_type="off_topic",
        action="retry_hybrid",
        required_tool="hybrid_search",
        verdict="retry",
        overall_score=2.5,
        relevance=2.5,
        entity_completeness=2.5,
        logical_consistency=2.5,
        context_sufficiency=2.5,
        missing_information=["Unable to determine exact evidence gaps."],
        missing_entities=[],
        missing_relationships=[],
        coverage_gap_sources=[],
        candidate_fix_paths=["expand_recall"],
        preferred_tools=[],
        preferred_providers=[],
        retry_scope="tool_escalation",
        reasoning=reasoning,
        failure_type="insufficient_context",
        recommended_action="expand_recall",
        should_retry=True,
        should_rewrite_query=False,
        should_rerank_again=False,
        comparison_to_previous="Unable to compare reliably.",
    )


def evaluate_reflection(
    query: str,
    results: list[SearchResult],
    openai_client: OpenAI | None = None,
    *,
    reflection_history: list[ReflectionStep] | None = None,
    workflow_memory: list[WorkflowMemoryEntry] | None = None,
    tool_name: str = "",
    attempt: int = 0,
) -> ReflectionStep:
    """Evaluate retrieval quality with structured multi-dimensional reflection."""
    cfg = get_settings()
    if openai_client is None:
        openai_client = make_openai_client(cfg)

    if not results:
        logger.warning("No results to evaluate")
        return _fallback_reflection(query, results, tool_name=tool_name, attempt=attempt)

    prompt = (
        f"{REFLECTION_PROMPT}\n\n"
        f"Query: {query}\n\n"
        f"Previous reflection history:\n{_history_summary(reflection_history)}\n\n"
        f"Workflow memory:\n{_memory_summary(workflow_memory)}\n\n"
        f"Retrieved chunks:\n{_results_summary(results)}\n"
    )

    try:
        response = None
        try:
            response = openai_client.chat.completions.create(
                model=cfg.openai.llm_model_mini,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format=_reflection_response_format(),
            )
        except Exception as exc:
            if not _is_response_format_unsupported(exc):
                raise
            logger.warning(
                "Reflection provider does not support response_format json_schema, "
                "falling back to prompt-only JSON mode: %s",
                exc,
            )
            response = openai_client.chat.completions.create(
                model=cfg.openai.llm_model_mini,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
        payload = _extract_json(response.choices[0].message.content or "")
        if not payload:
            return _policy_stop_reflection(
                query,
                tool_name=tool_name,
                attempt=attempt,
                error_message="Reflection policy guard: model returned invalid JSON.",
            )
        validation_error = _validate_reflection_payload(payload)
        if validation_error:
            return _policy_stop_reflection(
                query,
                tool_name=tool_name,
                attempt=attempt,
                error_message=f"Reflection policy guard: {validation_error}.",
            )
        step = _build_reflection_step(payload, query=query, tool_name=tool_name, attempt=attempt)
        logger.info(
            "Reflection score %.2f (rel=%.2f, ent=%.2f, logic=%.2f, ctx=%.2f)",
            step.overall_score,
            step.relevance,
            step.entity_completeness,
            step.logical_consistency,
            step.context_sufficiency,
        )
        return step

    except Exception as exc:
        logger.error("Error evaluating reflection: %s", exc)
        return _policy_stop_reflection(
            query,
            tool_name=tool_name,
            attempt=attempt,
            error_message=f"Reflection policy guard: {exc}",
        )


def evaluate_relevance(
    query: str,
    results: list[SearchResult],
    openai_client: OpenAI | None = None,
) -> float:
    """Backwards-compatible wrapper returning the overall reflection score."""
    step = evaluate_reflection(query, results, openai_client=openai_client)
    return step.overall_score


def generate_retry_query(
    query: str,
    results: list[SearchResult],
    openai_client: OpenAI | None = None,
    *,
    reflection: ReflectionStep | None = None,
    reflection_history: list[ReflectionStep] | None = None,
    workflow_memory: list[WorkflowMemoryEntry] | None = None,
) -> str:
    """Generate a targeted retry query based on reflection gaps and history."""
    cfg = get_settings()
    if openai_client is None:
        openai_client = make_openai_client(cfg)

    if reflection is None:
        reflection = _fallback_reflection(query, results, tool_name="", attempt=0)

    missing = ", ".join(reflection.missing_information) or "No explicit missing information."
    prompt = (
        f"{RETRY_QUERY_PROMPT}\n\n"
        f"Original query: {query}\n"
        f"Current reflection score: {reflection.overall_score:.2f}\n"
        f"Failure type: {reflection.failure_type or 'unknown'}\n"
        f"Recommended action: {reflection.recommended_action or 'unknown'}\n"
        f"Missing information: {missing}\n\n"
        f"Previous reflection history:\n{_history_summary(reflection_history)}\n\n"
        f"Workflow memory:\n{_memory_summary(workflow_memory)}\n\n"
        f"Found content:\n{_results_summary(results, limit=3)}\n"
    )

    try:
        response = openai_client.chat.completions.create(
            model=cfg.openai.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        retry_query = (response.choices[0].message.content or "").strip()
        if not retry_query:
            return query
        retry_query = retry_query.splitlines()[0].strip()
        logger.info("Generated retry query: %s", retry_query)
        return retry_query or query

    except Exception as exc:
        logger.error("Error generating retry query: %s", exc)
        return query


def evaluate_completeness(
    query: str,
    answer: str,
    openai_client: OpenAI | None = None,
) -> bool:
    """Check whether the generated answer is complete for the query."""
    normalized_query = " ".join(query.strip().lower().split())
    normalized_answer = " ".join(answer.strip().lower().split())

    if not normalized_answer:
        logger.info("Completeness check: incomplete (empty answer)")
        return False

    if normalized_answer.startswith("error generating answer:"):
        logger.info("Completeness check: incomplete (generation error)")
        return False

    if any(cue in normalized_answer for cue in _INCOMPLETE_ANSWER_CUES):
        logger.info("Completeness check: incomplete (answer admits missing evidence)")
        return False

    looks_like_enumeration = any(cue in normalized_query for cue in _ENUMERATION_CUES)
    if looks_like_enumeration:
        bullet_like_lines = sum(
            1
            for line in answer.splitlines()
            if line.strip().startswith(("-", "*", "•"))
            or re.match(r"^\s*\d+\.", line)
        )
        if bullet_like_lines >= 2:
            logger.info("Completeness check: complete (enumeration structure detected)")
            return True
        if len(answer.strip()) < 120:
            logger.info("Completeness check: incomplete (enumeration answer too short)")
            return False

    cfg = get_settings()
    if openai_client is None:
        openai_client = make_openai_client(cfg)

    prompt = (
        "You are evaluating an answer for completeness.\n"
        f"Query: {query}\n\n"
        f"Answer: {answer}\n\n"
        "Is this answer COMPLETE? Does it cover ALL aspects of the query? "
        "Consider: does it list all items asked for? Does it acknowledge gaps?\n"
        "Respond with ONLY 'YES' or 'NO' followed by a brief explanation."
    )

    try:
        response = openai_client.chat.completions.create(
            model=cfg.openai.llm_model_mini,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        text = (response.choices[0].message.content or "").strip().upper()
        is_complete = text.startswith("YES")
        logger.info("Completeness check: %s", "complete" if is_complete else "incomplete")
        return is_complete

    except Exception as exc:
        logger.error("Error evaluating completeness: %s", exc)
        return True
