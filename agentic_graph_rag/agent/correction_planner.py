"""LLM-guided correction planning for failed answer verification."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from rag_core.config import get_settings
from rag_core.models import ClaimVerificationStep

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)

AllowedCorrectionTool = Literal[
    "vector_search",
    "bm25_search",
    "cypher_traverse",
    "hybrid_search",
    "full_document_read",
]
CorrectionAction = Literal["retry_with_tool", "finish_partial", "stop"]

ALLOWED_CORRECTION_TOOLS: tuple[AllowedCorrectionTool, ...] = (
    "vector_search",
    "bm25_search",
    "cypher_traverse",
    "hybrid_search",
    "full_document_read",
)
_ALLOWED_TOOL_SET = set(ALLOWED_CORRECTION_TOOLS)

_TOOL_DESCRIPTIONS = {
    "vector_search": "semantic passage retrieval",
    "bm25_search": "exact lexical and numeric retrieval",
    "cypher_traverse": "entity relationship and graph evidence",
    "hybrid_search": "multi-provider recall with rerank",
    "full_document_read": "broad source expansion",
}

_NUMERIC_PATTERN = re.compile(
    r"(?:[<>]=?|≤|≥|=)\s*\d+(?:\.\d+)?"
    r"|\d+(?:\.\d+)?\s*(?:%|mg|ml|mmol|μg|ug|pg|g/l|iu|次|个月|天|年|分钟|小时)"
    r"|\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?",
    re.IGNORECASE,
)

_PROMPT = """You choose one correction tool for a failed Graph RAG answer verification.
Use only the fixed tool allowlist. Do not judge whether the medical fact is true.
Choose the next retrieval action that can gather missing evidence.

Tool guidance:
- bm25_search: exact numbers, thresholds, doses, frequencies, names.
- cypher_traverse: entity relationships and graph relations.
- vector_search: semantic recall when wording may differ.
- hybrid_search: mixed recall when both exact and semantic evidence may be needed.
- full_document_read: broad source expansion for incomplete global coverage.

Return ONLY JSON:
{
  "action": "retry_with_tool|finish_partial|stop",
  "tool": "vector_search|bm25_search|cypher_traverse|hybrid_search|full_document_read|null",
  "focus_query": "short focused query",
  "reason": "short reason"
}
"""


@dataclass(frozen=True, slots=True)
class CorrectionGap:
    gap_type: str
    claim_text: str
    missing_entities: list[str] = field(default_factory=list)
    missing_facts: list[str] = field(default_factory=list)
    relation_actions: list[str] = field(default_factory=list)

    def as_payload(self) -> dict[str, object]:
        return {
            "gap_type": self.gap_type,
            "claim_text": self.claim_text,
            "missing_entities": self.missing_entities,
            "missing_facts": self.missing_facts,
            "relation_actions": self.relation_actions,
        }


@dataclass(frozen=True, slots=True)
class CorrectionPlan:
    action: CorrectionAction
    tool: AllowedCorrectionTool | None
    focus_query: str
    reason: str


def build_gap_report(verification: ClaimVerificationStep) -> list[CorrectionGap]:
    """Convert unsupported claims into narrow planner gaps."""
    gaps: list[CorrectionGap] = []
    for claim in verification.unsupported_claims:
        if claim.verification_level not in {"incorrect", "possible_correct"}:
            continue
        missing_facts = list(claim.numeric_constraints)
        gap_type = "missing_relation"
        if missing_facts or _NUMERIC_PATTERN.search(claim.text):
            gap_type = "missing_numeric_fact"
        elif claim.entities:
            gap_type = "missing_entity"

        gaps.append(
            CorrectionGap(
                gap_type=gap_type,
                claim_text=claim.text,
                missing_entities=list(claim.entities),
                missing_facts=missing_facts,
                relation_actions=list(claim.relation_actions),
            )
        )
    return gaps


def plan_correction(
    *,
    query: str,
    answer: str,
    verification_status: str,
    gaps: list[CorrectionGap],
    openai_client: OpenAI,
) -> CorrectionPlan:
    """Ask an LLM to choose one correction tool, with deterministic fallback."""
    if not gaps:
        return CorrectionPlan(
            action="finish_partial",
            tool=None,
            focus_query=query,
            reason="no verification gaps available",
        )

    prompt_payload = {
        "query": query[:800],
        "answer": answer[:1200],
        "verification_status": verification_status,
        "gaps": [gap.as_payload() for gap in gaps[:3]],
        "available_tools": _TOOL_DESCRIPTIONS,
    }
    try:
        cfg = get_settings()
        response = openai_client.chat.completions.create(
            model=cfg.openai.llm_model_mini,
            messages=[
                {"role": "system", "content": _PROMPT},
                {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)},
            ],
            temperature=0.0,
        )
        raw = response.choices[0].message.content or ""
        return _coerce_plan(_extract_json(raw), gaps, query)
    except Exception as exc:
        logger.warning("Correction planner failed (%s); using deterministic fallback", exc)
        return _fallback_plan(gaps, query, reason="planner fallback after exception")


def _coerce_plan(payload: dict[str, object], gaps: list[CorrectionGap], query: str) -> CorrectionPlan:
    action = str(payload.get("action") or "").strip()
    tool = payload.get("tool")
    focus_query = str(payload.get("focus_query") or "").strip()
    reason = str(payload.get("reason") or "").strip()

    if action not in {"retry_with_tool", "finish_partial", "stop"}:
        return _fallback_plan(gaps, query, reason="planner fallback after invalid action")
    if action != "retry_with_tool":
        return CorrectionPlan(
            action=action,  # type: ignore[arg-type]
            tool=None,
            focus_query=focus_query or query,
            reason=reason or f"planner chose {action}",
        )
    if not isinstance(tool, str) or tool not in _ALLOWED_TOOL_SET:
        return _fallback_plan(gaps, query, reason="planner fallback after invalid tool")
    return CorrectionPlan(
        action="retry_with_tool",
        tool=tool,  # type: ignore[arg-type]
        focus_query=focus_query or _fallback_focus_query(gaps, query),
        reason=reason or "planner selected correction tool",
    )


def _fallback_plan(gaps: list[CorrectionGap], query: str, *, reason: str) -> CorrectionPlan:
    first_type = gaps[0].gap_type if gaps else ""
    tool: AllowedCorrectionTool = "cypher_traverse"
    if first_type == "missing_numeric_fact":
        tool = "bm25_search"
    elif first_type == "missing_entity":
        tool = "hybrid_search"
    return CorrectionPlan(
        action="retry_with_tool",
        tool=tool,
        focus_query=_fallback_focus_query(gaps, query),
        reason=reason,
    )


def _fallback_focus_query(gaps: list[CorrectionGap], query: str) -> str:
    parts: list[str] = []
    for gap in gaps[:2]:
        parts.append(gap.claim_text)
        parts.extend(gap.missing_entities)
        parts.extend(gap.missing_facts)
        parts.extend(gap.relation_actions)
    if not parts:
        parts.append(query)
    return _dedupe_join(parts)


def _dedupe_join(parts: list[str]) -> str:
    seen: set[str] = set()
    unique: list[str] = []
    for part in parts:
        normalized = " ".join(str(part).split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return " ".join(unique)


def _extract_json(text: str) -> dict[str, object]:
    text = text.strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
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
                    payload = json.loads(text[start : idx + 1])
                except json.JSONDecodeError:
                    start = -1
                    continue
                return payload if isinstance(payload, dict) else {}
    return {}
