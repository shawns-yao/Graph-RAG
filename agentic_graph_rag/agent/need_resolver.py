"""Small-model retrieval need resolution.

The resolver classifies evidence needs, not tools. Tool selection remains a
deterministic planning step in the workflow.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from rag_core.config import get_settings

from agentic_graph_rag.agent.query_signals import QuerySignals

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)

RetrievalNeed = Literal[
    "semantic_passage",
    "exact_numeric",
    "graph_relation",
    "broad_recall",
    "temporal_evidence",
]

ALLOWED_RETRIEVAL_NEEDS: tuple[RetrievalNeed, ...] = (
    "semantic_passage",
    "exact_numeric",
    "graph_relation",
    "broad_recall",
    "temporal_evidence",
)
_ALLOWED_NEED_SET = set(ALLOWED_RETRIEVAL_NEEDS)

_PROMPT = """You classify retrieval evidence needs for a Graph RAG medical QA system.
Return evidence needs, not tool names. Do not answer the medical question.

Allowed retrieval_needs:
- semantic_passage: semantic passage evidence from source text.
- exact_numeric: exact thresholds, doses, frequencies, units, or symbols.
- graph_relation: entity relationship, causal relation, replacement, comparison,
  contraindication, or interaction evidence.
- broad_recall: broad listing or overview evidence.
- temporal_evidence: date, timeline, or before/after evidence.

Rules:
- Always include semantic_passage unless the query is empty.
- Include graph_relation when the user asks how entities relate, compare, cause,
  replace, affect, interact, or imply an action through another entity.
- Include exact_numeric when exact numeric evidence is needed.
- Use only the allowed values.

Return ONLY JSON:
{
  "retrieval_needs": ["semantic_passage"],
  "reason": "short reason"
}
"""


@dataclass(frozen=True, slots=True)
class NeedResolution:
    retrieval_needs: tuple[RetrievalNeed, ...]
    reason: str = ""


def resolve_retrieval_needs(
    *,
    query: str,
    signals: QuerySignals,
    openai_client: OpenAI,
) -> NeedResolution:
    """Ask the small model for evidence needs with schema validation."""
    payload = {
        "query": query[:800],
        "anchors": [anchor.model_dump() for anchor in signals.anchors[:12]],
        "allowed_retrieval_needs": list(ALLOWED_RETRIEVAL_NEEDS),
    }
    try:
        cfg = get_settings()
        response = openai_client.chat.completions.create(
            model=cfg.openai.llm_model_mini,
            messages=[
                {"role": "system", "content": _PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.0,
        )
        raw = response.choices[0].message.content or ""
        return coerce_need_resolution(_extract_json(raw))
    except Exception as exc:
        logger.warning("Need resolver failed (%s); using deterministic fallback", exc)
        return NeedResolution(("semantic_passage",), "resolver fallback after exception")


def coerce_need_resolution(payload: dict[str, object]) -> NeedResolution:
    """Validate model output and drop unsupported need values."""
    raw_needs = payload.get("retrieval_needs")
    if not isinstance(raw_needs, list):
        return NeedResolution(("semantic_passage",), "resolver fallback after invalid needs")

    needs: list[RetrievalNeed] = []
    for item in raw_needs:
        if not isinstance(item, str):
            continue
        need = item.strip()
        if need not in _ALLOWED_NEED_SET or need in needs:
            continue
        needs.append(need)  # type: ignore[arg-type]
    if "semantic_passage" in needs:
        needs.remove("semantic_passage")
    needs.insert(0, "semantic_passage")
    reason = str(payload.get("reason") or "").strip()
    return NeedResolution(tuple(needs), reason or "resolver selected evidence needs")


def _extract_json(raw: str) -> dict[str, object]:
    stripped = raw.strip()
    if not stripped:
        raise ValueError("empty need resolver response")
    if stripped.startswith("{"):
        return json.loads(stripped)
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        raise ValueError("need resolver response did not contain JSON")
    return json.loads(match.group(0))
