"""Agentic Query Router — classifies queries and selects retrieval tools.

Uses pattern matching for fast classification with optional LLM fallback.
Categories: simple, relation, multi_hop, global, temporal.
"""

from __future__ import annotations

import logging
import re

from openai import OpenAI
from rag_core.config import get_settings, make_openai_client
from rag_core.models import QueryType, RouterDecision

from agentic_graph_rag.agent.routing_rules import (
    GLOBAL_PATTERNS,
    GLOBAL_QUERY_KEYWORDS,
    MULTI_HOP_PATTERNS,
    RELATION_PATTERNS,
    RELATION_QUERY_KEYWORDS,
    TEMPORAL_PATTERNS,
)
from agentic_graph_rag.agent.tool_registry import DEFAULT_TOOL_BY_QUERY_TYPE

logger = logging.getLogger(__name__)

def _match_patterns(query: str, patterns: list[str]) -> int:
    """Count how many patterns match in the query."""
    count = 0
    for pat in patterns:
        if re.search(pat, query, re.IGNORECASE):
            count += 1
    return count


def _looks_like_medical_decision_query(query: str) -> bool:
    """Detect condition-heavy medical treatment selection questions."""
    lowered = query.casefold()
    has_decision_intent = any(
        phrase in lowered
        for phrase in (
            "什么方案",
            "推荐什么",
            "应该使用",
            "应该用什么",
            "治疗方案",
            "recommended",
            "which regimen",
            "which treatment",
        )
    )
    has_condition = any(
        token in query
        for token in ("如果", "且", "并且", "≥", "≤", "<", ">", "/μl", "次/年")
    ) or bool(re.search(r"\b\d+(?:\.\d+)?\b", query))
    return has_decision_intent and has_condition


def _intent_override_decision(query: str) -> RouterDecision | None:
    """Detect high-signal intents before the optional LLM/pattern fallback."""
    lowered_query = query.casefold()
    relation_keyword_hit = any(keyword in lowered_query for keyword in RELATION_QUERY_KEYWORDS)
    global_keyword_hit = any(keyword in lowered_query for keyword in GLOBAL_QUERY_KEYWORDS)
    relation_pattern_hits = _match_patterns(query, RELATION_PATTERNS)
    multi_hop_pattern_hits = _match_patterns(query, MULTI_HOP_PATTERNS)
    global_pattern_hits = _match_patterns(query, GLOBAL_PATTERNS)

    if global_keyword_hit or global_pattern_hits > 0:
        return RouterDecision(
            query_type=QueryType.GLOBAL,
            confidence=0.9,
            reasoning="Global summary intent detected; retrieval planner may add broad recall.",
            suggested_tool=DEFAULT_TOOL_BY_QUERY_TYPE[QueryType.GLOBAL],
        )

    if (
        relation_keyword_hit
        or multi_hop_pattern_hits > 0
        or relation_pattern_hits > 0
        or _looks_like_medical_decision_query(query)
    ):
        query_type = QueryType.MULTI_HOP if multi_hop_pattern_hits > 0 else QueryType.RELATION
        return RouterDecision(
            query_type=query_type,
            confidence=0.96 if query_type == QueryType.RELATION else 0.94,
            reasoning=(
                "Relation intent detected; retrieval planner may add graph companion."
                if query_type == QueryType.RELATION
                else "Multi-hop intent detected; retrieval planner may add graph companion."
            ),
            suggested_tool=DEFAULT_TOOL_BY_QUERY_TYPE[query_type],
        )

    if _match_patterns(query, TEMPORAL_PATTERNS) > 0:
        return RouterDecision(
            query_type=QueryType.TEMPORAL,
            confidence=0.94,
            reasoning="Temporal intent detected; retrieval planner may add temporal companion.",
            suggested_tool=DEFAULT_TOOL_BY_QUERY_TYPE[QueryType.TEMPORAL],
        )

    return None


# ---------------------------------------------------------------------------
# Pattern-based classification (fast, no LLM)
# ---------------------------------------------------------------------------

def classify_query_by_patterns(query: str) -> RouterDecision:
    """Classify query using regex pattern matching.

    Returns RouterDecision with confidence based on match count.
    """
    scores: dict[QueryType, int] = {
        QueryType.TEMPORAL: _match_patterns(query, TEMPORAL_PATTERNS),
        QueryType.MULTI_HOP: _match_patterns(query, MULTI_HOP_PATTERNS),
        QueryType.RELATION: _match_patterns(query, RELATION_PATTERNS),
        QueryType.GLOBAL: _match_patterns(query, GLOBAL_PATTERNS),
    }

    best_type = max(scores, key=lambda k: scores[k])
    best_count = scores[best_type]

    if best_count == 0:
        query_type = QueryType.SIMPLE
        confidence = 0.5
        reasoning = "No specific patterns matched; defaulting to vector retrieval first."
    else:
        query_type = best_type
        confidence = min(0.5 + best_count * 0.2, 0.95)
        reasoning = f"Matched {best_count} {query_type.value} pattern(s)."

    return RouterDecision(
        query_type=query_type,
        confidence=confidence,
        reasoning=reasoning,
        suggested_tool=DEFAULT_TOOL_BY_QUERY_TYPE[query_type],
    )


# ---------------------------------------------------------------------------
# LLM-based classification (high-quality, slower)
# ---------------------------------------------------------------------------

_CLASSIFY_PROMPT = """Classify the following query into exactly ONE category:
- simple: factual lookup, single entity question
- relation: asks about relationships between entities
- multi_hop: requires traversing multiple connections or comparing entities
- global: asks about all/every/overview of something
- temporal: asks about time, dates, history, changes over time

Query: {query}

Respond with ONLY the category name (simple/relation/multi_hop/global/temporal):"""


def classify_query_by_llm(
    query: str, openai_client: OpenAI | None = None,
) -> RouterDecision:
    """Classify query using LLM for higher accuracy."""
    cfg = get_settings()
    if openai_client is None:
        openai_client = make_openai_client(cfg)

    try:
        response = openai_client.chat.completions.create(
            model=cfg.openai.llm_model_mini,
            messages=[{"role": "user", "content": _CLASSIFY_PROMPT.format(query=query)}],
            temperature=0.0,
        )
        raw = (response.choices[0].message.content or "simple").strip().lower()

        type_map = {
            "simple": QueryType.SIMPLE,
            "relation": QueryType.RELATION,
            "multi_hop": QueryType.MULTI_HOP,
            "global": QueryType.GLOBAL,
            "temporal": QueryType.TEMPORAL,
        }
        query_type = type_map.get(raw, QueryType.SIMPLE)

        return RouterDecision(
            query_type=query_type,
            confidence=0.85,
            reasoning=f"LLM classified as '{raw}'.",
            suggested_tool=DEFAULT_TOOL_BY_QUERY_TYPE[query_type],
        )

    except Exception as e:
        logger.error("LLM classification failed: %s — falling back to patterns", e)
        return classify_query_by_patterns(query)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def classify_query(
    query: str,
    use_llm: bool = False,
    openai_client: OpenAI | None = None,
) -> RouterDecision:
    """Classify query and suggest retrieval tool.

    High-signal intent checks are tried first, followed by the optional LLM router or pattern fallback.
    """
    intent_result = _intent_override_decision(query)
    if intent_result is not None:
        return intent_result

    if use_llm:
        return classify_query_by_llm(query, openai_client=openai_client)
    return classify_query_by_patterns(query)
