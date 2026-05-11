"""Agentic Query Router — classifies queries without selecting retrieval tools.

The deterministic path is intentionally conservative. Companion retrieval
channels are planned later from query signals and verification gaps.
"""

from __future__ import annotations

import logging

from openai import OpenAI
from rag_core.config import get_settings, make_openai_client
from rag_core.models import QueryType, RouterDecision

from agentic_graph_rag.agent.tool_registry import DEFAULT_TOOL_BY_QUERY_TYPE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deterministic fallback (fast, no LLM)
# ---------------------------------------------------------------------------

def classify_query_by_patterns(query: str) -> RouterDecision:
    """Return a conservative default when the optional LLM router is disabled."""
    query_type = QueryType.SIMPLE

    return RouterDecision(
        query_type=query_type,
        confidence=0.5,
        reasoning="Deterministic default; retrieval planner handles companion channels.",
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

    Optional LLM classification can set intent. Without it, Router stays neutral
    and retrieval planning is driven by downstream query signals.
    """
    if use_llm:
        return classify_query_by_llm(query, openai_client=openai_client)
    return classify_query_by_patterns(query)
