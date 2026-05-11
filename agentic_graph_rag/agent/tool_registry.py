"""Shared retrieval tool definitions."""

from __future__ import annotations

from typing import Literal, get_args

from rag_core.models import QueryType

ToolName = Literal[
    "vector_search",
    "bm25_search",
    "cypher_traverse",
    "hybrid_search",
    "comprehensive_search",
    "temporal_query",
    "full_document_read",
]

TOOL_NAMES: tuple[ToolName, ...] = get_args(ToolName)

DEFAULT_TOOL_BY_QUERY_TYPE: dict[QueryType, str] = {
    QueryType.SIMPLE: "vector_search",
    QueryType.RELATION: "vector_search",
    QueryType.MULTI_HOP: "vector_search",
    QueryType.GLOBAL: "vector_search",
    QueryType.TEMPORAL: "temporal_query",
}

QUERY_TYPE_BY_TOOL: dict[str, QueryType] = {
    "vector_search": QueryType.SIMPLE,
    "bm25_search": QueryType.SIMPLE,
    "cypher_traverse": QueryType.RELATION,
    "hybrid_search": QueryType.MULTI_HOP,
    "comprehensive_search": QueryType.GLOBAL,
    "full_document_read": QueryType.GLOBAL,
    "temporal_query": QueryType.TEMPORAL,
}
