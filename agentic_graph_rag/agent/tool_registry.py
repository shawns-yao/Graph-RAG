"""Shared retrieval tool definitions."""

from __future__ import annotations

from typing import Literal, get_args

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
