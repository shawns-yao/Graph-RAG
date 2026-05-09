"""Fusion primitives for heterogeneous retrieval results."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from rag_core.models import QueryType, SearchResult

_QUERY_TYPE_CHANNEL_WEIGHTS: dict[QueryType, dict[str, float]] = {
    QueryType.SIMPLE: {"vector": 1.0, "bm25": 0.85, "graph": 0.7},
    QueryType.RELATION: {"vector": 0.85, "bm25": 0.75, "graph": 1.35},
    QueryType.MULTI_HOP: {"vector": 0.8, "bm25": 0.7, "graph": 1.5},
    QueryType.GLOBAL: {"vector": 1.0, "bm25": 1.0, "graph": 0.9},
    QueryType.TEMPORAL: {"vector": 0.9, "bm25": 1.05, "graph": 0.85},
}


def resolve_channel_weights(
    query_type: QueryType | str | None = None,
    overrides: dict[str, float] | None = None,
) -> dict[str, float]:
    """Resolve per-channel fusion weights from query type and optional overrides."""
    weights = {"vector": 1.0, "bm25": 1.0, "graph": 1.0}
    if isinstance(query_type, str):
        try:
            query_type = QueryType(query_type)
        except ValueError:
            query_type = None
    if query_type is not None:
        weights.update(_QUERY_TYPE_CHANNEL_WEIGHTS.get(query_type, {}))
    if overrides:
        weights.update(overrides)
    return weights


def calibrate_channel_weights(
    query: str,
    provider_results: dict[str, list[SearchResult]],
    *,
    query_type: QueryType | str | None = None,
    overrides: dict[str, float] | None = None,
    retrieval_settings: Any | None = None,
) -> dict[str, float]:
    """Adjust base channel weights using live provider diagnostics."""
    weights = resolve_channel_weights(query_type, overrides)
    if not provider_results:
        return weights

    cfg = retrieval_settings
    empty_penalty = getattr(cfg, "empty_channel_penalty", 0.35)
    sparse_penalty = getattr(cfg, "sparse_channel_penalty", 0.75)
    weak_min_results = getattr(cfg, "weak_channel_min_results", 2)
    bm25_lexical_boost = getattr(cfg, "bm25_lexical_boost", 1.2)
    graph_evidence_boost = getattr(cfg, "graph_evidence_boost", 1.1)
    lexical_overlap_threshold = getattr(cfg, "lexical_overlap_threshold", 0.5)

    if isinstance(query_type, str):
        try:
            query_type = QueryType(query_type)
        except ValueError:
            query_type = None

    for source, results in provider_results.items():
        if source not in weights:
            continue
        if not results:
            weights[source] *= empty_penalty
            continue
        if len(results) < weak_min_results:
            weights[source] *= sparse_penalty

    bm25_results = provider_results.get("bm25", [])
    if bm25_results and _has_strong_lexical_match(
        query,
        bm25_results[0].chunk.content,
        lexical_overlap_threshold,
    ):
        weights["bm25"] = weights.get("bm25", 1.0) * bm25_lexical_boost

    if query_type in {QueryType.RELATION, QueryType.MULTI_HOP} and provider_results.get("graph"):
        weights["graph"] = weights.get("graph", 1.0) * graph_evidence_boost

    return weights


def _has_strong_lexical_match(
    query: str,
    content: str,
    overlap_threshold: float,
) -> bool:
    """Detect exact lexical overlap that should help sparse retrieval."""
    query_terms = _tokenize_for_overlap(query)
    if not query_terms:
        return False
    content_terms = set(_tokenize_for_overlap(content))
    matched = sum(1 for term in query_terms if term in content_terms)
    return (matched / len(query_terms)) >= overlap_threshold


def _tokenize_for_overlap(text: str) -> list[str]:
    """Tokenize mixed English/CJK text for lightweight overlap heuristics."""
    if not text:
        return []
    return [token.lower() for token in re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9_-]+", text)]


@dataclass(frozen=True, slots=True)
class FusionView:
    """Read-only fusion scoring view over an upstream SearchResult."""

    result: SearchResult
    fusion_score: float
    fusion_rank: int
    fused_source: str = "hybrid"


@dataclass(slots=True)
class FusionEngine:
    """Weighted RRF fusion over normalized SearchResult lists."""

    rrf_k: int = 60

    def build_views(
        self,
        *result_lists: list[SearchResult],
        top_k: int,
        weights: dict[str, float] | None = None,
    ) -> list[FusionView]:
        """Build fusion score views without rewriting upstream SearchResult objects."""
        scores: dict[str, float] = {}
        result_map: dict[str, SearchResult] = {}
        source_weights = weights or {}

        for results in result_lists:
            for rank, result in enumerate(results, start=1):
                key = result.chunk.id or result.chunk.content[:50]
                weight = source_weights.get(result.source, 1.0)
                scores[key] = scores.get(key, 0.0) + weight * (1.0 / (self.rrf_k + rank))
                if key not in result_map:
                    result_map[key] = result

        sorted_keys = sorted(scores, key=lambda item: scores[item], reverse=True)[:top_k]
        return [
            FusionView(
                result=result_map[key],
                fusion_score=scores[key],
                fusion_rank=index,
            )
            for index, key in enumerate(sorted_keys, start=1)
        ]

    def fuse(
        self,
        *result_lists: list[SearchResult],
        top_k: int,
        weights: dict[str, float] | None = None,
    ) -> list[SearchResult]:
        views = self.build_views(*result_lists, top_k=top_k, weights=weights)
        return [
            view.result.model_copy(
                update={
                    "score": view.fusion_score,
                    "score_normalized": None,
                    "rank": view.fusion_rank,
                    "source": view.fused_source,
                }
            )
            for view in views
        ]
