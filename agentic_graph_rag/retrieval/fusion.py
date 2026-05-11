"""Fusion primitives for heterogeneous retrieval results."""

from __future__ import annotations

from dataclasses import dataclass

from rag_core.models import SearchResult

_DEFAULT_CHANNEL_PRIORITY = ["vector", "bm25", "graph"]


def resolve_channel_priority(
    query_type: object | None = None,
    enabled_channels: list[str] | None = None,
) -> list[str]:
    """Resolve deterministic channel order without numeric fusion weights."""
    del query_type
    priority = list(_DEFAULT_CHANNEL_PRIORITY)
    if enabled_channels is not None:
        enabled = set(enabled_channels)
        priority = [channel for channel in priority if channel in enabled]
        priority.extend(channel for channel in enabled_channels if channel not in priority)
    return priority


def _result_key(result: SearchResult) -> str:
    if result.chunk.id:
        return result.chunk.id
    return " ".join(result.chunk.content.casefold().split())[:240]


def _collect_normalized_scores(
    result_lists: tuple[list[SearchResult], ...],
) -> dict[str, list[float]]:
    normalized_scores: dict[str, list[float]] = {}
    for results in result_lists:
        for result in results:
            if result.score_normalized is None:
                continue
            normalized_scores.setdefault(_result_key(result), []).append(result.score_normalized)
    return normalized_scores


@dataclass(frozen=True, slots=True)
class FusionView:
    """Read-only fusion view over an upstream SearchResult."""

    result: SearchResult
    fusion_rank: int
    fused_source: str = "hybrid"
    preserved_normalized: float | None = None


@dataclass(slots=True)
class FusionEngine:
    """Priority-ordered channel merge with deterministic dedupe."""

    def build_views(
        self,
        *result_lists: list[SearchResult],
        top_k: int,
        query_type: object | None = None,
        channel_priority: list[str] | None = None,
    ) -> list[FusionView]:
        """Merge channels by priority, preserving each provider's internal order."""
        by_source: dict[str, list[SearchResult]] = {}
        for results in result_lists:
            for result in results:
                by_source.setdefault(result.source, []).append(result)

        if channel_priority is None:
            priority = resolve_channel_priority(query_type)
            priority.extend(source for source in by_source if source not in priority)
        else:
            priority = list(channel_priority)

        normalized_scores = _collect_normalized_scores(result_lists)
        selected: list[SearchResult] = []
        seen: set[str] = set()
        for source in priority:
            for result in by_source.get(source, []):
                key = _result_key(result)
                if not key or key in seen:
                    continue
                seen.add(key)
                selected.append(result)
                if len(selected) >= top_k:
                    break
            if len(selected) >= top_k:
                break

        views: list[FusionView] = []
        for index, result in enumerate(selected, start=1):
            upstream_normalized = normalized_scores.get(_result_key(result))
            preserved = max(upstream_normalized) if upstream_normalized else None
            views.append(
                FusionView(
                    result=result,
                    fusion_rank=index,
                    preserved_normalized=preserved,
                )
            )
        return views

    def fuse(
        self,
        *result_lists: list[SearchResult],
        top_k: int,
        query_type: object | None = None,
        channel_priority: list[str] | None = None,
    ) -> list[SearchResult]:
        views = self.build_views(
            *result_lists,
            top_k=top_k,
            query_type=query_type,
            channel_priority=channel_priority,
        )
        return [
            view.result.model_copy(
                update={
                    "score_normalized": view.preserved_normalized,
                    "rank": view.fusion_rank,
                    "source": view.fused_source,
                }
            )
            for view in views
        ]
