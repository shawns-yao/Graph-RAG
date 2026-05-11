"""Orchestrates pluggable retrieval providers, fusion, and reranking."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, wait

from rag_core.config import get_settings
from rag_core.models import QueryType, SearchResult
from rag_core.reranker import rerank

from agentic_graph_rag.retrieval.fusion import FusionEngine, resolve_channel_priority
from agentic_graph_rag.retrieval.providers import (
    RetrievalProvider,
    RetrievalRequest,
    attach_passage_embeddings,
)


class RetrievalOrchestrator:
    """Parallel fan-out + fusion pipeline across retrieval providers."""

    def __init__(
        self,
        driver,
        providers: list[RetrievalProvider],
        fusion_engine: FusionEngine | None = None,
    ) -> None:
        self._driver = driver
        self._providers = providers
        self._cfg = get_settings().retrieval
        self._fusion_engine = fusion_engine or FusionEngine()

    def search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int,
        query_type: QueryType | str | None = None,
        provider_top_k: dict[str, int] | None = None,
        provider_filters: dict[str, dict] | None = None,
        seed_results: dict[str, list[SearchResult]] | None = None,
        enabled_providers: list[str] | None = None,
        provider_results: dict[str, list[SearchResult]] | None = None,
        rerank_enabled: bool = True,
    ) -> list[SearchResult]:
        """Execute providers in parallel, merge by channel priority, then rerank."""
        if not self._providers:
            return []

        provider_top_k = provider_top_k or {}
        provider_filters = provider_filters or {}
        seed_results = seed_results or {}
        selected_providers = self._select_providers(enabled_providers, seed_results)
        if not selected_providers:
            return []

        candidate_top_k = max(top_k * 3, *(provider_top_k.values() or [top_k]))
        requests = {
            provider.name: RetrievalRequest(
                query=query,
                query_embedding=query_embedding,
                top_k=provider_top_k.get(provider.name, top_k),
                filters=provider_filters.get(provider.name, {}),
            )
            for provider in selected_providers
            if provider.name not in seed_results
        }

        fanout_results = dict(seed_results)
        fanout_results.update(self._fanout(requests, selected_providers))
        if provider_results is not None:
            provider_results.clear()
            provider_results.update({
                provider.name: list(fanout_results.get(provider.name, []))
                for provider in selected_providers
            })
        channel_priority = resolve_channel_priority(
            query_type,
            [provider.name for provider in selected_providers],
        )
        fused = self._fusion_engine.fuse(
            *(fanout_results.get(provider.name, []) for provider in selected_providers),
            top_k=candidate_top_k,
            query_type=query_type,
            channel_priority=channel_priority,
        )
        if not fused:
            return []

        fused = attach_passage_embeddings(fused, self._driver)
        if rerank_enabled:
            ranked = rerank(query, fused, top_k=top_k)
        else:
            ranked = fused[:top_k]
        return self._finalize_results(ranked[:top_k])

    @staticmethod
    def _finalize_results(results: list[SearchResult]) -> list[SearchResult]:
        return [
            result.model_copy(
                update={
                    "source": "hybrid",
                }
            )
            for result in results
        ]

    def _select_providers(
        self,
        enabled_providers: list[str] | None,
        seed_results: dict[str, list[SearchResult]],
    ) -> list[RetrievalProvider]:
        """Resolve providers participating in this retrieval pass."""
        enabled = None if enabled_providers is None else set(enabled_providers)
        return [
            provider
            for provider in self._providers
            if enabled is None
            or provider.name in enabled
            or provider.name in seed_results
        ]

    def _fanout(
        self,
        requests: dict[str, RetrievalRequest],
        providers: list[RetrievalProvider],
    ) -> dict[str, list[SearchResult]]:
        """Run all providers in parallel and degrade on timeouts/failures."""
        if not requests:
            return {}

        provider_map = {provider.name: provider for provider in providers}
        executor = ThreadPoolExecutor(max_workers=self._cfg.fanout_max_workers)
        try:
            future_map: dict[Future, str] = {
                executor.submit(provider_map[name].retrieve, request): name
                for name, request in requests.items()
            }
            return _wait_for_future_results(
                future_map,
                timeout_ms=self._cfg.fanout_timeout_ms,
            )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)


def _wait_for_future_results(
    future_map: dict[Future, str],
    timeout_ms: int,
) -> dict[str, list[SearchResult]]:
    """Collect provider results with timeout-based graceful degradation."""
    results: dict[str, list[SearchResult]] = {}
    timeout_s = max(timeout_ms, 1) / 1000.0
    done, not_done = wait(set(future_map), timeout=timeout_s)

    for future in done:
        provider_name = future_map[future]
        try:
            results[provider_name] = future.result()
        except Exception:  # pragma: no cover - defensive degradation
            results[provider_name] = []

    for future in not_done:
        provider_name = future_map[future]
        future.cancel()
        results[provider_name] = []

    return results
