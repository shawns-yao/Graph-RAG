from rag_core.models import Chunk, QueryType, SearchResult

from agentic_graph_rag.retrieval.fusion import FusionEngine, resolve_channel_priority


def _result(
    chunk_id: str,
    source: str,
    rank: int = 1,
    *,
    score_normalized: float | None = 0.9,
) -> SearchResult:
    return SearchResult(
        chunk=Chunk(id=chunk_id, content=f"content {chunk_id}"),
        score=1.0 / rank,
        score_normalized=score_normalized,
        rank=rank,
        source=source,
    )


def test_resolve_channel_priority_uses_query_type_order_without_weights():
    assert resolve_channel_priority(QueryType.RELATION) == ["graph", "vector", "bm25"]
    assert resolve_channel_priority(QueryType.TEMPORAL) == ["bm25", "vector", "graph"]


def test_priority_fusion_keeps_primary_channel_order_and_appends_deduped_supplements():
    engine = FusionEngine()

    fused = engine.fuse(
        [_result("v1", "vector", 1), _result("shared", "vector", 2)],
        [_result("b1", "bm25", 1), _result("shared", "bm25", 2)],
        [_result("g1", "graph", 1), _result("g2", "graph", 2)],
        top_k=5,
        query_type=QueryType.RELATION,
    )

    assert [result.chunk.id for result in fused] == ["g1", "g2", "v1", "shared", "b1"]
    assert [result.rank for result in fused] == [1, 2, 3, 4, 5]
    assert [result.score for result in fused] == [1.0, 0.5, 1.0, 0.5, 1.0]
    assert all(result.source == "hybrid" for result in fused)


def test_priority_fusion_can_be_restricted_to_enabled_channels():
    engine = FusionEngine()

    fused = engine.fuse(
        [_result("v1", "vector", 1)],
        [_result("b1", "bm25", 1)],
        [_result("g1", "graph", 1)],
        top_k=3,
        query_type=QueryType.RELATION,
        channel_priority=["bm25", "vector"],
    )

    assert [result.chunk.id for result in fused] == ["b1", "v1"]


def test_priority_fusion_preserves_max_upstream_normalized_signal_for_duplicates():
    engine = FusionEngine()

    fused = engine.fuse(
        [_result("shared", "vector", 1, score_normalized=0.65)],
        [_result("shared", "bm25", 1, score_normalized=0.91)],
        top_k=1,
        query_type=QueryType.SIMPLE,
    )

    assert len(fused) == 1
    assert fused[0].chunk.id == "shared"
    assert fused[0].score == 1.0
    assert fused[0].score_normalized == 0.91
