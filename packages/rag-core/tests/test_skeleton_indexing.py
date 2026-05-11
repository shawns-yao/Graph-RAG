"""Tests for skeleton indexing safeguards."""

from rag_core.models import Chunk

from agentic_graph_rag.indexing.skeleton import build_knn_graph


def _chunk(idx: int) -> Chunk:
    return Chunk(id=f"c{idx}", content=f"chunk {idx}")


def test_build_knn_graph_uses_local_adaptive_threshold_not_absolute_similarity():
    chunks = [_chunk(i) for i in range(4)]
    embeddings = [
        [1.0, 0.0],
        [0.6, 0.8],
        [0.0, 1.0],
        [-1.0, 0.0],
    ]

    graph = build_knn_graph(chunks, embeddings, k=2)

    assert graph.has_edge(0, 1)
    assert graph[0][1]["weight"] < 0.7
    assert not graph.has_edge(0, 2)
