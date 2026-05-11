"""Tests for skeleton indexing safeguards."""

from rag_core.models import Chunk

from agentic_graph_rag.indexing.skeleton import build_knn_graph, filter_low_information_chunks


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


def test_filter_low_information_chunks_preserves_short_numeric_facts():
    chunks = [
        Chunk(id="fact", content="eGFR < 30"),
        Chunk(id="noise1", content="the and or"),
        Chunk(id="noise2", content="the and or"),
    ]
    embeddings = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.9]]

    kept, kept_embeddings, dropped = filter_low_information_chunks(chunks, embeddings)

    assert [chunk.id for chunk in kept] == ["fact"]
    assert kept_embeddings == [[1.0, 0.0]]
    assert {chunk.id for chunk in dropped} == {"noise1", "noise2"}
    assert chunks[0].metadata["low_information_chunk"] is False
    assert "tfidf_signal_score" not in chunks[0].metadata
    assert chunks[0].metadata["local_information_score"] >= 0.0
