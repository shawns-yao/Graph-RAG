"""Tests for small-model retrieval need resolution."""

from agentic_graph_rag.agent.need_resolver import coerce_need_resolution


def test_coerce_need_resolution_keeps_allowlisted_needs():
    resolution = coerce_need_resolution(
        {
            "retrieval_needs": [
                "graph_relation",
                "semantic_passage",
                "bad_tool_name",
                "exact_numeric",
                "graph_relation",
            ],
            "reason": "needs graph and exact evidence",
        }
    )

    assert resolution.retrieval_needs == (
        "semantic_passage",
        "graph_relation",
        "exact_numeric",
    )
    assert resolution.reason == "needs graph and exact evidence"


def test_coerce_need_resolution_falls_back_on_invalid_shape():
    resolution = coerce_need_resolution({"retrieval_needs": "cypher_traverse"})

    assert resolution.retrieval_needs == ("semantic_passage",)
    assert "fallback" in resolution.reason
