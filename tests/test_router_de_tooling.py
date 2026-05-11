from rag_core.models import QueryType

from agentic_graph_rag.agent.router import classify_query


def test_short_query_no_longer_uses_length_hard_rule():
    decision = classify_query("噻托溴铵", use_llm=False)

    assert decision.query_type == QueryType.SIMPLE
    assert decision.suggested_tool == "vector_search"
    assert not decision.reasoning.startswith("Hard rule: short factual query")


def test_long_query_no_longer_forces_global_by_length():
    query = " ".join(f"term{i}" for i in range(60))

    decision = classify_query(query, use_llm=False)

    assert decision.query_type == QueryType.SIMPLE
    assert decision.suggested_tool == "vector_search"
    assert "long query detected" not in decision.reasoning


def test_lexical_anchor_no_longer_forces_bm25_router_tool():
    decision = classify_query("ERR-42", use_llm=False)

    assert decision.query_type == QueryType.SIMPLE
    assert decision.suggested_tool == "vector_search"
    assert "lexical anchor detected" not in decision.reasoning
