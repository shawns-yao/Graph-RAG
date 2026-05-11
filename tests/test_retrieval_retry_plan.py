from rag_core.models import QueryType, ReflectionStep, RouterDecision

from agentic_graph_rag.agent.retrieval_agent import _get_next_tool


def _reflection(**updates) -> ReflectionStep:
    data = {
        "attempt": 0,
        "tool_name": "vector_search",
        "query_used": "短 query",
        "evidence_status": "partial",
        "gap_type": "missing_relation",
        "action": "retry_graph",
        "required_tool": "none",
        "verdict": "retry",
        "failure_type": "relation_missing",
        "recommended_action": "",
    }
    data.update(updates)
    return ReflectionStep(**data)


def test_retry_plan_uses_gap_type_before_query_shape_rules():
    next_tool = _get_next_tool(
        "vector_search",
        {"vector_search"},
        _reflection(gap_type="missing_relation", failure_type="relation_missing"),
        RouterDecision(query_type=QueryType.SIMPLE, suggested_tool="vector_search"),
    )

    assert next_tool == "cypher_traverse"


def test_retry_plan_uses_bm25_for_numeric_gap():
    next_tool = _get_next_tool(
        "vector_search",
        {"vector_search"},
        _reflection(
            gap_type="missing_numeric_threshold",
            failure_type="insufficient_context",
        ),
        RouterDecision(query_type=QueryType.SIMPLE, suggested_tool="vector_search"),
    )

    assert next_tool == "bm25_search"


def test_retry_plan_honors_required_tool_first():
    next_tool = _get_next_tool(
        "vector_search",
        {"vector_search"},
        _reflection(
            gap_type="missing_entity",
            failure_type="missing_entity",
            required_tool="hybrid_search",
        ),
        RouterDecision(query_type=QueryType.SIMPLE, suggested_tool="vector_search"),
    )

    assert next_tool == "hybrid_search"

