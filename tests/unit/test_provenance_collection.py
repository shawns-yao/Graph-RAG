"""Tests for provenance collection in retrieval agent."""
from unittest.mock import MagicMock, patch

from rag_core.models import (
    Chunk,
    PipelineTrace,
    QAResult,
    QueryType,
    ReflectionStep,
    RouterDecision,
    SearchResult,
)


def _make_results(n=3):
    return [
        SearchResult(chunk=Chunk(id=f"c{i}", content=f"text {i}"), score=0.9 - i * 0.1, rank=i + 1)
        for i in range(n)
    ]


def _make_decision(tool="vector_search", qtype=QueryType.SIMPLE):
    return RouterDecision(
        query_type=qtype, confidence=0.5, reasoning="test", suggested_tool=tool,
    )


@patch(
    "agentic_graph_rag.agent.retrieval_agent.evaluate_reflection",
    return_value=ReflectionStep(overall_score=4.0, failure_type="acceptable"),
)
@patch("agentic_graph_rag.agent.retrieval_agent._TOOL_REGISTRY")
def test_self_correction_loop_records_tool_step(mock_registry, mock_eval):
    mock_fn = MagicMock(return_value=_make_results())
    mock_registry.__getitem__ = MagicMock(return_value=mock_fn)
    mock_registry.__contains__ = MagicMock(return_value=True)

    from agentic_graph_rag.agent.retrieval_agent import self_correction_loop

    decision = _make_decision()
    trace = PipelineTrace(trace_id="tr_test", timestamp="T", query="q")

    results, retries = self_correction_loop(
        "q", MagicMock(), MagicMock(), decision, trace=trace,
    )
    assert len(trace.tool_steps) >= 1
    assert trace.tool_steps[0].tool_name == "vector_search"
    assert trace.tool_steps[0].results_count == 3
    assert trace.tool_steps[0].relevance_score == 4.0
    assert len(trace.reflection_steps) == 1


@patch(
    "agentic_graph_rag.agent.retrieval_agent.evaluate_reflection",
    return_value=ReflectionStep(
        overall_score=1.0,
        failure_type="insufficient_recall",
        recommended_action="expand_recall",
    ),
)
@patch("agentic_graph_rag.agent.retrieval_agent.generate_retry_query", return_value="rephrased")
@patch("agentic_graph_rag.agent.retrieval_agent._TOOL_REGISTRY")
def test_escalation_recorded_in_trace(mock_registry, mock_retry, mock_eval):
    mock_fn = MagicMock(return_value=_make_results())
    mock_registry.__getitem__ = MagicMock(return_value=mock_fn)
    mock_registry.__contains__ = MagicMock(return_value=True)

    from agentic_graph_rag.agent.retrieval_agent import self_correction_loop

    decision = _make_decision()
    trace = PipelineTrace(trace_id="tr_esc", timestamp="T", query="q")

    results, retries = self_correction_loop(
        "q", MagicMock(), MagicMock(), decision, trace=trace,
        max_retries=1, relevance_threshold=3.0,
    )
    assert len(trace.escalation_steps) >= 1
    assert trace.escalation_steps[0].from_tool == "vector_search"
    assert len(trace.reflection_steps) >= 1


@patch("agentic_graph_rag.agent.retrieval_agent.classify_query")
@patch("agentic_graph_rag.agent.retrieval_agent.self_correction_loop")
@patch("agentic_graph_rag.agent.retrieval_agent.generate_answer")
def test_run_returns_qa_with_trace(mock_gen, mock_loop, mock_classify):
    mock_classify.return_value = _make_decision()
    mock_loop.return_value = (_make_results(), 0)
    mock_gen.return_value = QAResult(answer="test", query="q", sources=_make_results())

    from agentic_graph_rag.agent.retrieval_agent import run

    qa = run("q", MagicMock(), MagicMock())
    assert qa.trace is not None
    assert qa.trace.trace_id.startswith("tr_")
    assert qa.trace.router_step is not None


@patch(
    "agentic_graph_rag.agent.retrieval_agent.evaluate_reflection",
    return_value=ReflectionStep(overall_score=4.0, failure_type="acceptable"),
)
@patch("agentic_graph_rag.agent.retrieval_agent._TOOL_REGISTRY")
def test_hybrid_tool_step_records_provider_diagnostics(mock_registry, mock_eval):
    def _hybrid_tool(_query, _driver, _client, **kwargs):
        kwargs["provider_results"].update({
            "vector": _make_results(2),
            "bm25": _make_results(1),
            "graph": [],
        })
        return _make_results(3)

    mock_registry.__getitem__ = MagicMock(return_value=_hybrid_tool)
    mock_registry.__contains__ = MagicMock(return_value=True)

    from agentic_graph_rag.agent.retrieval_agent import self_correction_loop

    decision = _make_decision(tool="hybrid_search")
    trace = PipelineTrace(trace_id="tr_hybrid", timestamp="T", query="q")

    _results, _retries = self_correction_loop(
        "q", MagicMock(), MagicMock(), decision, trace=trace,
    )
    assert trace.tool_steps[0].tool_name == "hybrid_search"
    diagnostics = {item.source: item for item in trace.tool_steps[0].provider_diagnostics}
    assert diagnostics["vector"].results_count == 2
    assert diagnostics["bm25"].results_count == 1
    assert diagnostics["graph"].results_count == 0
