"""Tests for PipelineService."""
from unittest.mock import MagicMock, patch

from rag_core.models import Chunk, PipelineTrace, QAResult, ReflectionStep, SearchResult, WorkflowMemoryEntry


def _mock_qa():
    return QAResult(answer="test answer", query="q", sources=[
        SearchResult(chunk=Chunk(id="c1", content="text"), score=0.9, rank=1),
    ], confidence=0.8)


@patch("agentic_graph_rag.service.agent_run")
def test_service_query_returns_qa_with_trace(mock_run):
    mock_run.return_value = _mock_qa()
    mock_run.return_value.trace = MagicMock(trace_id="tr_abc")

    from agentic_graph_rag.service import PipelineService

    svc = PipelineService(driver=MagicMock(), openai_client=MagicMock())
    qa = svc.query("test question")
    assert qa.answer == "test answer"
    mock_run.assert_called_once()


@patch("agentic_graph_rag.service.agent_run")
def test_service_caches_trace(mock_run):
    qa = _mock_qa()
    qa.trace = PipelineTrace(trace_id="tr_cached", timestamp="T", query="q")
    mock_run.return_value = qa

    from agentic_graph_rag.service import PipelineService

    svc = PipelineService(driver=MagicMock(), openai_client=MagicMock())
    svc.query("test")
    assert svc.get_trace("tr_cached") is not None
    assert svc.get_trace("nonexistent") is None


@patch("agentic_graph_rag.service.agent_run")
def test_service_reuses_session_memory_and_history(mock_run):
    prior_trace = PipelineTrace(
        trace_id="tr_prior",
        timestamp="T0",
        query="old q",
        final_answer="old a",
        session_id="sess-1",
        workflow_memory=[
            WorkflowMemoryEntry(stage="retrieval", message="old bad case"),
        ],
        reflection_steps=[
            ReflectionStep(
                tool_name="vector_search",
                overall_score=1.0,
                failure_type="insufficient_context",
            )
        ],
    )
    current_qa = _mock_qa()
    current_qa.trace = PipelineTrace(
        trace_id="tr_new",
        timestamp="T1",
        query="new q",
        session_id="sess-1",
    )
    mock_run.return_value = current_qa

    from agentic_graph_rag.service import PipelineService
    from agentic_graph_rag.trace_store import InMemoryTraceStore

    store = InMemoryTraceStore()
    store.put(prior_trace)
    svc = PipelineService(
        driver=MagicMock(),
        openai_client=MagicMock(),
        trace_store=store,
    )
    qa = svc.query("new question", session_id="sess-1")

    assert qa.trace is not None
    assert qa.trace.session_id == "sess-1"
    call_kwargs = mock_run.call_args.kwargs
    assert call_kwargs["session_id"] == "sess-1"
    assert call_kwargs["workflow_memory_seed"][0].message == "old bad case"
    assert call_kwargs["reflection_history_seed"][0].failure_type == "insufficient_context"
    assert [trace.trace_id for trace in store.get_session_traces("sess-1")] == [
        "tr_prior",
        "tr_new",
    ]


@patch("agentic_graph_rag.service.agent_run")
def test_service_contextualizes_follow_up_query(mock_run):
    prior_trace = PipelineTrace(
        trace_id="tr_prior",
        timestamp="T0",
        query="二型糖尿病常用什么药？",
        final_answer="常见药物包括二甲双胍，也会根据病情选择其他降糖药。",
        session_id="sess-2",
    )
    current_qa = _mock_qa()
    current_qa.trace = PipelineTrace(
        trace_id="tr_new",
        timestamp="T1",
        query="placeholder",
        session_id="sess-2",
    )
    mock_run.return_value = current_qa

    from agentic_graph_rag.service import PipelineService
    from agentic_graph_rag.trace_store import InMemoryTraceStore

    store = InMemoryTraceStore()
    store.put(prior_trace)
    svc = PipelineService(
        driver=MagicMock(),
        openai_client=MagicMock(),
        trace_store=store,
    )
    qa = svc.query("它的副作用呢？", session_id="sess-2")

    routed_query = mock_run.call_args.args[0]
    assert "Conversation history:" in routed_query
    assert "二型糖尿病常用什么药？" in routed_query
    assert "常见药物包括二甲双胍" in routed_query
    assert "它的副作用呢？" in routed_query
    assert qa.query == "它的副作用呢？"
    assert qa.expanded_query == routed_query
    assert qa.trace is not None
    assert qa.trace.query == "它的副作用呢？"
    assert qa.trace.expanded_query == routed_query


def test_service_health():
    from agentic_graph_rag.service import PipelineService

    driver = MagicMock()
    session = MagicMock()
    session.run.return_value.single.return_value = [1]
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)

    svc = PipelineService(driver=driver, openai_client=MagicMock())
    health = svc.health()
    assert health["status"] == "ok"


def test_service_search_dispatches_tool():
    from unittest.mock import patch as _patch

    from rag_core.models import Chunk, SearchResult

    from agentic_graph_rag.service import PipelineService

    fake_result = [SearchResult(chunk=Chunk(id="c1", content="found"), score=0.9, rank=1)]

    driver = MagicMock()
    client = MagicMock()
    svc = PipelineService(driver=driver, openai_client=client)
    with _patch("agentic_graph_rag.agent.tools.vector_search", return_value=fake_result) as mock_vs:
        results = svc.search("test query", tool="vector_search")
        assert results == fake_result
        mock_vs.assert_called_once_with("test query", driver, client)


def test_service_search_unknown_tool():
    import pytest as _pytest

    from agentic_graph_rag.service import PipelineService

    svc = PipelineService(driver=MagicMock(), openai_client=MagicMock())
    with _pytest.raises(ValueError, match="Unknown tool"):
        svc.search("test", tool="nonexistent_tool")


@patch("agentic_graph_rag.service.agent_run")
def test_service_stream_query_yields_status_tokens_and_done(mock_agent_run):
    from agentic_graph_rag.service import PipelineService

    qa = _mock_qa()
    qa.answer = "Hello world"
    qa.retries = 1
    qa.trace = PipelineTrace(trace_id="tr_stream", timestamp="T", query="q")
    qa.trace.router_step = MagicMock()
    qa.trace.router_step.decision = MagicMock(suggested_tool="vector_search")
    mock_agent_run.return_value = qa

    svc = PipelineService(driver=MagicMock(), openai_client=MagicMock())

    events = list(svc.stream_query("test question", session_id="sess-1"))

    assert [event["event"] for event in events[:3]] == [
        "status",
        "status",
        "status",
    ]
    assert events[-1]["event"] == "done"
    token_events = [event for event in events if event["event"] == "token"]
    assert events[0]["data"]["stage"] == "routing_started"
    assert events[1]["data"]["stage"] == "retrieval_started"
    assert events[2]["data"]["stage"] == "generation_started"
    assert "".join(event["data"]["text"] for event in token_events) == "Hello world"
    assert events[-1]["data"]["answer"] == "Hello world"
    assert events[-1]["data"]["confidence"] == qa.confidence
    assert events[-1]["data"]["session_id"] == "sess-1"
    assert events[-1]["data"]["retries"] == 1
    assert "trace_id" in events[-1]["data"]
    mock_agent_run.assert_called_once()


def test_stream_output_adapter_converts_qa_result_to_event_contract():
    from agentic_graph_rag.service import PipelineService

    qa = _mock_qa()
    qa.answer = "OK"
    qa.retries = 2
    qa.trace = PipelineTrace(trace_id="tr_adapter", timestamp="T", query="q")
    qa.trace.router_step = MagicMock()
    qa.trace.router_step.decision = MagicMock(suggested_tool="hybrid_search")

    events = list(
        PipelineService._adapt_qa_result_to_stream_events(
            qa,
            text="question",
            session_id="sess-adapter",
        )
    )

    assert [event["event"] for event in events] == [
        "status",
        "status",
        "token",
        "token",
        "done",
    ]
    assert events[0]["data"] == {
        "stage": "retrieval_started",
        "tool": "hybrid_search",
    }
    assert events[1]["data"] == {
        "stage": "generation_started",
        "sources": 1,
    }
    assert "".join(event["data"]["text"] for event in events[2:4]) == "OK"
    assert events[-1]["data"]["trace_id"] == "tr_adapter"
    assert events[-1]["data"]["session_id"] == "sess-adapter"


def test_service_trace_cache_bounded():
    from agentic_graph_rag.service import PipelineService

    svc = PipelineService(driver=MagicMock(), openai_client=MagicMock())
    from rag_core.models import PipelineTrace

    # Fill cache beyond limit
    for i in range(105):
        trace = PipelineTrace(trace_id=f"tr_{i:04d}", timestamp="T", query="q")
        svc._cache_trace(trace)

    # Oldest should be evicted (cache max 100)
    assert svc.get_trace("tr_0000") is None
    assert svc.get_trace("tr_0104") is not None
