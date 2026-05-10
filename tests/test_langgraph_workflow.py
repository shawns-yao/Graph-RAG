"""Tests for the LangGraph self-correction workflow."""

from unittest.mock import MagicMock, patch

import pytest

from rag_core.models import (
    Chunk,
    PipelineTrace,
    QAResult,
    QueryType,
    ReflectionStep,
    RouterDecision,
    SearchResult,
    WorkflowMemoryEntry,
)

from agentic_graph_rag.agent.langgraph_workflow import (
    AgentWorkflowOps,
    SelfCorrectionOps,
    _evaluate_reflection_node,
    _interpret_verdict_node,
    run_agent_workflow,
    run_self_correction_workflow,
)


def _make_results(prefix: str, count: int) -> list[SearchResult]:
    return [
        SearchResult(
            chunk=Chunk(id=f"{prefix}-{index}", content=f"{prefix} content {index}"),
            score=0.9 - index * 0.1,
            rank=index + 1,
        )
        for index in range(count)
    ]


def _make_decision(
    tool: str = "vector_search",
    query_type: QueryType = QueryType.SIMPLE,
) -> RouterDecision:
    return RouterDecision(
        query_type=query_type,
        confidence=0.8,
        reasoning="test",
        suggested_tool=tool,
    )


def _make_settings() -> MagicMock:
    settings = MagicMock()
    settings.agent.reflection_skip_score_threshold = 0.85
    settings.retrieval.reflection_score_scale = 5.0
    return settings


class _FakeWorkflowRuntime:
    def __init__(
        self,
        *,
        tool_results: dict[str, list[SearchResult]],
        reflections: list[ReflectionStep],
        next_tools: dict[str, str | None] | None = None,
        rewrite_query: str = "rewritten query",
        rewrite_enabled: bool = False,
        reranked_results: list[SearchResult] | None = None,
    ) -> None:
        self.tool_results = tool_results
        self.reflections = list(reflections)
        self.next_tools = next_tools or {}
        self.rewrite_query = rewrite_query
        self.rewrite_enabled = rewrite_enabled
        self.reranked_results = reranked_results
        self.run_calls: list[tuple[str, str]] = []
        self.reflection_calls = 0
        self.retry_query_calls: list[str] = []
        self.rerank_calls: list[tuple[str, list[str]]] = []

    def as_ops(self) -> SelfCorrectionOps:
        return SelfCorrectionOps(
            execution_sources=self.execution_sources,
            run_tool=self.run_tool,
            cache_tool_results=self.cache_tool_results,
            cache_hybrid_provider_results=self.cache_hybrid_provider_results,
            build_provider_diagnostics=self.build_provider_diagnostics,
            evaluate_reflection=self.evaluate_reflection,
            plan_incremental_retry=self.plan_incremental_retry,
            get_next_tool=self.get_next_tool,
            should_rewrite_query=self.should_rewrite_query,
            generate_retry_query=self.generate_retry_query,
            rerank_results=self.rerank_results,
        )

    def execution_sources(self, tool_name, query, channel_cache, forced_hybrid_providers):
        del query, channel_cache, forced_hybrid_providers
        return [], [tool_name]

    def run_tool(
        self,
        tool_name,
        query,
        driver,
        openai_client,
        decision,
        *,
        channel_cache=None,
        hybrid_enabled_providers=None,
        provider_results_sink=None,
    ):
        del driver, openai_client, decision, channel_cache, hybrid_enabled_providers, provider_results_sink
        self.run_calls.append((tool_name, query))
        return self.tool_results[tool_name]

    def cache_tool_results(self, query, tool_name, results, channel_cache):
        del tool_name, results
        channel_cache.setdefault(query, {})

    def cache_hybrid_provider_results(self, query, provider_results, channel_cache):
        del provider_results
        channel_cache.setdefault(query, {})

    def build_provider_diagnostics(self, provider_results, reused_sources, executed_sources):
        del provider_results, reused_sources, executed_sources
        return []

    def evaluate_reflection(
        self,
        query,
        results,
        *,
        openai_client,
        reflection_history,
        workflow_memory,
        tool_name,
        attempt,
    ):
        del query, results, openai_client, reflection_history, workflow_memory, tool_name, attempt
        self.reflection_calls += 1
        return self.reflections.pop(0)

    def plan_incremental_retry(self, query, current_tool, reflection, channel_cache, provider_results):
        del query, current_tool, reflection, channel_cache, provider_results
        return None

    def get_next_tool(self, current_tool, tried_tools, reflection, decision):
        del tried_tools, reflection, decision
        return self.next_tools.get(current_tool)

    def should_rewrite_query(self, next_tool, reflection, cached_sources_reused):
        del next_tool, reflection, cached_sources_reused
        return self.rewrite_enabled

    def generate_retry_query(
        self,
        query,
        results,
        *,
        openai_client,
        reflection,
        reflection_history,
        workflow_memory,
    ):
        del results, openai_client, reflection, reflection_history, workflow_memory
        self.retry_query_calls.append(query)
        return self.rewrite_query

    def rerank_results(self, query, results, *, openai_client):
        del openai_client
        self.rerank_calls.append((query, [result.chunk.id for result in results]))
        if self.reranked_results is not None:
            return self.reranked_results
        return list(reversed(results))


class TestLangGraphSelfCorrectionWorkflow:
    def test_reflection_evaluation_and_verdict_interpretation_are_separate_nodes(self):
        vector_results = _make_results("vector", 2)
        reflection = ReflectionStep(
            overall_score=1.0,
            verdict="retry",
            failure_type="insufficient_recall",
            recommended_action="expand_recall",
            should_retry=True,
            retry_scope="tool_escalation",
        )
        runtime = _FakeWorkflowRuntime(
            tool_results={"vector_search": vector_results},
            reflections=[reflection],
        )
        state = {
            "ops": runtime.as_ops(),
            "openai_client": MagicMock(),
            "current_query": "test query",
            "current_tool": "vector_search",
            "attempt": 0,
            "results": vector_results,
            "reflection_history": [],
            "memory": [],
            "best_results": [],
            "best_score": 0.0,
            "best_attempt": 0,
            "best_rank": (-1, -1, -1, -10**9),
            "trace": None,
            "relevance_threshold": 3.0,
            "max_retries": 2,
            "total_reranks": 0,
            "max_reranks": 1,
        }

        evaluated = _evaluate_reflection_node(state)

        assert evaluated["pending_reflection"] == reflection
        assert "reflection_history" not in evaluated
        assert "last_reflection" not in evaluated

        interpreted = _interpret_verdict_node({**state, **evaluated})

        assert interpreted["last_reflection"] == reflection
        assert interpreted["reflection_history"] == [reflection]
        assert interpreted["next_step"] == "prepare_retry"

    def test_returns_after_first_successful_reflection(self):
        vector_results = _make_results("vector", 2)
        runtime = _FakeWorkflowRuntime(
            tool_results={"vector_search": vector_results},
            reflections=[
                ReflectionStep(
                    overall_score=4.0,
                    failure_type="acceptable",
                    recommended_action="answer_ready",
                )
            ],
        )

        results, retries = run_self_correction_workflow(
            query="test query",
            driver=MagicMock(),
            openai_client=MagicMock(),
            decision=_make_decision(),
            max_retries=2,
            relevance_threshold=3.0,
            trace=None,
            ops=runtime.as_ops(),
        )

        assert results == vector_results
        assert retries == 0
        assert runtime.run_calls == [("vector_search", "test query")]

    def test_relation_query_falls_back_to_graph_when_reflection_transport_fails(self):
        vector_results = _make_results("vector", 2)
        graph_results = _make_results("graph", 3)
        runtime = _FakeWorkflowRuntime(
            tool_results={
                "vector_search": vector_results,
                "cypher_traverse": graph_results,
            },
            reflections=[
                ReflectionStep(
                    evidence_status="insufficient",
                    gap_type="off_topic",
                    action="stop",
                    required_tool="none",
                    verdict="stop",
                    overall_score=0.0,
                    failure_type="insufficient_context",
                    recommended_action="stop_due_to_invalid_reflection",
                    should_retry=False,
                    retry_scope="stop",
                    reasoning="Reflection policy guard: Error code: 522 connection timed out retryable.",
                ),
                ReflectionStep(
                    evidence_status="sufficient",
                    gap_type="none",
                    action="answer",
                    required_tool="none",
                    verdict="answer",
                    overall_score=5.0,
                    failure_type="acceptable",
                    recommended_action="answer_ready",
                    should_retry=False,
                    retry_scope="stop",
                ),
            ],
            next_tools={"vector_search": "cypher_traverse"},
        )

        results, retries = run_self_correction_workflow(
            query="FEV1和肺功能检查有什么关系？",
            driver=MagicMock(),
            openai_client=MagicMock(),
            decision=_make_decision(query_type=QueryType.RELATION),
            max_retries=2,
            relevance_threshold=3.0,
            trace=None,
            ops=runtime.as_ops(),
        )

        assert results == graph_results
        assert retries == 1
        assert runtime.run_calls == [
            ("vector_search", "FEV1和肺功能检查有什么关系？"),
            ("cypher_traverse", "FEV1和肺功能检查有什么关系？"),
        ]

    def test_escalates_with_rewritten_query_and_records_trace(self):
        vector_results = _make_results("vector", 1)
        bm25_results = _make_results("bm25", 3)
        trace = PipelineTrace(
            trace_id="tr_langgraph",
            timestamp="2026-05-07T00:00:00Z",
            query="original query",
        )
        runtime = _FakeWorkflowRuntime(
            tool_results={
                "vector_search": vector_results,
                "bm25_search": bm25_results,
            },
            reflections=[
                ReflectionStep(
                    overall_score=1.0,
                    failure_type="insufficient_recall",
                    recommended_action="expand_recall",
                ),
                ReflectionStep(
                    overall_score=4.5,
                    failure_type="acceptable",
                    recommended_action="answer_ready",
                ),
            ],
            next_tools={"vector_search": "bm25_search"},
            rewrite_query="rewritten query",
            rewrite_enabled=True,
        )

        results, retries = run_self_correction_workflow(
            query="original query",
            driver=MagicMock(),
            openai_client=MagicMock(),
            decision=_make_decision(),
            max_retries=2,
            relevance_threshold=3.0,
            max_query_rewrites=1,
            trace=trace,
            ops=runtime.as_ops(),
        )

        assert results == bm25_results
        assert retries == 1
        assert runtime.run_calls == [
            ("vector_search", "original query"),
            ("bm25_search", "rewritten query"),
        ]
        assert runtime.retry_query_calls == ["original query"]
        assert len(trace.tool_steps) == 2
        assert len(trace.reflection_steps) == 2
        assert len(trace.escalation_steps) == 1
        assert trace.escalation_steps[0].to_tool == "bm25_search"
        assert trace.escalation_steps[0].rephrased_query == "rewritten query"

    def test_runs_dedicated_rerank_node_when_reflection_requests_it(self):
        vector_results = _make_results("vector", 2)
        reranked_results = list(reversed(vector_results))
        trace = PipelineTrace(
            trace_id="tr_rerank",
            timestamp="2026-05-07T00:00:00Z",
            query="original query",
        )
        runtime = _FakeWorkflowRuntime(
            tool_results={"vector_search": vector_results},
            reflections=[
                ReflectionStep(
                    overall_score=1.5,
                    verdict="rerank",
                    failure_type="insufficient_context",
                    recommended_action="expand_recall",
                    should_rerank_again=True,
                    retry_scope="rerank_only",
                ),
                ReflectionStep(
                    overall_score=4.2,
                    failure_type="acceptable",
                    recommended_action="answer_ready",
                    should_retry=False,
                ),
            ],
            reranked_results=reranked_results,
        )

        results, retries = run_self_correction_workflow(
            query="original query",
            driver=MagicMock(),
            openai_client=MagicMock(),
            decision=_make_decision(),
            max_retries=2,
            relevance_threshold=3.0,
            trace=trace,
            ops=runtime.as_ops(),
        )

        assert results == reranked_results
        assert retries == 0
        assert runtime.run_calls == [("vector_search", "original query")]
        assert runtime.rerank_calls == [("original query", ["vector-0", "vector-1"])]
        assert [step.tool_name for step in trace.tool_steps] == [
            "vector_search",
            "rerank_results",
        ]
        assert any(entry.stage == "rerank" for entry in trace.workflow_memory)

    @patch("agentic_graph_rag.agent.langgraph_workflow.get_settings")
    def test_skips_llm_reflection_for_high_normalized_score(self, mock_settings):
        mock_settings.return_value = _make_settings()
        high_score_results = [
            SearchResult(
                chunk=Chunk(id="hit-1", content="strong evidence"),
                score=8.0,
                score_normalized=0.92,
                rank=1,
                source="bm25",
            )
        ]
        trace = PipelineTrace(
            trace_id="tr_skip_reflection",
            timestamp="2026-05-09T00:00:00Z",
            query="original query",
        )
        runtime = _FakeWorkflowRuntime(
            tool_results={"vector_search": high_score_results},
            reflections=[],
        )

        results, retries = run_self_correction_workflow(
            query="original query",
            driver=MagicMock(),
            openai_client=MagicMock(),
            decision=_make_decision(),
            max_retries=1,
            relevance_threshold=2.0,
            trace=trace,
            ops=runtime.as_ops(),
        )

        assert results == high_score_results
        assert retries == 0
        assert runtime.reflections == []
        assert len(trace.reflection_steps) == 1
        assert trace.reflection_steps[0].reasoning.startswith("Skipped LLM reflection")
        assert trace.workflow_memory[-1].metadata["top_score"] == 0.92
        assert trace.workflow_memory[-1].metadata["skip_threshold"] == 0.85

    def test_does_not_rerank_more_than_once_globally(self):
        vector_results = _make_results("vector", 2)
        bm25_results = _make_results("bm25", 2)
        runtime = _FakeWorkflowRuntime(
            tool_results={
                "vector_search": vector_results,
                "bm25_search": bm25_results,
            },
            reflections=[
                ReflectionStep(
                    overall_score=1.5,
                    verdict="rerank",
                    failure_type="insufficient_context",
                    recommended_action="expand_recall",
                    should_rerank_again=True,
                    retry_scope="rerank_only",
                ),
                ReflectionStep(
                    overall_score=1.0,
                    verdict="retry",
                    failure_type="insufficient_recall",
                    recommended_action="expand_recall",
                    retry_scope="tool_escalation",
                    should_retry=True,
                ),
                ReflectionStep(
                    overall_score=1.2,
                    verdict="rerank",
                    failure_type="insufficient_context",
                    recommended_action="expand_recall",
                    should_rerank_again=True,
                    retry_scope="rerank_only",
                    should_retry=True,
                ),
            ],
            next_tools={"vector_search": "bm25_search", "bm25_search": None},
        )

        results, retries = run_self_correction_workflow(
            query="original query",
            driver=MagicMock(),
            openai_client=MagicMock(),
            decision=_make_decision(),
            max_retries=2,
            relevance_threshold=3.0,
            max_query_rewrites=1,
            trace=None,
            ops=runtime.as_ops(),
        )

        assert results == vector_results
        assert retries == 1
        assert runtime.rerank_calls == [("original query", ["vector-0", "vector-1"])]

    def test_rewrite_happens_at_most_once(self):
        vector_results = _make_results("vector", 2)
        bm25_results = _make_results("bm25", 2)
        hybrid_results = _make_results("hybrid", 2)
        runtime = _FakeWorkflowRuntime(
            tool_results={
                "vector_search": vector_results,
                "bm25_search": bm25_results,
                "hybrid_search": hybrid_results,
            },
            reflections=[
                ReflectionStep(
                    overall_score=1.0,
                    verdict="retry",
                    failure_type="missing_entity",
                    recommended_action="target_missing_entity",
                    should_retry=True,
                    should_rewrite_query=True,
                    retry_scope="tool_escalation",
                ),
                ReflectionStep(
                    overall_score=1.1,
                    verdict="retry",
                    failure_type="insufficient_recall",
                    recommended_action="expand_recall",
                    should_retry=True,
                    should_rewrite_query=True,
                    retry_scope="tool_escalation",
                ),
                ReflectionStep(
                    overall_score=4.0,
                    verdict="answer",
                    failure_type="acceptable",
                    recommended_action="answer_ready",
                    should_retry=False,
                    retry_scope="stop",
                ),
            ],
            next_tools={
                "vector_search": "bm25_search",
                "bm25_search": "hybrid_search",
            },
            rewrite_query="rewritten once",
            rewrite_enabled=True,
        )

        results, retries = run_self_correction_workflow(
            query="original query",
            driver=MagicMock(),
            openai_client=MagicMock(),
            decision=_make_decision(),
            max_retries=2,
            relevance_threshold=3.0,
            max_query_rewrites=1,
            trace=None,
            ops=runtime.as_ops(),
        )

        assert results == hybrid_results
        assert retries == 2
        assert runtime.retry_query_calls == ["original query"]
        assert runtime.run_calls == [
            ("vector_search", "original query"),
            ("bm25_search", "rewritten once"),
            ("hybrid_search", "rewritten once"),
        ]

    def test_request_time_budget_stops_before_extra_retry(self):
        vector_results = _make_results("vector", 2)
        runtime = _FakeWorkflowRuntime(
            tool_results={
                "vector_search": vector_results,
                "bm25_search": _make_results("bm25", 2),
            },
            reflections=[
                ReflectionStep(
                    overall_score=1.0,
                    verdict="retry",
                    failure_type="insufficient_recall",
                    recommended_action="expand_recall",
                    should_retry=True,
                    retry_scope="tool_escalation",
                )
            ],
            next_tools={"vector_search": "bm25_search"},
        )

        results, retries = run_self_correction_workflow(
            query="original query",
            driver=MagicMock(),
            openai_client=MagicMock(),
            decision=_make_decision(),
            max_retries=2,
            relevance_threshold=3.0,
            request_time_budget_ms=0,
            trace=None,
            ops=runtime.as_ops(),
        )

        assert results == vector_results
        assert retries == 0
        assert runtime.run_calls == [("vector_search", "original query")]
        assert runtime.reflection_calls == 0

    def test_reflection_hallucination_guard_blocks_repeated_missing_claims(self):
        vector_results = [
            SearchResult(
                chunk=Chunk(id="vector-1", content="JDK 21 virtual thread default stack size is 1MB."),
                score=0.9,
                rank=1,
            )
        ]
        runtime = _FakeWorkflowRuntime(
            tool_results={
                "vector_search": vector_results,
                "bm25_search": _make_results("bm25", 2),
            },
            reflections=[
                ReflectionStep(
                    overall_score=1.0,
                    verdict="retry",
                    failure_type="insufficient_context",
                    recommended_action="expand_recall",
                    should_retry=True,
                    retry_scope="tool_escalation",
                    missing_information=["default stack size"],
                )
            ],
            next_tools={"vector_search": "bm25_search"},
        )

        results, retries = run_self_correction_workflow(
            query="JDK 21 virtual thread default stack size",
            driver=MagicMock(),
            openai_client=MagicMock(),
            decision=_make_decision(),
            max_retries=2,
            relevance_threshold=3.0,
            trace=None,
            ops=runtime.as_ops(),
        )

        assert results == vector_results
        assert retries == 0
        assert runtime.run_calls == [("vector_search", "JDK 21 virtual thread default stack size")]

    def test_best_results_prefers_anchor_hits_over_result_count(self):
        exact_results = [
            SearchResult(
                chunk=Chunk(
                    id="exact-1",
                    content="JDK 21 virtual thread default stack size is managed lazily.",
                ),
                score=0.42,
                rank=1,
            )
        ]
        noisy_results = [
            SearchResult(
                chunk=Chunk(id=f"noisy-{index}", content="General Java guide without the target detail."),
                score=0.95 - index * 0.01,
                rank=index + 1,
            )
            for index in range(10)
        ]
        runtime = _FakeWorkflowRuntime(
            tool_results={
                "vector_search": exact_results,
                "hybrid_search": noisy_results,
            },
            reflections=[
                ReflectionStep(
                    overall_score=1.0,
                    verdict="retry",
                    failure_type="insufficient_context",
                    recommended_action="expand_recall",
                    should_retry=True,
                    retry_scope="tool_escalation",
                ),
                ReflectionStep(
                    overall_score=4.8,
                    verdict="retry",
                    failure_type="insufficient_context",
                    recommended_action="expand_recall",
                    should_retry=True,
                    retry_scope="tool_escalation",
                ),
            ],
            next_tools={"vector_search": "hybrid_search", "hybrid_search": None},
        )

        results, retries = run_self_correction_workflow(
            query="JDK 21 virtual thread default stack size",
            driver=MagicMock(),
            openai_client=MagicMock(),
            decision=_make_decision(),
            max_retries=1,
            relevance_threshold=3.0,
            trace=None,
            ops=runtime.as_ops(),
        )

        assert results == exact_results
        assert retries == 1


class TestTopLevelAgentWorkflow:
    def test_passes_last_reflection_score_into_answer_generation(self):
        decision = _make_decision()
        results = _make_results("vector", 2)
        captured: dict[str, float | None] = {}

        def _run_self_correction(*args, **kwargs):
            kwargs["reflection_history_sink"].append(
                ReflectionStep(
                    overall_score=2.8,
                    verdict="answer",
                    should_retry=False,
                )
            )
            return results, 0

        def _generate(query, retrieved, *, openai_client, reflection_score=None):
            del query, retrieved, openai_client
            captured["reflection_score"] = reflection_score
            return QAResult(answer="final answer", sources=results, confidence=0.8, query="q")

        ops = AgentWorkflowOps(
            classify_query=lambda query, *, use_llm, openai_client, reasoning: decision,
            is_cross_language_global=lambda query: False,
            run_self_correction=_run_self_correction,
            generate_answer=_generate,
            evaluate_completeness=lambda query, answer, *, openai_client: True,
            comprehensive_search=lambda query, driver, openai_client: [],
        )

        run_agent_workflow(
            query="q",
            driver=MagicMock(),
            openai_client=MagicMock(),
            use_llm_router=False,
            reasoning=None,
            trace=PipelineTrace(trace_id="tr_reflect", timestamp="2026-05-07T00:00:00Z", query="q"),
            ops=ops,
        )

        assert captured["reflection_score"] == 2.8

    def test_routes_retrieves_generates_and_finishes_for_simple_query(self):
        decision = _make_decision()
        results = _make_results("vector", 2)
        qa_result = QAResult(answer="final answer", sources=results, confidence=0.8, query="q")

        ops = AgentWorkflowOps(
            classify_query=lambda query, *, use_llm, openai_client, reasoning: decision,
            is_cross_language_global=lambda query: False,
            run_self_correction=lambda *args, **kwargs: (results, 0),
            generate_answer=lambda query, retrieved, *, openai_client, reflection_score=None: qa_result,
            evaluate_completeness=lambda query, answer, *, openai_client: True,
            comprehensive_search=lambda query, driver, openai_client: [],
        )

        final_result = run_agent_workflow(
            query="q",
            driver=MagicMock(),
            openai_client=MagicMock(),
            use_llm_router=False,
            reasoning=None,
            trace=PipelineTrace(trace_id="tr_run", timestamp="2026-05-07T00:00:00Z", query="q"),
            ops=ops,
        )

        assert final_result.answer == "final answer"
        assert final_result.retries == 0
        assert final_result.router_decision == decision

    def test_guard_blocked_retrieval_lowers_answer_confidence(self):
        decision = _make_decision()
        results = _make_results("vector", 2)
        qa_result = QAResult(answer="direct answer", sources=results, confidence=0.82, query="q")

        def _run_self_correction(*args, **kwargs):
            kwargs["memory_sink"].append(
                WorkflowMemoryEntry(
                    stage="retry",
                    message="reflection requested already-covered gap: default stack size",
                    metadata={},
                )
            )
            kwargs["reflection_history_sink"].append(
                ReflectionStep(
                    verdict="stop",
                    failure_type="insufficient_context",
                    recommended_action="stop_due_to_invalid_reflection",
                    reasoning="Reflection policy guard stopped retry.",
                )
            )
            return results, 0

        ops = AgentWorkflowOps(
            classify_query=lambda query, *, use_llm, openai_client, reasoning: decision,
            is_cross_language_global=lambda query: False,
            run_self_correction=_run_self_correction,
            generate_answer=lambda query, retrieved, *, openai_client, reflection_score=None: qa_result,
            evaluate_completeness=lambda query, answer, *, openai_client: True,
            comprehensive_search=lambda query, driver, openai_client: [],
        )

        final_result = run_agent_workflow(
            query="q",
            driver=MagicMock(),
            openai_client=MagicMock(),
            use_llm_router=False,
            reasoning=None,
            trace=PipelineTrace(trace_id="tr_guard", timestamp="2026-05-07T00:00:00Z", query="q"),
            ops=ops,
        )

        assert final_result.confidence < 0.82  # Guard applies a scale-down
        assert final_result.confidence == pytest.approx(0.82 * 0.6, abs=0.01)
        assert "not decisive enough" in final_result.answer

    def test_global_query_runs_completeness_retry_nodes(self):
        decision = _make_decision(tool="comprehensive_search")
        decision.query_type = QueryType.GLOBAL
        initial_results = _make_results("initial", 1)
        comprehensive_results = _make_results("comp", 2)

        generated_answers = [
            QAResult(
                answer="first answer",
                sources=initial_results,
                confidence=0.4,
                query="q",
            ),
            QAResult(
                answer="second answer",
                sources=initial_results + comprehensive_results,
                confidence=0.6,
                query="q",
            ),
        ]

        def _generate(_query, _results, *, openai_client, reflection_score=None):
            del openai_client
            del reflection_score
            return generated_answers.pop(0)

        completeness_checks = iter([False, False])

        ops = AgentWorkflowOps(
            classify_query=lambda query, *, use_llm, openai_client, reasoning: decision,
            is_cross_language_global=lambda query: False,
            run_self_correction=lambda *args, **kwargs: (initial_results, 0),
            generate_answer=_generate,
            evaluate_completeness=lambda query, answer, *, openai_client: next(completeness_checks),
            comprehensive_search=lambda query, driver, openai_client: comprehensive_results,
        )

        final_result = run_agent_workflow(
            query="q",
            driver=MagicMock(),
            openai_client=MagicMock(),
            use_llm_router=False,
            reasoning=None,
            trace=PipelineTrace(trace_id="tr_global", timestamp="2026-05-07T00:00:00Z", query="q"),
            ops=ops,
        )

        assert final_result.answer == "second answer"
        assert len(final_result.sources) == 3
        assert final_result.retries == 1

    def test_retrieve_node_captures_nested_reflection_history_and_memory(self):
        decision = _make_decision()
        results = _make_results("vector", 2)
        qa_result = QAResult(answer="final answer", sources=results, confidence=0.8, query="q")

        def _run_self_correction(*args, **kwargs):
            memory_sink = kwargs["memory_sink"]
            reflection_history_sink = kwargs["reflection_history_sink"]
            memory_sink.append(
                WorkflowMemoryEntry(
                    stage="retrieval",
                    message="vector search returned partial evidence",
                    metadata={"tool": "vector_search"},
                )
            )
            reflection_history_sink.append(
                ReflectionStep(
                    attempt=0,
                    tool_name="vector_search",
                    overall_score=2.8,
                    failure_type="insufficient_context",
                    recommended_action="expand_recall",
                )
            )
            return results, 1

        ops = AgentWorkflowOps(
            classify_query=lambda query, *, use_llm, openai_client, reasoning: decision,
            is_cross_language_global=lambda query: False,
            run_self_correction=_run_self_correction,
            generate_answer=lambda query, retrieved, *, openai_client, reflection_score=None: qa_result,
            evaluate_completeness=lambda query, answer, *, openai_client: True,
            comprehensive_search=lambda query, driver, openai_client: [],
        )

        final_result = run_agent_workflow(
            query="q",
            driver=MagicMock(),
            openai_client=MagicMock(),
            use_llm_router=False,
            reasoning=None,
            trace=PipelineTrace(trace_id="tr_mem", timestamp="2026-05-07T00:00:00Z", query="q"),
            ops=ops,
        )

        assert final_result.retries == 1
        assert len(final_result.trace.reflection_steps) == 1
        assert final_result.trace.reflection_steps[0].failure_type == "insufficient_context"
        assert any(
            entry.message == "vector search returned partial evidence"
            for entry in final_result.trace.workflow_memory
        )

    def test_reflection_can_stop_retry_even_below_threshold(self):
        vector_results = _make_results("vector", 2)
        runtime = _FakeWorkflowRuntime(
            tool_results={"vector_search": vector_results},
            reflections=[
                ReflectionStep(
                    overall_score=1.0,
                    failure_type="acceptable",
                    recommended_action="answer_ready",
                    should_retry=False,
                    retry_scope="stop",
                )
            ],
        )

        results, retries = run_self_correction_workflow(
            query="test query",
            driver=MagicMock(),
            openai_client=MagicMock(),
            decision=_make_decision(),
            max_retries=2,
            relevance_threshold=3.0,
            trace=None,
            ops=runtime.as_ops(),
        )

        assert results == vector_results
        assert retries == 0
        assert runtime.run_calls == [("vector_search", "test query")]

    def test_low_score_answer_verdict_retries_when_budget_remains(self):
        vector_results = _make_results("vector", 2)
        retry_results = _make_results("bm25", 2)
        runtime = _FakeWorkflowRuntime(
            tool_results={
                "vector_search": vector_results,
                "bm25_search": retry_results,
            },
            next_tools={"vector_search": "bm25_search"},
            reflections=[
                ReflectionStep(
                    overall_score=1.0,
                    failure_type="acceptable",
                    recommended_action="answer_ready",
                    should_retry=True,
                    retry_scope="tool_escalation",
                ),
                ReflectionStep(
                    overall_score=4.0,
                    failure_type="acceptable",
                    recommended_action="answer_ready",
                    should_retry=False,
                    retry_scope="stop",
                ),
            ],
        )

        results, retries = run_self_correction_workflow(
            query="test query",
            driver=MagicMock(),
            openai_client=MagicMock(),
            decision=_make_decision(),
            max_retries=2,
            relevance_threshold=3.0,
            trace=None,
            ops=runtime.as_ops(),
        )

        assert results == retry_results
        assert retries == 1
        assert runtime.run_calls == [("vector_search", "test query"), ("bm25_search", "test query")]
