"""Tests for agentic_graph_rag.agent.retrieval_agent."""

from unittest.mock import ANY, MagicMock, patch

from rag_core.models import (
    Chunk,
    QAResult,
    QueryType,
    ReflectionStep,
    RouterDecision,
    SearchResult,
)

from agentic_graph_rag.agent.retrieval_agent import (
    _REFLECTION_RULES,
    _RETRY_TOOL_MATRIX,
    _TOOL_REGISTRY,
    _get_next_tool,
    _run_tool,
    run,
    select_tool,
    self_correction_loop,
)


def _make_results(n: int) -> list[SearchResult]:
    return [
        SearchResult(
            chunk=Chunk(id=f"c{i}", content=f"Content {i}"),
            score=0.9 - i * 0.1,
            rank=i + 1,
        )
        for i in range(n)
    ]


def _make_decision(
    query_type: QueryType = QueryType.SIMPLE,
    tool: str = "vector_search",
) -> RouterDecision:
    return RouterDecision(
        query_type=query_type,
        confidence=0.8,
        reasoning="test",
        suggested_tool=tool,
    )


def _mock_tool(results=None):
    """Create a mock tool function."""
    mock = MagicMock()
    mock.return_value = results if results is not None else _make_results(3)
    return mock


# ---------------------------------------------------------------------------
# select_tool
# ---------------------------------------------------------------------------

class TestSelectTool:
    def test_valid_tool(self):
        d = _make_decision(tool="cypher_traverse")
        assert select_tool(d) == "cypher_traverse"

    def test_unknown_tool_defaults(self):
        d = _make_decision(tool="nonexistent_tool")
        assert select_tool(d) == "vector_search"

    def test_all_known_tools(self):
        for tool in ["vector_search", "bm25_search", "cypher_traverse", "community_search",
                      "hybrid_search", "temporal_query", "full_document_read"]:
            d = _make_decision(tool=tool)
            assert select_tool(d) == tool


class TestRunTool:
    def test_passes_query_type_to_hybrid_search(self):
        mock_tool = _mock_tool(_make_results(2))
        decision = _make_decision(query_type=QueryType.MULTI_HOP, tool="hybrid_search")
        driver = MagicMock()
        client = MagicMock()

        with patch.dict(_TOOL_REGISTRY, {"hybrid_search": mock_tool}):
            _run_tool("hybrid_search", "test", driver, client, decision)

        mock_tool.assert_called_once_with(
            "test",
            driver,
            client,
            query_type=QueryType.MULTI_HOP,
        )


# ---------------------------------------------------------------------------
# _get_next_tool
# ---------------------------------------------------------------------------

class TestGetNextTool:
    def test_fallback_matrix_prefers_bm25_after_vector(self):
        nxt = _get_next_tool("vector_search", {"vector_search"})
        assert nxt == "bm25_search"

    def test_fallback_matrix_prefers_vector_after_bm25(self):
        nxt = _get_next_tool("bm25_search", {"vector_search", "bm25_search"})
        assert nxt == "cypher_traverse"

    def test_fallback_matrix_prefers_hybrid_after_cypher(self):
        nxt = _get_next_tool("cypher_traverse", {"cypher_traverse"})
        assert nxt == "bm25_search"

    def test_skips_tried(self):
        nxt = _get_next_tool("vector_search", {"vector_search", "bm25_search", "cypher_traverse"})
        assert nxt == "hybrid_search"

    def test_skips_to_hybrid_only_after_lightweight_simple_tools(self):
        decision = _make_decision(query_type=QueryType.SIMPLE, tool="vector_search")
        reflection = ReflectionStep(
            verdict="retry",
            failure_type="no_results",
            recommended_action="target_missing_entity",
            query_used="Type 2 diabetes treatment",
        )
        nxt = _get_next_tool(
            "vector_search",
            {"vector_search", "bm25_search", "cypher_traverse"},
            reflection,
            decision,
        )
        assert nxt == "hybrid_search"

    def test_no_more_tools(self):
        nxt = _get_next_tool("full_document_read", {"full_document_read"})
        assert nxt is None

    def test_all_tried(self):
        all_tools = {
            "vector_search",
            "bm25_search",
            "cypher_traverse",
            "hybrid_search",
            "comprehensive_search",
            "full_document_read",
        }
        nxt = _get_next_tool("vector_search", all_tools)
        assert nxt is None

    def test_unknown_current(self):
        nxt = _get_next_tool("temporal_query", {"temporal_query"})
        assert nxt == "bm25_search"

    def test_prefers_graph_tool_for_relation_failure(self):
        reflection = ReflectionStep(
            failure_type="relation_missing",
            recommended_action="use_graph_traversal",
            verdict="retry",
        )
        nxt = _get_next_tool("vector_search", {"vector_search"}, reflection)
        assert nxt == "cypher_traverse"

    def test_prefers_bm25_for_factual_missing_entity(self):
        decision = _make_decision(query_type=QueryType.SIMPLE, tool="vector_search")
        reflection = ReflectionStep(
            verdict="retry",
            failure_type="missing_entity",
            recommended_action="target_missing_entity",
            preferred_tools=["bm25_search", "cypher_traverse"],
            retry_scope="tool_escalation",
        )
        nxt = _get_next_tool("vector_search", {"vector_search"}, reflection, decision)
        assert nxt == "bm25_search"

    def test_rule_first_prefers_bm25_for_error_code_queries(self):
        decision = _make_decision(query_type=QueryType.SIMPLE, tool="vector_search")
        reflection = ReflectionStep(
            verdict="retry",
            query_used="ERR-902X on JDK 21",
            failure_type="insufficient_recall",
            recommended_action="expand_recall",
        )
        nxt = _get_next_tool("vector_search", {"vector_search"}, reflection, decision)
        assert nxt == "bm25_search"

    def test_rule_first_prefers_graph_for_relation_queries(self):
        decision = _make_decision(query_type=QueryType.RELATION, tool="vector_search")
        reflection = ReflectionStep(
            verdict="retry",
            query_used="Kafka 和 RabbitMQ 的关系与区别是什么",
            failure_type="insufficient_context",
            recommended_action="expand_recall",
            preferred_tools=["bm25_search"],
        )
        nxt = _get_next_tool("vector_search", {"vector_search"}, reflection, decision)
        assert nxt == "cypher_traverse"

    def test_rule_first_prefers_bm25_for_short_queries(self):
        decision = _make_decision(query_type=QueryType.SIMPLE, tool="vector_search")
        reflection = ReflectionStep(
            verdict="retry",
            query_used="JDK 21 stack size",
            failure_type="insufficient_context",
            recommended_action="expand_recall",
        )
        nxt = _get_next_tool("vector_search", {"vector_search"}, reflection, decision)
        assert nxt == "bm25_search"

    def test_downgrades_hybrid_to_bm25_for_lexical_retry(self):
        decision = _make_decision(query_type=QueryType.SIMPLE, tool="hybrid_search")
        reflection = ReflectionStep(
            verdict="retry",
            query_used="ERR-902X on JDK 21",
            failure_type="insufficient_recall",
            recommended_action="expand_recall",
        )
        nxt = _get_next_tool("hybrid_search", {"hybrid_search"}, reflection, decision)
        assert nxt == "bm25_search"

    def test_full_document_read_only_after_comprehensive_for_global_query(self):
        decision = _make_decision(query_type=QueryType.GLOBAL, tool="comprehensive_search")
        reflection = ReflectionStep(
            verdict="retry",
            query_used="请总结整个设计文档",
            failure_type="insufficient_context",
            recommended_action="use_comprehensive_search",
        )
        nxt = _get_next_tool(
            "comprehensive_search",
            {"comprehensive_search", "hybrid_search", "bm25_search", "vector_search"},
            reflection,
            decision,
        )
        assert nxt == "full_document_read"

    def test_reflection_tool_maps_are_explicit(self):
        assert _REFLECTION_RULES["target_missing_entity"]["tools"] == [
            "bm25_search",
            "vector_search",
            "cypher_traverse",
            "hybrid_search",
        ]
        assert _REFLECTION_RULES["insufficient_context"]["tools"] == [
            "hybrid_search",
            "comprehensive_search",
        ]
        assert _RETRY_TOOL_MATRIX["comprehensive_search"][0] == "full_document_read"


# ---------------------------------------------------------------------------
# self_correction_loop
# ---------------------------------------------------------------------------

class TestSelfCorrectionLoop:
    @patch("agentic_graph_rag.agent.retrieval_agent.evaluate_reflection")
    def test_no_retry_when_relevant(self, mock_eval):
        results = _make_results(3)
        mock_tool = _mock_tool(results)
        mock_eval.return_value = ReflectionStep(
            failure_type="acceptable",
            verdict="answer",
            evidence_status="sufficient",
        )

        driver = MagicMock()
        client = MagicMock()
        decision = _make_decision()

        with patch.dict(_TOOL_REGISTRY, {"vector_search": mock_tool}):
            out, retries = self_correction_loop(
                "test", driver, client, decision,
                max_retries=2, relevance_threshold=3.0,
            )
        assert retries == 0
        assert out == results
        mock_tool.assert_called_once()

    @patch("agentic_graph_rag.agent.retrieval_agent.evaluate_reflection")
    def test_records_reflection_in_trace(self, mock_eval):
        from rag_core.models import PipelineTrace

        results = _make_results(2)
        mock_tool = _mock_tool(results)
        mock_eval.return_value = ReflectionStep(
            failure_type="acceptable",
            reasoning="enough evidence",
        )

        trace = PipelineTrace(trace_id="tr_1", timestamp="2026-05-07T00:00:00Z", query="test")
        driver = MagicMock()
        client = MagicMock()
        decision = _make_decision()

        with patch.dict(_TOOL_REGISTRY, {"vector_search": mock_tool}):
            out, retries = self_correction_loop(
                "test",
                driver,
                client,
                decision,
                max_retries=1,
                relevance_threshold=3.0,
                trace=trace,
            )

        assert retries == 0
        assert out == results
        assert len(trace.tool_steps) == 1
        assert len(trace.reflection_steps) == 1
        assert trace.reflection_steps[0].failure_type == "acceptable"

    @patch("agentic_graph_rag.agent.retrieval_agent.evaluate_reflection")
    def test_escalates_on_low_relevance(self, mock_eval):
        mock_vs = _mock_tool(_make_results(2))
        mock_bm25 = _mock_tool(_make_results(3))
        mock_eval.side_effect = [
            ReflectionStep(
                failure_type="insufficient_recall",
                recommended_action="expand_recall",
            ),
            ReflectionStep(
                failure_type="acceptable",
                verdict="answer",
                evidence_status="sufficient",
            ),
        ]

        driver = MagicMock()
        client = MagicMock()
        decision = _make_decision()

        with patch.dict(_TOOL_REGISTRY, {
            "vector_search": mock_vs,
            "bm25_search": mock_bm25,
        }):
            out, retries = self_correction_loop(
                "test", driver, client, decision,
                max_retries=2, relevance_threshold=3.0,
            )
        assert retries == 1
        assert mock_vs.call_count == 1
        assert mock_bm25.call_count == 1

    @patch("agentic_graph_rag.agent.retrieval_agent.generate_retry_query")
    @patch("agentic_graph_rag.agent.retrieval_agent.evaluate_reflection")
    def test_incremental_retry_replays_only_graph_channel(self, mock_eval, mock_retry):
        from rag_core.models import PipelineTrace

        vector_results = _make_results(2)
        hybrid_results = _make_results(4)
        mock_vs = _mock_tool(vector_results)
        mock_hs = _mock_tool(hybrid_results)
        mock_eval.side_effect = [
            ReflectionStep(
                failure_type="relation_missing",
                recommended_action="use_graph_traversal",
            ),
            ReflectionStep(
                failure_type="acceptable",
                verdict="answer",
                evidence_status="sufficient",
            ),
        ]
        trace = PipelineTrace(trace_id="tr_cache", timestamp="2026-05-07T00:00:00Z", query="test")

        driver = MagicMock()
        client = MagicMock()
        decision = _make_decision()

        with patch.dict(_TOOL_REGISTRY, {
            "vector_search": mock_vs,
            "hybrid_search": mock_hs,
        }):
            out, retries = self_correction_loop(
                "test",
                driver,
                client,
                decision,
                max_retries=1,
                relevance_threshold=3.0,
                trace=trace,
            )

        assert retries == 1
        assert out == hybrid_results
        mock_retry.assert_not_called()
        mock_hs.assert_called_once_with(
            "test",
            driver,
            client,
            query_type=QueryType.SIMPLE,
            seed_results={
                "vector": vector_results,
            },
            enabled_providers=["graph"],
            provider_results=ANY,
        )
        assert trace.tool_steps[-1].tool_name == "hybrid_search"
        assert trace.tool_steps[-1].cache_hit is True
        assert trace.tool_steps[-1].reused_sources == ["vector"]
        assert trace.tool_steps[-1].executed_sources == ["graph"]
        assert trace.escalation_steps[-1].cached_sources_reused == ["vector"]

    @patch("agentic_graph_rag.agent.retrieval_agent.evaluate_reflection")
    def test_enum_reflection_missing_relation_falls_back_to_graph(self, mock_eval):
        vector_results = _make_results(2)
        graph_results = _make_results(3)
        for result in graph_results:
            result.source = "graph"
        mock_vs = _mock_tool(vector_results)
        mock_graph = _mock_tool(graph_results)
        mock_eval.side_effect = [
            ReflectionStep(
                evidence_status="partial",
                gap_type="missing_relation",
                action="retry_graph",
                required_tool="cypher_traverse",
                failure_type="relation_missing",
                recommended_action="use_graph_traversal",
                preferred_tools=["cypher_traverse"],
                should_retry=True,
                retry_scope="tool_escalation",
            ),
            ReflectionStep(
                evidence_status="sufficient",
                gap_type="none",
                action="answer",
                failure_type="acceptable",
                recommended_action="answer_ready",
                should_retry=False,
                retry_scope="stop",
            ),
        ]

        decision = _make_decision(query_type=QueryType.RELATION)
        with patch.dict(_TOOL_REGISTRY, {
            "vector_search": mock_vs,
            "cypher_traverse": mock_graph,
        }):
            out, retries = self_correction_loop(
                "FEV1和肺功能检查有什么关系？",
                MagicMock(),
                MagicMock(),
                decision,
                max_retries=2,
                relevance_threshold=3.0,
            )

        assert out == graph_results
        assert retries == 1
        assert mock_vs.call_count == 1
        assert mock_graph.call_count == 1

    @patch("agentic_graph_rag.agent.retrieval_agent.generate_retry_query")
    @patch("agentic_graph_rag.agent.retrieval_agent.evaluate_reflection")
    def test_hybrid_retry_reuses_initial_provider_results(self, mock_eval, mock_retry):
        from rag_core.models import PipelineTrace

        hybrid_results_1 = _make_results(3)
        hybrid_results_2 = _make_results(4)
        trace = PipelineTrace(trace_id="tr_hybrid", timestamp="2026-05-07T00:00:00Z", query="test")

        def _hybrid_side_effect(query, driver, client, **kwargs):
            provider_results = kwargs["provider_results"]
            if "seed_results" not in kwargs:
                provider_results.update({
                    "vector": _make_results(2),
                    "bm25": _make_results(2),
                    "graph": _make_results(2),
                })
                return hybrid_results_1
            provider_results.update({
                "vector": kwargs["seed_results"]["vector"],
                "bm25": kwargs["seed_results"]["bm25"],
                "graph": _make_results(1),
            })
            return hybrid_results_2

        mock_hs = MagicMock(side_effect=_hybrid_side_effect)
        mock_eval.side_effect = [
            ReflectionStep(
                failure_type="relation_missing",
                recommended_action="use_graph_traversal",
            ),
            ReflectionStep(
                failure_type="acceptable",
                verdict="answer",
                evidence_status="sufficient",
            ),
        ]

        driver = MagicMock()
        client = MagicMock()
        decision = _make_decision(tool="hybrid_search")

        with patch.dict(_TOOL_REGISTRY, {"hybrid_search": mock_hs}):
            out, retries = self_correction_loop(
                "test",
                driver,
                client,
                decision,
                max_retries=1,
                relevance_threshold=3.0,
                trace=trace,
            )

        assert out == hybrid_results_2
        assert retries == 1
        mock_retry.assert_not_called()
        assert mock_hs.call_count == 2
        second_call = mock_hs.call_args_list[1]
        assert second_call.args == ("test", driver, client)
        assert second_call.kwargs["query_type"] == QueryType.SIMPLE
        assert second_call.kwargs["enabled_providers"] == ["graph"]
        assert second_call.kwargs["seed_results"].keys() == {"vector", "bm25"}
        assert "provider_results" in second_call.kwargs
        assert trace.tool_steps[-1].tool_name == "hybrid_search"
        assert trace.tool_steps[-1].reused_sources == ["vector", "bm25"]
        assert trace.tool_steps[-1].executed_sources == ["graph"]

    @patch("agentic_graph_rag.agent.retrieval_agent.generate_retry_query")
    @patch("agentic_graph_rag.agent.retrieval_agent.evaluate_reflection")
    def test_hybrid_insufficient_context_replays_empty_graph_first(self, mock_eval, mock_retry):
        hybrid_results_1 = _make_results(2)
        hybrid_results_2 = _make_results(3)

        def _hybrid_side_effect(query, driver, client, **kwargs):
            provider_results = kwargs["provider_results"]
            if "seed_results" not in kwargs:
                provider_results.update({
                    "vector": _make_results(2),
                    "bm25": _make_results(1),
                    "graph": [],
                })
                return hybrid_results_1
            provider_results.update({
                "vector": kwargs["seed_results"]["vector"],
                "bm25": kwargs["seed_results"]["bm25"],
                "graph": _make_results(2),
            })
            return hybrid_results_2

        mock_hs = MagicMock(side_effect=_hybrid_side_effect)
        mock_eval.side_effect = [
            ReflectionStep(
                failure_type="insufficient_context",
                recommended_action="answer_ready",
            ),
            ReflectionStep(
                failure_type="acceptable",
                verdict="answer",
                evidence_status="sufficient",
            ),
        ]

        driver = MagicMock()
        client = MagicMock()
        decision = _make_decision(tool="hybrid_search")

        with patch.dict(_TOOL_REGISTRY, {"hybrid_search": mock_hs}):
            out, retries = self_correction_loop(
                "test",
                driver,
                client,
                decision,
                max_retries=1,
                relevance_threshold=3.0,
            )

        assert out == hybrid_results_2
        assert retries == 1
        mock_retry.assert_not_called()
        second_call = mock_hs.call_args_list[1]
        assert second_call.kwargs["enabled_providers"] == ["graph"]
        assert second_call.kwargs["seed_results"].keys() == {"vector", "bm25"}

    @patch("agentic_graph_rag.agent.retrieval_agent.evaluate_reflection")
    def test_empty_results_triggers_escalation(self, mock_eval):
        mock_vs = _mock_tool([])
        mock_bm25 = _mock_tool(_make_results(2))
        mock_eval.side_effect = [
            ReflectionStep(
                failure_type="no_results",
                recommended_action="expand_recall",
            ),
            ReflectionStep(
                failure_type="acceptable",
                verdict="answer",
                evidence_status="sufficient",
            ),
        ]

        driver = MagicMock()
        client = MagicMock()
        decision = _make_decision()

        with patch.dict(_TOOL_REGISTRY, {
            "vector_search": mock_vs,
            "bm25_search": mock_bm25,
        }):
            out, retries = self_correction_loop(
                "test", driver, client, decision,
                max_retries=2, relevance_threshold=3.0,
            )
        assert retries == 1

    @patch("agentic_graph_rag.agent.retrieval_agent.evaluate_reflection")
    def test_max_retries_exhausted(self, mock_eval):
        mock_vs = _mock_tool(_make_results(1))
        mock_bm25 = _mock_tool(_make_results(1))
        mock_ct = _mock_tool(_make_results(1))
        mock_hs = _mock_tool(_make_results(1))
        mock_fdr = _mock_tool(_make_results(1))
        mock_eval.return_value = ReflectionStep(
            failure_type="insufficient_recall",
            recommended_action="expand_recall",
        )

        driver = MagicMock()
        client = MagicMock()
        decision = _make_decision()

        with patch.dict(_TOOL_REGISTRY, {
            "vector_search": mock_vs,
            "bm25_search": mock_bm25,
            "cypher_traverse": mock_ct,
            "hybrid_search": mock_hs,
            "full_document_read": mock_fdr,
        }):
            out, retries = self_correction_loop(
                "test", driver, client, decision,
                max_retries=2, relevance_threshold=3.0,
            )
        assert retries == 2

    @patch("agentic_graph_rag.agent.retrieval_agent.evaluate_reflection")
    def test_uses_settings_defaults(self, mock_eval):
        mock_tool = _mock_tool(_make_results(2))
        mock_eval.return_value = ReflectionStep(
            failure_type="acceptable",
            verdict="answer",
            evidence_status="sufficient",
        )

        driver = MagicMock()
        client = MagicMock()
        decision = _make_decision()

        with patch.dict(_TOOL_REGISTRY, {"vector_search": mock_tool}):
            with patch("agentic_graph_rag.agent.retrieval_agent.get_settings") as mock_cfg:
                cfg = MagicMock()
                cfg.agent.max_retries = 1
                cfg.agent.relevance_threshold = 2.0
                mock_cfg.return_value = cfg

                out, retries = self_correction_loop(
                    "test", driver, client, decision,
                )
        assert retries == 0


# ---------------------------------------------------------------------------
# run (full pipeline)
# ---------------------------------------------------------------------------

class TestRun:
    @patch("agentic_graph_rag.agent.retrieval_agent.generate_answer")
    @patch("agentic_graph_rag.agent.retrieval_agent.self_correction_loop")
    @patch("agentic_graph_rag.agent.retrieval_agent.classify_query")
    def test_full_pipeline(self, mock_classify, mock_loop, mock_gen):
        decision = _make_decision()
        mock_classify.return_value = decision
        results = _make_results(3)
        mock_loop.return_value = (results, 0)
        mock_gen.return_value = QAResult(
            answer="Test answer", sources=results, confidence_level="high", evidence_score=0.8, query="test",
        )

        driver = MagicMock()
        client = MagicMock()

        qa = run("test query", driver, openai_client=client)

        assert isinstance(qa, QAResult)
        assert qa.answer == "Test answer"
        assert qa.retries == 0
        assert qa.router_decision == decision
        mock_classify.assert_called_once()
        mock_loop.assert_called_once()
        mock_gen.assert_called_once()

    @patch("agentic_graph_rag.agent.retrieval_agent.generate_answer")
    @patch("agentic_graph_rag.agent.retrieval_agent.self_correction_loop")
    @patch("agentic_graph_rag.agent.retrieval_agent.classify_query")
    def test_uses_llm_router(self, mock_classify, mock_loop, mock_gen):
        mock_classify.return_value = _make_decision()
        mock_loop.return_value = (_make_results(1), 0)
        mock_gen.return_value = QAResult(answer="A", query="q")

        driver = MagicMock()
        client = MagicMock()

        run("q", driver, openai_client=client, use_llm_router=True)
        mock_classify.assert_called_once_with("q", use_llm=True, openai_client=client, reasoning=None)

    @patch("agentic_graph_rag.agent.retrieval_agent.evaluate_completeness", return_value=True)
    @patch("agentic_graph_rag.agent.retrieval_agent.generate_answer")
    @patch("agentic_graph_rag.agent.retrieval_agent.self_correction_loop")
    @patch("agentic_graph_rag.agent.retrieval_agent.classify_query")
    @patch("agentic_graph_rag.agent.retrieval_agent.get_settings")
    def test_creates_client_when_none(self, mock_settings, mock_classify, mock_loop, mock_gen, _mock_compl):
        cfg = MagicMock()
        cfg.openai.api_key = "key"
        cfg.openai.base_url = ""
        mock_settings.return_value = cfg
        mock_classify.return_value = _make_decision()
        mock_loop.return_value = (_make_results(1), 0)
        mock_gen.return_value = QAResult(answer="A", query="q")

        driver = MagicMock()

        with patch("rag_core.config.make_openai_client") as mock_make:
            mock_make.return_value = MagicMock()
            run("q", driver)
            mock_make.assert_called_once_with(cfg)

    @patch("agentic_graph_rag.agent.retrieval_agent.run_agent_workflow")
    def test_does_not_override_workflow_confidence(self, mock_run_workflow):
        qa_result = QAResult(answer="A", query="q", confidence_level="high", evidence_score=0.73)
        mock_run_workflow.return_value = qa_result

        driver = MagicMock()
        client = MagicMock()

        result = run("q", driver, openai_client=client)

        assert result.evidence_score == 0.73
        assert result.confidence_level == "high"


# ---------------------------------------------------------------------------
# Retry matrix structure
# ---------------------------------------------------------------------------

class TestRetryToolMatrix:
    def test_comprehensive_search_can_open_full_document_first(self):
        assert _RETRY_TOOL_MATRIX["comprehensive_search"][0] == "full_document_read"

    def test_full_document_read_has_no_default_follow_up(self):
        assert _RETRY_TOOL_MATRIX["full_document_read"] == []

    def test_comprehensive_search_in_registry(self):
        assert "comprehensive_search" in _TOOL_REGISTRY


# ---------------------------------------------------------------------------
# self_correction_loop with retry query
# ---------------------------------------------------------------------------

class TestRetryQuery:
    @patch("agentic_graph_rag.agent.retrieval_agent.get_settings")
    @patch("agentic_graph_rag.agent.retrieval_agent.generate_retry_query")
    @patch("agentic_graph_rag.agent.retrieval_agent.evaluate_reflection")
    def test_rephrases_query_before_retry_tool_execution(self, mock_eval, mock_retry, mock_get_settings):
        mock_vs = _mock_tool(_make_results(2))
        mock_cs = _mock_tool(_make_results(3))
        mock_bm25 = _mock_tool(_make_results(1))
        mock_cypher = _mock_tool(_make_results(1))
        mock_hybrid = _mock_tool(_make_results(1))
        mock_eval.side_effect = [
            ReflectionStep(
                failure_type="insufficient_context",
                recommended_action="use_comprehensive_search",
                missing_information=["Need broader supporting evidence"],
            ),
            ReflectionStep(
                failure_type="acceptable",
                verdict="answer",
                evidence_status="sufficient",
            ),
        ]
        mock_retry.return_value = "rephrased query"
        mock_get_settings.return_value.agent.max_query_rewrites = 1
        mock_get_settings.return_value.agent.max_reranks = 1
        mock_get_settings.return_value.agent.request_time_budget_ms = 1500

        driver = MagicMock()
        client = MagicMock()
        decision = _make_decision()

        with patch.dict(_TOOL_REGISTRY, {
            "vector_search": mock_vs,
            "comprehensive_search": mock_cs,
            "bm25_search": mock_bm25,
            "cypher_traverse": mock_cypher,
            "hybrid_search": mock_hybrid,
        }):
            out, retries = self_correction_loop(
                "test", driver, client, decision,
                max_retries=2, relevance_threshold=3.0,
            )

        mock_retry.assert_called_once()
        follow_up_calls = []
        for candidate in (mock_bm25, mock_cypher, mock_hybrid, mock_cs):
            if candidate.call_args is not None:
                follow_up_calls.append(candidate.call_args.args[0])
        assert "rephrased query" in follow_up_calls


# ---------------------------------------------------------------------------
# Completeness check in run()
# ---------------------------------------------------------------------------

class TestCompletenessCheck:
    @patch("agentic_graph_rag.agent.retrieval_agent.comprehensive_search")
    @patch("agentic_graph_rag.agent.retrieval_agent.evaluate_completeness")
    @patch("agentic_graph_rag.agent.retrieval_agent.generate_answer")
    @patch("agentic_graph_rag.agent.retrieval_agent.self_correction_loop")
    @patch("agentic_graph_rag.agent.retrieval_agent.classify_query")
    def test_completeness_retry_for_global(self, mock_classify, mock_loop, mock_gen, mock_compl, mock_cs):
        decision = _make_decision(query_type=QueryType.GLOBAL, tool="comprehensive_search")
        mock_classify.return_value = decision
        results = _make_results(3)
        mock_loop.return_value = (results, 0)

        # First generate returns incomplete, second (after comprehensive) returns complete
        qa_incomplete = QAResult(answer="Partial answer", sources=results, confidence_level="medium", evidence_score=0.4, query="list all")
        qa_complete = QAResult(answer="Full answer: A, B, C, D", sources=results, confidence_level="high", evidence_score=0.9, query="list all")
        mock_gen.side_effect = [qa_incomplete, qa_complete]

        # First completeness check fails, second succeeds
        mock_compl.side_effect = [False, True]
        mock_cs.return_value = _make_results(5)

        driver = MagicMock()
        client = MagicMock()

        qa = run("list all items", driver, openai_client=client)

        assert mock_compl.call_count == 0
        mock_cs.assert_called_once()
        assert mock_gen.call_count == 2
        assert qa.retries == 1

    @patch("agentic_graph_rag.agent.retrieval_agent.evaluate_completeness")
    @patch("agentic_graph_rag.agent.retrieval_agent.generate_answer")
    @patch("agentic_graph_rag.agent.retrieval_agent.self_correction_loop")
    @patch("agentic_graph_rag.agent.retrieval_agent.classify_query")
    def test_no_completeness_check_for_simple(self, mock_classify, mock_loop, mock_gen, mock_compl):
        decision = _make_decision(query_type=QueryType.SIMPLE, tool="vector_search")
        mock_classify.return_value = decision
        mock_loop.return_value = (_make_results(3), 0)
        mock_gen.return_value = QAResult(answer="Answer", query="q")

        driver = MagicMock()
        client = MagicMock()

        run("what is X?", driver, openai_client=client)
        # evaluate_completeness should NOT be called for SIMPLE queries
        mock_compl.assert_not_called()
