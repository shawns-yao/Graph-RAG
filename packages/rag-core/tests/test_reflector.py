"""Tests for rag_core.reflector."""

import json
from unittest.mock import MagicMock, patch

from rag_core.models import Chunk, ReflectionStep, SearchResult, WorkflowMemoryEntry
from rag_core.reflector import (
    evaluate_completeness,
    evaluate_reflection,
    evaluate_relevance,
    generate_retry_query,
    reflection_to_confidence,
)


def _mock_openai_response(content: str | None) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _make_result(text: str = "chunk content") -> SearchResult:
    return SearchResult(chunk=Chunk(content=text), score=0.8, rank=1)


def _reflection_payload(**overrides) -> str:
    payload = {
        "evidence_status": "sufficient",
        "gap_type": "none",
        "action": "answer",
        "required_tool": "none",
        "missing_information": [],
        "missing_entities": [],
        "missing_relationships": [],
        "coverage_gap_sources": [],
        "candidate_fix_paths": [],
        "preferred_tools": [],
        "preferred_providers": [],
        "reasoning": "ok",
        "failure_type": "acceptable",
        "comparison_to_previous": "n/a",
    }
    payload.update(overrides)
    return json.dumps(payload)


class TestEvaluateReflection:
    def test_empty_results_returns_no_results_reflection(self):
        client = MagicMock()
        step = evaluate_reflection("q", [], openai_client=client, tool_name="vector_search")
        assert step.overall_score == 0.0
        assert step.failure_type == "no_results"
        assert "No evidence retrieved." in step.missing_information
        assert step.should_retry is True
        assert step.should_rewrite_query is True
        client.chat.completions.create.assert_not_called()

    def test_parses_structured_json(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response(
            _reflection_payload(
                evidence_status="partial",
                gap_type="missing_relation",
                action="retry_hybrid",
                required_tool="hybrid_search",
                missing_information=["timeline detail"],
                missing_entities=["Samuel Pepys"],
                missing_relationships=["visited_at"],
                coverage_gap_sources=["graph"],
                candidate_fix_paths=["graph -> rerank"],
                preferred_tools=["hybrid_search", "full_document_read"],
                preferred_providers=["graph", "bm25"],
                reasoning="Most entities are present but one detail is missing.",
                failure_type="insufficient_context",
                comparison_to_previous="better than previous",
            )
        )

        step = evaluate_reflection(
            "q",
            [_make_result("c1"), _make_result("c2")],
            openai_client=client,
            tool_name="hybrid_search",
            attempt=1,
        )

        assert step.tool_name == "hybrid_search"
        assert step.attempt == 1
        assert step.evidence_status == "partial"
        assert step.gap_type == "missing_relation"
        assert step.action == "retry_hybrid"
        assert step.required_tool == "hybrid_search"
        assert step.verdict == "retry"
        assert step.relevance == 3.0
        assert step.entity_completeness == 3.0
        assert step.logical_consistency == 3.0
        assert step.context_sufficiency == 3.0
        assert step.failure_type == "insufficient_context"
        assert step.recommended_action == "use_graph_traversal"
        assert step.missing_information == ["timeline detail"]
        assert step.missing_entities == ["Samuel Pepys"]
        assert step.missing_relationships == ["visited_at"]
        assert step.coverage_gap_sources == ["graph"]
        assert step.candidate_fix_paths == ["graph -> rerank"]
        assert step.preferred_tools == ["hybrid_search", "full_document_read"]
        assert step.preferred_providers == ["graph", "bm25"]
        assert step.retry_scope == "tool_escalation"
        assert step.should_retry is True
        assert step.should_rewrite_query is False
        assert step.should_rerank_again is False
        assert step.overall_score == 3.0
        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["response_format"]["type"] == "json_schema"
        assert call_kwargs["response_format"]["json_schema"]["strict"] is True
        schema = call_kwargs["response_format"]["json_schema"]["schema"]
        assert "evidence_status" in schema["properties"]
        assert "action" in schema["properties"]
        assert "relevance" not in schema["properties"]
        assert "overall_score" not in schema["properties"]

    def test_reflection_to_confidence_maps_weighted_scores(self):
        step = ReflectionStep(
            overall_score=4.175,
            relevance=4.5,
            entity_completeness=4.0,
            logical_consistency=5.0,
            context_sufficiency=3.5,
        )
        assert reflection_to_confidence(step) == 0.845

    def test_falls_back_when_response_format_is_unsupported(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            RuntimeError("response_format json_schema unsupported"),
            _mock_openai_response(_reflection_payload()),
        ]

        step = evaluate_reflection("q", [_make_result()], openai_client=client)

        assert step.verdict == "answer"
        assert client.chat.completions.create.call_count == 2
        first_call = client.chat.completions.create.call_args_list[0].kwargs
        second_call = client.chat.completions.create.call_args_list[1].kwargs
        assert "response_format" in first_call
        assert "response_format" not in second_call

    def test_extracts_first_valid_json_object(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response(
            f"""
            Here is the JSON:
            {_reflection_payload(reasoning="usable")}
            And here is another example:
            {{"wrong": "json"}}
            """
        )

        step = evaluate_reflection("q", [_make_result()], openai_client=client)
        assert step.overall_score == 5.0
        assert step.verdict == "answer"
        assert step.recommended_action == "answer_ready"

    def test_invalid_json_is_stopped_by_policy_guard(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response("not valid json")

        step = evaluate_reflection("q", [_make_result()], openai_client=client)

        assert step.overall_score == 0.0
        assert step.verdict == "stop"
        assert step.failure_type == "insufficient_context"
        assert step.recommended_action == "stop_due_to_invalid_reflection"

    def test_invalid_verdict_is_safely_coerced_to_stop(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response(
            """
            {
              "verdict": "need_more_information",
              "relevance": 4.5,
              "entity_completeness": 4.0,
              "logical_consistency": 4.0,
              "context_sufficiency": 4.0,
              "missing_information": [],
              "missing_entities": [],
              "missing_relationships": [],
              "coverage_gap_sources": [],
              "candidate_fix_paths": [],
              "preferred_tools": ["bm25_search", "totally_fake_tool"],
              "preferred_providers": ["graph", "fake_provider"],
              "retry_scope": "invented_scope",
              "reasoning": "bad verdict",
              "failure_type": "i_dont_know",
              "recommended_action": "expand_recall",
              "should_retry": true,
              "should_rewrite_query": false,
              "should_rerank_again": false,
              "comparison_to_previous": "n/a"
            }
            """
        )

        step = evaluate_reflection("q", [_make_result()], openai_client=client)

        assert step.verdict == "stop"
        assert step.should_retry is False
        assert step.retry_scope == "stop"
        assert step.preferred_tools == []
        assert step.preferred_providers == []

    def test_contradictory_answer_verdict_downgrades_to_stop(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response(
            """
            {
              "verdict": "answer",
              "relevance": 5.0,
              "entity_completeness": 5.0,
              "logical_consistency": 5.0,
              "context_sufficiency": 5.0,
              "missing_information": ["still missing exact value"],
              "missing_entities": [],
              "missing_relationships": [],
              "coverage_gap_sources": [],
              "candidate_fix_paths": [],
              "preferred_tools": [],
              "preferred_providers": [],
              "retry_scope": "stop",
              "reasoning": "contradictory answer",
              "failure_type": "insufficient_context",
              "recommended_action": "answer_ready",
              "should_retry": false,
              "should_rewrite_query": false,
              "should_rerank_again": false,
              "comparison_to_previous": "n/a"
            }
            """
        )

        step = evaluate_reflection("q", [_make_result()], openai_client=client)

        assert step.verdict == "stop"
        assert step.should_retry is False
        assert step.failure_type == "insufficient_context"

    def test_verdict_defaults_to_rerank_for_rerank_only_scope(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response(
            _reflection_payload(
                evidence_status="partial",
                action="rerank",
                gap_type="none",
                reasoning="ranking issue",
                failure_type="insufficient_context",
            )
        )

        step = evaluate_reflection("q", [_make_result()], openai_client=client)

        assert step.verdict == "rerank"
        assert step.should_rerank_again is True

    def test_retry_graph_choice_derives_graph_tool_and_relation_failure(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response(
            _reflection_payload(
                evidence_status="partial",
                gap_type="missing_relation",
                action="retry_graph",
                required_tool="cypher_traverse",
                failure_type="relation_missing",
            )
        )

        step = evaluate_reflection("q", [_make_result()], openai_client=client)

        assert step.verdict == "retry"
        assert step.failure_type == "relation_missing"
        assert step.recommended_action == "use_graph_traversal"
        assert step.preferred_tools[0] == "cypher_traverse"
        assert step.retry_scope == "tool_escalation"
        assert step.should_retry is True

    def test_handles_api_error(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("API error")

        step = evaluate_reflection("q", [_make_result()], openai_client=client)
        assert step.overall_score == 0.0
        assert step.verdict == "stop"
        assert step.failure_type == "insufficient_context"

    def test_includes_workflow_memory_in_reflection_prompt(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response(
            _reflection_payload()
        )

        _ = evaluate_reflection(
            "q",
            [_make_result()],
            openai_client=client,
            workflow_memory=[
                WorkflowMemoryEntry(
                    stage="retrieval",
                    message="vector recall missed one entity",
                    metadata={"tool": "vector_search"},
                )
            ],
        )

        call_args = client.chat.completions.create.call_args
        user_msg = call_args[1]["messages"][0]["content"]
        assert "Workflow memory" in user_msg
        assert "vector recall missed one entity" in user_msg

    @patch("rag_core.reflector.make_openai_client")
    @patch("rag_core.reflector.get_settings")
    def test_creates_client_when_none(self, mock_settings, mock_make_client):
        cfg = MagicMock()
        cfg.openai.llm_model_mini = "gpt-4o-mini"
        mock_settings.return_value = cfg

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            _reflection_payload()
        )
        mock_make_client.return_value = mock_client

        step = evaluate_reflection("q", [_make_result()])
        mock_make_client.assert_called_once_with(cfg)
        assert step.overall_score == 5.0


class TestEvaluateRelevance:
    def test_wraps_structured_reflection_score(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response(
            _reflection_payload()
        )

        score = evaluate_relevance("q", [_make_result()], openai_client=client)
        assert score == 5.0


class TestGenerateRetryQuery:
    def test_generates_targeted_retry_query(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response(
            "Bob projects and overlap with Alice projects"
        )
        reflection = ReflectionStep(
            overall_score=1.5,
            failure_type="missing_entity",
            recommended_action="target_missing_entity",
            missing_information=["Bob's project membership"],
        )

        retry = generate_retry_query(
            "original",
            [_make_result("partial content")],
            openai_client=client,
            reflection=reflection,
        )
        assert retry == "Bob projects and overlap with Alice projects"

    def test_empty_results_mentions_no_content(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response("retry q")
        reflection = ReflectionStep(
            overall_score=0.0,
            failure_type="no_results",
            recommended_action="expand_recall",
            missing_information=["No evidence retrieved."],
        )

        retry = generate_retry_query(
            "original",
            [],
            openai_client=client,
            reflection=reflection,
        )
        assert retry == "retry q"

        call_args = client.chat.completions.create.call_args
        user_msg = call_args[1]["messages"][0]["content"]
        assert "No relevant content found" in user_msg
        assert "No evidence retrieved." in user_msg

    def test_retry_prompt_includes_workflow_memory(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response("retry q")
        reflection = ReflectionStep(
            overall_score=1.0,
            failure_type="insufficient_recall",
            recommended_action="expand_recall",
        )

        _ = generate_retry_query(
            "original",
            [_make_result("partial content")],
            openai_client=client,
            reflection=reflection,
            workflow_memory=[
                WorkflowMemoryEntry(
                    stage="reflection",
                    message="graph channel returned zero edges",
                    metadata={"provider": "graph"},
                )
            ],
        )
        call_args = client.chat.completions.create.call_args
        user_msg = call_args[1]["messages"][0]["content"]
        assert "Workflow memory" in user_msg
        assert "graph channel returned zero edges" in user_msg

    def test_handles_api_error_returns_original(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("fail")
        reflection = ReflectionStep(
            overall_score=1.0,
            failure_type="insufficient_recall",
            recommended_action="expand_recall",
        )

        result = generate_retry_query(
            "original query",
            [],
            openai_client=client,
            reflection=reflection,
        )
        assert result == "original query"

    def test_handles_none_content(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response(None)
        reflection = ReflectionStep(
            overall_score=1.0,
            failure_type="insufficient_recall",
            recommended_action="expand_recall",
        )

        result = generate_retry_query(
            "fallback",
            [],
            openai_client=client,
            reflection=reflection,
        )
        assert result == "fallback"

    @patch("rag_core.reflector.make_openai_client")
    @patch("rag_core.reflector.get_settings")
    def test_creates_client_when_none(self, mock_settings, mock_make_client):
        cfg = MagicMock()
        cfg.openai.llm_model = "gpt-4o-mini"
        mock_settings.return_value = cfg

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response("better query")
        mock_make_client.return_value = mock_client

        result = generate_retry_query(
            "q",
            [],
            reflection=ReflectionStep(
                overall_score=1.0,
                failure_type="insufficient_recall",
                recommended_action="expand_recall",
            ),
        )
        mock_make_client.assert_called_once_with(cfg)
        assert result == "better query"


class TestEvaluateCompleteness:
    def test_returns_false_when_answer_admits_missing_context(self):
        client = MagicMock()
        assert evaluate_completeness(
            "COPD的诊断和分级需要哪些检查指标？",
            "Available evidence covers part of the question, but the retrieval guard blocked further expansion.",
            openai_client=client,
        ) is False
        client.chat.completions.create.assert_not_called()

    def test_returns_true_for_structured_enumeration_without_llm(self):
        client = MagicMock()
        answer = "\n".join(
            [
                "1. 肺功能检查：用于确诊。",
                "2. FEV1/FVC：用于判断是否存在持续气流受限。",
                "3. FEV1占预计值百分比：用于GOLD分级。",
            ]
        )
        assert evaluate_completeness(
            "COPD的诊断和分级需要哪些检查指标？这些指标如何使用？",
            answer,
            openai_client=client,
        ) is True
        client.chat.completions.create.assert_not_called()

    def test_returns_false_for_short_enumeration_answer_without_llm(self):
        client = MagicMock()
        assert evaluate_completeness(
            "COPD的诊断和分级需要哪些检查指标？",
            "需要肺功能检查。",
            openai_client=client,
        ) is False
        client.chat.completions.create.assert_not_called()

    def test_returns_true_when_yes(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response(
            "YES, the answer covers all aspects."
        )
        assert evaluate_completeness("what is X", "X is a chronic disease.", openai_client=client) is True

    def test_returns_false_when_no(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response(
            "NO, the answer only mentions 2 out of 5 items."
        )
        assert evaluate_completeness("list all X", "Here are X: A, B", openai_client=client) is False

    def test_returns_true_on_error_to_avoid_extra_retries(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("boom")
        assert evaluate_completeness("q", "answer", openai_client=client) is True

    @patch("rag_core.reflector.make_openai_client")
    @patch("rag_core.reflector.get_settings")
    def test_creates_client_when_none(self, mock_settings, mock_make_client):
        cfg = MagicMock()
        cfg.openai.llm_model_mini = "gpt-4o-mini"
        mock_settings.return_value = cfg

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response("YES, complete")
        mock_make_client.return_value = mock_client

        result = evaluate_completeness("q", "answer")
        mock_make_client.assert_called_once_with(cfg)
        assert result is True
