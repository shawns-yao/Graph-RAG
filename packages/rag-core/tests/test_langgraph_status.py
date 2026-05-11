from rag_core.models import (
    Chunk,
    ClaimVerificationStep,
    PipelineTrace,
    QAResult,
    QueryType,
    RouterDecision,
    SearchResult,
    ToolStep,
    VerifiedClaim,
    WorkflowMemoryEntry,
)

from agentic_graph_rag.agent.budget import BudgetTracker
from agentic_graph_rag.agent.correction_planner import CorrectionGap, CorrectionPlan
from agentic_graph_rag.agent.langgraph_workflow import (
    AgentWorkflowOps,
    _after_execute_correction_tool,
    _after_verify,
    _answer_guard_status,
    _answer_has_verifiable_claims,
    _claim_focus_query,
    _execute_correction_tool_node,
    _generate_answer_node,
    _initial_tool_plan,
    _plan_correction_node,
    _retrieve_evidence,
    _verify_answer_node,
)


def test_answer_guard_timeout_does_not_invalidate_complete_answer():
    status = _answer_guard_status(
        "time budget exhausted before reflection",
        has_answer=True,
        retrieval_status="complete",
    )

    assert status == "partial"


def test_answer_guard_timeout_without_answer_is_timeout():
    status = _answer_guard_status(
        "time budget exhausted before reflection",
        has_answer=False,
        retrieval_status="empty",
    )

    assert status == "skipped_timeout"


def test_answer_guard_transport_failure_is_partial():
    status = _answer_guard_status(
        "Reflection policy guard: Connection error.",
        has_answer=True,
        retrieval_status="complete",
    )

    assert status == "partial"


def test_short_numeric_threshold_answer_is_verifiable():
    assert _answer_has_verifiable_claims("FEV1/FVC < 0.70即可确诊。")


def test_short_dose_answer_is_verifiable():
    assert _answer_has_verifiable_claims("噻托溴铵18 μg，每日1次。")


def test_short_non_factual_answer_is_not_verifiable():
    assert not _answer_has_verifiable_claims("可以。")


def test_after_verify_requires_planner_for_retry_required_claim():
    trace = PipelineTrace(trace_id="tr_test", timestamp="2026-05-11T00:00:00Z", query="q")
    trace.verification_step = ClaimVerificationStep(
        claims_total=1,
        claims_incorrect=1,
        status="retry_required",
        unsupported_claims=[
            VerifiedClaim(
                text="错误声明",
                claim_role="core",
                supported=False,
                verification_level="incorrect",
                failure_type="soft_fail",
            )
        ],
    )
    ops = AgentWorkflowOps(
        classify_query=lambda *_args, **_kwargs: None,
        is_cross_language_global=lambda *_args, **_kwargs: False,
        run_self_correction=lambda *_args, **_kwargs: ([], 0),
        generate_answer=lambda *_args, **_kwargs: None,
        evaluate_completeness=lambda *_args, **_kwargs: True,
        comprehensive_search=lambda *_args, **_kwargs: [],
    )
    state = {
        "trace": trace,
        "ops": ops,
        "decision": RouterDecision(
            query_type=QueryType.SIMPLE,
            suggested_tool="vector_search",
        ),
    }

    assert _after_verify({**state, "verification_retry_attempt": 0}) == "finish"
    assert _after_verify({**state, "verification_retry_attempt": 1}) == "finish"


def test_after_verify_prefers_planner_when_available():
    trace = PipelineTrace(trace_id="tr_test", timestamp="2026-05-11T00:00:00Z", query="q")
    trace.verification_step = ClaimVerificationStep(
        claims_total=1,
        claims_incorrect=1,
        status="retry_required",
        unsupported_claims=[
            VerifiedClaim(
                text="错误声明",
                claim_role="core",
                supported=False,
                verification_level="incorrect",
                failure_type="soft_fail",
            )
        ],
    )
    ops = AgentWorkflowOps(
        classify_query=lambda *_args, **_kwargs: None,
        is_cross_language_global=lambda *_args, **_kwargs: False,
        run_self_correction=lambda *_args, **_kwargs: ([], 0),
        generate_answer=lambda *_args, **_kwargs: None,
        evaluate_completeness=lambda *_args, **_kwargs: True,
        comprehensive_search=lambda *_args, **_kwargs: [],
        plan_correction=lambda *_args, **_kwargs: None,
        run_correction_tool=lambda *_args, **_kwargs: [],
    )
    state = {
        "trace": trace,
        "ops": ops,
        "decision": RouterDecision(
            query_type=QueryType.SIMPLE,
            suggested_tool="vector_search",
        ),
        "verification_retry_attempt": 0,
        "correction_gaps": [
            CorrectionGap(
                gap_type="missing_entity",
                claim_text="错误声明",
                claim_role="core",
            )
        ],
    }

    assert _after_verify(state) == "plan_correction"


def test_after_verify_retries_partial_numeric_gap_with_planner():
    trace = PipelineTrace(trace_id="tr_test", timestamp="2026-05-11T00:00:00Z", query="q")
    trace.verification_step = ClaimVerificationStep(
        claims_total=1,
        claims_possible=1,
        status="partial",
        unsupported_claims=[
            VerifiedClaim(
                text="eGFR < 30时存在用药禁忌",
                claim_role="core",
                numeric_constraints=["< 30"],
                supported=False,
                verification_level="possible_correct",
                failure_type="hard_fail",
            )
        ],
    )
    ops = AgentWorkflowOps(
        classify_query=lambda *_args, **_kwargs: None,
        is_cross_language_global=lambda *_args, **_kwargs: False,
        run_self_correction=lambda *_args, **_kwargs: ([], 0),
        generate_answer=lambda *_args, **_kwargs: None,
        evaluate_completeness=lambda *_args, **_kwargs: True,
        comprehensive_search=lambda *_args, **_kwargs: [],
        plan_correction=lambda *_args, **_kwargs: None,
        run_correction_tool=lambda *_args, **_kwargs: [],
    )
    state = {
        "trace": trace,
        "ops": ops,
        "decision": RouterDecision(query_type=QueryType.SIMPLE, suggested_tool="vector_search"),
        "verification_retry_attempt": 0,
        "correction_gaps": [
            CorrectionGap(
                gap_type="missing_numeric_fact",
                claim_text="eGFR < 30时存在用药禁忌",
                claim_role="core",
                missing_facts=["< 30"],
            )
        ],
    }

    assert _after_verify(state) == "plan_correction"


def test_after_verify_does_not_retry_supporting_gap():
    trace = PipelineTrace(trace_id="tr_test", timestamp="2026-05-11T00:00:00Z", query="q")
    trace.verification_step = ClaimVerificationStep(
        claims_total=1,
        claims_possible=1,
        status="partial",
        unsupported_claims=[
            VerifiedClaim(
                text="背景发生率为15-20%",
                claim_role="supporting",
                numeric_constraints=["15-20%"],
                supported=False,
                verification_level="possible_correct",
                failure_type="hard_fail",
            )
        ],
    )
    ops = AgentWorkflowOps(
        classify_query=lambda *_args, **_kwargs: None,
        is_cross_language_global=lambda *_args, **_kwargs: False,
        run_self_correction=lambda *_args, **_kwargs: ([], 0),
        generate_answer=lambda *_args, **_kwargs: None,
        evaluate_completeness=lambda *_args, **_kwargs: True,
        comprehensive_search=lambda *_args, **_kwargs: [],
        plan_correction=lambda *_args, **_kwargs: None,
        run_correction_tool=lambda *_args, **_kwargs: [],
    )
    state = {
        "trace": trace,
        "ops": ops,
        "decision": RouterDecision(query_type=QueryType.SIMPLE, suggested_tool="vector_search"),
        "verification_retry_attempt": 0,
        "correction_gaps": [
            CorrectionGap(
                gap_type="missing_numeric_fact",
                claim_text="背景发生率为15-20%",
                claim_role="supporting",
                missing_facts=["15-20%"],
            )
        ],
    }

    assert _after_verify(state) == "finish"


def test_execute_correction_tool_uses_planned_tool_and_appends_unique_results():
    base = SearchResult(chunk=Chunk(id="base", content="旧证据"), score=0.5)
    extra = SearchResult(chunk=Chunk(id="extra", content="噻托溴铵18 μg每日1次"), score=1.0)
    calls = []

    def run_correction_tool(tool, query, *_args, **_kwargs):
        calls.append((tool, query))
        return [extra]

    ops = AgentWorkflowOps(
        classify_query=lambda *_args, **_kwargs: None,
        is_cross_language_global=lambda *_args, **_kwargs: False,
        run_self_correction=lambda *_args, **_kwargs: ([], 0),
        generate_answer=lambda *_args, **_kwargs: None,
        evaluate_completeness=lambda *_args, **_kwargs: True,
        comprehensive_search=lambda *_args, **_kwargs: [],
        run_correction_tool=run_correction_tool,
    )
    trace = PipelineTrace(trace_id="tr_test", timestamp="2026-05-11T00:00:00Z", query="q")
    state = {
        "ops": ops,
        "query": "噻托溴铵剂量是多少？",
        "qa_result": QAResult(answer="噻托溴铵18 μg每日1次。"),
        "trace": trace,
        "driver": object(),
        "openai_client": object(),
        "decision": RouterDecision(query_type=QueryType.SIMPLE, suggested_tool="vector_search"),
        "results": [base],
        "existing_ids": ["base"],
        "retries": 0,
        "memory": [],
        "correction_gaps": [
            CorrectionGap(
                gap_type="missing_numeric_fact",
                claim_text="噻托溴铵18 μg每日1次",
                missing_entities=["噻托溴铵"],
                missing_facts=["18 μg", "每日1次"],
            )
        ],
        "correction_plan": CorrectionPlan(
            action="retry_with_tool",
            tool="bm25_search",
            focus_query="噻托溴铵 18 μg 每日1次",
            reason="Need exact numeric evidence",
        ),
    }

    update = _execute_correction_tool_node(state)

    assert calls == [("bm25_search", "噻托溴铵 18 μg 每日1次")]
    assert [result.chunk.id for result in update["results"]] == ["base", "extra"]
    assert update["verification_retry_attempt"] == 1
    assert update["total_retries"] == 1
    assert update["correction_added_results"] == 1
    assert trace.tool_steps[-1].tool_name == "bm25_search"


def test_execute_correction_tool_marks_zero_added_results():
    base = SearchResult(chunk=Chunk(id="base", content="旧证据"), score=0.5)

    def run_correction_tool(*_args, **_kwargs):
        return []

    ops = AgentWorkflowOps(
        classify_query=lambda *_args, **_kwargs: None,
        is_cross_language_global=lambda *_args, **_kwargs: False,
        run_self_correction=lambda *_args, **_kwargs: ([], 0),
        generate_answer=lambda *_args, **_kwargs: None,
        evaluate_completeness=lambda *_args, **_kwargs: True,
        comprehensive_search=lambda *_args, **_kwargs: [],
        run_correction_tool=run_correction_tool,
    )

    update = _execute_correction_tool_node(
        {
            "ops": ops,
            "query": "eGFR < 30 怎么办？",
            "trace": PipelineTrace(trace_id="tr_test", timestamp="2026-05-11T00:00:00Z", query="q"),
            "driver": object(),
            "openai_client": object(),
            "decision": RouterDecision(query_type=QueryType.SIMPLE, suggested_tool="vector_search"),
            "results": [base],
            "existing_ids": ["base"],
            "retries": 0,
            "memory": [],
            "correction_plan": CorrectionPlan(
                action="retry_with_tool",
                tool="bm25_search",
                focus_query="eGFR < 30",
                reason="Need exact threshold evidence",
            ),
        }
    )

    assert update["correction_added_results"] == 0
    assert _after_execute_correction_tool(update) == "finish"


def test_initial_tool_plan_adds_bm25_for_strong_anchor():
    decision = RouterDecision(query_type=QueryType.SIMPLE, suggested_tool="vector_search")

    assert _initial_tool_plan("FEV1/FVC < 0.70 说明什么？", decision) == [
        "vector_search",
        "bm25_search",
    ]


def test_initial_tool_plan_does_not_add_bm25_for_phrase_only_query():
    decision = RouterDecision(query_type=QueryType.SIMPLE, suggested_tool="vector_search")

    assert _initial_tool_plan("噻托溴铵剂量是多少？", decision) == ["vector_search"]


def test_initial_tool_plan_preserves_router_tool_when_threshold_present():
    decision = RouterDecision(query_type=QueryType.RELATION, suggested_tool="cypher_traverse")

    assert _initial_tool_plan("eGFR < 30 怎么办？", decision) == [
        "cypher_traverse",
        "bm25_search",
    ]


def test_retrieve_evidence_runs_companion_tool_and_merges_results():
    calls = []

    def run_self_correction(*, decision, trace, memory_sink, reflection_history_sink, **_kwargs):
        calls.append(decision.suggested_tool)
        memory_sink.append(
            WorkflowMemoryEntry(
                stage="retrieval",
                message=f"{decision.suggested_tool} done",
            )
        )
        result = SearchResult(
            chunk=Chunk(id=decision.suggested_tool, content=decision.suggested_tool),
            score=1.0,
            source=decision.suggested_tool,
        )
        trace.tool_steps.append(
            ToolStep(
                tool_name=decision.suggested_tool,
                results_count=1,
            )
        )
        return [result], 0

    ops = AgentWorkflowOps(
        classify_query=lambda *_args, **_kwargs: None,
        is_cross_language_global=lambda *_args, **_kwargs: False,
        run_self_correction=run_self_correction,
        generate_answer=lambda *_args, **_kwargs: None,
        evaluate_completeness=lambda *_args, **_kwargs: True,
        comprehensive_search=lambda *_args, **_kwargs: [],
    )
    trace = PipelineTrace(trace_id="tr_test", timestamp="2026-05-11T00:00:00Z", query="q")

    update = _retrieve_evidence(
        {
            "query": "FEV1/FVC < 0.70 说明什么？",
            "driver": object(),
            "openai_client": object(),
            "decision": RouterDecision(query_type=QueryType.SIMPLE, suggested_tool="vector_search"),
            "trace": trace,
            "ops": ops,
            "memory": [],
            "reflection_history": [],
        }
    )

    assert calls == ["vector_search", "bm25_search"]
    assert [result.chunk.id for result in update["results"]] == [
        "vector_search",
        "bm25_search",
    ]
    assert [step.tool_name for step in trace.tool_steps] == ["vector_search", "bm25_search"]


def test_generate_answer_skips_when_llm_budget_exhausted():
    calls = []
    ops = AgentWorkflowOps(
        classify_query=lambda *_args, **_kwargs: None,
        is_cross_language_global=lambda *_args, **_kwargs: False,
        run_self_correction=lambda *_args, **_kwargs: ([], 0),
        generate_answer=lambda *_args, **_kwargs: calls.append("generate"),
        evaluate_completeness=lambda *_args, **_kwargs: True,
        comprehensive_search=lambda *_args, **_kwargs: [],
    )
    budget = BudgetTracker(max_llm_calls=0)

    update = _generate_answer_node(
        {
            "ops": ops,
            "query": "q",
            "results": [SearchResult(chunk=Chunk(id="c1", content="evidence"), score=1.0)],
            "openai_client": object(),
            "reflection_history": [],
            "memory": [],
            "budget": budget,
        }
    )

    assert calls == []
    assert update["qa_result"].answer_status == "partial"
    assert update["memory"][-1].stage == "budget"


def test_verify_answer_skips_claim_extraction_when_llm_budget_exhausted():
    calls = []
    trace = PipelineTrace(trace_id="tr_test", timestamp="2026-05-11T00:00:00Z", query="q")
    ops = AgentWorkflowOps(
        classify_query=lambda *_args, **_kwargs: None,
        is_cross_language_global=lambda *_args, **_kwargs: False,
        run_self_correction=lambda *_args, **_kwargs: ([], 0),
        generate_answer=lambda *_args, **_kwargs: None,
        evaluate_completeness=lambda *_args, **_kwargs: True,
        comprehensive_search=lambda *_args, **_kwargs: [],
        extract_claims=lambda *_args, **_kwargs: calls.append("extract"),
        verify_claims=lambda *_args, **_kwargs: None,
    )

    update = _verify_answer_node(
        {
            "ops": ops,
            "query": "q",
            "qa_result": QAResult(answer="FEV1/FVC < 0.70。"),
            "trace": trace,
            "memory": [],
            "budget": BudgetTracker(max_llm_calls=0),
        }
    )

    assert calls == []
    assert trace.verification_step.skipped_reason == "llm_budget_exhausted"
    assert update["qa_result"].verification_status == "skipped"


def test_plan_correction_skips_when_llm_budget_exhausted():
    calls = []
    trace = PipelineTrace(trace_id="tr_test", timestamp="2026-05-11T00:00:00Z", query="q")
    trace.verification_step = ClaimVerificationStep(status="retry_required")
    ops = AgentWorkflowOps(
        classify_query=lambda *_args, **_kwargs: None,
        is_cross_language_global=lambda *_args, **_kwargs: False,
        run_self_correction=lambda *_args, **_kwargs: ([], 0),
        generate_answer=lambda *_args, **_kwargs: None,
        evaluate_completeness=lambda *_args, **_kwargs: True,
        comprehensive_search=lambda *_args, **_kwargs: [],
        plan_correction=lambda *_args, **_kwargs: calls.append("plan"),
    )

    update = _plan_correction_node(
        {
            "ops": ops,
            "query": "q",
            "qa_result": QAResult(answer="answer"),
            "trace": trace,
            "openai_client": object(),
            "correction_gaps": [CorrectionGap(gap_type="missing_entity", claim_text="claim")],
            "memory": [],
            "budget": BudgetTracker(max_llm_calls=0),
        }
    )

    assert calls == []
    assert "correction_plan" not in update
    assert update["memory"][-1].stage == "budget"


def test_claim_focus_query_deduplicates_structured_terms():
    claim = VerifiedClaim(
        text="达比加群在eGFR<30时禁用",
        entities=["达比加群"],
        numeric_constraints=["eGFR<30"],
        relation_actions=["禁用"],
        key_terms=["达比加群", "eGFR<30", "禁用"],
        supported=False,
        verification_level="incorrect",
        failure_type="soft_fail",
    )

    assert _claim_focus_query(claim) == "达比加群在eGFR<30时禁用 达比加群 eGFR<30 禁用"
