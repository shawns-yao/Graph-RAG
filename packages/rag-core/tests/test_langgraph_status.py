from rag_core.models import (
    ClaimVerificationStep,
    PipelineTrace,
    QueryType,
    RouterDecision,
    VerifiedClaim,
)

from agentic_graph_rag.agent.langgraph_workflow import (
    AgentWorkflowOps,
    _after_verify,
    _answer_guard_status,
    _answer_has_verifiable_claims,
    _claim_focus_query,
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


def test_after_verify_retries_once_for_incorrect_claim():
    trace = PipelineTrace(trace_id="tr_test", timestamp="2026-05-11T00:00:00Z", query="q")
    trace.verification_step = ClaimVerificationStep(
        claims_total=1,
        claims_incorrect=1,
        status="retry_required",
        unsupported_claims=[
            VerifiedClaim(
                text="错误声明",
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
        targeted_graph_search=lambda *_args, **_kwargs: [],
    )
    state = {
        "trace": trace,
        "ops": ops,
        "decision": RouterDecision(
            query_type=QueryType.SIMPLE,
            suggested_tool="vector_search",
        ),
    }

    assert _after_verify({**state, "verification_retry_attempt": 0}) == "augment_from_verification"
    assert _after_verify({**state, "verification_retry_attempt": 1}) == "finish"


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
