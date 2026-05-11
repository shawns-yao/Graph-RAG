from unittest.mock import MagicMock

from rag_core.models import ClaimVerificationStep, VerifiedClaim

from agentic_graph_rag.agent.correction_planner import (
    CorrectionGap,
    CorrectionPlan,
    build_gap_report,
    plan_correction,
)


def _client(payload: str) -> MagicMock:
    client = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = payload
    client.chat.completions.create.return_value = response
    return client


def test_build_gap_report_marks_missing_numeric_fact_for_hard_numeric_claim():
    verification = ClaimVerificationStep(
        status="partial",
        unsupported_claims=[
            VerifiedClaim(
                text="噻托溴铵18 μg每日1次",
                entities=["噻托溴铵"],
                numeric_constraints=["18 μg", "每日1次"],
                relation_actions=["剂量"],
                supported=False,
                verification_level="possible_correct",
                failure_type="hard_fail",
            )
        ],
    )

    gaps = build_gap_report(verification)

    assert gaps == [
        CorrectionGap(
            gap_type="missing_numeric_fact",
            claim_text="噻托溴铵18 μg每日1次",
            missing_entities=["噻托溴铵"],
            missing_facts=["18 μg", "每日1次"],
            relation_actions=["剂量"],
        )
    ]


def test_plan_correction_accepts_allowlisted_bm25_tool_for_numeric_gap():
    client = _client(
        '{"action":"retry_with_tool","tool":"bm25_search",'
        '"focus_query":"噻托溴铵 18 μg 每日1次","reason":"Need exact numeric evidence"}'
    )
    gaps = [
        CorrectionGap(
            gap_type="missing_numeric_fact",
            claim_text="噻托溴铵18 μg每日1次",
            missing_entities=["噻托溴铵"],
            missing_facts=["18 μg", "每日1次"],
        )
    ]

    plan = plan_correction(
        query="噻托溴铵剂量是多少？",
        answer="噻托溴铵18 μg每日1次。",
        verification_status="partial",
        gaps=gaps,
        openai_client=client,
    )

    assert plan == CorrectionPlan(
        action="retry_with_tool",
        tool="bm25_search",
        focus_query="噻托溴铵 18 μg 每日1次",
        reason="Need exact numeric evidence",
    )


def test_plan_correction_falls_back_to_bm25_when_llm_returns_invalid_tool_for_numeric_gap():
    client = _client(
        '{"action":"retry_with_tool","tool":"web_search",'
        '"focus_query":"噻托溴铵 18 μg 每日1次","reason":"bad tool"}'
    )
    gaps = [
        CorrectionGap(
            gap_type="missing_numeric_fact",
            claim_text="噻托溴铵18 μg每日1次",
            missing_entities=["噻托溴铵"],
            missing_facts=["18 μg", "每日1次"],
        )
    ]

    plan = plan_correction(
        query="噻托溴铵剂量是多少？",
        answer="噻托溴铵18 μg每日1次。",
        verification_status="partial",
        gaps=gaps,
        openai_client=client,
    )

    assert plan.tool == "bm25_search"
    assert plan.focus_query == "噻托溴铵18 μg每日1次 噻托溴铵 18 μg 每日1次"
    assert "fallback" in plan.reason
