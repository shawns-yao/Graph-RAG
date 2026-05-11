from unittest.mock import MagicMock

from rag_core.models import Chunk, SearchResult

from agentic_graph_rag.generation.claim_verifier import (
    ExtractedClaim,
    _verification_evidence_text,
    extract_claims,
    verify_claims,
)


def _result(text: str) -> SearchResult:
    return SearchResult(chunk=Chunk(id="c1", content=text), score=1.0)


def _client(verdict: str = "correct") -> MagicMock:
    client = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = f'{{"verdict": "{verdict}"}}'
    client.chat.completions.create.return_value = response
    return client


def _cypher_noop(*_args, **_kwargs):
    return []


def test_extract_claims_includes_question_context_in_prompt():
    client = _client("correct")
    client.chat.completions.create.return_value.choices[0].message.content = (
        '{"claims": [{"text": "GOLD 3-4级且嗜酸性粒细胞≥100/μL推荐LABA+LAMA+ICS", '
        '"role": "core", '
        '"entities": ["GOLD 3-4级", "嗜酸性粒细胞", "LABA", "LAMA", "ICS"], '
        '"numeric_constraints": ["≥100/μL"], "relation_actions": ["推荐"]}]}'
    )

    result = extract_claims(
        "推荐方案是LABA+LAMA+ICS三联治疗。",
        query="GOLD 3-4级患者如果嗜酸性粒细胞≥100/μL，推荐什么治疗方案？",
        openai_client=client,
    )

    prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "Question:" in prompt
    assert "GOLD 3-4级患者" in prompt
    assert "Background explanation is not core" in prompt
    assert result.claims[0].entities[0] == "GOLD 3-4级"
    assert result.claims[0].role == "core"


def test_extract_claims_defaults_invalid_role_to_supporting():
    client = _client("correct")
    client.chat.completions.create.return_value.choices[0].message.content = (
        '{"claims": [{"text": "背景说明不是直接答案", '
        '"role": "everything", '
        '"entities": [], "numeric_constraints": [], "relation_actions": []}]}'
    )

    result = extract_claims("背景说明不是直接答案。", openai_client=client)

    assert result.claims[0].role == "supporting"


def test_verify_claim_carries_claim_role_into_verified_claim():
    claim = ExtractedClaim(
        text="噻托溴铵18μg每日1次",
        role="core",
        entities=("噻托溴铵",),
        numeric_constraints=("18μg",),
        relation_actions=("每日1次",),
    )

    step = verify_claims(
        [claim],
        cypher_traverse=_cypher_noop,
        driver=MagicMock(),
        openai_client=_client(),
        existing_evidence=[_result("沙丁胺醇起效5-15分钟")],
    )

    assert step.unsupported_claims[0].claim_role == "core"


def test_possible_correct_when_entity_or_number_missing_after_retry():
    claim = ExtractedClaim(
        text="噻托溴铵18μg每日1次",
        entities=("噻托溴铵",),
        numeric_constraints=("18μg",),
        relation_actions=("每日1次",),
    )

    step = verify_claims(
        [claim],
        cypher_traverse=_cypher_noop,
        driver=MagicMock(),
        openai_client=_client(),
        existing_evidence=[_result("沙丁胺醇起效5-15分钟")],
    )

    assert step.status == "partial"
    assert step.claims_possible == 1
    assert step.claims_incorrect == 0
    assert step.unsupported_claims[0].verification_level == "possible_correct"
    assert step.unsupported_claims[0].failure_type == "hard_fail"


def test_hard_fail_can_be_promoted_to_correct_by_targeted_retrieval():
    claim = ExtractedClaim(
        text="噻托溴铵18μg每日1次",
        entities=("噻托溴铵",),
        numeric_constraints=("18μg",),
        relation_actions=("每日1次",),
    )

    def cypher_retrieve(*_args, **_kwargs):
        return [_result("噻托溴铵（Tiotropium）：18 μg，每天一次。")]

    step = verify_claims(
        [claim],
        cypher_traverse=cypher_retrieve,
        driver=MagicMock(),
        openai_client=_client("correct"),
        existing_evidence=[_result("沙丁胺醇起效5-15分钟")],
    )

    assert step.status == "passed"
    assert step.claims_supported == 1
    assert step.claims_possible == 0
    assert step.verified_claims[0].verification_level == "correct"


def test_canonical_quantitative_fact_check_normalizes_equivalent_frequency_forms():
    claim = ExtractedClaim(
        text="噻托溴铵每日使用1次",
        entities=("噻托溴铵",),
        numeric_constraints=("1次/日",),
        relation_actions=("每日使用",),
    )

    step = verify_claims(
        [claim],
        cypher_traverse=_cypher_noop,
        driver=MagicMock(),
        openai_client=_client("correct"),
        existing_evidence=[_result("噻托溴铵 --剂量--> 18 μg每日1次")],
    )

    assert step.status == "passed"
    assert step.claims_supported == 1


def test_possible_correct_when_relation_is_unclear():
    claim = ExtractedClaim(
        text="噻托溴铵18μg每日1次",
        entities=("噻托溴铵",),
        numeric_constraints=("18μg",),
        relation_actions=("每日1次",),
    )

    step = verify_claims(
        [claim],
        cypher_traverse=_cypher_noop,
        driver=MagicMock(),
        openai_client=_client("possible_correct"),
        existing_evidence=[_result("噻托溴铵（Tiotropium）：18μg。")],
    )

    assert step.status == "partial"
    assert step.claims_possible == 1
    assert step.claims_incorrect == 0
    assert step.unsupported_claims[0].verification_level == "possible_correct"
    assert step.unsupported_claims[0].failure_type == "hard_fail"


def test_direct_relation_evidence_bypasses_soft_llm():
    claim = ExtractedClaim(
        text="推荐治疗方案是LABA+LAMA+ICS三联方案",
        entities=("LABA", "LAMA", "ICS"),
        numeric_constraints=(),
        relation_actions=("推荐", "治疗方案"),
    )
    client = _client("possible_correct")

    step = verify_claims(
        [claim],
        cypher_traverse=_cypher_noop,
        driver=MagicMock(),
        openai_client=client,
        existing_evidence=[
            _result("GOLD 3-4级且嗜酸性粒细胞≥100/μL --推荐治疗--> LABA+LAMA+ICS")
        ],
    )

    assert step.status == "passed"
    assert step.claims_supported == 1
    assert step.claims_possible == 0
    client.chat.completions.create.assert_not_called()


def test_incorrect_when_relation_is_contradicted():
    claim = ExtractedClaim(
        text="噻托溴铵18μg每日1次",
        entities=("噻托溴铵",),
        numeric_constraints=("18μg",),
        relation_actions=("每日1次",),
    )

    step = verify_claims(
        [claim],
        cypher_traverse=_cypher_noop,
        driver=MagicMock(),
        openai_client=_client("incorrect"),
        existing_evidence=[_result("噻托溴铵（Tiotropium）：18μg，每周一次。")],
    )

    assert step.status == "retry_required"
    assert step.claims_possible == 0
    assert step.claims_incorrect == 1
    assert step.unsupported_claims[0].verification_level == "incorrect"
    assert step.unsupported_claims[0].failure_type == "hard_fail"
    _client("incorrect").chat.completions.create.assert_not_called()


def test_passed_when_canonical_facts_and_entities_match():
    claim = ExtractedClaim(
        text="噻托溴铵18μg每日1次",
        entities=("噻托溴铵",),
        numeric_constraints=("18μg",),
        relation_actions=("每日1次",),
    )

    client = _client("incorrect")
    step = verify_claims(
        [claim],
        cypher_traverse=_cypher_noop,
        driver=MagicMock(),
        openai_client=client,
        existing_evidence=[_result("噻托溴铵（Tiotropium）：18 μg，每天一次。")],
    )

    assert step.status == "passed"
    assert step.claims_supported == 1
    assert step.claims_possible == 0
    assert step.claims_incorrect == 0
    assert step.verified_claims[0].verification_level == "correct"
    assert step.verified_claims[0].failure_type == "none"
    client.chat.completions.create.assert_not_called()


def test_verification_evidence_text_prefers_evidence_section():
    content = (
        "Graph paths:\nA -[rel]-> B\n\n"
        "Entities:\nA (Drug)\n\n"
        "Evidence:\n原文证据句。"
    )

    assert _verification_evidence_text(content) == "原文证据句。"
