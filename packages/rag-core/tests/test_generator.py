"""Tests for rag_core.generator."""

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

from rag_core.generator import generate_answer, stream_answer
from rag_core.models import Chunk, SearchResult


def _mock_openai_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _make_result(text: str = "chunk content") -> SearchResult:
    return SearchResult(chunk=Chunk(content=text), score=0.8, rank=1)


def _make_scored_result(label: str, score: float) -> SearchResult:
    return SearchResult(chunk=Chunk(content=label), score=score, rank=1)


def _make_graph_result(text: str, score: float = 0.8) -> SearchResult:
    return SearchResult(chunk=Chunk(content=text), score=score, rank=1, source="graph")


class _StreamChunk:
    def __init__(self, content: str | None) -> None:
        delta = MagicMock()
        delta.content = content
        choice = MagicMock()
        choice.delta = delta
        self.choices = [choice]


def _mock_stream_response(parts: list[str]) -> Iterator[_StreamChunk]:
    for part in parts:
        yield _StreamChunk(part)


class TestGenerateAnswer:
    def test_no_results_returns_fallback(self):
        client = MagicMock()
        result = generate_answer("question?", [], openai_client=client)
        assert result.answer_status == "failed"
        assert result.retrieval_status == "empty"
        assert "don't have enough context" in result.answer
        assert result.query == "question?"
        assert result.sources == []
        client.chat.completions.create.assert_not_called()

    def test_generates_answer_from_chunks(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response("The answer based on Chunk 1 is X.")

        results = [_make_result("relevant content")]
        qa = generate_answer("What is X?", results, openai_client=client)

        assert qa.answer == "The answer based on Chunk 1 is X."
        assert qa.answer_status == "unverified"
        assert qa.retrieval_status == "complete"
        assert qa.query == "What is X?"
        assert len(qa.sources) == 1
        client.chat.completions.create.assert_called_once()

    def test_reflection_verdict_sets_discrete_answer_status(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response("answer")

        results = [_make_scored_result("relevant content", 0.8)]
        qa_retry = generate_answer(
            "What is X?",
            results,
            openai_client=client,
            reflection_verdict="retry",
        )
        qa_answer = generate_answer(
            "What is X?",
            results,
            openai_client=client,
            reflection_verdict="answer",
        )

        assert qa_retry.answer_status == "retry_required"
        assert qa_answer.answer_status == "unverified"

    def test_includes_all_chunks_in_context(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response("answer")

        results = [_make_result(f"content {i}") for i in range(3)]
        generate_answer("q", results, openai_client=client)

        call_args = client.chat.completions.create.call_args
        user_msg = call_args[1]["messages"][1]["content"]
        assert "[Chunk 1]" in user_msg
        assert "[Chunk 2]" in user_msg
        assert "[Chunk 3]" in user_msg

    @patch("rag_core.generator.get_settings")
    def test_caps_chunks_before_prompt(self, mock_settings):
        cfg = MagicMock()
        cfg.retrieval.prompt_max_chunks = 10
        cfg.retrieval.prompt_max_chars = 10_000
        cfg.openai.llm_model = "test-model"
        cfg.openai.llm_temperature = 0.0
        mock_settings.return_value = cfg

        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response("answer")
        results = [_make_result(f"content {i}") for i in range(4)]

        qa = generate_answer("q", results, openai_client=client)

        user_msg = client.chat.completions.create.call_args[1]["messages"][1]["content"]
        assert "[Chunk 1]" in user_msg
        assert "[Chunk 2]" in user_msg
        assert "[Chunk 3]" in user_msg
        assert "[Chunk 4]" in user_msg
        assert len(qa.sources) == 4

    @patch("rag_core.generator.get_settings")
    def test_short_query_does_not_drop_strong_anchor_evidence(self, mock_settings):
        cfg = MagicMock()
        cfg.retrieval.prompt_max_chunks = 4
        cfg.retrieval.prompt_max_chars = 10_000
        cfg.openai.llm_model = "test-model"
        cfg.openai.llm_temperature = 0.0
        mock_settings.return_value = cfg

        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response("answer")
        results = [
            _make_scored_result("generic high score", 0.99),
            _make_scored_result("another generic chunk", 0.98),
            _make_scored_result("more generic content", 0.97),
            _make_scored_result("FEV1/FVC < 0.70 is diagnostic evidence", 0.2),
            _make_scored_result("tail generic content", 0.1),
        ]

        qa = generate_answer("FEV1/FVC < 0.70?", results, openai_client=client)

        user_msg = client.chat.completions.create.call_args[1]["messages"][1]["content"]
        assert "FEV1/FVC < 0.70 is diagnostic evidence" in user_msg
        assert len(qa.sources) == 4

    @patch("rag_core.generator.get_settings")
    def test_phrase_anchor_orders_subject_evidence_before_generic_threshold_chunks(self, mock_settings):
        cfg = MagicMock()
        cfg.retrieval.prompt_max_chunks = 2
        cfg.retrieval.prompt_max_chars = 10_000
        cfg.openai.llm_model = "test-model"
        cfg.openai.llm_temperature = 0.0
        mock_settings.return_value = cfg

        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response("answer")
        results = [
            _make_scored_result("COPD eGFR < 30 unrelated high score", 0.99),
            _make_scored_result("噻托溴铵 eGFR < 30 unrelated high score", 0.98),
            _make_scored_result("二甲双胍 eGFR < 30 时禁用", 0.20),
        ]

        qa = generate_answer("eGFR < 30 时二甲双胍怎么处理？", results, openai_client=client)

        user_msg = client.chat.completions.create.call_args[1]["messages"][1]["content"]
        assert "二甲双胍 eGFR < 30 时禁用" in user_msg
        assert "COPD eGFR < 30 unrelated high score" in user_msg
        assert "噻托溴铵 eGFR < 30 unrelated high score" not in user_msg
        assert qa.sources[0].chunk.content == "二甲双胍 eGFR < 30 时禁用"

    @patch("rag_core.generator.get_settings")
    def test_caps_enumeration_queries_more_aggressively(self, mock_settings):
        cfg = MagicMock()
        cfg.retrieval.prompt_max_chunks = 10
        cfg.retrieval.prompt_max_chars = 20_000
        cfg.openai.llm_model = "test-model"
        cfg.openai.llm_temperature = 0.0
        mock_settings.return_value = cfg

        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response("answer")
        results = [_make_result(f"content {i}") for i in range(8)]

        qa = generate_answer("list all COPD indicators", results, openai_client=client)

        assert len(qa.sources) == 5

    @patch("rag_core.generator.get_settings")
    def test_caps_chunks_by_prompt_character_budget(self, mock_settings):
        cfg = MagicMock()
        cfg.retrieval.prompt_max_chunks = 10
        cfg.retrieval.prompt_max_chars = 15
        cfg.openai.llm_model = "test-model"
        cfg.openai.llm_temperature = 0.0
        mock_settings.return_value = cfg

        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response("answer")
        results = [
            _make_scored_result("1234567890", 0.9),
            _make_scored_result("abcdefghij", 0.8),
            _make_scored_result("klmnopqrst", 0.7),
        ]

        qa = generate_answer("q", results, openai_client=client)

        user_msg = client.chat.completions.create.call_args[1]["messages"][1]["content"]
        assert "1234567890" in user_msg
        assert "abcdefghij" not in user_msg
        assert "klmnopqrst" not in user_msg
        assert len(qa.sources) == 1

    @patch("rag_core.generator.get_settings")
    def test_reorders_context_to_bias_edges_for_high_scores(self, mock_settings):
        cfg = MagicMock()
        cfg.retrieval.prompt_max_chunks = 3
        cfg.retrieval.prompt_max_chars = 10_000
        cfg.openai.llm_model = "test-model"
        cfg.openai.llm_temperature = 0.0
        mock_settings.return_value = cfg

        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response("answer")

        results = [
            _make_scored_result("alpha", 0.95),
            _make_scored_result("beta", 0.90),
            _make_scored_result("gamma", 0.80),
            _make_scored_result("delta", 0.70),
        ]
        generate_answer("q", results, openai_client=client)

        call_args = client.chat.completions.create.call_args
        user_msg = call_args[1]["messages"][1]["content"]
        alpha_pos = user_msg.index("alpha")
        beta_pos = user_msg.index("beta")
        gamma_pos = user_msg.index("gamma")
        assert "delta" not in user_msg
        assert alpha_pos < gamma_pos < beta_pos

    @patch("rag_core.generator.get_settings")
    def test_compresses_large_graph_evidence_before_prompt(self, mock_settings):
        cfg = MagicMock()
        cfg.retrieval.prompt_max_chunks = 10
        cfg.retrieval.prompt_max_chars = 1800
        cfg.openai.llm_model = "test-model"
        cfg.openai.llm_temperature = 0.0
        mock_settings.return_value = cfg

        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response("answer")
        graph_text = (
            "Graph paths:\n"
            + "\n".join(f"A{i} -[REL]-> B{i}" for i in range(10))
            + "\n\nEntities:\n"
            + "\n".join(f"Entity{i} (Type)" for i in range(12))
            + "\n\nEvidence:\n"
            + ("Long evidence paragraph. " * 200)
        )

        generate_answer("q", [_make_graph_result(graph_text)], openai_client=client)

        user_msg = client.chat.completions.create.call_args[1]["messages"][1]["content"]
        assert len(user_msg) < len(graph_text)
        assert "A0 -[REL]-> B0" in user_msg
        assert "Entity0 (Type)" in user_msg
        assert "Long evidence paragraph." in user_msg

    def test_completeness_instruction_in_prompt(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response("answer")

        results = [_make_result("text")]
        generate_answer("list all items", results, openai_client=client)

        call_args = client.chat.completions.create.call_args
        system_msg = call_args[1]["messages"][0]["content"]
        assert "enumeration" in system_msg.lower()
        assert "NUMBERED LIST" in system_msg

    def test_non_enumeration_prompt_limits_unasked_facts(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response("answer")

        generate_answer("噻托溴铵每日使用几次？剂量是多少？", [_make_result("text")], openai_client=client)

        call_args = client.chat.completions.create.call_args
        system_msg = call_args[1]["messages"][0]["content"]
        user_msg = call_args[1]["messages"][1]["content"]
        assert "Answer only the fields explicitly asked" in system_msg
        assert "do not add related facts" in system_msg
        assert "Do not include facts that are not needed" in user_msg

    def test_handles_api_error(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("API down")

        results = [_make_result()]
        qa = generate_answer("q", results, openai_client=client)

        assert qa.answer_status == "failed"
        assert qa.retrieval_status == "complete"
        assert "Error" in qa.answer
        assert len(qa.sources) == 1

    def test_handles_none_content(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response(None)

        results = [_make_result()]
        qa = generate_answer("q", results, openai_client=client)
        assert qa.answer == ""

    @patch("rag_core.generator.make_openai_client")
    @patch("rag_core.generator.get_settings")
    def test_creates_client_when_none(self, mock_settings, mock_make_client):
        cfg = MagicMock()
        cfg.retrieval.prompt_max_chunks = 10
        cfg.retrieval.prompt_max_chars = 10_000
        cfg.openai.llm_model = "gpt-4o-mini"
        cfg.openai.llm_temperature = 0.1
        mock_settings.return_value = cfg

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response("answer")
        mock_make_client.return_value = mock_client

        qa = generate_answer("q", [_make_result()])
        mock_make_client.assert_called_once_with(cfg)
        assert qa.answer == "answer"


class TestStreamAnswer:
    def test_streams_delta_tokens_from_llm(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_stream_response(["Hello", " ", "world"])

        parts = list(stream_answer("q", [_make_result("ctx")], openai_client=client))

        assert parts == ["Hello", " ", "world"]
        call_args = client.chat.completions.create.call_args
        assert call_args[1]["stream"] is True

    def test_stream_returns_no_tokens_without_results(self):
        client = MagicMock()

        parts = list(stream_answer("q", [], openai_client=client))

        assert parts == []
        client.chat.completions.create.assert_not_called()


def test_phrase_anchor_balances_two_entity_relation_evidence():
    cfg = MagicMock()
    cfg.retrieval.prompt_max_chunks = 2
    cfg.retrieval.prompt_max_chars = 10_000
    cfg.openai.llm_model = "test-model"
    cfg.openai.llm_temperature = 0.0

    with patch("rag_core.generator.get_settings", return_value=cfg):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response("answer")
        results = [
            _make_scored_result("ACEI 可导致干咳，需要停用", 0.99),
            _make_scored_result("泛化背景内容", 0.98),
            _make_scored_result("ARB 干咳风险较低，可作为替代", 0.20),
        ]

        qa = generate_answer("ACEI 和 ARB 的区别是什么？", results, openai_client=client)

    user_msg = client.chat.completions.create.call_args[1]["messages"][1]["content"]
    assert "ACEI 可导致干咳，需要停用" in user_msg
    assert "ARB 干咳风险较低，可作为替代" in user_msg
    assert "泛化背景内容" not in user_msg
    assert [source.chunk.content for source in qa.sources] == [
        "ACEI 可导致干咳，需要停用",
        "ARB 干咳风险较低，可作为替代",
    ]


def test_build_evidence_contract_extracts_graph_and_numeric_facts():
    from rag_core.generator import build_evidence_contract

    results = [
        _make_graph_result("- ACEI --不良反应--> 干咳"),
        _make_scored_result("噻托溴铵 --剂量--> 18 μg每日1次", 0.9),
    ]

    contract = build_evidence_contract(results)

    assert len(contract.facts) >= 2
    assert any("ACEI --不良反应--> 干咳" in fact.text for fact in contract.facts)
    assert any("18 μg" in fact.text for fact in contract.facts)
    assert contract.completeness_status == "complete"


def test_evidence_contract_keeps_structured_fact_when_subject_matches_query():
    from rag_core.generator import build_evidence_contract

    results = [_make_scored_result("噻托溴铵 --剂量--> 18 μg每日1次", 0.9)]

    contract = build_evidence_contract(results, query="噻托溴铵每日用几次？")

    assert any("噻托溴铵 --剂量--> 18 μg每日1次" in fact.text for fact in contract.facts)

def test_prompt_includes_evidence_contract_and_fact_citation_instruction():
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_openai_response("answer [fact:f_1_1_c1]")
    results = [_make_scored_result("噻托溴铵 --剂量--> 18 μg每日1次", 0.9)]
    results[0].chunk.id = "c1"

    qa = generate_answer("噻托溴铵剂量是多少？", results, openai_client=client)

    user_msg = client.chat.completions.create.call_args[1]["messages"][1]["content"]
    assert "Evidence Contract:" in user_msg
    assert "[fact:" in user_msg
    assert "Attach [fact:<id>]" in user_msg
    assert qa.evidence_contract is not None
    assert qa.evidence_contract.facts
    assert qa.evidence_contract.citation_coverage["coverage_status"] == "passed"


def test_contract_citation_check_reports_unknown_fact_ids():
    from rag_core.generator import build_evidence_contract, check_contract_citations

    contract = build_evidence_contract([_make_scored_result("FEV1/FVC < 0.70", 0.9)])
    checked = check_contract_citations("answer [fact:not_real]", contract)

    assert checked.citation_coverage["coverage_status"] == "partial"
    assert checked.citation_coverage["unknown_fact_ids"] == ["not_real"]
