"""Tests for rag_core.generator."""

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

from rag_core.generator import calculate_answer_confidence, generate_answer, stream_answer
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
        assert result.confidence_level == "low"
        assert result.evidence_score == 0.0
        assert "don't have enough context" in result.answer
        assert result.query == "question?"
        assert result.sources == []
        client.chat.completions.create.assert_not_called()

    def test_generates_answer_from_chunks(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response(
            "The answer based on Chunk 1 is X."
        )

        results = [_make_result("relevant content")]
        qa = generate_answer("What is X?", results, openai_client=client)

        assert qa.answer == "The answer based on Chunk 1 is X."
        assert qa.evidence_score == 0.8
        assert qa.confidence_level == "medium"
        assert qa.query == "What is X?"
        assert len(qa.sources) == 1
        client.chat.completions.create.assert_called_once()

    def test_reflection_verdict_does_not_affect_evidence_score(self):
        """Reflection is a policy decision (answer/retry/stop), not a numeric signal.

        Evidence score should reflect retrieval quality only. Verdict only
        affects the derived confidence_level, not the numeric evidence_score.
        """
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response("answer")

        results = [_make_scored_result("relevant content", 0.8)]
        qa_with_verdict = generate_answer(
            "What is X?",
            results,
            openai_client=client,
            reflection_verdict="answer",
        )
        qa_without_verdict = generate_answer(
            "What is X?",
            results,
            openai_client=client,
            reflection_verdict="",
        )

        # Evidence score is driven by retrieval, not reflection
        assert qa_with_verdict.evidence_score == qa_without_verdict.evidence_score
        assert qa_with_verdict.evidence_score == 0.8

    def test_public_confidence_helper_matches_generate_answer(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response("answer")
        results = [_make_scored_result("relevant content", 0.8)]

        qa = generate_answer(
            "What is X?",
            results,
            openai_client=client,
            reflection_verdict="answer",
        )

        # Both helpers should return the same evidence score (reflection is
        # intentionally ignored at the numeric layer).
        assert calculate_answer_confidence(results) == qa.evidence_score

    def test_confidence_prefers_normalized_scores(self):
        results = [
            SearchResult(
                chunk=Chunk(content="a"),
                score=7.5,
                score_normalized=0.75,
                rank=1,
                source="bm25",
            ),
            SearchResult(
                chunk=Chunk(content="b"),
                score=5.0,
                score_normalized=0.5,
                rank=2,
                source="bm25",
            ),
        ]

        assert calculate_answer_confidence(results) == 0.625

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
        assert "[Chunk 4]" not in user_msg
        assert len(qa.sources) == 3

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

    def test_reorders_context_to_bias_edges_for_high_scores(self):
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

    def test_handles_api_error(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("API down")

        results = [_make_result()]
        qa = generate_answer("q", results, openai_client=client)

        assert qa.confidence_level == "low"
        assert qa.evidence_score == 0.0
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
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            "answer"
        )
        mock_make_client.return_value = mock_client

        qa = generate_answer("q", [_make_result()])
        mock_make_client.assert_called_once_with(cfg)
        assert qa.answer == "answer"


class TestStreamAnswer:
    def test_streams_delta_tokens_from_llm(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_stream_response(
            ["Hello", " ", "world"]
        )

        parts = list(stream_answer("q", [_make_result("ctx")], openai_client=client))

        assert parts == ["Hello", " ", "world"]
        call_args = client.chat.completions.create.call_args
        assert call_args[1]["stream"] is True

    def test_stream_returns_no_tokens_without_results(self):
        client = MagicMock()

        parts = list(stream_answer("q", [], openai_client=client))

        assert parts == []
        client.chat.completions.create.assert_not_called()
