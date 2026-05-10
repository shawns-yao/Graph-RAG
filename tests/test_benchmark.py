"""Tests for the rebuilt GraphRAG-Benchmark style evaluation stack."""

from __future__ import annotations

import json
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark.metrics.answer_accuracy import compute_answer_correctness
from benchmark.metrics.coverage import compute_coverage_score
from benchmark.metrics.evidence_recall import compute_evidence_recall
from benchmark.generation_eval import evaluate_dataset as evaluate_generation_dataset
from scripts.run_medical_format_smoke import _print_utf8_payload
from benchmark.runner import (
    _is_empty_generation_row,
    _load_questions,
    _run_mode,
    _split_evidences,
    _write_payload,
    run_benchmark,
)
from rag_core.llm_resilience import LLMCircuitOpenError


class _FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses

    async def ainvoke(self, prompt: str, config=None):  # noqa: ANN001
        del prompt, config
        if not self._responses:
            raise RuntimeError("no fake responses left")
        return SimpleNamespace(content=self._responses.pop(0))


class _FakeEmbeddings:
    def __init__(self, vectors: list[list[float]]) -> None:
        self._vectors = vectors

    async def aembed_query(self, text: str) -> list[float]:
        del text
        if not self._vectors:
            raise RuntimeError("no fake vectors left")
        return self._vectors.pop(0)


def test_load_questions_accepts_graphrag_benchmark_shape(tmp_path: Path) -> None:
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "Medical-1",
                    "question": "What is BCC?",
                    "answer": "Basal cell carcinoma.",
                    "question_type": "Fact Retrieval",
                    "evidence": "Basal cell carcinoma.",
                }
            ]
        ),
        encoding="utf-8",
    )
    rows = _load_questions(str(path))
    assert len(rows) == 1
    assert rows[0]["question_type"] == "Fact Retrieval"


def test_split_evidences_handles_semicolon_payload() -> None:
    assert _split_evidences("one; two ; ; three") == ["one", "two", "three"]


@pytest.mark.asyncio
async def test_answer_correctness_scores_high_for_matching_answer() -> None:
    llm = _FakeLLM(
        [
            '["Basal cell carcinoma is the most common type of skin cancer."]',
            '["Basal cell carcinoma is the most common type of skin cancer."]',
            '{"TP":[{"statement":"Basal cell carcinoma is the most common type of skin cancer.","reason":"same"}],"FP":[],"FN":[]}',
        ]
    )
    embeddings = _FakeEmbeddings([[1.0, 0.0], [1.0, 0.0]])
    score = await compute_answer_correctness(
        "What is the most common type of skin cancer?",
        "Basal cell carcinoma is the most common type of skin cancer.",
        "Basal cell carcinoma is the most common type of skin cancer.",
        llm,
        embeddings,
    )
    assert score > 0.9


@pytest.mark.asyncio
async def test_coverage_score_returns_partial_credit() -> None:
    llm = _FakeLLM(
        [
            '{"facts":["Fact A","Fact B"]}',
            '{"classifications":[{"statement":"Fact A","attributed":1},{"statement":"Fact B","attributed":0}]}',
        ]
    )
    score = await compute_coverage_score("Q", "Reference", "Response", llm)
    assert score == 0.5


@pytest.mark.asyncio
async def test_evidence_recall_returns_average_attribution() -> None:
    llm = _FakeLLM(
        [
            '{"classifications":[{"statement":"E1","reason":"yes","attributed":1},{"statement":"E2","reason":"no","attributed":0}]}'
        ]
    )
    score = await compute_evidence_recall("Q", ["ctx"], ["E1", "E2"], llm)
    assert score == 0.5


@pytest.mark.asyncio
async def test_generation_eval_tolerates_metric_failure() -> None:
    llm = _FakeLLM([])
    embeddings = _FakeEmbeddings([])
    dataset = [
        {
            "question": "Q",
            "answer": "A",
            "contexts": ["ctx"],
            "ground_truth": "GT",
        }
    ]
    result = await evaluate_generation_dataset(
        dataset,
        metrics=["answer_correctness"],
        llm=llm,
        embeddings=embeddings,
    )
    assert result["answer_correctness"] == pytest.approx(0.0)


def test_query_embedding_retry_controller_opens_on_repeated_timeout(monkeypatch):
    from agentic_graph_rag.agent.tools import _embed_query

    fake_cfg = SimpleNamespace(
        openai=SimpleNamespace(embedding_model="demo", embedding_dimensions=3),
        agent=SimpleNamespace(
            max_retries=1,
            request_time_budget_ms=5000,
        ),
    )
    monkeypatch.setattr("agentic_graph_rag.agent.tools.get_settings", lambda: fake_cfg)

    client = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("502 gateway timeout"))
        )
    )

    with pytest.raises(LLMCircuitOpenError):
        _embed_query("What is BCC?", client)


def test_write_payload_falls_back_to_utf8_buffer(monkeypatch):
    class _AsciiStdout:
        encoding = "gbk"

        def __init__(self) -> None:
            self.buffer = io.BytesIO()

        def write(self, _text: str):
            raise UnicodeEncodeError("gbk", "\uf0dc", 0, 1, "illegal multibyte sequence")

        def flush(self):
            return None

    fake_stdout = _AsciiStdout()
    monkeypatch.setattr("benchmark.runner.sys.stdout", fake_stdout)

    _write_payload({"symbol": "\uf0dc"})

    assert '"symbol": "\\uf0dc"' not in fake_stdout.buffer.getvalue().decode("utf-8")


def test_medical_smoke_print_payload_falls_back_to_utf8_buffer(monkeypatch):
    class _AsciiStdout:
        encoding = "gbk"

        def __init__(self) -> None:
            self.buffer = io.BytesIO()

        def write(self, _text: str):
            raise UnicodeEncodeError("gbk", "级", 0, 1, "illegal multibyte sequence")

        def flush(self):
            return None

    fake_stdout = _AsciiStdout()
    monkeypatch.setattr("scripts.run_medical_format_smoke.sys.stdout", fake_stdout)

    _print_utf8_payload({"query": "COPD诊断和分级"})

    assert "COPD诊断和分级" in fake_stdout.buffer.getvalue().decode("utf-8")


def test_empty_generation_row_detects_no_context_fallback() -> None:
    row = {
        "answer": "I don't have enough context to answer this question.",
        "contexts": [],
    }
    assert _is_empty_generation_row(row) is True


def test_empty_generation_row_ignores_non_empty_contexts() -> None:
    row = {
        "answer": "I don't have enough context to answer this question.",
        "contexts": ["real context"],
    }
    assert _is_empty_generation_row(row) is False


def test_run_mode_routes_graph_h2_with_fixed_hops(monkeypatch) -> None:
    captured: dict[str, int] = {}

    def _fake_cypher_traverse(query, driver, client, top_k=None, max_hops=None, entry_top_k=None):  # noqa: ANN001
        del query, driver, client, top_k, entry_top_k
        captured["max_hops"] = max_hops
        return []

    monkeypatch.setattr("benchmark.runner.cypher_traverse", _fake_cypher_traverse)
    monkeypatch.setattr(
        "benchmark.runner.generate_answer",
        lambda question, results, client: SimpleNamespace(answer="A", sources=results),
    )

    _run_mode("graph_h2", "q", SimpleNamespace(), SimpleNamespace())

    assert captured["max_hops"] == 2


def test_run_mode_routes_graph_h3_with_fixed_hops(monkeypatch) -> None:
    captured: dict[str, int] = {}

    def _fake_cypher_traverse(query, driver, client, top_k=None, max_hops=None, entry_top_k=None):  # noqa: ANN001
        del query, driver, client, top_k, entry_top_k
        captured["max_hops"] = max_hops
        return []

    monkeypatch.setattr("benchmark.runner.cypher_traverse", _fake_cypher_traverse)
    monkeypatch.setattr(
        "benchmark.runner.generate_answer",
        lambda question, results, client: SimpleNamespace(answer="A", sources=results),
    )

    _run_mode("graph_h3", "q", SimpleNamespace(), SimpleNamespace())

    assert captured["max_hops"] == 3


def test_run_benchmark_exposes_parallelism_settings(monkeypatch, tmp_path: Path):
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(
        json.dumps(
            [
                {
                    "id": "q1",
                    "question": "What is BCC?",
                    "answer": "Basal cell carcinoma.",
                    "question_type": "Fact Retrieval",
                    "evidence": "Basal cell carcinoma.",
                }
            ]
        ),
        encoding="utf-8",
    )

    fake_cfg = SimpleNamespace(
        neo4j=SimpleNamespace(uri="bolt://fake", user="neo4j", password="neo4j"),
        openai=SimpleNamespace(llm_model_mini="judge", embedding_model="embed"),
    )
    monkeypatch.setattr("benchmark.runner.get_settings", lambda: fake_cfg)
    monkeypatch.setattr("benchmark.runner.make_openai_client", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr("benchmark.runner.GraphDatabase.driver", lambda *args, **kwargs: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr("benchmark.runner.OpenAIAsyncJudge", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr("benchmark.runner.OpenAIAsyncEmbeddings", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(
        "benchmark.runner._run_mode",
        lambda mode, question, driver, client: SimpleNamespace(answer="A", sources=[]),
    )
    async def _fake_score_mode(rows, llm, embeddings, eval_concurrency):  # noqa: ANN001
        del llm, embeddings
        return {
            "rows": len(rows),
            "eval_concurrency": eval_concurrency,
        }

    monkeypatch.setattr("benchmark.runner._score_mode", _fake_score_mode)

    payload = run_benchmark(
        questions_path=str(questions_path),
        modes=["bm25_only"],
        max_workers=2,
        eval_concurrency=3,
    )

    assert payload["max_workers"] == 2
    assert payload["eval_concurrency"] == 3


def test_run_benchmark_aborts_after_consecutive_empty_results(monkeypatch, tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(
        json.dumps(
            [
                {
                    "id": "q1",
                    "question": "Q1",
                    "answer": "A1",
                    "question_type": "Fact Retrieval",
                    "evidence": "E1",
                },
                {
                    "id": "q2",
                    "question": "Q2",
                    "answer": "A2",
                    "question_type": "Fact Retrieval",
                    "evidence": "E2",
                },
                {
                    "id": "q3",
                    "question": "Q3",
                    "answer": "A3",
                    "question_type": "Fact Retrieval",
                    "evidence": "E3",
                },
                {
                    "id": "q4",
                    "question": "Q4",
                    "answer": "A4",
                    "question_type": "Fact Retrieval",
                    "evidence": "E4",
                },
            ]
        ),
        encoding="utf-8",
    )

    fake_cfg = SimpleNamespace(
        neo4j=SimpleNamespace(uri="bolt://fake", user="neo4j", password="neo4j"),
        openai=SimpleNamespace(llm_model_mini="judge", embedding_model="embed"),
    )
    monkeypatch.setattr("benchmark.runner.get_settings", lambda: fake_cfg)
    monkeypatch.setattr("benchmark.runner.make_openai_client", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr("benchmark.runner.GraphDatabase.driver", lambda *args, **kwargs: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr("benchmark.runner.OpenAIAsyncJudge", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr("benchmark.runner.OpenAIAsyncEmbeddings", lambda *args, **kwargs: SimpleNamespace())

    answers = iter(
        [
            SimpleNamespace(
                answer="I don't have enough context to answer this question.",
                sources=[],
            ),
            SimpleNamespace(
                answer="I don't have enough context to answer this question.",
                sources=[],
            ),
            SimpleNamespace(
                answer="I don't have enough context to answer this question.",
                sources=[],
            ),
            SimpleNamespace(
                answer="should never run",
                sources=[],
            ),
        ]
    )
    monkeypatch.setattr("benchmark.runner._run_mode", lambda *args, **kwargs: next(answers))

    async def _fake_score_mode(rows, llm, embeddings, eval_concurrency):  # noqa: ANN001
        del llm, embeddings, eval_concurrency
        return {"rows": len(rows)}

    monkeypatch.setattr("benchmark.runner._score_mode", _fake_score_mode)

    payload = run_benchmark(
        questions_path=str(questions_path),
        modes=["bm25_only"],
        max_workers=1,
        max_consecutive_empty_results=3,
    )

    assert payload["summary"]["bm25_only"]["rows"] == 3
    assert payload["summary"]["bm25_only"]["empty_result_count"] == 3
    assert payload["summary"]["bm25_only"]["processed_questions"] == 3
    assert payload["summary"]["bm25_only"]["requested_questions"] == 4
    assert "aborted after 3 consecutive empty" in payload["summary"]["bm25_only"]["abort_reason"]
