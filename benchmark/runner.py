"""Local benchmark runner using GraphRAG-Benchmark style evaluation metrics."""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys
import time
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase

from benchmark.adapters import OpenAIAsyncEmbeddings, OpenAIAsyncJudge
from benchmark.generation_eval import evaluate_dataset as evaluate_generation_dataset
from benchmark.retrieval_eval import evaluate_dataset as evaluate_retrieval_dataset
from rag_core.config import get_settings, make_openai_client
from rag_core.generator import generate_answer

from agentic_graph_rag.agent.retrieval_agent import run as agent_run
from agentic_graph_rag.agent.tools import bm25_search, cypher_traverse, hybrid_search, vector_search

QUESTION_TYPE_TO_GENERATION_METRICS = {
    "Fact Retrieval": ["rouge_score", "answer_correctness"],
    "Relation Reasoning": ["rouge_score", "answer_correctness"],
    "Multi-hop Reasoning": ["rouge_score", "answer_correctness"],
    "Temporal Reasoning": ["rouge_score", "answer_correctness"],
    "Global Summarization": ["answer_correctness", "coverage_score"],
}

EMPTY_ANSWER_PREFIX = "I don't have enough context to answer this question."
DEFAULT_MAX_CONSECUTIVE_EMPTY_RESULTS = 3
LEGACY_QUESTIONS_PATH = Path(__file__).with_name("questions.json")
MEDICAL_QUESTIONS_PATH = (
    Path(__file__).resolve().parent.parent
    / "test"
    / "medical_benchmark"
    / "questions_master.json"
)


def _write_payload(payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    try:
        print(text)
    except UnicodeEncodeError:
        if getattr(sys.stdout, "buffer", None) is not None:
            sys.stdout.buffer.write((text + "\n").encode("utf-8"))
            sys.stdout.buffer.flush()
        else:
            raise


def _load_questions(path: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("questions"), list):
        payload = payload["questions"]
    if not isinstance(payload, list):
        raise ValueError("questions file must be a JSON array")
    questions: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        row = dict(item)
        if "question" not in row and "query" in row:
            row["question"] = row["query"]
        if "question_type" in row:
            row["question_type"] = _normalize_question_type(str(row["question_type"]))
        if "evidence" not in row and isinstance(row.get("evidence_chunks"), list):
            row["evidence"] = "; ".join(str(value) for value in row["evidence_chunks"])
        questions.append(row)
    return questions


def load_questions(path: str | None = None) -> list[dict[str, Any]]:
    """Load legacy benchmark questions when no explicit path is supplied."""
    if path is not None:
        return _load_questions(path)
    if LEGACY_QUESTIONS_PATH.exists():
        return _load_questions(str(LEGACY_QUESTIONS_PATH))
    return _load_questions(str(MEDICAL_QUESTIONS_PATH))


def _normalize_question_type(value: str) -> str:
    mapping = {
        "simple": "Fact Retrieval",
        "fact": "Fact Retrieval",
        "fact retrieval": "Fact Retrieval",
        "relation": "Relation Reasoning",
        "relation reasoning": "Relation Reasoning",
        "multi_hop": "Multi-hop Reasoning",
        "multihop": "Multi-hop Reasoning",
        "multi-hop reasoning": "Multi-hop Reasoning",
        "temporal": "Temporal Reasoning",
        "temporal reasoning": "Temporal Reasoning",
        "global": "Global Summarization",
        "global summarization": "Global Summarization",
    }
    return mapping.get(value.strip().casefold(), value)


def _split_evidences(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(";") if part.strip()]


def _run_mode(mode: str, question: str, driver: Any, client: Any) -> Any:
    if mode == "graph_only":
        results = cypher_traverse(question, driver, client)
    elif mode == "graph_h2":
        results = cypher_traverse(question, driver, client, max_hops=2)
    elif mode == "graph_h3":
        results = cypher_traverse(question, driver, client, max_hops=3)
    elif mode == "vector_only":
        results = vector_search(question, driver, client)
    elif mode == "bm25_only":
        results = bm25_search(question, driver, client)
    elif mode == "hybrid":
        results = hybrid_search(question, driver, client, rerank_enabled=False)
    elif mode == "hybrid_rerank":
        results = hybrid_search(question, driver, client, rerank_enabled=True)
    elif mode == "vector_chain":
        return agent_run(
            question,
            driver,
            openai_client=client,
            use_llm_router=False,
            forced_tool="vector_search",
        )
    elif mode == "graph_chain":
        return agent_run(
            question,
            driver,
            openai_client=client,
            use_llm_router=False,
            forced_tool="cypher_traverse",
        )
    elif mode == "bm25_chain":
        return agent_run(
            question,
            driver,
            openai_client=client,
            use_llm_router=False,
            forced_tool="bm25_search",
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return generate_answer(question, results, client)


def _legacy_vector(question: str, driver: Any, client: Any) -> Any:
    return generate_answer(question, vector_search(question, driver, client), client)


def _legacy_cypher(question: str, driver: Any, client: Any) -> Any:
    return generate_answer(question, cypher_traverse(question, driver, client), client)


def _legacy_hybrid(question: str, driver: Any, client: Any) -> Any:
    return generate_answer(question, hybrid_search(question, driver, client), client)


def _legacy_agent_pattern(question: str, driver: Any, client: Any) -> Any:
    return agent_run(question, driver, openai_client=client, use_llm_router=False)


def _legacy_agent_llm(question: str, driver: Any, client: Any) -> Any:
    return agent_run(question, driver, openai_client=client, use_llm_router=True)


def _legacy_agent_mangle(question: str, driver: Any, client: Any) -> Any:
    return agent_run(question, driver, openai_client=client, use_llm_router=False, reasoning="mangle")


MODES = {
    "vector": _legacy_vector,
    "cypher": _legacy_cypher,
    "hybrid": _legacy_hybrid,
    "agent_pattern": _legacy_agent_pattern,
    "agent_llm": _legacy_agent_llm,
    "agent_mangle": _legacy_agent_mangle,
}


def _is_empty_generation_row(row: dict[str, Any]) -> bool:
    if row["contexts"]:
        return False
    answer = row["answer"].strip()
    return not answer or answer.startswith(EMPTY_ANSWER_PREFIX)


def _to_eval_row(question: dict[str, Any], qa: Any, latency: float) -> dict[str, Any]:
    contexts = [
        (result.chunk.enriched_content or result.chunk.content or "").strip()
        for result in qa.sources
        if (result.chunk.enriched_content or result.chunk.content or "").strip()
    ]
    return {
        "id": question.get("id"),
        "question": str(question.get("question") or ""),
        "question_type": str(question.get("question_type") or "Fact Retrieval"),
        "answer": qa.answer,
        "ground_truth": str(question.get("answer") or ""),
        "contexts": contexts,
        "evidences": _split_evidences(str(question.get("evidence") or "")),
        "latency": latency,
    }

async def _score_mode(
    rows: list[dict[str, Any]],
    llm: Any,
    embeddings: Any,
    *,
    eval_concurrency: int,
) -> dict[str, Any]:
    grouped_rows: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        metrics = tuple(
            QUESTION_TYPE_TO_GENERATION_METRICS.get(
                row["question_type"],
                ["rouge_score", "answer_correctness"],
            )
        )
        grouped_rows.setdefault(metrics, []).append(row)

    generation_scores: dict[str, list[float]] = {}
    for metrics, metric_rows in grouped_rows.items():
        result = await evaluate_generation_dataset(
            metric_rows,
            list(metrics),
            llm,
            embeddings,
            max_concurrent=eval_concurrency,
        )
        for metric_name, score in result.items():
            generation_scores.setdefault(metric_name, []).append(score)

    retrieval_scores = await evaluate_retrieval_dataset(
        rows,
        llm,
        max_concurrent=eval_concurrency,
    )
    avg_latency = sum(row["latency"] for row in rows) / len(rows) if rows else 0.0
    return {
        "generation": {
            metric_name: sum(scores) / len(scores)
            for metric_name, scores in generation_scores.items()
        },
        "retrieval": retrieval_scores,
        "avg_latency": avg_latency,
        "total": len(rows),
    }


def run_benchmark(
    *,
    questions_path: str,
    modes: list[str],
    limit: int | None = None,
    max_workers: int = 1,
    eval_concurrency: int = 1,
    max_consecutive_empty_results: int = DEFAULT_MAX_CONSECUTIVE_EMPTY_RESULTS,
) -> dict[str, Any]:
    cfg = get_settings()
    driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password))
    client = make_openai_client(cfg, profile="benchmark")
    async_llm = OpenAIAsyncJudge(client, model=cfg.openai.llm_model_mini)
    async_embeddings = OpenAIAsyncEmbeddings(client, model=cfg.openai.embedding_model)
    questions = _load_questions(questions_path)
    if limit is not None:
        questions = questions[:limit]

    raw_results: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}
    mode_stats: dict[str, dict[str, Any]] = {}
    try:
        for mode in modes:
            consecutive_empty_results = 0
            empty_result_count = 0
            abort_reason = ""

            def _execute(question: dict[str, Any]) -> dict[str, Any]:
                started = time.perf_counter()
                qa = _run_mode(mode, str(question.get("question") or ""), driver, client)
                latency = time.perf_counter() - started
                return _to_eval_row(question, qa, latency)

            if max_workers <= 1:
                for question in questions:
                    row = _execute(question)
                    raw_results[mode].append(row)
                    if _is_empty_generation_row(row):
                        empty_result_count += 1
                        consecutive_empty_results += 1
                        if consecutive_empty_results >= max_consecutive_empty_results:
                            abort_reason = (
                                f"aborted after {consecutive_empty_results} consecutive empty retrieval/generation results"
                            )
                            break
                    else:
                        consecutive_empty_results = 0
                mode_stats[mode] = {
                    "empty_result_count": empty_result_count,
                    "processed_questions": len(raw_results[mode]),
                    "requested_questions": len(questions),
                    "abort_reason": abort_reason,
                }
                continue

            ordered_rows: list[dict[str, Any] | None] = [None] * len(questions)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {
                    executor.submit(_execute, question): index
                    for index, question in enumerate(questions)
                }
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    ordered_rows[index] = future.result()
            raw_results[mode] = [row for row in ordered_rows if row is not None]
            for row in raw_results[mode]:
                if _is_empty_generation_row(row):
                    empty_result_count += 1
                    consecutive_empty_results += 1
                    if consecutive_empty_results >= max_consecutive_empty_results and not abort_reason:
                        abort_reason = (
                            "parallel execution hit consecutive empty retrieval/generation "
                            f"threshold ({max_consecutive_empty_results})"
                        )
                else:
                    consecutive_empty_results = 0
            mode_stats[mode] = {
                "empty_result_count": empty_result_count,
                "processed_questions": len(raw_results[mode]),
                "requested_questions": len(questions),
                "abort_reason": abort_reason,
            }
    finally:
        driver.close()

    summary: dict[str, Any] = {}
    for mode, rows in raw_results.items():
        summary[mode] = asyncio.run(
            _score_mode(
                rows,
                async_llm,
                async_embeddings,
                eval_concurrency=eval_concurrency,
            )
        )
        summary[mode]["empty_result_count"] = mode_stats.get(mode, {}).get("empty_result_count", 0)
        summary[mode]["processed_questions"] = mode_stats.get(mode, {}).get("processed_questions", len(rows))
        summary[mode]["requested_questions"] = mode_stats.get(mode, {}).get("requested_questions", len(rows))
        summary[mode]["abort_reason"] = mode_stats.get(mode, {}).get("abort_reason", "")
    return {
        "questions_path": questions_path,
        "modes": modes,
        "max_workers": max_workers,
        "eval_concurrency": eval_concurrency,
        "max_consecutive_empty_results": max_consecutive_empty_results,
        "summary": summary,
        "results": raw_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GraphRAG-Benchmark style evaluation locally")
    parser.add_argument("--questions", required=True, help="Path to GraphRAG-Benchmark question JSON")
    parser.add_argument(
        "--modes",
        default="graph_only,graph_h2,graph_h3,vector_only,bm25_only,hybrid,hybrid_rerank,vector_chain,graph_chain,bm25_chain",
        help="Comma-separated retrieval modes",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max questions to run")
    parser.add_argument("--max-workers", type=int, default=1, help="Execution concurrency per mode")
    parser.add_argument("--eval-concurrency", type=int, default=1, help="Metric evaluation concurrency")
    parser.add_argument(
        "--max-consecutive-empty-results",
        type=int,
        default=DEFAULT_MAX_CONSECUTIVE_EMPTY_RESULTS,
        help="Abort a mode after this many consecutive empty retrieval/generation results",
    )
    parser.add_argument("--output", default="", help="Optional output JSON file")
    args = parser.parse_args()

    payload = run_benchmark(
        questions_path=args.questions,
        modes=[mode.strip() for mode in args.modes.split(",") if mode.strip()],
        limit=args.limit,
        max_workers=max(1, args.max_workers),
        eval_concurrency=max(1, args.eval_concurrency),
        max_consecutive_empty_results=max(1, args.max_consecutive_empty_results),
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_payload(payload)


if __name__ == "__main__":
    main()
