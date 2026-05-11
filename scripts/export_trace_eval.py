#!/usr/bin/env python3
"""Run fixed trace-evaluation questions and export chain behavior as JSON."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

QUESTIONS: list[dict[str, str]] = [
    {
        "id": "graph_direct_001",
        "group": "graph_direct",
        "query": "ACEI 导致干咳时应该如何处理？",
    },
    {
        "id": "graph_direct_002",
        "group": "graph_direct",
        "query": "ACEI、干咳、ARB 之间是什么关系？",
    },
    {
        "id": "graph_direct_003",
        "group": "graph_direct",
        "query": "ARB 相比 ACEI 在干咳方面有什么优势？",
    },
    {
        "id": "numeric_exact_001",
        "group": "numeric_exact",
        "query": "噻托溴铵每日用几次？",
    },
    {
        "id": "numeric_exact_002",
        "group": "numeric_exact",
        "query": "噻托溴铵剂量是多少？",
    },
    {
        "id": "numeric_exact_003",
        "group": "numeric_exact",
        "query": "噻托溴铵 18 μg 每日1次是否正确？",
    },
    {
        "id": "self_correction_001",
        "group": "self_correction",
        "query": "eGFR < 30 时二甲双胍怎么处理？",
    },
    {
        "id": "self_correction_002",
        "group": "self_correction",
        "query": "eGFR 小于30的患者可以用二甲双胍吗？",
    },
    {
        "id": "boundary_001",
        "group": "boundary",
        "query": "二甲双胍什么时候禁用？",
    },
    {
        "id": "boundary_002",
        "group": "boundary",
        "query": "eGFR < 30 时有哪些药物需要注意？",
    },
]


def _anchor_summary(query: str) -> dict[str, Any]:
    from agentic_graph_rag.agent.query_signals import extract_query_signals

    signals = extract_query_signals(query)
    anchors = [anchor.model_dump() for anchor in signals.anchors]
    by_kind: dict[str, int] = {}
    for anchor in anchors:
        kind = anchor["kind"]
        by_kind[kind] = by_kind.get(kind, 0) + 1
    return {"anchors": anchors, "by_kind": by_kind}


def _initial_tools(trace_dump: dict[str, Any]) -> list[str]:
    for entry in trace_dump.get("workflow_memory") or []:
        if entry.get("stage") == "route":
            metadata = entry.get("metadata") or {}
            tools = metadata.get("initial_tools")
            if isinstance(tools, list):
                return [str(tool) for tool in tools]
    return []


def _provider_hits(trace_dump: dict[str, Any]) -> dict[str, int]:
    hits: dict[str, int] = {}
    for step in trace_dump.get("tool_steps") or []:
        tool = step.get("tool_name") or "unknown"
        hits[tool] = hits.get(tool, 0) + int(step.get("results_count") or 0)
        for diag in step.get("provider_diagnostics") or []:
            source = diag.get("source")
            if not source:
                continue
            hits[str(source)] = hits.get(str(source), 0) + int(diag.get("results_count") or 0)
    return hits


def _claim_role_summary(trace_dump: dict[str, Any]) -> dict[str, int]:
    step = trace_dump.get("verification_step") or {}
    roles: dict[str, int] = {}
    for claim in step.get("verified_claims") or []:
        role = str(claim.get("claim_role") or "unknown")
        roles[role] = roles.get(role, 0) + 1
    return roles


def _retry_tools(trace_dump: dict[str, Any]) -> list[str]:
    tools: list[str] = []
    for step in trace_dump.get("reflection_steps") or []:
        required = str(step.get("required_tool") or "")
        if required and required != "none":
            tools.append(required)
        for preferred in step.get("preferred_tools") or []:
            if preferred and preferred not in tools:
                tools.append(str(preferred))
    return tools


def run_one(question: dict[str, str]) -> dict[str, Any]:
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from neo4j import GraphDatabase
    from rag_core.config import get_settings, make_openai_client

    from agentic_graph_rag.service import PipelineService

    load_dotenv(ROOT / ".env")
    started = time.perf_counter()
    cfg = get_settings()
    driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password))
    try:
        service = PipelineService(driver, make_openai_client(cfg))
        qa = service.query(
            question["query"],
            mode="agent_pattern",
            session_id=f"trace-eval::{question['id']}",
        )
        trace_dump = qa.trace.model_dump(mode="json") if qa.trace else {}
        verification = trace_dump.get("verification_step") or {}
        generator = trace_dump.get("generator_step") or {}
        contract = generator.get("evidence_contract") or {}
        citation_coverage = contract.get("citation_coverage") or {}
        row = {
            **question,
            "status": "ok",
            "error": "",
            "duration_seconds": round(time.perf_counter() - started, 2),
            "answer_status": qa.answer_status,
            "retrieval_status": qa.retrieval_status,
            "verification_status": qa.verification_status,
            "source_count": len(qa.sources),
            "retries": qa.retries,
            "generation_duration_ms": generator.get("duration_ms", 0),
            "evidence_contract_fact_count": len(contract.get("facts") or []),
            "citation_coverage": citation_coverage,
            "signal_extractor_output": _anchor_summary(question["query"]),
            "initial_tools": _initial_tools(trace_dump),
            "retrieval_hits_by_provider": _provider_hits(trace_dump),
            "reflection_verdicts": [
                {
                    "verdict": step.get("verdict"),
                    "gap_type": step.get("gap_type"),
                    "failure_type": step.get("failure_type"),
                    "required_tool": step.get("required_tool"),
                    "preferred_tools": step.get("preferred_tools"),
                    "preferred_providers": step.get("preferred_providers"),
                }
                for step in trace_dump.get("reflection_steps") or []
            ],
            "retry_tools": _retry_tools(trace_dump),
            "claim_roles": _claim_role_summary(trace_dump),
            "regenerated": qa.retries > 0 and qa.verification_status in {"passed", "partial", "retry_required"},
            "verification": {
                "status": verification.get("status"),
                "claims_total": verification.get("claims_total", 0),
                "claims_supported": verification.get("claims_supported", 0),
                "claims_possible": verification.get("claims_possible", 0),
                "claims_incorrect": verification.get("claims_incorrect", 0),
                "skipped_reason": verification.get("skipped_reason", ""),
            },
            "answer_preview": (qa.answer or "")[:300],
            "source_previews": [
                {
                    "rank": source.rank,
                    "score": round(source.score, 4),
                    "source": source.source,
                    "content": (source.chunk.content or "")[:260],
                }
                for source in qa.sources[:5]
            ],
            "trace": trace_dump,
        }
        return row
    except Exception as exc:
        return {
            **question,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "duration_seconds": round(time.perf_counter() - started, 2),
            "signal_extractor_output": _anchor_summary(question["query"]),
        }
    finally:
        driver.close()


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, dict[str, int]] = {}
    for row in rows:
        group = row["group"]
        bucket = by_group.setdefault(group, {"total": 0, "ok": 0, "verified": 0, "retried": 0})
        bucket["total"] += 1
        if row.get("status") == "ok":
            bucket["ok"] += 1
        if row.get("answer_status") == "verified":
            bucket["verified"] += 1
        if int(row.get("retries") or 0) > 0:
            bucket["retried"] += 1
    return {
        "total": len(rows),
        "ok": sum(1 for row in rows if row.get("status") == "ok"),
        "verified": sum(1 for row in rows if row.get("answer_status") == "verified"),
        "retried": sum(1 for row in rows if int(row.get("retries") or 0) > 0),
        "by_group": by_group,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="test/medical_benchmark/results/trace_eval_fixed_questions.json",
    )
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--per-question-timeout", type=int, default=120)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    pending = {question["id"]: question for question in QUESTIONS}
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(run_one, question): question for question in QUESTIONS}
        for future in as_completed(futures, timeout=args.per_question_timeout * len(QUESTIONS)):
            question = futures[future]
            pending.pop(question["id"], None)
            try:
                rows.append(future.result(timeout=args.per_question_timeout))
            except TimeoutError:
                rows.append(
                    {
                        **question,
                        "status": "timeout",
                        "error": f"exceeded {args.per_question_timeout}s",
                        "signal_extractor_output": _anchor_summary(question["query"]),
                    }
                )
            output.write_text(
                json.dumps(
                    {
                        "summary": _summarize(rows),
                        "pending": sorted(pending),
                        "elapsed_seconds": round(time.perf_counter() - started, 2),
                        "results": sorted(rows, key=lambda item: item["id"]),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    payload = {
        "summary": _summarize(rows),
        "pending": [],
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "results": sorted(rows, key=lambda item: item["id"]),
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **payload["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
