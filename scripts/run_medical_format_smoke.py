#!/usr/bin/env python3
"""Run a small md/docx/pdf smoke benchmark against the medical benchmark corpus."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_graph_rag.service import PipelineService  # noqa: E402
from rag_core.config import get_settings, make_openai_client  # noqa: E402
from rag_core.neo4j_utils import open_neo4j_session  # noqa: E402
from scripts.ingest import ingest_file  # noqa: E402


QUESTION_FILES = [
    ("simple", ROOT / "test" / "medical_benchmark" / "questions_doc_001_simple.json"),
    ("relation", ROOT / "test" / "medical_benchmark" / "questions_doc_001_relation.json"),
    ("multi_hop", ROOT / "test" / "medical_benchmark" / "questions_doc_001_multihop.json"),
    ("global_temporal", ROOT / "test" / "medical_benchmark" / "questions_doc_001_global_temporal.json"),
]

EXPORT_DIR = ROOT / "test" / "medical_benchmark" / "corpus_exports"
OUTPUT_PATH = ROOT / ".tmp" / "medical_format_smoke.json"
logger = logging.getLogger("medical_format_smoke")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _print_utf8_payload(payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    try:
        print(text)
    except UnicodeEncodeError:
        if getattr(sys.stdout, "buffer", None) is not None:
            sys.stdout.buffer.write((text + "\n").encode("utf-8"))
            sys.stdout.buffer.flush()
        else:
            raise


def _load_smoke_questions() -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    for label, path in QUESTION_FILES:
        rows = _load_json(path)
        if not isinstance(rows, list) or not rows:
            continue
        row = dict(rows[0])
        row["_source_group"] = label
        questions.append(row)
    return questions


def _clear_database() -> None:
    cfg = get_settings()
    driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password))
    try:
        with open_neo4j_session(driver) as session:
            session.run("MATCH (n) DETACH DELETE n")
    finally:
        driver.close()


def _collect_graph_counts() -> dict[str, int]:
    cfg = get_settings()
    driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password))
    try:
        with open_neo4j_session(driver) as session:
            return {
                "passage_nodes": session.run("MATCH (n:PassageNode) RETURN count(n) AS c").single()["c"],
                "phrase_nodes": session.run("MATCH (n:PhraseNode) RETURN count(n) AS c").single()["c"],
                "mentioned_in_edges": session.run(
                    "MATCH ()-[r:MENTIONED_IN]->() RETURN count(r) AS c"
                ).single()["c"],
                "related_to_edges": session.run(
                    "MATCH ()-[r:RELATED_TO]->() RETURN count(r) AS c"
                ).single()["c"],
            }
    finally:
        driver.close()


def _run_queries(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cfg = get_settings()
    driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password))
    client = make_openai_client(cfg)
    service = PipelineService(driver=driver, openai_client=client)
    rows: list[dict[str, Any]] = []
    try:
        for question in questions:
            started = time.perf_counter()
            row = {
                "question_id": question["question_id"],
                "group": question["_source_group"],
                "question_type": question["question_type"],
                "expected_best_route": question.get("expected_best_route", ""),
                "query": question["question"],
            }
            try:
                qa = service.query(
                    question["question"],
                    mode="agent_pattern",
                    session_id=f"medical-format-smoke::{question['question_id']}",
                )
                elapsed = time.perf_counter() - started
                trace = qa.trace.model_dump() if qa.trace else {}
                router = trace.get("router_step", {}) or {}
                tool_steps = trace.get("tool_steps", []) or []
                row.update(
                    {
                        "suggested_tool": (router.get("decision") or {}).get("suggested_tool", ""),
                        "first_tool": tool_steps[0]["tool_name"] if tool_steps else "",
                        "retries": qa.retries,
                        "source_count": len(qa.sources),
                        "duration_seconds": round(elapsed, 2),
                        "answer_preview": qa.answer[:220],
                        "status": "ok",
                        "error": "",
                    }
                )
            except Exception as exc:
                elapsed = time.perf_counter() - started
                logger.exception("Smoke query failed for %s", question["question_id"])
                row.update(
                    {
                        "suggested_tool": "",
                        "first_tool": "",
                        "retries": 0,
                        "source_count": 0,
                        "duration_seconds": round(elapsed, 2),
                        "answer_preview": "",
                        "status": "error",
                        "error": str(exc),
                    }
                )
            rows.append(row)
    finally:
        driver.close()
    return rows


def run_once(file_path: Path) -> dict[str, Any]:
    _clear_database()
    ingest_started = time.perf_counter()
    ingest_file(str(file_path))
    ingest_seconds = time.perf_counter() - ingest_started
    questions = _load_smoke_questions()
    return {
        "file": str(file_path),
        "ingest_seconds": round(ingest_seconds, 2),
        "graph_counts": _collect_graph_counts(),
        "results": _run_queries(questions),
    }


def main() -> None:
    payload = {
        "database": get_settings().neo4j.resolved_database,
        "runs": [],
    }
    for suffix in ("md", "docx", "pdf"):
        file_path = EXPORT_DIR / f"med_doc_001.{suffix}"
        if not file_path.exists():
            continue
        payload["runs"].append(
            {
                "format": suffix,
                **run_once(file_path),
            }
        )
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_utf8_payload(payload)


if __name__ == "__main__":
    main()
