#!/usr/bin/env python3
"""Check fixed trace-evaluation output for retrieval-chain regressions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_TRACE_PATH = Path("test/medical_benchmark/results/trace_eval_fixed_questions.json")
DEFAULT_MIN_VERIFIED = 8


def _needs(row: dict[str, Any]) -> set[str]:
    trace = row.get("trace") or {}
    for entry in trace.get("workflow_memory") or []:
        if entry.get("stage") == "route":
            metadata = entry.get("metadata") or {}
            raw = metadata.get("retrieval_needs") or []
            return {str(item) for item in raw}
    return set()


def _has_strong_numeric_anchor(row: dict[str, Any]) -> bool:
    signal = row.get("signal_extractor_output") or {}
    anchors = signal.get("anchors") or []
    return any(str(anchor.get("kind")) in {"numeric", "threshold", "symbolic", "quoted"} for anchor in anchors)


def _failures(payload: dict[str, Any], min_verified: int) -> list[str]:
    rows = payload.get("results") or []
    summary = payload.get("summary") or {}
    failures: list[str] = []
    if summary.get("ok") != summary.get("total"):
        failures.append(f"not all questions completed: ok={summary.get('ok')} total={summary.get('total')}")
    if int(summary.get("verified") or 0) < min_verified:
        failures.append(f"verified below baseline: {summary.get('verified')} < {min_verified}")

    for row in rows:
        row_id = row.get("id")
        tools = set(row.get("initial_tools") or [])
        if row.get("group") == "graph_direct" and "cypher_traverse" not in tools:
            failures.append(f"{row_id}: graph_direct missing cypher_traverse in initial_tools")
        if row.get("group") == "numeric_exact" and _has_strong_numeric_anchor(row) and "bm25_search" not in tools:
            failures.append(f"{row_id}: numeric_exact missing bm25_search for strong anchor")
    return failures


def _mixed_need_report(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("results") or []
    mixed = [row for row in rows if {"exact_numeric", "graph_relation"}.issubset(_needs(row))]
    verified = sum(1 for row in mixed if row.get("answer_status") == "verified")
    return {
        "count": len(mixed),
        "verified": verified,
        "verified_rate": round(verified / len(mixed), 4) if mixed else None,
        "tool_sets": [
            {
                "id": row.get("id"),
                "initial_tools": row.get("initial_tools") or [],
                "answer_status": row.get("answer_status"),
                "verification_status": row.get("verification_status"),
            }
            for row in mixed
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_TRACE_PATH))
    parser.add_argument("--min-verified", type=int, default=DEFAULT_MIN_VERIFIED)
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(json.dumps({"status": "error", "error": f"trace file not found: {path}"}, ensure_ascii=False))
        return 2

    payload = json.loads(path.read_text(encoding="utf-8"))
    failures = _failures(payload, args.min_verified)
    report = {
        "status": "failed" if failures else "passed",
        "summary": payload.get("summary") or {},
        "failures": failures,
        "mixed_exact_numeric_graph_relation": _mixed_need_report(payload),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
