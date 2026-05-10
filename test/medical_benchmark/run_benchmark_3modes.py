"""3-mode benchmark runner: vector, cypher, agent_pattern on 30 medical questions.

Usage:
    py test/medical_benchmark/run_benchmark_3modes.py

Output:
    test/medical_benchmark/results/benchmark_results.json
    test/medical_benchmark/results/benchmark_summary.txt
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

# Fix Windows console encoding BEFORE imports
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "pymangle"))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from neo4j import GraphDatabase

from rag_core.config import get_settings, make_openai_client
from agentic_graph_rag.service import PipelineService


MODES = ["vector", "cypher", "agent_pattern"]
QUESTIONS_PATH = _ROOT / "test" / "medical_benchmark" / "questions_master.json"
RESULTS_DIR = _ROOT / "test" / "medical_benchmark" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def keyword_match_score(answer: str, keywords: list[str]) -> tuple[int, int, float]:
    """Count how many expected keywords appear in the answer."""
    if not keywords:
        return 0, 0, 0.0
    hits = sum(1 for kw in keywords if kw in answer)
    total = len(keywords)
    ratio = hits / total if total else 0.0
    return hits, total, ratio


def is_pass(ratio: float) -> bool:
    """Pass criterion: at least 50% of expected keywords must appear."""
    return ratio >= 0.5


def llm_judge_score(client, query: str, expected: str, actual: str, model: str) -> dict:
    """Use LLM to judge answer correctness (1-5 scale)."""
    prompt = f"""你是一个医疗知识评测专家。请评估以下答案的正确性。

问题: {query}

标准答案: {expected}

系统答案: {actual}

评分标准（1-5分）:
5分: 完全正确，信息完整准确
4分: 基本正确，有轻微遗漏或表述差异
3分: 部分正确，缺少关键信息
2分: 大部分错误，仅有少量正确信息
1分: 完全错误或答非所问

返回JSON格式，不要任何其他文字：
{{"score": <1-5>, "reason": "<简短理由>"}}
"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        text = (response.choices[0].message.content or "").strip()
        # Extract JSON if wrapped in code blocks
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        result = json.loads(text)
        return {
            "score": int(result.get("score", 0)),
            "reason": str(result.get("reason", ""))[:200],
        }
    except Exception as e:
        return {"score": 0, "reason": f"judge_error: {e}"[:200]}


def run_single_question(svc, q: dict, mode: str) -> dict:
    """Execute one question against the pipeline."""
    started = time.perf_counter()
    try:
        qa = svc.query(q["query"], mode=mode)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        answer = qa.answer or ""
        keywords = q.get("keywords", [])
        hits, total, ratio = keyword_match_score(answer, keywords)

        router_tool = ""
        router_method = ""
        if qa.trace and qa.trace.router_step:
            router_method = qa.trace.router_step.method
            router_tool = qa.trace.router_step.decision.suggested_tool

        return {
            "id": q["id"],
            "query": q["query"],
            "query_type": q["query_type"],
            "mode": mode,
            "answer": answer,
            "expected": q["answer"],
            "confidence": round(qa.confidence, 3),
            "retries": qa.retries,
            "sources": len(qa.sources),
            "elapsed_ms": elapsed_ms,
            "router_method": router_method,
            "router_tool": router_tool,
            "kw_hits": hits,
            "kw_total": total,
            "kw_ratio": round(ratio, 2),
            "kw_pass": is_pass(ratio),
            "error": None,
        }
    except Exception as e:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return {
            "id": q["id"],
            "query": q["query"],
            "query_type": q["query_type"],
            "mode": mode,
            "error": str(e)[:300],
            "elapsed_ms": elapsed_ms,
            "kw_pass": False,
        }


def summarize(results: list[dict]) -> dict:
    """Compute per-mode and per-query-type statistics."""
    by_mode: dict[str, dict] = {}
    by_type: dict[str, dict] = {}
    by_mode_type: dict[str, dict[str, dict]] = {}

    for r in results:
        mode = r["mode"]
        qt = r["query_type"]
        passed = r.get("kw_pass", False)
        elapsed = r.get("elapsed_ms", 0)
        retries = r.get("retries", 0)
        conf = r.get("confidence", 0.0)

        m = by_mode.setdefault(mode, {"total": 0, "pass": 0, "ms": 0, "retries": 0, "conf": 0.0})
        m["total"] += 1
        m["pass"] += int(passed)
        m["ms"] += elapsed
        m["retries"] += retries
        m["conf"] += conf

        t = by_type.setdefault(qt, {"total": 0, "pass": 0})
        t["total"] += 1
        t["pass"] += int(passed)

        mt = by_mode_type.setdefault(mode, {}).setdefault(qt, {"total": 0, "pass": 0})
        mt["total"] += 1
        mt["pass"] += int(passed)

    # Compute averages
    for mode, stats in by_mode.items():
        n = stats["total"]
        stats["accuracy"] = round(stats["pass"] / n, 3) if n else 0.0
        stats["avg_ms"] = int(stats["ms"] / n) if n else 0
        stats["avg_retries"] = round(stats["retries"] / n, 2) if n else 0.0
        stats["avg_confidence"] = round(stats["conf"] / n, 3) if n else 0.0

    for qt, stats in by_type.items():
        n = stats["total"]
        stats["accuracy"] = round(stats["pass"] / n, 3) if n else 0.0

    return {
        "by_mode": by_mode,
        "by_query_type": by_type,
        "by_mode_type": by_mode_type,
    }


def format_report(summary: dict, results: list[dict]) -> str:
    """Format a human-readable report."""
    lines = []
    lines.append("=" * 80)
    lines.append("Medical Graph RAG Benchmark — 30 Questions, 3 Modes")
    lines.append("=" * 80)

    lines.append("\n## Accuracy by Mode\n")
    lines.append(f"{'Mode':<16} {'Pass/Total':<12} {'Accuracy':<10} {'Avg ms':<10} {'Avg Retries':<12} {'Avg Conf':<10}")
    lines.append("-" * 76)
    for mode, s in summary["by_mode"].items():
        lines.append(
            f"{mode:<16} {s['pass']}/{s['total']:<10} "
            f"{s['accuracy']*100:>6.1f}%   "
            f"{s['avg_ms']:<10} {s['avg_retries']:<12} {s['avg_confidence']:<10}"
        )

    lines.append("\n## Accuracy by Query Type (aggregated across all modes)\n")
    lines.append(f"{'Type':<14} {'Pass/Total':<12} {'Accuracy':<10}")
    lines.append("-" * 40)
    for qt, s in summary["by_query_type"].items():
        lines.append(f"{qt:<14} {s['pass']}/{s['total']:<10} {s['accuracy']*100:>6.1f}%")

    lines.append("\n## Accuracy by Mode × Query Type\n")
    header = f"{'Query Type':<14}"
    modes = list(summary["by_mode"].keys())
    for mode in modes:
        header += f" {mode:<16}"
    lines.append(header)
    lines.append("-" * len(header))

    query_types = sorted(summary["by_query_type"].keys())
    for qt in query_types:
        row = f"{qt:<14}"
        for mode in modes:
            mt = summary["by_mode_type"].get(mode, {}).get(qt, {"total": 0, "pass": 0})
            if mt["total"]:
                acc = mt["pass"] / mt["total"] * 100
                row += f" {mt['pass']}/{mt['total']} ({acc:.0f}%){'':<6}"
            else:
                row += f" {'N/A':<16}"
        lines.append(row)

    lines.append("\n## Per-Question Results\n")
    mode_header = "  ".join(f"{m:<14}" for m in modes)
    lines.append(f"{'ID':<6} {'Type':<12} {mode_header}")
    lines.append("-" * (6 + 12 + len(mode_header) + 4))

    # Group results by question id
    by_qid: dict[str, dict[str, dict]] = {}
    for r in results:
        by_qid.setdefault(r["id"], {})[r["mode"]] = r

    for qid in sorted(by_qid.keys()):
        q_results = by_qid[qid]
        qt = next(iter(q_results.values())).get("query_type", "")
        row = f"{qid:<6} {qt:<12}"
        for mode in modes:
            r = q_results.get(mode)
            if r is None:
                row += f" {'N/A':<14}"
            elif r.get("error"):
                row += f" {'ERR':<14}"
            else:
                mark = "PASS" if r.get("kw_pass") else "FAIL"
                conf = r.get("confidence", 0)
                row += f" {mark}({conf:.2f})    "
        lines.append(row)

    return "\n".join(lines)


def main():
    cfg = get_settings()
    driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password))
    client = make_openai_client(cfg)
    svc = PipelineService(driver, client)

    # Load questions
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    print(f"Loaded {len(questions)} questions\n")

    all_results: list[dict] = []
    overall_start = time.perf_counter()

    for mode in MODES:
        print(f"\n{'=' * 70}")
        print(f"Running mode: {mode}")
        print(f"{'=' * 70}")
        mode_start = time.perf_counter()

        for i, q in enumerate(questions, start=1):
            print(f"  [{i:2d}/{len(questions)}] {q['id']} ({q['query_type']}): {q['query'][:50]}...", flush=True)
            result = run_single_question(svc, q, mode)
            all_results.append(result)

            if result.get("error"):
                print(f"      ERROR: {result['error'][:100]}")
            else:
                mark = "PASS" if result["kw_pass"] else "FAIL"
                print(
                    f"      {mark} | conf={result['confidence']:.2f} | "
                    f"retries={result['retries']} | {result['elapsed_ms']}ms | "
                    f"kw={result['kw_hits']}/{result['kw_total']}"
                )

        mode_elapsed = time.perf_counter() - mode_start
        print(f"\n  Mode '{mode}' done in {mode_elapsed:.1f}s")

        # Save intermediate results after each mode (safety)
        with open(RESULTS_DIR / "benchmark_results.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

    total_elapsed = time.perf_counter() - overall_start
    print(f"\n\nTotal time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    # Summary
    summary = summarize(all_results)

    # Save results
    with open(RESULTS_DIR / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    with open(RESULTS_DIR / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    report = format_report(summary, all_results)
    print("\n" + report)

    with open(RESULTS_DIR / "benchmark_summary.txt", "w", encoding="utf-8") as f:
        f.write(report)

    driver.close()
    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
