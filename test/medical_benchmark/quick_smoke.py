"""Quick smoke test: run a few queries across vector, bm25, graph, and agent modes."""

import io
import os
import sys

# Fix Windows console encoding BEFORE any imports that might print
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"))

from neo4j import GraphDatabase
from rag_core.config import get_settings, make_openai_client
from agentic_graph_rag.service import PipelineService


def main():
    cfg = get_settings()
    driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password))
    client = make_openai_client(cfg)
    svc = PipelineService(driver, client)

    tests = [
        # (question_id, query, mode, expected_keywords)
        ("Q001", "2型糖尿病的诊断标准是什么？", "agent_pattern", ["7.0", "HbA1c", "6.5"]),
        ("Q009", "NT-proBNP多少可以排除心衰？", "agent_pattern", ["300", "排除"]),
        ("Q005", "ACEI导致干咳时应该如何处理？", "agent_pattern", ["ARB", "干咳"]),
        ("Q002", "eGFR小于30时二甲双胍应该如何调整？", "agent_pattern", ["禁用", "30"]),
        ("Q004", "糖尿病合并高血压患者的目标血压是多少？首选药物是什么？", "agent_pattern", ["130/80", "ACEI", "ARB"]),
        ("Q011", "PCI术后使用药物洗脱支架的患者，双抗治疗应该持续多久？", "agent_pattern", ["12", "月"]),
    ]

    results = []
    for qid, query, mode, expected_kw in tests:
        print("=" * 70)
        print(f"[{qid}] mode={mode}")
        print(f"Q: {query}")

        try:
            qa = svc.query(query, mode=mode)
            answer = qa.answer or ""
            # Check if expected keywords appear in answer
            hits = [kw for kw in expected_kw if kw in answer]
            passed = len(hits) >= len(expected_kw) * 0.5  # at least 50% keywords present

            print(f"A: {answer[:200]}")
            print(
                f"Status: {qa.answer_status}/{qa.retrieval_status}/{qa.verification_status} "
                f"| Sources: {len(qa.sources)} | Retries: {qa.retries}"
            )
            if qa.trace and qa.trace.router_step:
                d = qa.trace.router_step.decision
                print(f"Router: {qa.trace.router_step.method} -> {d.suggested_tool} ({d.query_type.value})")
            print(f"Keywords hit: {hits} / {expected_kw}")
            print(f"PASS" if passed else "FAIL")
            results.append({
                "id": qid,
                "passed": passed,
                "answer_status": qa.answer_status,
                "retries": qa.retries,
            })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"id": qid, "passed": False, "error": str(e)})

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    print(f"Passed: {passed}/{total} ({passed/total*100:.0f}%)")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        detail = f"status={r.get('answer_status', '')}" if "answer_status" in r else r.get("error", "")[:40]
        print(f"  {r['id']}: {status} ({detail})")

    driver.close()


if __name__ == "__main__":
    main()
