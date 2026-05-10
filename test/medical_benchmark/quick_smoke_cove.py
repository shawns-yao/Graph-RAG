"""Quick smoke test for CoVe verification node.

Runs 3 questions covering relation / multi_hop / global types to verify
the new verify_answer node fires and the trace records verification_step.
"""

import io
import os
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "pymangle"))

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

    questions = [
        ("Q005 (relation)", "ACEI导致干咳时应该如何处理？", "hybrid"),
        ("Q008 (multi_hop)", "美托洛尔缓释片的目标剂量是多少？滴定周期是多久？", "hybrid"),
        ("Q024 (global)", "患者张某的入院诊断有哪些？", "agent_pattern"),
    ]

    for label, query, mode in questions:
        print("=" * 70)
        print(f"[{label}] mode={mode}")
        print(f"Q: {query}")
        try:
            qa = svc.query(query, mode=mode)
            trace = qa.trace
            print(f"Answer preview: {(qa.answer or '')[:200]}")
            print(f"evidence_score: {qa.evidence_score:.3f}")
            print(f"confidence_level: {qa.confidence_level}")
            if trace and trace.router_step:
                d = trace.router_step.decision
                print(f"Router: method={trace.router_step.method} type={d.query_type.value} tool={d.suggested_tool}")
            if trace and trace.verification_step:
                v = trace.verification_step
                print(f"CoVe: {v.claims_supported}/{v.claims_total} supported "
                      f"(rate={v.support_rate:.2f}, {v.duration_ms}ms)")
                if v.unsupported_claims:
                    print(f"Unsupported claims:")
                    for c in v.unsupported_claims[:3]:
                        print(f"  - {c.text}")
                elif v.skipped_reason:
                    print(f"Verification skipped: {v.skipped_reason}")
            else:
                print("Verification: NOT TRIGGERED")
        except Exception as e:
            print(f"ERROR: {e}")

    driver.close()


if __name__ == "__main__":
    main()
