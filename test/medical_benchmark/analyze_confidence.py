"""Analyze discrete answer and verification statuses."""

import io
import json
import sys
from collections import Counter
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
with open(ROOT / "test" / "medical_benchmark" / "results" / "benchmark_results.json", "r", encoding="utf-8") as f:
    results = json.load(f)

print("Status Distribution Analysis")
print("=" * 70)

for mode in ["vector", "cypher"]:
    rows = [r for r in results if r["mode"] == mode and not r.get("error")]
    print(f"\n=== Mode: {mode} ({len(rows)} questions) ===")
    print("  Answer status:")
    answer_counts = Counter(r.get("answer_status", "unknown") for r in rows)
    for status, count in sorted(answer_counts.items()):
        print(f"    {status}: {count}")

    print("  Verification status:")
    verification_counts = Counter(r.get("verification_status", "unknown") for r in rows)
    for status, count in sorted(verification_counts.items()):
        print(f"    {status}: {count}")

    print("\n  Passed with non-verified answer status:")
    for r in rows:
        if r.get("kw_pass") and r.get("answer_status") != "verified":
            print(
                f"    {r['id']} status={r.get('answer_status', 'unknown')} "
                f"verification={r.get('verification_status', 'unknown')} "
                f"retries={r.get('retries', 0)} sources={r.get('sources', 0)}"
            )
