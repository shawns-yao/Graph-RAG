"""Analyze why confidence is low."""

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

print("Confidence Distribution Analysis")
print("=" * 70)

for mode in ["vector", "cypher"]:
    rows = [r for r in results if r["mode"] == mode and not r.get("error")]
    confs = [r.get("confidence", 0) for r in rows]
    print(f"\n=== Mode: {mode} ({len(rows)} questions) ===")
    print(f"  Confidence buckets:")
    buckets = Counter()
    for c in confs:
        if c < 0.2:
            buckets["[0.0-0.2)"] += 1
        elif c < 0.4:
            buckets["[0.2-0.4)"] += 1
        elif c < 0.6:
            buckets["[0.4-0.6)"] += 1
        elif c < 0.8:
            buckets["[0.6-0.8)"] += 1
        else:
            buckets["[0.8-1.0]"] += 1
    for bucket in ["[0.0-0.2)", "[0.2-0.4)", "[0.4-0.6)", "[0.6-0.8)", "[0.8-1.0]"]:
        count = buckets.get(bucket, 0)
        bar = "#" * count
        print(f"    {bucket}: {count:2d} {bar}")
    
    # Low-confidence but passed questions
    print(f"\n  Low confidence (< 0.4) but PASSED:")
    for r in rows:
        if r.get("confidence", 1.0) < 0.4 and r.get("kw_pass"):
            print(f"    {r['id']} conf={r['confidence']:.2f} retries={r['retries']} "
                  f"sources={r.get('sources', 0)}")
    
    # Specific confidence values
    print(f"\n  Exact confidence value counts:")
    exact_counts = Counter(round(c, 2) for c in confs)
    for conf, count in sorted(exact_counts.items())[:10]:
        if count >= 2:
            print(f"    {conf}: {count} times")
