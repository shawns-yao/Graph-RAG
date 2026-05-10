"""Analyze partial benchmark results (while full run is still in progress)."""

import json
import sys
import io
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
results_path = ROOT / "test" / "medical_benchmark" / "results" / "benchmark_results.json"

with open(results_path, "r", encoding="utf-8") as f:
    results = json.load(f)

# Group by mode
modes = {}
for r in results:
    modes.setdefault(r["mode"], []).append(r)

for mode, rows in modes.items():
    total = len(rows)
    with_kw = [r for r in rows if r.get("kw_total", 0) > 0]
    no_kw = [r for r in rows if r.get("kw_total", 0) == 0]
    errors = [r for r in rows if r.get("error")]
    passed_with_kw = sum(1 for r in with_kw if r.get("kw_pass"))
    
    print(f"\n=== Mode: {mode} ({total} questions) ===")
    print(f"  Questions with keywords: {len(with_kw)}")
    print(f"    - PASS: {passed_with_kw}/{len(with_kw)} = {passed_with_kw/len(with_kw)*100:.1f}%" if with_kw else "")
    print(f"  Questions without keywords (skipped in accuracy calc): {len(no_kw)}")
    if no_kw:
        print(f"    IDs: {[r['id'] for r in no_kw]}")
    print(f"  Errors: {len(errors)}")
    if errors:
        print(f"    IDs: {[r['id'] for r in errors]}")
    
    # Confidence distribution
    confs = [r.get("confidence", 0) for r in rows if not r.get("error")]
    if confs:
        print(f"  Avg confidence: {sum(confs)/len(confs):.3f}")
    
    # Elapsed time
    times = [r.get("elapsed_ms", 0) for r in rows]
    if times:
        print(f"  Avg time: {sum(times)/len(times)/1000:.1f}s")
        print(f"  Total time: {sum(times)/1000:.1f}s ({sum(times)/60000:.1f} min)")
