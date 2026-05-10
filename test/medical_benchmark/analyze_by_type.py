"""Break down accuracy by mode x query_type."""

import io
import json
import sys
from collections import defaultdict
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
with open(ROOT / "test" / "medical_benchmark" / "results" / "benchmark_results.json", "r", encoding="utf-8") as f:
    results = json.load(f)

stats = defaultdict(lambda: defaultdict(lambda: {"total": 0, "pass": 0, "no_kw": 0, "error": 0}))
for r in results:
    m = r["mode"]
    qt = r["query_type"]
    s = stats[m][qt]
    s["total"] += 1
    if r.get("error"):
        s["error"] += 1
    elif r.get("kw_total", 0) > 0:
        if r.get("kw_pass"):
            s["pass"] += 1
    else:
        s["no_kw"] += 1

modes = list(stats.keys())
query_types = ["simple", "relation", "multi_hop", "global", "temporal"]

print("\nAccuracy by Mode x Query Type (evaluable only, excl. no-keyword questions)")
print("=" * 80)

# Header
TYPE_W = 12
CELL_W = 18
header = f"{'Type':<{TYPE_W}}"
for m in modes:
    header += f" {m:<{CELL_W}}"
print(header)
print("-" * 80)

for qt in query_types:
    row = f"{qt:<{TYPE_W}}"
    for m in modes:
        s = stats[m].get(qt, {"total": 0, "pass": 0, "no_kw": 0, "error": 0})
        evaluable = s["total"] - s["no_kw"] - s["error"]
        if evaluable == 0:
            if s["total"] > 0:
                cell = f"--- ({s['total']} skip)"
                row += f" {cell:<{CELL_W}}"
            else:
                row += f" {'N/A':<{CELL_W}}"
            continue
        acc = s["pass"] / evaluable * 100
        cell = f"{s['pass']}/{evaluable}={acc:.0f}%"
        row += f" {cell:<{CELL_W}}"
    print(row)

print("\n" + "=" * 80)
print("Totals (evaluable only)")
print("=" * 80)

total_by_mode = {}
for m in modes:
    total = 0
    passed = 0
    no_kw = 0
    errors = 0
    for qt in query_types:
        s = stats[m].get(qt, {"total": 0, "pass": 0, "no_kw": 0, "error": 0})
        total += s["total"]
        passed += s["pass"]
        no_kw += s["no_kw"]
        errors += s["error"]
    total_by_mode[m] = {"total": total, "pass": passed, "no_kw": no_kw, "error": errors}

for m, s in total_by_mode.items():
    evaluable = s["total"] - s["no_kw"] - s["error"]
    if evaluable > 0:
        acc = s["pass"] / evaluable * 100
        print(f"  {m:<16}: {s['pass']}/{evaluable} = {acc:.1f}%  (skipped {s['no_kw']}, errors {s['error']})")
