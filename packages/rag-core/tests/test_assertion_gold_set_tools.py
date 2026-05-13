"""Tests for assertion gold-set review tooling."""

import csv
import json

import pytest

from scripts.compile_assertion_gold_set import compile_gold_csv
from scripts.export_assertion_gold_review import export_review_csv


def _write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_export_review_csv_and_compile_gold_set(tmp_path):
    source = tmp_path / "review.jsonl"
    _write_jsonl(
        source,
        [
            {
                "text": "患者既往无高血压、糖尿病史。",
                "entity": "糖尿病",
                "label": "negated",
                "start": 9,
                "end": 12,
                "cue": "无",
                "source": "dialogue_weak:test",
            },
            {
                "text": "目前不能排除冠心病。",
                "entity": "冠心病",
                "label": "speculated",
                "start": 6,
                "end": 9,
                "cue": "不能排除",
                "source": "dialogue_weak:test",
            },
        ],
    )
    csv_path = tmp_path / "todo.csv"

    summary = export_review_csv(source, csv_path, per_label=10, limit=0)

    assert summary["total"] == 2
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["start"] == "9"
    assert rows[0]["end"] == "12"
    rows[0]["gold_label"] = "negated"
    rows[1]["gold_label"] = ""
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    output = tmp_path / "gold.jsonl"
    stats = tmp_path / "stats.json"
    gold_summary = compile_gold_csv(csv_path, output, stats)

    assert gold_summary["total"] == 1
    assert gold_summary["skipped"]["missing_gold_label"] == 1
    compiled = json.loads(output.read_text(encoding="utf-8").strip())
    assert compiled["label"] == "negated"
    assert compiled["text"][compiled["start"]:compiled["end"]] == compiled["entity"]
    assert compiled["confidence"] == 1.0


def test_compile_gold_set_uses_exported_offsets_for_repeated_entity(tmp_path):
    csv_path = tmp_path / "todo.csv"
    text = "糖尿病筛查提示异常，患者否认糖尿病史。"
    start = text.rfind("糖尿病")
    csv_path.write_text(
        "id,gold_label,weak_label,entity,answer_span,start,end,cue,text,source,notes\n"
        f"gold-1,negated,negated,糖尿病,糖尿病,{start},{start + 3},否认,{text},src,\n",
        encoding="utf-8-sig",
    )

    compile_gold_csv(csv_path, tmp_path / "gold.jsonl", tmp_path / "stats.json")

    compiled = json.loads((tmp_path / "gold.jsonl").read_text(encoding="utf-8").strip())
    assert compiled["start"] == start
    assert compiled["text"][compiled["start"] - 2 : compiled["start"]] == "否认"


def test_compile_gold_set_rejects_invalid_label(tmp_path):
    csv_path = tmp_path / "todo.csv"
    csv_path.write_text(
        "id,gold_label,weak_label,entity,answer_span,cue,text,source,notes\n"
        "gold-1,bad_label,negated,糖尿病,糖尿病,无,患者无糖尿病史。,src,\n",
        encoding="utf-8-sig",
    )

    with pytest.raises(ValueError, match="Invalid gold_label"):
        compile_gold_csv(csv_path, tmp_path / "gold.jsonl", tmp_path / "stats.json")
