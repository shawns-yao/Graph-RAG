import json
from pathlib import Path


MEDICAL_BENCHMARK_DIR = Path("test/medical_benchmark")


def test_medical_benchmark_json_assets_are_valid():
    for path in MEDICAL_BENCHMARK_DIR.rglob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))


def test_medical_validation_report_counts_are_consistent():
    report = json.loads((MEDICAL_BENCHMARK_DIR / "validation_report.json").read_text(encoding="utf-8"))

    validated_count = len(report["validated_questions"])
    rejected_count = len(report["rejected_questions"])
    stats = report["summary_stats"]

    assert stats["total_checked"] == validated_count + rejected_count
    assert stats["total_passed"] == validated_count
    assert stats["total_rejected"] == rejected_count


def test_medical_master_question_count_matches_metadata():
    master = json.loads((MEDICAL_BENCHMARK_DIR / "questions_master.json").read_text(encoding="utf-8"))

    assert len(master["questions"]) == master["metadata"]["total_questions"]
