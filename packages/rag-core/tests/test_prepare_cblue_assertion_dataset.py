"""Tests for CBLUE assertion dataset conversion."""

import json

from scripts.prepare_cblue_assertion_dataset import convert_cblue_root


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_convert_cblue_files_to_assertion_examples(tmp_path):
    _write_json(
        tmp_path / "CMeEE" / "CMeEE_train.json",
        [
            {
                "text": "患者既往无高血压、糖尿病史。",
                "entities": [
                    {"start_idx": 5, "end_idx": 7, "type": "dis", "entity": "高血压"},
                    {"start_idx": 9, "end_idx": 11, "type": "dis", "entity": "糖尿病"},
                ],
            }
        ],
    )
    _write_json(
        tmp_path / "CHIP-CTC" / "CHIP-CTC_train.json",
        [
            {
                "id": "ctc-1",
                "text": "若 eGFR < 30，则禁用二甲双胍。",
                "label": "Laboratory Examinations",
            }
        ],
    )
    _write_json(
        tmp_path / "CHIP-CDN" / "CHIP-CDN_train.json",
        [{"text": "冠心病", "normalized_result": "冠状动脉粥样硬化性心脏病"}],
    )

    examples = convert_cblue_root(tmp_path)

    by_entity = {example.entity: example for example in examples}
    assert by_entity["高血压"].label == "negated"
    assert by_entity["糖尿病"].label == "negated"
    assert by_entity["二甲双胍"].label == "conditional"
    assert by_entity["冠心病"].label == "affirmed"
    for example in examples:
        assert example.text[example.start:example.end] == example.entity
        assert example.source.startswith("cblue_weak:")
