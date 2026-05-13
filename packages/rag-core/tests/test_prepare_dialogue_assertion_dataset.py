"""Tests for Chinese medical dialogue assertion dataset cleaning."""

from scripts.prepare_dialogue_assertion_dataset import (
    build_dialogue_examples,
    clean_text,
    detect_encoding,
    repair_mojibake,
)


def test_dialogue_cleaner_converts_csv_to_assertion_examples(tmp_path):
    csv_path = tmp_path / "dialogue.csv"
    csv_path.write_text(
        "\n".join(
            [
                "department,title,ask,answer",
                "内科,高血压,我没有糖尿病,患者既往无高血压、糖尿病史。",
                "心内科,冠心病,胸闷怎么办,目前不能排除冠心病，建议完善心电图。",
                "肾内科,用药,eGFR低怎么办,若 eGFR < 30，则禁用二甲双胍。",
            ]
        ),
        encoding="gb18030",
    )

    examples, stats = build_dialogue_examples(csv_path, fields=("answer",))

    by_entity = {example.entity: example for example in examples}
    assert by_entity["糖尿病"].label == "negated"
    assert by_entity["冠心病"].label == "speculated"
    assert by_entity["二甲双胍"].label == "conditional"
    assert stats.rows == 3
    assert stats.examples >= 3
    assert stats.encodings["gb18030"] == 3
    for example in examples:
        assert example.text[example.start:example.end] == example.entity
        assert example.source.startswith("dialogue_weak:")


def test_dialogue_cleaner_can_use_ask_field(tmp_path):
    csv_path = tmp_path / "dialogue.csv"
    csv_path.write_text(
        "department,title,ask,answer\n内科,糖尿病,我没有糖尿病,建议就诊。\n",
        encoding="gb18030",
    )

    examples, stats = build_dialogue_examples(csv_path, fields=("ask",))

    assert any(example.entity == "糖尿病" and example.label == "negated" for example in examples)
    assert stats.fields["ask"] >= 1


def test_clean_text_removes_html_and_boilerplate():
    text = clean_text("<p>患者无高血压。</p>感谢您的咨询", max_chars=100)

    assert text == "患者无高血压"


def test_repair_mojibake_department_names():
    assert repair_mojibake("ç¥žç»ç§‘") == "神经科"
    assert repair_mojibake("Éñ¾­¿Æ") == "神经科"


def test_detect_encoding_for_gb18030_csv(tmp_path):
    csv_path = tmp_path / "dialogue.csv"
    csv_path.write_text("department,title,ask,answer\n内科,a,b,c\n", encoding="gb18030")

    assert detect_encoding(csv_path) == "gb18030"
