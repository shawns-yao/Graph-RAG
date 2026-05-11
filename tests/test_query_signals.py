from agentic_graph_rag.agent.query_signals import (
    extract_query_signals,
    has_strong_form_anchor,
)


def _kinds(query: str) -> set[str]:
    return {anchor.kind for anchor in extract_query_signals(query).anchors}


def test_extracts_threshold_and_symbolic_anchor():
    signals = extract_query_signals("FEV1/FVC < 0.70 说明什么？")

    assert ("FEV1/FVC", "symbolic") in {(a.text, a.kind) for a in signals.anchors}
    assert ("FEV1/FVC < 0.70", "threshold") in {(a.text, a.kind) for a in signals.anchors}
    assert has_strong_form_anchor(signals)


def test_extracts_dose_and_frequency_as_numeric():
    signals = extract_query_signals("噻托溴铵 18 μg 每日1次 是否正确？")
    anchors = {(anchor.text, anchor.kind) for anchor in signals.anchors}

    assert ("18 μg", "numeric") in anchors
    assert ("每日1次", "numeric") in anchors
    assert has_strong_form_anchor(signals)


def test_extracts_egfr_threshold():
    signals = extract_query_signals("eGFR < 30 怎么办？")

    assert ("eGFR < 30", "threshold") in {(a.text, a.kind) for a in signals.anchors}
    assert has_strong_form_anchor(signals)


def test_plain_medical_term_is_phrase_only():
    signals = extract_query_signals("噻托溴铵剂量是多少？")

    assert _kinds("噻托溴铵剂量是多少？") == {"phrase"}
    assert [anchor.text for anchor in signals.anchors] == ["噻托溴铵剂量"]
    assert not has_strong_form_anchor(signals)


def test_generic_phrase_does_not_create_strong_anchor():
    signals = extract_query_signals("诊断标准是什么？")

    assert [(anchor.text, anchor.kind) for anchor in signals.anchors] == [
        ("诊断标准", "phrase")
    ]
    assert not has_strong_form_anchor(signals)

