"""Tests for Chinese medical assertion-status rules."""

from rag_core.config import IndexingSettings

from agentic_graph_rag.indexing.assertion_classifier import (
    RuleAssertionClassifier,
    load_assertion_classifier,
    mark_entity,
)
from agentic_graph_rag.indexing.assertion_rules import (
    classify_assertion_by_rules,
    extract_assertion_candidates,
    find_entity_offsets,
)


def test_negated_assertion_scope():
    decision = classify_assertion_by_rules("患者既往无高血压、糖尿病史。", "糖尿病")

    assert decision.label == "negated"
    assert decision.cue == "无"


def test_speculated_assertion_scope():
    decision = classify_assertion_by_rules("考虑冠心病可能。", "冠心病")

    assert decision.label == "speculated"


def test_conditional_assertion_scope():
    decision = classify_assertion_by_rules("若 eGFR < 30，则禁用二甲双胍。", "二甲双胍")

    assert decision.label == "conditional"


def test_historical_assertion_scope():
    decision = classify_assertion_by_rules("患者既往诊断为肺炎。", "肺炎")

    assert decision.label == "historical"


def test_family_history_assertion_scope():
    decision = classify_assertion_by_rules("父亲有高血压病史。", "高血压")

    assert decision.label == "family_history"


def test_extract_candidates_preserves_offsets():
    examples = extract_assertion_candidates("患者既往无高血压、糖尿病史。考虑冠心病可能。")

    assert examples
    for example in examples:
        assert example.text[example.start:example.end] == example.entity

    labels = {example.entity: example.label for example in examples}
    assert labels["糖尿病"] == "negated"
    assert labels["冠心病"] == "speculated"


def test_find_entity_offsets_missing_entity():
    assert find_entity_offsets("患者无糖尿病史。", "冠心病") is None


def test_mark_entity_for_classifier_input():
    assert mark_entity("考虑冠心病可能。", "冠心病") == "考虑[E]冠心病[/E]可能。"


def test_load_assertion_classifier_defaults_to_rules():
    settings = IndexingSettings(_env_file=None)

    classifier = load_assertion_classifier(settings)

    assert isinstance(classifier, RuleAssertionClassifier)
