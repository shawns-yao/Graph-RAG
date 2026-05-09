from agentic_graph_rag.medical_aliases import expand_medical_aliases


def test_expands_chinese_medical_term_to_abbreviation_and_english_name():
    expanded = expand_medical_aliases("2型糖尿病的诊断标准是什么？")

    assert "Medical aliases:" in expanded
    assert "T2DM" in expanded
    assert "Type 2 Diabetes Mellitus" in expanded


def test_expands_abbreviation_to_chinese_medical_surface():
    expanded = expand_medical_aliases("AECOPD是什么的缩写？")

    assert "急性加重" in expanded


def test_leaves_non_medical_query_unchanged():
    assert expand_medical_aliases("苹果公司的产品有哪些？") == "苹果公司的产品有哪些？"
