from rag_core.models import Entity, GraphContext

from agentic_graph_rag.retrieval.providers import graph_context_to_search_results, _query_terms


def test_graph_result_preserves_paths_weak_hints_and_evidence():
    ctx = GraphContext(
        triplets=[
            {"source": "沙丁胺醇", "relation": "起效时间", "target": "5-15分钟"},
        ],
        weak_triplets=[
            {
                "source": "噻托溴铵",
                "relation": "CO_OCCURS_WITH",
                "target": "每日1次",
                "evidence": "噻托溴铵：18 μg，每日1次。",
            }
        ],
        passages=[
            "沙丁胺醇起效时间：5-15分钟。噻托溴铵：18 μg，每日1次。",
        ],
        entities=[
            Entity(name="沙丁胺醇", entity_type="Drug"),
            Entity(name="噻托溴铵", entity_type="Drug"),
        ],
        source_ids=["chunk-1"],
    )

    results = graph_context_to_search_results(
        ctx,
        source="graph",
        include_graph_structure=True,
        top_k=1,
        query="沙丁胺醇和噻托溴铵有什么区别",
    )

    content = results[0].chunk.content
    assert "Graph paths:" in content
    assert "沙丁胺醇 -[起效时间]-> 5-15分钟" in content
    assert "Weak graph hints:" in content
    assert "噻托溴铵 -[CO_OCCURS_WITH?]-> 每日1次" in content
    assert "Evidence:" in content
    assert content.index("噻托溴铵：18 μg，每日1次") > content.index("Evidence:")
    assert "噻托溴铵：18 μg，每日1次" in content
    assert "Entities:" in content


def test_graph_evidence_uses_source_body_not_indexing_hints():
    ctx = GraphContext(
        triplets=[
            {"source": "沙丁胺醇", "relation": "起效时间", "target": "5-15分钟"},
            {"source": "噻托溴铵", "relation": "给药频率", "target": "每日1次"},
        ],
        passages=[
            "\n".join(
                [
                    "Document summary: 诊断依据肺功能检查，FEV1/FVC < 0.70。",
                    "Section: 慢性阻塞性肺疾病（COPD）诊疗指南",
                    "Chunk position: 1/4",
                    "Focus: FEV1/FVC < 0.70 GOLD分级 诊断 评分",
                    "",
                    "**短效β2受体激动剂（SABA）**：",
                    "- 沙丁胺醇（Salbutamol）：100-200 μg/次，按需使用，每日不超过8次",
                    "- 起效时间：5-15分钟",
                    "- 持续时间：4-6小时",
                    "**长效抗胆碱能药物（LAMA）**：",
                    "- 噻托溴铵（Tiotropium）：18 μg，每日1次",
                    "- 起效时间：30-60分钟",
                    "- 持续时间：24小时",
                ]
            )
        ],
        entities=[
            Entity(name="沙丁胺醇", entity_type="Drug"),
            Entity(name="噻托溴铵", entity_type="Drug"),
        ],
        source_ids=["chunk-1"],
    )

    results = graph_context_to_search_results(
        ctx,
        source="graph",
        include_graph_structure=True,
        top_k=1,
        query="为什么沙丁胺醇适合按需使用而噻托溴铵适合每日规律使用",
    )

    content = results[0].chunk.content
    evidence = content.split("Evidence:", 1)[1]
    assert "Document summary:" not in evidence
    assert "Focus:" not in evidence
    assert "诊断依据肺功能检查" not in evidence
    assert "沙丁胺醇（Salbutamol）" in evidence
    assert "噻托溴铵（Tiotropium）" in evidence
    assert "每日1次" in evidence


def test_graph_results_prioritize_source_evidence_over_relation_catalog():
    ctx = GraphContext(
        triplets=[
            {"source": "沙丁胺醇", "relation": "起效时间", "target": "5-15分钟"},
            {"source": "噻托溴铵", "relation": "给药频率", "target": "每日1次"},
        ],
        passages=[
            "\n".join(
                [
                    "- 沙丁胺醇 --起效时间--> 5-15分钟",
                    "- 沙丁胺醇 --持续时间--> 4-6小时",
                    "- 噻托溴铵 --剂量--> 18 μg每日1次",
                    "- 噻托溴铵 --起效时间--> 30-60分钟",
                ]
            ),
            "\n".join(
                [
                    "**短效β2受体激动剂（SABA）**：",
                    "- 沙丁胺醇（Salbutamol）：100-200 μg/次，按需使用",
                    "- 起效时间：5-15分钟",
                    "- 持续时间：4-6小时",
                    "**长效抗胆碱能药物（LAMA）**：",
                    "- 噻托溴铵（Tiotropium）：18 μg，每日1次",
                    "- 起效时间：30-60分钟",
                    "- 持续时间：24小时",
                ]
            ),
        ],
        entities=[
            Entity(name="沙丁胺醇", entity_type="Drug"),
            Entity(name="噻托溴铵", entity_type="Drug"),
        ],
        source_ids=["relations", "source"],
    )

    results = graph_context_to_search_results(
        ctx,
        source="graph",
        include_graph_structure=True,
        top_k=2,
        query="沙丁胺醇 噻托溴铵 起效时间 每日1次",
    )

    assert results[0].chunk.id == "source"
    assert "噻托溴铵（Tiotropium）" in results[0].chunk.content


def test_graph_query_terms_split_chinese_constraints():
    terms = _query_terms("GOLD 3-4级且嗜酸性粒细胞≥100/μL时推荐什么方案？")

    assert "gold" in terms
    assert "3-4" in terms
    assert "100" in terms
    assert "嗜酸性粒细胞" in terms
    assert all("推荐什么方案" not in term for term in terms)


def test_graph_results_prioritize_multi_constraint_match():
    ctx = GraphContext(
        passages=[
            "根据FEV1占预计值百分比分为4级：- GOLD 3级：30% ≤ FEV1 < 50%预计值",
            "| 3-4级 | 重 | ≥2次/年 | ≥100/μL | LABA+LAMA+ICS |",
        ],
        entities=[Entity(name="LABA+LAMA+ICS", entity_type="Therapy")],
        source_ids=["definition", "recommendation"],
    )

    results = graph_context_to_search_results(
        ctx,
        source="graph",
        include_graph_structure=True,
        top_k=2,
        query="GOLD 3-4级且嗜酸性粒细胞≥100/μL时推荐什么方案？",
    )

    assert results[0].chunk.id == "recommendation"
