"""Tests for skeleton indexing safeguards."""

from rag_core.models import Chunk, Entity

from agentic_graph_rag.indexing.skeleton import (
    _infer_sentence_relations,
    _inject_medical_phrase_entities,
    _merge_entities,
    _parse_extraction_response,
    _strip_candidate_noise,
    build_knn_graph,
    filter_low_information_chunks,
    link_peripheral_keywords,
)


def _chunk(idx: int) -> Chunk:
    return Chunk(id=f"c{idx}", content=f"chunk {idx}")


def test_build_knn_graph_uses_local_adaptive_threshold_not_absolute_similarity():
    chunks = [_chunk(i) for i in range(4)]
    embeddings = [
        [1.0, 0.0],
        [0.6, 0.8],
        [0.0, 1.0],
        [-1.0, 0.0],
    ]

    graph = build_knn_graph(chunks, embeddings, k=2)

    assert graph.has_edge(0, 1)
    assert graph[0][1]["weight"] < 0.7
    assert not graph.has_edge(0, 2)


def test_filter_low_information_chunks_preserves_short_numeric_facts():
    chunks = [
        Chunk(id="fact", content="eGFR < 30"),
        Chunk(id="noise1", content="the and or"),
        Chunk(id="noise2", content="the and or"),
    ]
    embeddings = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.9]]

    kept, kept_embeddings, dropped = filter_low_information_chunks(chunks, embeddings)

    assert [chunk.id for chunk in kept] == ["fact"]
    assert kept_embeddings == [[1.0, 0.0]]
    assert {chunk.id for chunk in dropped} == {"noise1", "noise2"}
    assert chunks[0].metadata["low_information_chunk"] is False
    assert "tfidf_signal_score" not in chunks[0].metadata
    assert chunks[0].metadata["local_information_score"] >= 0.0


def test_negated_pattern_entity_is_not_injected():
    chunk = Chunk(id="c_neg", content="患者既往无药物支架植入术。")

    entities = _inject_medical_phrase_entities(chunk)

    assert entities == []


def test_negated_sentence_does_not_create_positive_relation():
    chunk = Chunk(id="c_rel", content="患者无ACEI禁忌，可使用ACEI。")
    entities, _ = _parse_extraction_response(
        '{"entities": ['
        '{"chunk_id": "c_rel", "name": "ACEI", "type": "DrugClass", "confidence": 0.9},'
        '{"chunk_id": "c_rel", "name": "禁忌", "type": "Procedure", "confidence": 0.9}'
        ']}',
        chunk_text_by_id={"c_rel": chunk.enriched_content},
    )

    relationships = _infer_sentence_relations(chunk, entities)

    assert relationships == []


def test_llm_entities_in_negation_scope_are_discarded():
    entities, relationships = _parse_extraction_response(
        '{"entities": ['
        '{"chunk_id": "c1", "name": "糖尿病", "type": "Disease", "confidence": 0.9}'
        '], "relationships": ['
        '{"chunk_id": "c1", "from": "患者", "to": "糖尿病", "type": "患有", "confidence": 0.9}'
        ']}',
        chunk_text_by_id={"c1": "患者既往无高血压、糖尿病史。"},
    )

    assert entities == []
    assert relationships == []


def test_negated_peripheral_mention_does_not_link_entity():
    chunk = Chunk(id="p1", content="患者既往无糖尿病史。")
    entity = Entity(id="e1", name="糖尿病", entity_type="Disease", metadata={"confidence": 1.0})

    relationships = link_peripheral_keywords([chunk], [entity])

    assert relationships == []


def test_candidate_noise_cleanup_removes_markdown_relation_fragments():
    assert _strip_candidate_noise("--替代方案") == "替代方案"
    assert _strip_candidate_noise("eGFR < 30\n-") == "eGFR < 30"


def test_relation_fragment_candidate_is_not_injected_as_entity():
    chunk = Chunk(id="c_noise", content="- 干咳 --替代方案--> ARB")

    entities = _inject_medical_phrase_entities(chunk)

    assert all(entity.name != "替代方案" for entity in entities)
    assert all(not entity.name.startswith("--") for entity in entities)


def test_merge_entities_drops_low_value_heading_fragments():
    noisy = Entity(id="noise", name="关键事实", entity_type="Procedure")
    useful = Entity(id="acei", name="ACEI", entity_type="DrugClass")

    merged = _merge_entities([noisy, useful])

    assert [entity.name for entity in merged] == ["ACEI"]
