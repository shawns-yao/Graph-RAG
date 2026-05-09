"""Tests for agentic_graph_rag.indexing.skeleton."""

from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
from rag_core.models import Chunk, Entity

from agentic_graph_rag.indexing.skeleton import (
    _merge_entities,
    _parse_extraction_response,
    _rank_chunks_for_skeleton_selection,
    build_knn_graph,
    build_skeleton_index,
    compute_pagerank,
    extract_candidate_entities,
    extract_entities_full,
    extract_keywords,
    filter_low_information_chunks,
    infer_document_type,
    link_peripheral_keywords,
    resolve_pagerank_damping,
    resolve_skeleton_beta,
    select_skeletal_chunks,
)


def _make_chunks(n: int) -> list[Chunk]:
    return [Chunk(id=f"c{i}", content=f"Content of chunk {i}") for i in range(n)]


def _make_embeddings(n: int, dim: int = 4) -> list[list[float]]:
    """Generate deterministic embeddings for testing."""
    rng = np.random.default_rng(42)
    embs = rng.standard_normal((n, dim))
    # Normalize
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    normed = embs / np.where(norms == 0, 1.0, norms)
    return normed.tolist()


# ---------------------------------------------------------------------------
# build_knn_graph
# ---------------------------------------------------------------------------

class TestBuildKnnGraph:
    def test_empty_chunks(self):
        g = build_knn_graph([], [], k=3)
        assert g.number_of_nodes() == 0
        assert g.number_of_edges() == 0

    def test_single_chunk(self):
        chunks = _make_chunks(1)
        embs = _make_embeddings(1)
        g = build_knn_graph(chunks, embs, k=3)
        assert g.number_of_nodes() == 1
        assert g.number_of_edges() == 0  # no neighbours

    def test_two_chunks(self):
        chunks = _make_chunks(2)
        embs = [[1.0, 0.0], [0.95, 0.05]]
        g = build_knn_graph(chunks, embs, k=5)
        assert g.number_of_nodes() == 2
        # k=5 but only 1 neighbour available → 1 edge per node = 2 edges
        assert g.number_of_edges() == 2

    def test_five_chunks_k3(self):
        chunks = _make_chunks(5)
        embs = _make_embeddings(5)
        g = build_knn_graph(chunks, embs, k=3)
        assert g.number_of_nodes() == 5
        # Each node connects to 3 neighbours → at most 15 edges
        assert g.number_of_edges() <= 15
        assert g.number_of_edges() >= 5  # at least some edges

    def test_edges_have_weight(self):
        chunks = _make_chunks(3)
        embs = _make_embeddings(3)
        g = build_knn_graph(chunks, embs, k=2)
        for _, _, data in g.edges(data=True):
            assert "weight" in data
            assert -1.0 <= data["weight"] <= 1.0

    def test_filters_low_similarity_edges(self):
        chunks = _make_chunks(3)
        embs = [
            [1.0, 0.0],
            [0.95, 0.05],
            [-1.0, 0.0],
        ]
        g = build_knn_graph(chunks, embs, k=2)
        assert g.has_edge(0, 1)
        assert not g.has_edge(0, 2)

    @patch("agentic_graph_rag.indexing.skeleton.get_settings")
    def test_uses_settings_k(self, mock_settings):
        cfg = MagicMock()
        cfg.indexing.knn_k = 2
        mock_settings.return_value = cfg

        chunks = _make_chunks(5)
        embs = _make_embeddings(5)
        g = build_knn_graph(chunks, embs)
        # k=2 → each node gets 2 edges → 10 total max
        assert g.number_of_edges() <= 10


# ---------------------------------------------------------------------------
# compute_pagerank
# ---------------------------------------------------------------------------

class TestComputePagerank:
    def test_empty_graph(self):
        g = nx.DiGraph()
        scores = compute_pagerank(g, damping=0.85)
        assert scores == {}

    def test_simple_graph(self):
        g = nx.DiGraph()
        g.add_edges_from([(0, 1), (1, 2), (2, 0)])
        scores = compute_pagerank(g, damping=0.85)
        assert len(scores) == 3
        assert all(0 < s < 1 for s in scores.values())
        # Sum should be ~1.0
        assert abs(sum(scores.values()) - 1.0) < 1e-6

    def test_star_graph_center_highest(self):
        """Center of star should have highest PageRank."""
        g = nx.DiGraph()
        for i in range(1, 6):
            g.add_edge(i, 0)  # all point to center
        scores = compute_pagerank(g, damping=0.85)
        assert scores[0] == max(scores.values())

    @patch("agentic_graph_rag.indexing.skeleton.get_settings")
    def test_uses_settings_damping(self, mock_settings):
        cfg = MagicMock()
        cfg.indexing.pagerank_damping = 0.5
        mock_settings.return_value = cfg

        g = nx.DiGraph()
        g.add_edges_from([(0, 1), (1, 0)])
        scores = compute_pagerank(g)
        assert len(scores) == 2


# ---------------------------------------------------------------------------
# select_skeletal_chunks
# ---------------------------------------------------------------------------

class TestSelectSkeletalChunks:
    def test_empty(self):
        skeletal, peripheral = select_skeletal_chunks([], {}, beta=0.25)
        assert skeletal == []
        assert peripheral == []

    def test_selects_top_beta(self):
        chunks = _make_chunks(10)
        scores = {i: float(i) for i in range(10)}  # 9 is highest
        skeletal, peripheral = select_skeletal_chunks(chunks, scores, beta=0.3)
        # beta=0.3 → 3 chunks
        assert len(skeletal) == 3
        assert len(peripheral) == 7
        # Highest-scored chunks should be skeletal
        assert chunks[9] in skeletal
        assert chunks[8] in skeletal
        assert chunks[7] in skeletal

    def test_at_least_one_skeletal(self):
        chunks = _make_chunks(2)
        scores = {0: 0.5, 1: 0.3}
        skeletal, peripheral = select_skeletal_chunks(chunks, scores, beta=0.1)
        assert len(skeletal) >= 1  # min 1

    @patch("agentic_graph_rag.indexing.skeleton.get_settings")
    def test_uses_settings_beta(self, mock_settings):
        cfg = MagicMock()
        cfg.indexing.skeleton_beta = 0.5
        mock_settings.return_value = cfg

        chunks = _make_chunks(10)
        scores = {i: float(i) for i in range(10)}
        skeletal, peripheral = select_skeletal_chunks(chunks, scores)
        assert len(skeletal) == 5

    @patch("agentic_graph_rag.indexing.skeleton.get_settings")
    def test_dynamic_beta_for_short_docs(self, mock_settings):
        cfg = MagicMock()
        cfg.indexing.skeleton_beta = 0.25
        cfg.indexing.skeleton_beta_short_doc = 1.0
        cfg.indexing.skeleton_beta_medium_doc = 0.5
        cfg.indexing.skeleton_beta_long_doc = 0.3
        cfg.indexing.skeleton_short_doc_max_chunks = 8
        cfg.indexing.skeleton_medium_doc_max_chunks = 24
        mock_settings.return_value = cfg

        assert resolve_skeleton_beta(_make_chunks(4)) == 0.5

    @patch("agentic_graph_rag.indexing.skeleton.get_settings")
    def test_dynamic_beta_for_long_docs(self, mock_settings):
        cfg = MagicMock()
        cfg.indexing.skeleton_beta = 0.25
        cfg.indexing.skeleton_beta_short_doc = 1.0
        cfg.indexing.skeleton_beta_medium_doc = 0.5
        cfg.indexing.skeleton_beta_long_doc = 0.3
        cfg.indexing.skeleton_short_doc_max_chunks = 8
        cfg.indexing.skeleton_medium_doc_max_chunks = 24
        mock_settings.return_value = cfg

        assert resolve_skeleton_beta(_make_chunks(40)) == 0.3

    def test_short_docs_cap_skeletal_count(self):
        chunks = _make_chunks(8)
        scores = {i: float(8 - i) for i in range(8)}
        skeletal, peripheral = select_skeletal_chunks(chunks, scores, beta=1.0)
        assert len(skeletal) == 4
        assert len(peripheral) == 4

    @patch("agentic_graph_rag.indexing.skeleton.get_settings")
    def test_entity_density_can_promote_graph_chunk(self, mock_settings):
        cfg = MagicMock()
        cfg.indexing.skeleton_beta = 0.5
        cfg.indexing.skeleton_entity_density_weight = 0.6
        mock_settings.return_value = cfg
        chunks = [
            Chunk(
                id="c0",
                content="general notes",
                metadata={"graph_entity_count": 1, "graph_chunk_type": "peripheral_candidate"},
            ),
            Chunk(
                id="c1",
                content="Neo4j GraphRAG FastAPI PageRank",
                metadata={"graph_entity_count": 4, "graph_chunk_type": "skeleton_candidate"},
            ),
        ]
        scores = _rank_chunks_for_skeleton_selection(chunks, {0: 1.0, 1: 0.2})
        assert scores[1] > scores[0]


class TestFilterLowInformationChunks:
    def test_filters_low_signal_chunks_before_knn(self):
        chunks = [
            Chunk(id="c1", content="目录"),
            Chunk(id="c2", content="目录"),
            Chunk(id="c4", content="目录"),
            Chunk(id="c3", content="二型糖尿病 胰岛素 治疗 方案"),
        ]
        embeddings = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]

        kept_chunks, kept_embeddings, dropped_chunks = filter_low_information_chunks(
            chunks,
            embeddings,
        )

        assert [chunk.id for chunk in kept_chunks] == ["c3"]
        assert kept_embeddings == [[0.0, 1.0]]
        assert [chunk.id for chunk in dropped_chunks] == ["c1", "c2", "c4"]
        assert all(chunk.metadata["low_information_chunk"] is True for chunk in dropped_chunks)


# ---------------------------------------------------------------------------
# _parse_extraction_response
# ---------------------------------------------------------------------------

class TestParseExtractionResponse:
    def test_parses_json_entities_and_relationships(self):
        text = """
        {
          "entities": [
            {
              "chunk_id": "c1",
              "name": "COVID-19",
              "type": "Disease",
              "description": "Respiratory illness",
              "confidence": 0.95
            }
          ],
          "relationships": [
            {
              "chunk_id": "c1",
              "from": "COVID-19",
              "to": "SARS-CoV-2",
              "type": "caused_by",
              "confidence": 0.92
            }
          ]
        }
        """
        entities, rels = _parse_extraction_response(
            text,
            candidate_entities_by_chunk={"c1": ["COVID-19"]},
        )
        assert len(entities) == 1
        assert entities[0].metadata["confidence"] == 0.95
        assert entities[0].metadata["aliases"] == ["COVID-19"]
        assert len(rels) == 1
        assert rels[0].metadata["confidence"] == 0.92
        assert entities[0].entity_confidence == 0.95

    def test_parses_entities(self):
        text = "ENTITY: COVID-19 | Disease | Respiratory illness\nENTITY: WHO | Organization | World Health"
        entities, rels = _parse_extraction_response(text, "c1", candidate_entities=["WHO"])
        assert len(entities) == 2
        assert entities[0].name == "COVID-19"
        assert entities[0].entity_type == "Disease"
        assert entities[0].description == "Respiratory illness"
        assert entities[1].metadata["aliases"] == ["WHO"]

    def test_resolves_medical_abbreviation_aliases_only(self):
        text = """
        {
          "entities": [
            {
              "chunk_id": "c1",
              "name": "Diabetes Mellitus",
              "type": "Disease"
            },
            {
              "chunk_id": "c1",
              "name": "苹果公司",
              "type": "Organization"
            }
          ],
          "relationships": []
        }
        """
        entities, _rels = _parse_extraction_response(
            text,
            candidate_entities_by_chunk={"c1": ["DM", "Apple"]},
        )
        alias_map = {entity.name: entity.metadata["aliases"] for entity in entities}
        assert "DM" in alias_map["Diabetes Mellitus"]
        assert alias_map["苹果公司"] == []

    def test_parses_relationships(self):
        text = "RELATIONSHIP: COVID-19 | caused_by | SARS-CoV-2"
        entities, rels = _parse_extraction_response(text, "c1")
        assert len(rels) == 1
        assert rels[0].source == "COVID-19"
        assert rels[0].target == "SARS-CoV-2"
        assert rels[0].relation_type == "caused_by"

    def test_mixed_output(self):
        text = (
            "ENTITY: Python | Language | Programming language\n"
            "RELATIONSHIP: Python | used_for | Machine Learning\n"
            "ENTITY: ML | Field | Machine Learning\n"
        )
        entities, rels = _parse_extraction_response(text, "c1")
        assert len(entities) == 2
        assert len(rels) == 1

    def test_malformed_lines_skipped(self):
        text = "ENTITY: lonely\nGARBAGE LINE\nENTITY: A | B | C"
        entities, rels = _parse_extraction_response(text, "c1")
        # "lonely" has only 1 part → skipped (need ≥2)
        assert len(entities) == 1
        assert entities[0].name == "A"

    def test_empty_text(self):
        entities, rels = _parse_extraction_response("", "c1")
        assert entities == []
        assert rels == []


# ---------------------------------------------------------------------------
# extract_entities_full
# ---------------------------------------------------------------------------

class TestExtractEntitiesFull:
    def test_empty_chunks(self):
        entities, rels = extract_entities_full([])
        assert entities == []
        assert rels == []

    def test_extracts_from_llm(self):
        client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = """
        {
          "entities": [
            {
              "chunk_id": "c1",
              "name": "Machine Learning",
              "type": "Field",
              "description": "AI subfield",
              "confidence": 0.91
            }
          ],
          "relationships": [
            {
              "chunk_id": "c1",
              "from": "ML",
              "to": "AI",
              "type": "part_of",
              "confidence": 0.88
            }
          ]
        }
        """
        client.chat.completions.create.return_value = resp

        chunks = [Chunk(id="c1", content="Machine Learning is part of AI")]
        entities, rels = extract_entities_full(chunks, openai_client=client)

        assert len(entities) == 1
        assert len(rels) == 1


class TestMergeEntities:
    def test_merge_entities_considers_type(self):
        entities = [
            Entity(name="Apple", entity_type="Fruit", entity_confidence=0.9),
            Entity(name="Apple", entity_type="Company", entity_confidence=0.85),
        ]
        merged = _merge_entities(entities)
        assert len(merged) == 2

    def test_merge_entities_keeps_highest_confidence(self):
        entities = [
            Entity(
                name="Cancer",
                entity_type="Disease",
                entity_confidence=0.95,
                metadata={"source_chunk": "c1"},
            ),
            Entity(
                name="Cancer",
                entity_type="Disease",
                entity_confidence=0.81,
                metadata={"source_chunk": "c2"},
            ),
        ]
        merged = _merge_entities(entities)
        assert len(merged) == 1
        assert merged[0].entity_confidence == 0.95
        assert merged[0].metadata["confidence"] == 0.95

    def test_handles_api_error(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("API down")

        chunks = [Chunk(id="c1", content="test")]
        entities, rels = extract_entities_full(chunks, openai_client=client)
        assert entities == []
        assert rels == []


class TestDocumentTypeAndCandidates:
    def test_infers_medical(self):
        chunks = [Chunk(id="c1", content="content", metadata={"section_title": "Diagnosis and Treatment"})]
        assert infer_document_type(chunks) == "medical"

    def test_infers_paper(self):
        chunks = [Chunk(id="c1", content="content", metadata={"section_title": "Abstract"})]
        assert infer_document_type(chunks) == "paper"

    def test_infers_technical(self):
        chunks = [Chunk(id="c1", content="content", metadata={"section_title": "API Architecture"})]
        assert infer_document_type(chunks) == "technical"

    @patch("agentic_graph_rag.indexing.skeleton.get_settings")
    def test_resolves_paper_damping(self, mock_settings):
        cfg = MagicMock()
        cfg.indexing.pagerank_damping = 0.85
        cfg.indexing.pagerank_damping_technical = 0.8
        cfg.indexing.pagerank_damping_paper = 0.9
        mock_settings.return_value = cfg

        chunks = [Chunk(id="c1", content="content", metadata={"section_title": "Methodology"})]
        assert resolve_pagerank_damping(chunks) == 0.9

    def test_extract_candidate_entities(self):
        text = "GraphRAG uses Neo4j and PageRank in OpenAI pipelines."
        candidates = extract_candidate_entities(text)
        assert "GraphRAG" in candidates
        assert "Neo4j" in candidates

    def test_extract_candidate_entities_finds_medical_terms(self):
        text = "The patient developed meningitis and bacteremia after surgery."
        candidates = extract_candidate_entities(text)
        assert "meningitis" in [item.lower() for item in candidates]


# ---------------------------------------------------------------------------
# link_peripheral_keywords
# ---------------------------------------------------------------------------

class TestLinkPeripheralKeywords:
    def test_empty(self):
        assert link_peripheral_keywords([], []) == []

    def test_links_matching_entities(self):
        entities = [Entity(id="e1", name="Insulin", entity_type="Drug")]
        chunks = [
            Chunk(id="c1", content="Insulin is effective"),
            Chunk(id="c2", content="Java is also good"),
        ]
        rels = link_peripheral_keywords(chunks, entities)
        assert len(rels) == 1
        assert rels[0].source == "Insulin"
        assert rels[0].target == "c1"
        assert rels[0].relation_type == "MENTIONED_IN"

    def test_case_insensitive(self):
        entities = [Entity(id="e1", name="INSULIN", entity_type="Drug")]
        chunks = [Chunk(id="c1", content="insulin is effective")]
        rels = link_peripheral_keywords(chunks, entities)
        assert len(rels) == 1

    def test_uses_same_alias_normalization_as_primary_linking(self):
        entities = [Entity(id="e1", name="Type 2 Diabetes", entity_type="Disease", metadata={"aliases": ["Type-2-Diabetes"]})]
        chunks = [Chunk(id="c1", content="Type 2 Diabetes requires long-term monitoring")]
        rels = link_peripheral_keywords(chunks, entities)
        assert len(rels) == 1
        assert rels[0].source == "Type 2 Diabetes"

    def test_skips_short_names(self):
        entities = [Entity(id="e1", name="A")]
        chunks = [Chunk(id="c1", content="A is a letter")]
        rels = link_peripheral_keywords(chunks, entities)
        assert len(rels) == 0  # "A" too short (<2 chars)

    def test_multiple_matches(self):
        entities = [
            Entity(id="e1", name="Insulin", entity_type="Drug"),
            Entity(id="e2", name="Glucose", entity_type="Biomarker"),
        ]
        chunks = [Chunk(id="c1", content="Insulin reduced glucose levels")]
        rels = link_peripheral_keywords(chunks, entities)
        assert len(rels) == 2

    def test_only_links_medically_salient_entities(self):
        entities = [
            Entity(id="e1", name="General Study", entity_type="Concept"),
            Entity(id="e2", name="Insulin", entity_type="Drug"),
        ]
        chunks = [Chunk(id="c1", content="General Study compared outcomes and insulin dosing")]
        rels = link_peripheral_keywords(chunks, entities)
        assert len(rels) == 1
        assert rels[0].source == "Insulin"

    def test_promotes_high_confidence_aliases_from_peripheral_hits(self):
        entities = [
            Entity(
                id="e1",
                name="Diabetes Mellitus",
                entity_type="Disease",
                metadata={"candidate_entities": ["DM"]},
            )
        ]
        chunks = [
            Chunk(id="c1", content="DM management requires follow-up"),
            Chunk(id="c2", content="DM treatment needs monitoring"),
        ]
        rels = link_peripheral_keywords(chunks, entities)
        assert len(rels) == 2
        assert "DM" in entities[0].metadata["aliases"]
        assert "learned_aliases" not in entities[0].metadata
        assert "alias_counts" not in entities[0].metadata
        assert "alias_evidence" not in entities[0].metadata


# ---------------------------------------------------------------------------
# extract_keywords
# ---------------------------------------------------------------------------

class TestExtractKeywords:
    def test_basic(self):
        keywords = extract_keywords("Python machine learning is very good for data")
        assert "python" in keywords
        assert "machine" in keywords

    def test_stop_words_removed(self):
        keywords = extract_keywords("the quick brown fox is very fast")
        assert "the" not in keywords
        assert "quick" in keywords

    def test_max_keywords(self):
        words = ["alpha", "bravo", "charlie", "delta", "echo",
                 "foxtrot", "golf", "hotel", "india", "juliet"]
        text = " ".join(words * 2)  # repeat so all have freq
        keywords = extract_keywords(text, max_keywords=5)
        assert len(keywords) == 5

    def test_empty(self):
        assert extract_keywords("") == []


# ---------------------------------------------------------------------------
# build_skeleton_index (orchestrator)
# ---------------------------------------------------------------------------

class TestBuildSkeletonIndex:
    def test_empty(self):
        entities, rels, skel, peri = build_skeleton_index([], [])
        assert entities == []
        assert rels == []
        assert skel == []
        assert peri == []

    def test_full_pipeline(self):
        """Integration test with mock LLM."""
        client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "ENTITY: TestEntity | Concept | A test"
        client.chat.completions.create.return_value = resp

        chunks = _make_chunks(8)
        embs = _make_embeddings(8)

        entities, rels, skeletal, peripheral = build_skeleton_index(
            chunks, embs, openai_client=client
        )

        # short documents now cap skeletal extraction to avoid full-width LLM passes
        assert len(skeletal) == 4
        assert len(peripheral) == 4
        assert len(entities) >= 1  # at least something extracted
        # Only skeletal chunks trigger deep extraction calls
        assert client.chat.completions.create.call_count == 4

    @patch("agentic_graph_rag.indexing.skeleton.get_settings")
    @patch("agentic_graph_rag.indexing.skeleton.extract_entities_full")
    def test_short_docs_keep_all_chunks(self, mock_extract, mock_settings):
        cfg = MagicMock()
        cfg.indexing.knn_k = 2
        cfg.indexing.pagerank_damping = 0.85
        cfg.indexing.pagerank_damping_technical = 0.8
        cfg.indexing.pagerank_damping_paper = 0.9
        cfg.indexing.skeleton_beta = 0.25
        cfg.indexing.skeleton_beta_short_doc = 1.0
        cfg.indexing.skeleton_beta_medium_doc = 0.5
        cfg.indexing.skeleton_beta_long_doc = 0.3
        cfg.indexing.skeleton_short_doc_max_chunks = 8
        cfg.indexing.skeleton_medium_doc_max_chunks = 24
        mock_settings.return_value = cfg
        mock_extract.return_value = ([], [])

        chunks = _make_chunks(4)
        embs = _make_embeddings(4)
        entities, rels, skeletal, peripheral = build_skeleton_index(chunks, embs, openai_client=MagicMock())

        assert entities == []
        assert rels == []
        assert len(skeletal) == 2
        assert len(peripheral) == 2

    @patch("agentic_graph_rag.indexing.skeleton.persist_entity_alias_metadata")
    @patch("agentic_graph_rag.indexing.skeleton.get_settings")
    @patch("agentic_graph_rag.indexing.skeleton.extract_entities_full")
    def test_persists_medical_aliases_when_driver_present(self, mock_extract, mock_settings, mock_persist):
        cfg = MagicMock()
        cfg.indexing.knn_k = 2
        cfg.indexing.pagerank_damping = 0.85
        cfg.indexing.pagerank_damping_technical = 0.8
        cfg.indexing.pagerank_damping_paper = 0.9
        cfg.indexing.skeleton_beta = 0.25
        cfg.indexing.skeleton_beta_short_doc = 1.0
        cfg.indexing.skeleton_beta_medium_doc = 0.5
        cfg.indexing.skeleton_beta_long_doc = 0.3
        cfg.indexing.skeleton_short_doc_max_chunks = 8
        cfg.indexing.skeleton_medium_doc_max_chunks = 24
        mock_settings.return_value = cfg
        mock_extract.return_value = (
            [Entity(id="e1", name="Diabetes Mellitus", entity_type="Disease")],
            [],
        )

        chunks = _make_chunks(4)
        embs = _make_embeddings(4)
        driver = MagicMock()

        build_skeleton_index(chunks, embs, openai_client=MagicMock(), driver=driver)

        mock_persist.assert_called_once()
        assert mock_persist.call_args.args[1] is driver
