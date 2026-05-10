"""Tests for provider/orchestrator retrieval architecture."""

from unittest.mock import MagicMock, patch

from rag_core.models import Chunk, Entity, GraphContext, QueryType, SearchResult

from agentic_graph_rag.retrieval.fusion import (
    FusionView,
    FusionEngine,
    calibrate_channel_weights,
    resolve_channel_weights,
)
from agentic_graph_rag.retrieval.orchestrator import RetrievalOrchestrator
from agentic_graph_rag.retrieval.providers import (
    GraphRetrievalProvider,
    BM25RetrievalProvider,
    RetrievalRequest,
    VectorRetrievalProvider,
    _build_bm25_search_text,
    build_bm25_focus_query,
    graph_context_to_search_results,
)


def _mock_driver() -> MagicMock:
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return driver


class TestGraphContextSerialization:
    def test_includes_paths_entities_and_evidence(self):
        ctx = GraphContext(
            triplets=[{"source": "Alice", "relation": "WORKS_ON", "target": "Project X"}],
            entities=[Entity(name="Alice", entity_type="Person")],
            passages=["Alice works on Project X."],
            source_ids=["c1"],
        )

        results = graph_context_to_search_results(
            ctx,
            source="graph",
            include_graph_structure=True,
        )

        assert len(results) == 1
        assert "Graph paths:" in results[0].chunk.content
        assert "Entities:" in results[0].chunk.content
        assert "Evidence:" in results[0].chunk.content

    def test_binds_graph_paths_to_matching_passage_only(self):
        ctx = GraphContext(
            triplets=[
                {"source": "COPD", "relation": "RELATED_TO", "target": "FEV1"},
                {"source": "COPD", "relation": "TREATED_BY", "target": "LABA"},
            ],
            passages=[
                "COPD 诊断依据 FEV1/FVC < 0.70。",
                "COPD 治疗方案包括 LABA。",
            ],
            source_ids=["diagnosis", "treatment"],
        )

        results = graph_context_to_search_results(
            ctx,
            source="graph",
            include_graph_structure=True,
            query="COPD 诊断标准",
        )

        by_id = {item.chunk.id: item for item in results}
        assert "COPD -[RELATED_TO]-> FEV1" in by_id["diagnosis"].chunk.content
        assert "TREATED_BY" not in by_id["diagnosis"].chunk.content
        assert "COPD -[TREATED_BY]-> LABA" in by_id["treatment"].chunk.content

    def test_respects_top_k_cap(self):
        ctx = GraphContext(
            passages=["Text A", "Text B", "Text C"],
            source_ids=["c1", "c2", "c3"],
        )

        results = graph_context_to_search_results(ctx, source="vector", top_k=2)

        assert [item.chunk.id for item in results] == ["c1", "c2"]

    def test_build_bm25_search_text_keeps_lexical_anchors(self):
        search_text = _build_bm25_search_text(
            "During the events described in Samuel Pepys's diary, who was the lady that dined with him?"
        )

        assert "+lady" in search_text
        assert "+dined" in search_text
        assert '"Samuel Pepys"' in search_text
        assert "diary" not in search_text.casefold()

    def test_build_bm25_focus_query_removes_chatty_noise(self):
        focus_query = build_bm25_focus_query(
            "你好，我想了解一下二型糖尿病一般用什么药治疗比较合适呢？",
            [
                "免责声明 目录 版权声明",
                "二型糖尿病 胰岛素 二甲双胍 治疗",
                "高血压 ACEI ARB 治疗",
            ],
        )

        assert "二型糖尿病" in focus_query
        assert "你好" not in focus_query
        assert "了解一下" not in focus_query


class TestProviders:
    @patch("agentic_graph_rag.retrieval.providers.VectorStore")
    def test_vector_provider_returns_normalized_results(self, mock_vector_store):
        mock_vector_store.return_value.search.return_value = [
            SearchResult(chunk=Chunk(id="c1", content="Vector text"), score=0.91, rank=1)
        ]

        provider = VectorRetrievalProvider(_mock_driver())
        results = provider.retrieve(
            RetrievalRequest(query="q", query_embedding=[1.0, 0.0], top_k=5)
        )

        assert len(results) == 1
        assert results[0].chunk.id == "c1"
        assert results[0].source == "vector"
        assert results[0].score_normalized == 0.91
        mock_vector_store.return_value.search.assert_called_once_with([1.0, 0.0], top_k=15)

    @patch("agentic_graph_rag.retrieval.providers.VectorStore")
    def test_vector_provider_preserves_vector_store_ranking(self, mock_vector_store):
        mock_vector_store.return_value.search.return_value = [
            SearchResult(chunk=Chunk(id="c1", content="Vector A"), score=0.99, rank=1),
            SearchResult(chunk=Chunk(id="c2", content="Vector B"), score=0.88, rank=2),
            SearchResult(chunk=Chunk(id="c3", content="Vector C"), score=0.77, rank=3),
        ]

        provider = VectorRetrievalProvider(_mock_driver())
        results = provider.retrieve(
            RetrievalRequest(query="q", query_embedding=[1.0, 0.0], top_k=2)
        )

        assert [item.chunk.id for item in results] == ["c1", "c2"]
        assert [item.rank for item in results] == [1, 2]

    @patch("agentic_graph_rag.retrieval.providers.VectorStore")
    def test_vector_provider_deduplicates_near_identical_content(self, mock_vector_store):
        duplicate_prefix = "COPD 诊断标准 FEV1/FVC < 0.70 " * 30
        mock_vector_store.return_value.search.return_value = [
            SearchResult(chunk=Chunk(id="c1", content=duplicate_prefix + "A"), score=0.91, rank=1),
            SearchResult(chunk=Chunk(id="c2", content=duplicate_prefix + "B"), score=0.90, rank=2),
            SearchResult(chunk=Chunk(id="c3", content="其他 COPD 评估内容"), score=0.80, rank=3),
        ]

        provider = VectorRetrievalProvider(_mock_driver())
        results = provider.retrieve(
            RetrievalRequest(
                query="COPD 诊断标准",
                query_embedding=[1.0, 0.0],
                top_k=3,
            )
        )

        assert [item.chunk.id for item in results] == ["c1", "c3"]
        assert [item.rank for item in results] == [1, 2]
        assert results[0].score_normalized == 1.0

    @patch("agentic_graph_rag.retrieval.providers.vector_cypher_search")
    def test_graph_provider_serializes_graph_context(self, mock_search):
        mock_search.return_value = GraphContext(
            triplets=[{"source": "A", "relation": "LINKS", "target": "B"}],
            passages=["A links to B."],
            source_ids=["c9"],
        )

        provider = GraphRetrievalProvider(_mock_driver())
        results = provider.retrieve(
            RetrievalRequest(
                query="q",
                query_embedding=[1.0, 0.0],
                top_k=5,
                filters={"max_hops": 4},
            )
        )

        assert len(results) == 1
        assert results[0].source == "graph"
        assert "Graph paths:" in results[0].chunk.content
        assert results[0].score_normalized == 0.6
        mock_search.assert_called_once_with(
            [1.0, 0.0],
            provider._driver,
            top_k=5,
            max_hops=4,
        )

    @patch("agentic_graph_rag.retrieval.providers._ensure_passage_fulltext_index")
    def test_bm25_provider_uses_fulltext_query_and_normalizes_results(self, mock_ensure_index):
        driver = _mock_driver()
        session = driver.session.return_value.__enter__.return_value
        session.run.return_value = [
            {
                "id": "node-1",
                "chunk_id": "c1",
                "text": "The Lady dined with Samuel Pepys.",
                "score": 3.2,
            },
            {
                "id": "node-2",
                "chunk_id": "c2",
                "text": "Other diary entry.",
                "score": 1.1,
            },
        ]

        provider = BM25RetrievalProvider(driver)
        results = provider.retrieve(RetrievalRequest(query="The Lady", top_k=2))

        assert [item.chunk.id for item in results] == ["c1", "c2"]
        assert results[0].source == "bm25"
        assert results[0].score == 3.2
        assert results[0].score_normalized is not None
        assert results[0].score_normalized > 0.75
        mock_ensure_index.assert_called_once_with(driver)
        assert session.run.call_args.kwargs["search_text"] == "+lady"

    @patch("agentic_graph_rag.retrieval.providers._ensure_passage_fulltext_index")
    def test_bm25_provider_prefers_tighter_anchor_cooccurrence(self, mock_ensure_index):
        driver = _mock_driver()
        session = driver.session.return_value.__enter__.return_value
        session.run.return_value = [
            {
                "id": "node-1",
                "chunk_id": "c_title",
                "text": (
                    "Samuel Pepys wrote the diary. "
                    + ("x" * 240)
                    + " my Lady was mentioned here. "
                    + ("y" * 240)
                    + " Later he dined elsewhere."
                ),
                "score": 9.0,
            },
            {
                "id": "node-2",
                "chunk_id": "c_local",
                "text": "He dined with my Lady and was kindly treated by her.",
                "score": 1.5,
            },
        ]

        provider = BM25RetrievalProvider(driver)
        results = provider.retrieve(
            RetrievalRequest(
                query="Who was the lady that dined with him?",
                top_k=2,
            )
        )

        assert [item.chunk.id for item in results] == ["c_local", "c_title"]

    @patch("agentic_graph_rag.retrieval.providers._ensure_passage_fulltext_index")
    def test_bm25_provider_uses_focus_query_for_chatty_chinese_request(self, mock_ensure_index):
        driver = _mock_driver()
        session = driver.session.return_value.__enter__.return_value
        session.run.side_effect = [
            [
                {"text": "目录"},
                {"text": "二型糖尿病 胰岛素 二甲双胍 治疗"},
                {"text": "高血压 ACEI ARB 治疗"},
            ],
            [
                {
                    "id": "node-1",
                    "chunk_id": "c1",
                    "text": "二型糖尿病通常使用二甲双胍或胰岛素治疗。",
                    "score": 2.5,
                }
            ],
        ]

        provider = BM25RetrievalProvider(driver)
        provider.retrieve(
            RetrievalRequest(
                query="你好，我想了解一下二型糖尿病一般用什么药治疗比较合适呢？",
                top_k=3,
            )
        )

        search_kwargs = session.run.call_args_list[1].kwargs
        assert "二型糖尿病" in search_kwargs["search_text"]
        assert "你好" not in search_kwargs["search_text"]
        mock_ensure_index.assert_called_once_with(driver)

    @patch("agentic_graph_rag.retrieval.providers.vector_cypher_search")
    def test_graph_provider_caps_expanded_graph_context(self, mock_search):
        mock_search.return_value = GraphContext(
            triplets=[{"source": "A", "relation": "LINKS", "target": "B"}],
            passages=["A links to B.", "B links to C.", "C links to D."],
            source_ids=["c1", "c2", "c3"],
        )

        provider = GraphRetrievalProvider(_mock_driver())
        results = provider.retrieve(
            RetrievalRequest(
                query="q",
                query_embedding=[1.0, 0.0],
                top_k=2,
                filters={"max_hops": 4},
            )
        )

        assert [item.chunk.id for item in results] == ["c1", "c2"]

    @patch("agentic_graph_rag.retrieval.providers.vector_cypher_search")
    def test_graph_provider_prioritizes_medical_diagnostic_evidence(self, mock_search):
        mock_search.return_value = GraphContext(
            triplets=[
                {"source": "COPD", "relation": "RELATED_TO", "target": "FEV1"},
                {"source": "FEV1", "relation": "RELATED_TO", "target": "肺功能检查"},
            ],
            passages=[
                "COPD 治疗方案包括 LABA 或 LAMA。",
                "COPD 诊断依据肺功能检查，FEV1/FVC < 0.70 即可确诊。",
            ],
            source_ids=["treatment", "diagnosis"],
        )

        provider = GraphRetrievalProvider(_mock_driver())
        results = provider.retrieve(
            RetrievalRequest(
                query="COPD的诊断标准是什么？",
                query_embedding=[1.0, 0.0],
                top_k=2,
                filters={"max_hops": 4},
            )
        )

        assert results[0].chunk.id == "diagnosis"
        assert results[0].score_normalized >= 0.95

    @patch("agentic_graph_rag.retrieval.providers.get_settings")
    @patch("agentic_graph_rag.retrieval.providers.vector_cypher_search")
    def test_graph_provider_uses_entry_top_k_override(self, mock_search, mock_settings):
        cfg = MagicMock()
        cfg.retrieval.max_hops = 3
        cfg.retrieval.graph_entry_top_k = 4
        mock_settings.return_value = cfg
        mock_search.return_value = GraphContext(passages=["A links to B."], source_ids=["c1"])

        provider = GraphRetrievalProvider(_mock_driver())
        provider.retrieve(
            RetrievalRequest(
                query="q",
                query_embedding=[1.0, 0.0],
                top_k=10,
                filters={"max_hops": 2, "entry_top_k": 3},
            )
        )

        mock_search.assert_called_once_with(
            [1.0, 0.0],
            provider._driver,
            top_k=3,
            max_hops=2,
        )


class _FakeProvider:
    def __init__(self, name: str, results: list[SearchResult]) -> None:
        self.name = name
        self._results = results
        self.calls = 0

    def retrieve(self, request: RetrievalRequest) -> list[SearchResult]:
        _ = request
        self.calls += 1
        return self._results


class TestOrchestrator:
    @patch("agentic_graph_rag.retrieval.orchestrator.attach_passage_embeddings")
    @patch("agentic_graph_rag.retrieval.orchestrator.rerank")
    def test_applies_query_type_weights_during_fusion(self, mock_rerank, mock_attach):
        vector_results = [
            SearchResult(chunk=Chunk(id="c1", content="vector"), score=1.0, rank=1, source="vector")
        ]
        graph_results = [
            SearchResult(chunk=Chunk(id="c2", content="graph"), score=1.0, rank=1, source="graph")
        ]
        mock_attach.side_effect = lambda results, _driver: results
        mock_rerank.side_effect = lambda _query, results, **_kwargs: results

        orchestrator = RetrievalOrchestrator(
            _mock_driver(),
            providers=[
                _FakeProvider("vector", vector_results),
                _FakeProvider("graph", graph_results),
            ],
            fusion_engine=FusionEngine(rrf_k=60),
        )

        results = orchestrator.search(
            query="q",
            query_embedding=[1.0, 0.0],
            top_k=2,
            query_type=QueryType.MULTI_HOP,
            provider_top_k={"vector": 5, "graph": 5},
        )

        assert [item.chunk.id for item in results] == ["c2", "c1"]

    @patch("agentic_graph_rag.retrieval.orchestrator.attach_passage_embeddings")
    @patch("agentic_graph_rag.retrieval.orchestrator.rerank")
    def test_reuses_seed_results_without_recalling_cached_providers(self, mock_rerank, mock_attach):
        vector_results = [
            SearchResult(chunk=Chunk(id="c1", content="vector"), score=1.0, rank=1, source="vector")
        ]
        bm25_results = [
            SearchResult(chunk=Chunk(id="c2", content="bm25"), score=1.0, rank=1, source="bm25")
        ]
        graph_results = [
            SearchResult(chunk=Chunk(id="c3", content="graph"), score=1.0, rank=1, source="graph")
        ]
        vector_provider = _FakeProvider("vector", vector_results)
        bm25_provider = _FakeProvider("bm25", bm25_results)
        graph_provider = _FakeProvider("graph", graph_results)
        mock_attach.side_effect = lambda results, _driver: results
        mock_rerank.side_effect = lambda _query, results, **_kwargs: results

        orchestrator = RetrievalOrchestrator(
            _mock_driver(),
            providers=[vector_provider, bm25_provider, graph_provider],
            fusion_engine=FusionEngine(rrf_k=60),
        )

        results = orchestrator.search(
            query="q",
            query_embedding=[1.0, 0.0],
            top_k=3,
            query_type=QueryType.MULTI_HOP,
            provider_top_k={"vector": 5, "bm25": 5, "graph": 5},
            seed_results={"vector": vector_results, "bm25": bm25_results},
            enabled_providers=["graph"],
        )

        assert [item.chunk.id for item in results] == ["c3", "c1", "c2"]
        assert vector_provider.calls == 0
        assert bm25_provider.calls == 0
        assert graph_provider.calls == 1

    @patch("agentic_graph_rag.retrieval.orchestrator.attach_passage_embeddings")
    @patch("agentic_graph_rag.retrieval.orchestrator.rerank")
    def test_exposes_provider_results_for_incremental_reuse(self, mock_rerank, mock_attach):
        vector_results = [
            SearchResult(chunk=Chunk(id="c1", content="vector"), score=1.0, rank=1, source="vector")
        ]
        graph_results = [
            SearchResult(chunk=Chunk(id="c2", content="graph"), score=1.0, rank=1, source="graph")
        ]
        vector_provider = _FakeProvider("vector", vector_results)
        graph_provider = _FakeProvider("graph", graph_results)
        provider_results: dict[str, list[SearchResult]] = {}
        mock_attach.side_effect = lambda results, _driver: results
        mock_rerank.side_effect = lambda _query, results, **_kwargs: results

        orchestrator = RetrievalOrchestrator(
            _mock_driver(),
            providers=[vector_provider, graph_provider],
            fusion_engine=FusionEngine(rrf_k=60),
        )

        _ = orchestrator.search(
            query="q",
            query_embedding=[1.0, 0.0],
            top_k=2,
            provider_top_k={"vector": 5, "graph": 5},
            provider_results=provider_results,
        )

        assert provider_results["vector"] == vector_results
        assert provider_results["graph"] == graph_results

    @patch("agentic_graph_rag.retrieval.orchestrator.attach_passage_embeddings")
    @patch("agentic_graph_rag.retrieval.orchestrator.rerank")
    def test_skips_rerank_when_disabled(self, mock_rerank, mock_attach):
        vector_results = [
            SearchResult(chunk=Chunk(id="c1", content="vector"), score=1.0, rank=1, source="vector")
        ]
        mock_attach.side_effect = lambda results, _driver: results

        orchestrator = RetrievalOrchestrator(
            _mock_driver(),
            providers=[_FakeProvider("vector", vector_results)],
            fusion_engine=FusionEngine(rrf_k=60),
        )

        results = orchestrator.search(
            query="q",
            query_embedding=[1.0, 0.0],
            top_k=1,
            rerank_enabled=False,
        )

        assert len(results) == 1
        assert results[0].chunk.id == "c1"
        mock_rerank.assert_not_called()

    @patch("agentic_graph_rag.retrieval.orchestrator.attach_passage_embeddings")
    @patch("agentic_graph_rag.retrieval.orchestrator.rerank")
    def test_finalization_preserves_reranker_rank(self, mock_rerank, mock_attach):
        vector_results = [
            SearchResult(chunk=Chunk(id="c1", content="vector"), score=1.0, rank=1, source="vector")
        ]
        reranked = [
            SearchResult(
                chunk=Chunk(id="c1", content="vector"),
                score=0.7,
                score_normalized=1.0,
                rank=42,
                source="hybrid",
            )
        ]
        mock_attach.side_effect = lambda results, _driver: results
        mock_rerank.return_value = reranked

        orchestrator = RetrievalOrchestrator(
            _mock_driver(),
            providers=[_FakeProvider("vector", vector_results)],
            fusion_engine=FusionEngine(rrf_k=60),
        )

        results = orchestrator.search(
            query="q",
            query_embedding=[1.0, 0.0],
            top_k=1,
            rerank_enabled=True,
        )

        assert results[0].rank == 42
        assert results[0].source == "hybrid"


class TestFusionWeights:
    def test_fusion_views_keep_original_results_unchanged(self):
        vector_result = SearchResult(
            chunk=Chunk(id="c1", content="same evidence"),
            score=0.91,
            rank=7,
            source="vector",
        )

        views = FusionEngine(rrf_k=60).build_views(
            [vector_result],
            top_k=1,
            weights={"vector": 2.0},
        )

        assert views == [
            FusionView(
                result=vector_result,
                fusion_score=2.0 / 61.0,
                fusion_rank=1,
                fused_source="hybrid",
            )
        ]
        assert vector_result.score == 0.91
        assert vector_result.rank == 7
        assert vector_result.source == "vector"

    def test_resolve_channel_weights_for_multi_hop(self):
        weights = resolve_channel_weights(QueryType.MULTI_HOP)
        assert weights["graph"] > weights["vector"]
        assert weights["graph"] > weights["bm25"]

    def test_calibrate_channel_weights_boosts_exact_bm25_matches(self):
        provider_results = {
            "vector": [
                SearchResult(chunk=Chunk(id="c1", content="semantic summary"), score=1.0, rank=1, source="vector")
            ],
            "bm25": [
                SearchResult(chunk=Chunk(id="c2", content="ERR-902X retry failed"), score=1.0, rank=1, source="bm25")
            ],
            "graph": [],
        }

        weights = calibrate_channel_weights(
            "ERR-902X",
            provider_results,
            query_type=QueryType.SIMPLE,
        )

        assert weights["bm25"] > weights["vector"]
        assert weights["graph"] < resolve_channel_weights(QueryType.SIMPLE)["graph"]

    @patch("agentic_graph_rag.retrieval.orchestrator.attach_passage_embeddings")
    @patch("agentic_graph_rag.retrieval.orchestrator.rerank")
    def test_orchestrator_uses_diagnostic_weights(self, mock_rerank, mock_attach):
        vector_results = [
            SearchResult(chunk=Chunk(id="c1", content="semantic summary"), score=1.0, rank=1, source="vector")
        ]
        bm25_results = [
            SearchResult(chunk=Chunk(id="c2", content="ERR-902X retry failed"), score=1.0, rank=1, source="bm25")
        ]
        mock_attach.side_effect = lambda results, _driver: results
        mock_rerank.side_effect = lambda _query, results, **_kwargs: results

        orchestrator = RetrievalOrchestrator(
            _mock_driver(),
            providers=[
                _FakeProvider("vector", vector_results),
                _FakeProvider("bm25", bm25_results),
            ],
            fusion_engine=FusionEngine(rrf_k=60),
        )

        results = orchestrator.search(
            query="ERR-902X",
            query_embedding=[1.0, 0.0],
            top_k=2,
            query_type=QueryType.SIMPLE,
            provider_top_k={"vector": 5, "bm25": 5},
        )

        assert [item.chunk.id for item in results] == ["c2", "c1"]
