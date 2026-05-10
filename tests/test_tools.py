"""Tests for agentic_graph_rag.agent.tools."""

from unittest.mock import ANY, MagicMock, patch

from rag_core.models import Chunk, GraphContext, SearchResult

from agentic_graph_rag.agent.tools import (
    _cosine_similarity,
    _embed_query,
    _generate_sub_queries,
    _graph_context_to_results,
    _rrf_merge,
    bm25_search,
    community_search,
    comprehensive_search,
    full_document_read,
    hybrid_search,
    temporal_query,
    vector_search,
)
from agentic_graph_rag.retrieval.fusion import resolve_channel_weights as _get_channel_weights
from agentic_graph_rag.retrieval.providers import fetch_passage_embeddings as _fetch_passage_embeddings


def _mock_driver():
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return driver


def _mock_openai_client(embedding=None):
    client = MagicMock()
    resp = MagicMock()
    resp.data = [MagicMock()]
    resp.data[0].embedding = embedding or [1.0, 0.0]
    client.embeddings.create.return_value = resp
    return client


def _make_results(n: int, source: str = "vector") -> list[SearchResult]:
    return [
        SearchResult(
            chunk=Chunk(id=f"c{i}", content=f"Content {i}"),
            score=1.0 / (i + 1),
            rank=i + 1,
            source=source,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# _embed_query
# ---------------------------------------------------------------------------

class TestEmbedQuery:
    def test_returns_embedding(self):
        client = _mock_openai_client([0.5, 0.5])
        emb = _embed_query("test", client)
        assert emb == [0.5, 0.5]
        client.embeddings.create.assert_called_once()


# ---------------------------------------------------------------------------
# _graph_context_to_results
# ---------------------------------------------------------------------------

class TestGraphContextToResults:
    def test_empty_context(self):
        ctx = GraphContext()
        results = _graph_context_to_results(ctx, "test")
        assert results == []

    def test_converts_passages(self):
        ctx = GraphContext(
            passages=["Text A", "Text B"],
            source_ids=["c1", "c2"],
        )
        results = _graph_context_to_results(ctx, "graph")
        assert len(results) == 2
        assert results[0].chunk.content == "Text A"
        assert results[0].chunk.id == "c1"
        assert results[0].source == "graph"
        assert results[0].rank == 1
        assert results[1].rank == 2

    def test_handles_missing_source_ids(self):
        ctx = GraphContext(
            passages=["Text A", "Text B"],
            source_ids=["c1"],
        )
        results = _graph_context_to_results(ctx, "test")
        assert len(results) == 2
        assert results[0].chunk.id == "c1"
        assert results[1].chunk.id == ""  # no source_id for index 1


# ---------------------------------------------------------------------------
# _rrf_merge
# ---------------------------------------------------------------------------

class TestRRFMerge:
    def test_empty_lists(self):
        merged = _rrf_merge([], [])
        assert merged == []

    def test_single_list(self):
        results = _make_results(3)
        merged = _rrf_merge(results, [], top_k=5)
        assert len(merged) == 3

    def test_merges_two_lists(self):
        a = _make_results(3, source="vector")
        b = _make_results(3, source="graph")
        merged = _rrf_merge(a, b, top_k=5)
        # 3 unique IDs (c0, c1, c2) with combined scores
        assert len(merged) == 3
        assert merged[0].source == "hybrid"

    def test_deduplicates(self):
        a = [SearchResult(chunk=Chunk(id="c1", content="A"), score=0.9, rank=1, source="v")]
        b = [SearchResult(chunk=Chunk(id="c1", content="A"), score=0.8, rank=1, source="g")]
        merged = _rrf_merge(a, b, top_k=5)
        assert len(merged) == 1  # deduplicated by id

    def test_respects_top_k(self):
        a = _make_results(10)
        b = _make_results(10)
        merged = _rrf_merge(a, b, top_k=3)
        assert len(merged) == 3

    def test_ranks_assigned(self):
        merged = _rrf_merge(_make_results(3), _make_results(3), top_k=5)
        for i, r in enumerate(merged, start=1):
            assert r.rank == i

    def test_merges_three_lists(self):
        a = _make_results(2, source="vector")
        b = _make_results(2, source="bm25")
        c = _make_results(2, source="graph")
        merged = _rrf_merge(a, b, c, top_k=5)
        assert len(merged) == 2
        assert merged[0].source == "hybrid"

    def test_applies_source_weights(self):
        a = [SearchResult(chunk=Chunk(id="c1", content="Vector"), score=0.9, rank=1, source="vector")]
        b = [SearchResult(chunk=Chunk(id="c2", content="Graph"), score=0.9, rank=1, source="graph")]
        merged = _rrf_merge(
            a,
            b,
            top_k=2,
            weights={"vector": 0.5, "graph": 2.0},
        )
        assert [item.chunk.id for item in merged] == ["c2", "c1"]


class TestChannelWeights:
    def test_uses_query_type_defaults(self):
        weights = _get_channel_weights("multi_hop")
        assert weights["graph"] > weights["vector"]
        assert weights["graph"] > weights["bm25"]

    def test_override_wins(self):
        weights = _get_channel_weights("simple", overrides={"bm25": 1.4})
        assert weights["bm25"] == 1.4


# ---------------------------------------------------------------------------
# bm25_search
# ---------------------------------------------------------------------------

class TestBM25Search:
    @patch("agentic_graph_rag.retrieval.providers.BM25RetrievalProvider.retrieve")
    def test_returns_ranked_results(self, mock_retrieve):
        driver = _mock_driver()
        expected = [
            SearchResult(chunk=Chunk(id="c1", content="Alpha"), score=3.5, rank=1, source="bm25"),
            SearchResult(chunk=Chunk(id="c2", content="Beta"), score=1.2, rank=2, source="bm25"),
        ]
        mock_retrieve.return_value = expected

        results = bm25_search("alpha", driver, _mock_openai_client(), top_k=2)

        assert results == expected
        assert results[0].chunk.id == "c1"
        mock_retrieve.assert_called_once()


# ---------------------------------------------------------------------------
# _fetch_passage_embeddings
# ---------------------------------------------------------------------------

class TestFetchPassageEmbeddings:
    def test_returns_embeddings_for_known_ids(self):
        driver = _mock_driver()
        session = driver.session().__enter__()

        rec1 = MagicMock()
        rec1.__getitem__ = lambda self, key: {"chunk_id": "c1", "embedding": [0.5, 0.5]}[key]
        rec2 = MagicMock()
        rec2.__getitem__ = lambda self, key: {"chunk_id": "c2", "embedding": [1.0, 0.0]}[key]

        result_mock = MagicMock()
        result_mock.__iter__ = MagicMock(return_value=iter([rec1, rec2]))
        session.run.return_value = result_mock

        emb_map = _fetch_passage_embeddings(["c1", "c2"], driver)
        assert len(emb_map) == 2
        assert emb_map["c1"] == [0.5, 0.5]
        assert emb_map["c2"] == [1.0, 0.0]

    def test_empty_chunk_ids(self):
        driver = _mock_driver()
        emb_map = _fetch_passage_embeddings([], driver)
        assert emb_map == {}

    def test_skips_missing_embeddings(self):
        driver = _mock_driver()
        session = driver.session().__enter__()

        rec1 = MagicMock()
        rec1.__getitem__ = lambda self, key: {"chunk_id": "c1", "embedding": None}[key]

        result_mock = MagicMock()
        result_mock.__iter__ = MagicMock(return_value=iter([rec1]))
        session.run.return_value = result_mock

        emb_map = _fetch_passage_embeddings(["c1"], driver)
        assert emb_map == {}


# ---------------------------------------------------------------------------
# hybrid_search (cosine re-ranking)
# ---------------------------------------------------------------------------

class TestHybridSearch:
    @patch("agentic_graph_rag.agent.tools.RetrievalOrchestrator.search")
    def test_delegates_to_orchestrator_with_query_type(self, mock_search):
        driver = _mock_driver()
        client = _mock_openai_client([1.0, 0.0])
        mock_search.return_value = _make_results(2, source="hybrid")

        results = hybrid_search("test", driver, client, top_k=3, query_type="multi_hop")

        assert len(results) == 2
        _, kwargs = mock_search.call_args
        assert kwargs["query_type"] == "multi_hop"
        assert kwargs["provider_top_k"]["bm25"] > 0


# ---------------------------------------------------------------------------
# vector_search
# ---------------------------------------------------------------------------

class TestVectorSearch:
    @patch("agentic_graph_rag.retrieval.providers.VectorRetrievalProvider.retrieve")
    def test_returns_results(self, mock_retrieve):
        driver = _mock_driver()
        client = _mock_openai_client()
        mock_retrieve.return_value = [
            SearchResult(chunk=Chunk(id="c1", content="Result"), score=1.0, rank=1, source="vector")
        ]
        results = vector_search("test", driver, client, top_k=5)

        assert len(results) == 1
        assert results[0].chunk.content == "Result"
        client.embeddings.create.assert_called_once()
        mock_retrieve.assert_called_once()


# ---------------------------------------------------------------------------
# full_document_read
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0

    def test_orthogonal_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


class TestFullDocumentRead:
    @patch("agentic_graph_rag.agent.tools._passage_vector_search")
    def test_reads_and_ranks_passages(self, mock_pvs):
        driver = _mock_driver()
        # query embedding = [1.0, 0.0]
        client = _mock_openai_client([1.0, 0.0])

        mock_pvs.return_value = [
            {"id": "p2", "text": "Text two", "chunk_id": "c2", "score": 0.99},
            {"id": "p1", "text": "Text one", "chunk_id": "c1", "score": 0.71},
        ]

        results = full_document_read("overview", driver, client, top_k=5)
        assert len(results) == 2
        # rec2 has higher similarity, should be first
        assert results[0].chunk.content == "Text two"
        assert results[0].source == "vector"
        assert results[0].score > results[1].score
        assert results[1].chunk.content == "Text one"

    @patch("agentic_graph_rag.agent.tools._passage_vector_search")
    def test_passages_without_embedding(self, mock_pvs):
        driver = _mock_driver()
        client = _mock_openai_client([1.0, 0.0])

        mock_pvs.return_value = [
            {"id": "p1", "text": "No emb", "chunk_id": "c1", "score": 0.0},
        ]

        results = full_document_read("overview", driver, client, top_k=5)
        assert len(results) == 1
        assert results[0].score == 0.0

    @patch("agentic_graph_rag.agent.tools._passage_vector_search")
    def test_empty_passages(self, mock_pvs):
        driver = _mock_driver()
        client = _mock_openai_client()

        mock_pvs.return_value = []

        results = full_document_read("overview", driver, client, top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# community_search / temporal_query (wrappers)
# ---------------------------------------------------------------------------

class TestWrapperTools:
    @patch("agentic_graph_rag.agent.tools.vector_search")
    def test_community_search_fallback(self, mock_vs):
        mock_vs.return_value = _make_results(2)
        driver = _mock_driver()
        client = _mock_openai_client()

        results = community_search("test", driver, client)
        mock_vs.assert_called_once_with("test", driver, client)
        assert len(results) == 2

    @patch("agentic_graph_rag.agent.tools._passage_vector_search")
    def test_temporal_query_boosts_temporal_passages(self, mock_pvs):
        driver = _mock_driver()
        client = _mock_openai_client([1.0, 0.0])

        mock_pvs.return_value = [
            {"id": "p1", "text": "Компания основана в 2015 году", "chunk_id": "c1", "score": 0.71},
            {"id": "p2", "text": "Описание продукта и характеристики", "chunk_id": "c2", "score": 0.74},
        ]

        results = temporal_query("когда основана компания", driver, client)
        assert len(results) == 2
        assert results[0].source == "vector"
        # Temporal passage should be boosted above regular
        assert results[0].chunk.content == "Компания основана в 2015 году"

    @patch("agentic_graph_rag.agent.tools._passage_vector_search")
    def test_temporal_query_empty_falls_back(self, mock_pvs):
        driver = _mock_driver()
        client = _mock_openai_client()

        mock_pvs.return_value = []

        with patch("agentic_graph_rag.agent.tools.vector_search") as mock_vs:
            mock_vs.return_value = _make_results(2)
            results = temporal_query("when", driver, client)
            mock_vs.assert_called_once()
            assert len(results) == 2


# ---------------------------------------------------------------------------
# _generate_sub_queries
# ---------------------------------------------------------------------------

class TestGenerateSubQueries:
    def test_generates_sub_queries(self):
        client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "sub query 1\nsub query 2\nsub query 3"
        client.chat.completions.create.return_value = resp

        subs = _generate_sub_queries("list all features", client, "gpt-4o-mini", n=3)
        assert len(subs) == 3
        assert subs[0] == "sub query 1"

    def test_handles_api_error(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("fail")
        subs = _generate_sub_queries("test", client, "gpt-4o-mini")
        assert subs == ["test"]

    def test_handles_empty_response(self):
        client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = ""
        client.chat.completions.create.return_value = resp

        subs = _generate_sub_queries("test", client, "gpt-4o-mini")
        assert subs == ["test"]

    def test_limits_to_n(self):
        client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "q1\nq2\nq3\nq4\nq5"
        client.chat.completions.create.return_value = resp

        subs = _generate_sub_queries("test", client, "gpt-4o-mini", n=2)
        assert len(subs) == 2


class TestComprehensiveSearchFanout:
    @patch("agentic_graph_rag.agent.tools._rrf_merge")
    @patch("agentic_graph_rag.agent.tools._embed_query")
    @patch("agentic_graph_rag.agent.tools.rerank")
    @patch("agentic_graph_rag.agent.tools.full_document_read")
    @patch("agentic_graph_rag.agent.tools.vector_search")
    @patch("agentic_graph_rag.agent.tools._generate_sub_queries")
    @patch("agentic_graph_rag.agent.tools._detect_enumeration_count")
    @patch("agentic_graph_rag.agent.tools.get_settings")
    def test_comprehensive_search_caps_fanout_and_uses_light_hybrid(
        self,
        mock_settings,
        mock_detect_count,
        mock_generate_sub_queries,
        mock_vector_search,
        mock_full_document_read,
        mock_rerank,
        mock_embed_query,
        mock_rrf_merge,
    ):
        cfg = MagicMock()
        cfg.retrieval.top_k_final = 6
        cfg.retrieval.top_k_vector = 6
        cfg.openai.llm_model_mini = "test-mini"
        mock_settings.return_value = cfg

        mock_detect_count.return_value = 9
        mock_generate_sub_queries.return_value = ["q1", "q2", "q3", "q4", "q5", "q6"]
        mock_vector_search.return_value = _make_results(2, source="vector")
        mock_full_document_read.return_value = _make_results(2, source="vector")
        mock_rrf_merge.side_effect = lambda *lists, top_k=5, **_kwargs: lists[0][:top_k]
        mock_rerank.side_effect = lambda _query, results, **_kwargs: results
        mock_embed_query.return_value = [1.0, 0.0]

        results = comprehensive_search("list all COPD indicators", _mock_driver(), _mock_openai_client())

        assert results
        mock_generate_sub_queries.assert_called_once_with(
            "list all COPD indicators",
            ANY,
            "test-mini",
            n=6,
        )
        assert mock_vector_search.call_count == 6
        for call in mock_vector_search.call_args_list:
            assert call.kwargs["top_k"] == 4
        mock_full_document_read.assert_called_once()
        assert mock_full_document_read.call_args.kwargs["top_k"] == 8
        mock_rerank.assert_called_once()


# ---------------------------------------------------------------------------
# comprehensive_search
# ---------------------------------------------------------------------------

class TestComprehensiveSearch:
    @patch("agentic_graph_rag.agent.tools.rerank")
    @patch("agentic_graph_rag.agent.tools.full_document_read")
    @patch("agentic_graph_rag.agent.tools.vector_search")
    @patch("agentic_graph_rag.agent.tools._generate_sub_queries")
    def test_merges_sub_query_results(self, mock_gen, mock_vs, mock_full_read, mock_rerank):
        mock_gen.return_value = ["sub1", "sub2", "sub3"]
        mock_vs.side_effect = [
            _make_results(3, source="v"),  # sub1
            _make_results(3, source="v"),  # sub2
            _make_results(3, source="v"),  # sub3
        ]
        mock_full_read.return_value = _make_results(2, source="vector")
        mock_rerank.side_effect = lambda _query, results, **_kwargs: results

        driver = _mock_driver()
        client = _mock_openai_client()

        results = comprehensive_search("list all features", driver, client, top_k=10)
        assert len(results) > 0
        assert mock_vs.call_count == 3
        mock_full_read.assert_called_once()
        mock_rerank.assert_called_once()

    @patch("agentic_graph_rag.agent.tools.rerank")
    @patch("agentic_graph_rag.agent.tools.full_document_read")
    @patch("agentic_graph_rag.agent.tools.vector_search")
    @patch("agentic_graph_rag.agent.tools._generate_sub_queries")
    def test_falls_back_on_empty_sub_queries(self, mock_gen, mock_vs, mock_full_read, mock_rerank):
        mock_gen.return_value = []
        mock_vs.return_value = _make_results(5, source="v")
        mock_full_read.return_value = []
        mock_rerank.side_effect = lambda _query, results, **_kwargs: results

        driver = _mock_driver()
        client = _mock_openai_client()

        results = comprehensive_search("test", driver, client)
        assert len(results) == 0
        mock_vs.assert_not_called()
