"""Tests for rag_core.reranker."""

from unittest.mock import MagicMock, patch

from rag_core.models import Chunk, SearchResult
from rag_core.reranker import (
    rerank,
    rerank_cosine,
    rerank_cross_encoder,
    rerank_lexical_semantic,
)


def _make_result(embedding: list[float], score: float = 0.5, rank: int = 1) -> SearchResult:
    chunk = Chunk(content="test", embedding=embedding)
    return SearchResult(chunk=chunk, score=score, rank=rank)


class TestRerankCosine:
    def test_empty_results(self):
        assert rerank_cosine([1.0, 0.0], [], top_k=5) == []

    def test_no_embeddings(self):
        chunk = Chunk(content="no emb")
        results = [SearchResult(chunk=chunk, score=0.5, rank=1)]
        reranked = rerank_cosine([1.0, 0.0], results, top_k=5)
        assert len(reranked) == 1
        assert reranked[0].chunk.content == "no emb"

    def test_ranks_by_similarity(self):
        # r1 is more similar to query [1, 0] than r2
        r1 = _make_result([1.0, 0.0], rank=2)
        r2 = _make_result([0.0, 1.0], rank=1)

        reranked = rerank_cosine([1.0, 0.0], [r2, r1], top_k=2)
        assert len(reranked) == 2
        assert reranked[0].score > reranked[1].score
        assert reranked[0].rank == 1
        assert reranked[1].rank == 2

    def test_respects_top_k(self):
        results = [_make_result([1.0, 0.0]) for _ in range(5)]
        reranked = rerank_cosine([1.0, 0.0], results, top_k=2)
        assert len(reranked) == 2

    def test_zero_norm_query(self):
        results = [_make_result([1.0, 0.0])]
        reranked = rerank_cosine([0.0, 0.0], results, top_k=5)
        assert len(reranked) == 1

    def test_zero_norm_chunk(self):
        r = _make_result([0.0, 0.0])
        reranked = rerank_cosine([1.0, 0.0], [r], top_k=5)
        assert len(reranked) == 1
        assert reranked[0].score == 0.0

    def test_cosine_similarity_values(self):
        # Identical vectors → similarity ≈ 1.0
        r = _make_result([1.0, 0.0])
        reranked = rerank_cosine([1.0, 0.0], [r], top_k=1)
        assert abs(reranked[0].score - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        r = _make_result([0.0, 1.0])
        reranked = rerank_cosine([1.0, 0.0], [r], top_k=1)
        assert abs(reranked[0].score) < 1e-6

    def test_mixed_embeddings(self):
        """Results with and without embeddings — only embedded ones get reranked."""
        r1 = _make_result([1.0, 0.0], rank=1)
        r2 = SearchResult(chunk=Chunk(content="no emb"), score=0.9, rank=2)

        reranked = rerank_cosine([1.0, 0.0], [r1, r2], top_k=5)
        # Only r1 has embedding, r2 filtered out
        assert len(reranked) == 1


class TestRerank:
    @patch("rag_core.reranker.get_settings")
    def test_uses_settings_top_k(self, mock_settings):
        cfg = MagicMock()
        cfg.retrieval.top_k_final = 2
        cfg.retrieval.reranker_backend = "cosine"
        mock_settings.return_value = cfg

        results = [_make_result([1.0, 0.0]) for _ in range(5)]
        reranked = rerank([1.0, 0.0], results)
        assert len(reranked) == 2

    def test_explicit_top_k_overrides(self):
        results = [_make_result([1.0, 0.0]) for _ in range(5)]
        reranked = rerank([1.0, 0.0], results, top_k=3)
        assert len(reranked) == 3

    @patch("rag_core.reranker.get_settings")
    @patch("rag_core.reranker._load_cross_encoder")
    @patch("rag_core.reranker.rerank_cross_encoder")
    def test_uses_cross_encoder_for_text_query(
        self,
        mock_cross_encoder,
        mock_loader,
        mock_settings,
    ):
        cfg = MagicMock()
        cfg.retrieval.top_k_final = 2
        cfg.retrieval.reranker_backend = "cross_encoder"
        cfg.retrieval.reranker_model = "test-model"
        mock_settings.return_value = cfg
        mock_loader.return_value = MagicMock()
        mock_cross_encoder.return_value = [_make_result([1.0, 0.0], rank=1)]

        results = [_make_result([1.0, 0.0]) for _ in range(3)]
        reranked = rerank("test query", results, query_embedding=[1.0, 0.0])

        assert len(reranked) == 1
        mock_cross_encoder.assert_called_once()

    @patch("rag_core.reranker.get_settings")
    @patch("rag_core.reranker._load_cross_encoder", return_value=None)
    def test_falls_back_to_cosine_when_cross_encoder_returns_empty(
        self,
        _mock_loader,
        mock_settings,
    ):
        cfg = MagicMock()
        cfg.retrieval.top_k_final = 2
        cfg.retrieval.reranker_backend = "cross_encoder"
        cfg.retrieval.reranker_model = "test-model"
        mock_settings.return_value = cfg

        results = [_make_result([1.0, 0.0]) for _ in range(3)]
        reranked = rerank("test query", results, query_embedding=[1.0, 0.0])

        assert len(reranked) == 2

    @patch("rag_core.reranker.get_settings")
    @patch("rag_core.reranker._load_cross_encoder", return_value=None)
    def test_falls_back_to_lexical_semantic_when_cross_encoder_unavailable(
        self,
        _mock_loader,
        mock_settings,
    ):
        cfg = MagicMock()
        cfg.retrieval.top_k_final = 2
        cfg.retrieval.reranker_backend = "cross_encoder"
        cfg.retrieval.reranker_model = "test-model"
        mock_settings.return_value = cfg

        r1 = SearchResult(
            chunk=Chunk(content="Samuel Pepys met a lady and dined with her.", embedding=[1.0, 0.0]),
            score=0.4,
            rank=2,
        )
        r2 = SearchResult(
            chunk=Chunk(content="Samuel Pepys diary introduction.", embedding=[1.0, 0.0]),
            score=0.9,
            rank=1,
        )

        reranked = rerank(
            "Who was the lady that dined with Samuel Pepys?",
            [r2, r1],
            query_embedding=[1.0, 0.0],
        )

        assert reranked[0].chunk.content == "Samuel Pepys met a lady and dined with her."


class TestRerankLexicalSemantic:
    def test_prefers_lexical_anchor_hits_when_embeddings_tie(self):
        r1 = SearchResult(
            chunk=Chunk(content="Samuel Pepys met a lady and dined with her.", embedding=[1.0, 0.0]),
            score=0.4,
            rank=2,
        )
        r2 = SearchResult(
            chunk=Chunk(content="Samuel Pepys diary introduction.", embedding=[1.0, 0.0]),
            score=0.9,
            rank=1,
        )

        reranked = rerank_lexical_semantic(
            "Who was the lady that dined with Samuel Pepys?",
            [1.0, 0.0],
            [r2, r1],
            top_k=2,
        )

        assert reranked[0].chunk.content == "Samuel Pepys met a lady and dined with her."


class TestRerankCrossEncoder:
    @patch("rag_core.reranker._load_cross_encoder")
    @patch("rag_core.reranker.get_settings")
    def test_reranks_by_cross_encoder_scores(self, mock_settings, mock_loader):
        cfg = MagicMock()
        cfg.retrieval.reranker_model = "test-model"
        mock_settings.return_value = cfg

        encoder = MagicMock()
        encoder.predict.return_value = [0.2, 0.9]
        mock_loader.return_value = encoder

        r1 = SearchResult(chunk=Chunk(content="doc1"), score=0.1, rank=1, source="vector")
        r2 = SearchResult(chunk=Chunk(content="doc2"), score=0.1, rank=2, source="bm25")

        reranked = rerank_cross_encoder("query", [r1, r2], top_k=2)

        assert [item.chunk.content for item in reranked] == ["doc2", "doc1"]
        assert reranked[0].source == "bm25"

    @patch("rag_core.reranker._load_cross_encoder", return_value=None)
    def test_returns_original_top_k_when_model_unavailable(self, _mock_loader):
        r1 = SearchResult(chunk=Chunk(content="doc1"), score=0.1, rank=1)
        r2 = SearchResult(chunk=Chunk(content="doc2"), score=0.2, rank=2)

        reranked = rerank_cross_encoder("query", [r1, r2], top_k=1)

        assert len(reranked) == 1
        assert reranked[0].chunk.content == "doc1"
