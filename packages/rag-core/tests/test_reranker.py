"""Tests for cross-encoder-only reranking."""

from unittest.mock import MagicMock, patch

from rag_core.models import Chunk, SearchResult
from rag_core.reranker import rerank, rerank_cross_encoder


def _make_result(content: str, score: float = 0.5, rank: int = 1) -> SearchResult:
    return SearchResult(chunk=Chunk(content=content), score=score, rank=rank)


class TestRerank:
    @patch("rag_core.reranker.get_settings")
    @patch("rag_core.reranker.rerank_cross_encoder")
    def test_uses_cross_encoder_for_text_query(self, mock_cross_encoder, mock_settings):
        cfg = MagicMock()
        cfg.retrieval.top_k_final = 2
        cfg.retrieval.reranker_model = "test-model"
        mock_settings.return_value = cfg
        mock_cross_encoder.return_value = [_make_result("doc1", rank=1)]

        results = [_make_result("doc1"), _make_result("doc2"), _make_result("doc3")]
        reranked = rerank("test query", results)

        assert len(reranked) == 1
        mock_cross_encoder.assert_called_once_with(
            "test query",
            results,
            top_k=2,
            model_name="test-model",
        )

    @patch("rag_core.reranker.get_settings")
    def test_explicit_top_k_overrides_settings(self, mock_settings):
        cfg = MagicMock()
        cfg.retrieval.top_k_final = 10
        cfg.retrieval.reranker_model = "test-model"
        mock_settings.return_value = cfg

        results = [_make_result("doc1"), _make_result("doc2")]
        with patch("rag_core.reranker.rerank_cross_encoder", return_value=results[:1]) as mock_cross_encoder:
            rerank("test query", results, top_k=1)

        mock_cross_encoder.assert_called_once_with(
            "test query",
            results,
            top_k=1,
            model_name="test-model",
        )

    @patch("rag_core.reranker.get_settings")
    def test_embedding_only_legacy_input_returns_original_top_k(self, mock_settings):
        cfg = MagicMock()
        cfg.retrieval.top_k_final = 1
        cfg.retrieval.reranker_model = "test-model"
        mock_settings.return_value = cfg

        results = [_make_result("doc1"), _make_result("doc2")]
        reranked = rerank([1.0, 0.0], results)

        assert [item.chunk.content for item in reranked] == ["doc1"]


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
        assert reranked[0].score == 0.9
        assert reranked[0].rank == 1
        assert reranked[0].source == "bm25"

    @patch("rag_core.reranker._load_cross_encoder", return_value=None)
    def test_returns_original_top_k_when_model_unavailable(self, _mock_loader):
        r1 = SearchResult(chunk=Chunk(content="doc1"), score=0.1, rank=1)
        r2 = SearchResult(chunk=Chunk(content="doc2"), score=0.2, rank=2)

        reranked = rerank_cross_encoder("query", [r1, r2], top_k=1)

        assert len(reranked) == 1
        assert reranked[0].chunk.content == "doc1"
