"""Tests for agentic_graph_rag.agent.router."""

from unittest.mock import MagicMock, patch

from rag_core.models import QueryType, RouterDecision

from agentic_graph_rag.agent.router import (
    classify_query,
    classify_query_by_llm,
    classify_query_by_patterns,
)

# ---------------------------------------------------------------------------
# classify_query_by_patterns
# ---------------------------------------------------------------------------

class TestClassifyByPatterns:
    def test_simple_query(self):
        d = classify_query_by_patterns("What is Python?")
        assert d.query_type == QueryType.SIMPLE
        assert d.suggested_tool == "vector_search"

    def test_relation_query_en(self):
        d = classify_query_by_patterns("What is the relationship between A and B?")
        assert d.query_type == QueryType.RELATION
        assert d.suggested_tool == "cypher_traverse"

    def test_multi_hop_query(self):
        d = classify_query_by_patterns("Compare Python and Java through their ecosystems")
        assert d.query_type == QueryType.MULTI_HOP
        assert d.suggested_tool == "cypher_traverse"

    def test_global_query(self):
        d = classify_query_by_patterns("Show all entities in the system")
        assert d.query_type == QueryType.GLOBAL
        assert d.suggested_tool == "comprehensive_search"

    def test_temporal_query(self):
        d = classify_query_by_patterns("When was Python created?")
        assert d.query_type == QueryType.TEMPORAL
        assert d.suggested_tool == "temporal_query"

    def test_temporal_date_pattern(self):
        d = classify_query_by_patterns("What happened in 2024-01?")
        assert d.query_type == QueryType.TEMPORAL

    def test_confidence_increases_with_matches(self):
        d1 = classify_query_by_patterns("relationship")
        d2 = classify_query_by_patterns("relationship between connected entities")
        assert d2.confidence >= d1.confidence

    def test_confidence_capped(self):
        d = classify_query_by_patterns(
            "relationship link connect between related"
        )
        assert d.confidence <= 0.95

    def test_returns_router_decision(self):
        d = classify_query_by_patterns("test query")
        assert isinstance(d, RouterDecision)
        assert d.reasoning != ""

    def test_empty_query(self):
        d = classify_query_by_patterns("")
        assert d.query_type == QueryType.SIMPLE


# ---------------------------------------------------------------------------
# classify_query
# ---------------------------------------------------------------------------

class TestClassifyQueryHardRules:
    @patch("agentic_graph_rag.agent.router.classify_query_by_llm")
    def test_hard_rule_prefers_bm25_for_error_codes(self, mock_llm):
        d = classify_query("ERR-902X on JDK 21", use_llm=True)
        assert d.query_type == QueryType.SIMPLE
        assert d.suggested_tool == "bm25_search"
        assert d.reasoning.startswith("Hard rule:")
        mock_llm.assert_not_called()

    @patch("agentic_graph_rag.agent.router.classify_query_by_llm")
    def test_hard_rule_prefers_graph_for_relation_queries(self, mock_llm):
        d = classify_query("Kafka 和 RabbitMQ 的区别和依赖关系是什么", use_llm=True)
        assert d.query_type == QueryType.RELATION
        assert d.suggested_tool == "cypher_traverse"
        mock_llm.assert_not_called()

    @patch("agentic_graph_rag.agent.router.classify_query_by_llm")
    def test_hard_rule_relation_beats_lexical_anchor(self, mock_llm):
        d = classify_query("GOLD 2级患者如果嗜酸性粒细胞≥300/μL且急性加重≥2次/年，应该使用什么方案？", use_llm=True)
        assert d.query_type == QueryType.RELATION
        assert d.suggested_tool == "cypher_traverse"
        mock_llm.assert_not_called()

    @patch("agentic_graph_rag.agent.router.classify_query_by_llm")
    def test_hard_rule_multihop_beats_lexical_anchor(self, mock_llm):
        d = classify_query("一个GOLD 2级患者，症状中-重，急性加重≥2次/年，嗜酸性粒细胞80/μL，应该用什么方案？为什么不用含ICS的方案？", use_llm=True)
        assert d.query_type in {QueryType.RELATION, QueryType.MULTI_HOP}
        assert d.suggested_tool == "cypher_traverse"
        mock_llm.assert_not_called()

    @patch("agentic_graph_rag.agent.router.classify_query_by_llm")
    def test_hard_rule_prefers_comprehensive_for_chinese_global_query(self, mock_llm):
        d = classify_query("COPD的诊断和分级需要哪些检查指标？这些指标如何使用？", use_llm=True)
        assert d.query_type == QueryType.GLOBAL
        assert d.suggested_tool == "comprehensive_search"
        mock_llm.assert_not_called()

    @patch("agentic_graph_rag.agent.router.classify_query_by_llm")
    def test_hard_rule_prefers_vector_for_short_factual_query(self, mock_llm):
        d = classify_query("What causes BCC?", use_llm=True)
        assert d.query_type == QueryType.SIMPLE
        assert d.suggested_tool == "vector_search"
        mock_llm.assert_not_called()

    @patch("agentic_graph_rag.agent.router.classify_query_by_llm")
    def test_hard_rule_prefers_full_document_for_internal_alias_global_query(self, mock_llm):
        d = classify_query("describe all decisions for SCL", use_llm=True)
        assert d.query_type == QueryType.GLOBAL
        assert d.suggested_tool == "full_document_read"
        mock_llm.assert_not_called()


# ---------------------------------------------------------------------------
# classify_query_by_llm
# ---------------------------------------------------------------------------

class TestClassifyByLLM:
    def test_llm_classification(self):
        client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "relation"
        client.chat.completions.create.return_value = resp

        d = classify_query_by_llm("How are A and B related?", openai_client=client)
        assert d.query_type == QueryType.RELATION
        assert d.confidence == 0.85

    def test_llm_returns_multi_hop(self):
        client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "multi_hop"
        client.chat.completions.create.return_value = resp

        d = classify_query_by_llm("test", openai_client=client)
        assert d.query_type == QueryType.MULTI_HOP

    def test_llm_unknown_type_defaults_simple(self):
        client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "unknown_type"
        client.chat.completions.create.return_value = resp

        d = classify_query_by_llm("test", openai_client=client)
        assert d.query_type == QueryType.SIMPLE

    def test_llm_error_fallback_patterns(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("API error")

        d = classify_query_by_llm("Show all items", openai_client=client)
        # Falls back to patterns → global
        assert d.query_type == QueryType.GLOBAL

    @patch("agentic_graph_rag.agent.router.get_settings")
    def test_creates_client_when_none(self, mock_settings):
        cfg = MagicMock()
        cfg.openai.api_key = "test-key"
        cfg.openai.base_url = ""
        cfg.openai.llm_model_mini = "gpt-4o-mini"
        mock_settings.return_value = cfg

        with patch("agentic_graph_rag.agent.router.make_openai_client") as mock_make:
            mock_client = MagicMock()
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = "simple"
            mock_client.chat.completions.create.return_value = resp
            mock_make.return_value = mock_client

            d = classify_query_by_llm("test query")
            mock_make.assert_called_once_with(cfg)
            assert d.query_type == QueryType.SIMPLE


# ---------------------------------------------------------------------------
# classify_query (main entry)
# ---------------------------------------------------------------------------

class TestClassifyQuery:
    def test_default_uses_patterns(self):
        d = classify_query("What is Python?")
        assert isinstance(d, RouterDecision)
        assert d.query_type == QueryType.SIMPLE

    def test_use_llm_flag(self):
        client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "temporal"
        client.chat.completions.create.return_value = resp

        d = classify_query(
            "Explain the internal memory layout behavior in this subsystem",
            use_llm=True,
            openai_client=client,
        )
        assert d.query_type == QueryType.TEMPORAL
        assert d.confidence == 0.85
