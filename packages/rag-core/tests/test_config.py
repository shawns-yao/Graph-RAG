"""Tests for rag_core.config."""



from pathlib import Path
from unittest.mock import MagicMock

import pytest
from rag_core.config import (
    AgentSettings,
    BenchmarkSettings,
    IndexingSettings,
    IngestSettings,
    Neo4jSettings,
    OpenAISettings,
    RetrievalSettings,
    Settings,
    get_settings,
    make_openai_client,
)


def _clear_openai_env(monkeypatch):
    for key in (
        "LLM_API_KEY",
        "OPENAI_API_KEY",
        "LLM_BASE_URL",
        "OPENAI_BASE_URL",
        "EMBEDDING_API_KEY",
        "EMBEDDING_BASE_URL",
        "EMBEDDING_MODEL",
        "OPENAI_EMBEDDING_MODEL",
        "EMBEDDING_DIMENSIONS",
        "OPENAI_EMBEDDING_DIMENSIONS",
        "LLM_MODEL",
        "OPENAI_LLM_MODEL",
        "LLM_MODEL_MINI",
        "OPENAI_LLM_MODEL_MINI",
        "LLM_TEMPERATURE",
        "OPENAI_LLM_TEMPERATURE",
    ):
        monkeypatch.delenv(key, raising=False)


def _clear_neo4j_env(monkeypatch):
    for key in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "NEO4J_DATABASE"):
        monkeypatch.delenv(key, raising=False)


def _clear_indexing_env(monkeypatch):
    for key in (
        "INDEXING_STRUCTURED_CHUNKING_ENABLED",
        "INDEXING_HIERARCHICAL_CHUNKING_ENABLED",
        "INDEXING_GRAPH_CHUNKING_ENABLED",
        "INDEXING_CHUNK_SIZE",
        "INDEXING_CHUNK_OVERLAP",
        "INDEXING_PARENT_CHUNK_SIZE",
        "INDEXING_CONTEXT_WINDOW_CHARS",
        "INDEXING_GRAPH_SKELETON_CHUNK_SIZE",
        "INDEXING_GRAPH_SKELETON_CHUNK_MAX_SIZE",
        "INDEXING_GRAPH_PERIPHERAL_CHUNK_SIZE",
        "INDEXING_GRAPH_MIN_ENTITIES_PER_CHUNK",
        "INDEXING_GRAPH_SENTENCE_BOUNDARY_ONLY",
        "INDEXING_SKELETON_ENTITY_DENSITY_WEIGHT",
        "INDEXING_SKELETON_BETA",
        "INDEXING_SKELETON_BETA_SHORT_DOC",
        "INDEXING_SKELETON_BETA_MEDIUM_DOC",
        "INDEXING_SKELETON_BETA_LONG_DOC",
        "INDEXING_SKELETON_SHORT_DOC_MAX_CHUNKS",
        "INDEXING_SKELETON_MEDIUM_DOC_MAX_CHUNKS",
        "INDEXING_KNN_K",
        "INDEXING_PAGERANK_DAMPING",
        "INDEXING_PAGERANK_DAMPING_TECHNICAL",
        "INDEXING_PAGERANK_DAMPING_PAPER",
        "INDEXING_TFIDF_LOW_IDF_THRESHOLD",
        "INDEXING_TFIDF_LOW_INFO_CHUNK_SCORE_THRESHOLD",
        "INDEXING_TFIDF_MAX_KEYWORDS",
    ):
        monkeypatch.delenv(key, raising=False)


def _clear_retrieval_env(monkeypatch):
    for key in (
        "RETRIEVAL_TOP_K_VECTOR",
        "RETRIEVAL_TOP_K_BM25",
        "RETRIEVAL_TOP_K_FINAL",
        "RETRIEVAL_PROMPT_MAX_CHUNKS",
        "RETRIEVAL_PROMPT_MAX_CHARS",
        "RETRIEVAL_VECTOR_THRESHOLD",
        "RETRIEVAL_MAX_HOPS",
        "RETRIEVAL_GRAPH_ENTRY_TOP_K",
        "RETRIEVAL_GRAPH_COOCCURRENCE_LIMIT",
        "RETRIEVAL_GRAPH_PASSAGE_LIMIT",
        "RETRIEVAL_PPR_ALPHA",
        "RETRIEVAL_RRF_K",
        "RETRIEVAL_FANOUT_MAX_WORKERS",
        "RETRIEVAL_FANOUT_TIMEOUT_MS",
        "RETRIEVAL_FULLTEXT_INDEX_NAME",
        "RETRIEVAL_RERANKER_BACKEND",
        "RETRIEVAL_RERANKER_MODEL",
        "RETRIEVAL_EMPTY_CHANNEL_PENALTY",
        "RETRIEVAL_SPARSE_CHANNEL_PENALTY",
        "RETRIEVAL_WEAK_CHANNEL_MIN_RESULTS",
        "RETRIEVAL_BM25_LEXICAL_BOOST",
        "RETRIEVAL_GRAPH_EVIDENCE_BOOST",
        "RETRIEVAL_LEXICAL_OVERLAP_THRESHOLD",
        "RETRIEVAL_TFIDF_QUERY_MIN_IDF",
        "RETRIEVAL_TFIDF_QUERY_MAX_KEYWORDS",
        "RETRIEVAL_CONFIDENCE_MIN",
        "RETRIEVAL_RETRIEVAL_CONFIDENCE_WEIGHT",
        "RETRIEVAL_REFLECTION_CONFIDENCE_WEIGHT",
        "RETRIEVAL_REFLECTION_SCORE_SCALE",
    ):
        monkeypatch.delenv(key, raising=False)


class TestNeo4jSettings:
    def test_defaults(self, monkeypatch):
        _clear_neo4j_env(monkeypatch)
        s = Neo4jSettings(_env_file=None)
        assert s.uri == "bolt://localhost:7687"
        assert s.user == "neo4j"
        assert s.password == "neo4j"
        assert s.database == "neo4j"
        assert s.resolved_database == "neo4j"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("NEO4J_URI", "bolt://custom:7688")
        monkeypatch.setenv("NEO4J_DATABASE", "agr_exp_01")
        s = Neo4jSettings(_env_file=None)
        assert s.uri == "bolt://custom:7688"
        assert s.database == "agr_exp_01"
        assert s.resolved_database == "agr_exp_01"


class TestOpenAISettings:
    def test_defaults(self, monkeypatch):
        _clear_openai_env(monkeypatch)
        s = OpenAISettings(_env_file=None)
        assert s.api_key == ""
        assert s.base_url == ""
        assert s.embedding_api_key == ""
        assert s.embedding_base_url == ""
        assert s.embedding_model == "text-embedding-3-small"
        assert s.embedding_dimensions == 1536
        assert s.llm_model == "gpt-4o"
        assert s.llm_model_mini == "gpt-4o-mini"
        assert s.llm_temperature == 0.0

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "sk-test-123")
        monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
        monkeypatch.setenv("LLM_MODEL", "deepseek-chat")
        monkeypatch.setenv("EMBEDDING_API_KEY", "embed-key")
        monkeypatch.setenv("EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-v3")
        s = OpenAISettings(_env_file=None)
        assert s.api_key == "sk-test-123"
        assert s.base_url == "https://api.deepseek.com/v1"
        assert s.llm_model == "deepseek-chat"
        assert s.embedding_api_key == "embed-key"
        assert s.embedding_base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
        assert s.embedding_model == "text-embedding-v3"

    def test_embedding_provider_falls_back_to_llm(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "sk-test-123")
        monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
        monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
        monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)
        s = OpenAISettings(_env_file=None)
        assert s.resolved_embedding_api_key == "sk-test-123"
        assert s.resolved_embedding_base_url == "https://api.deepseek.com/v1"

    def test_legacy_openai_env_still_supported(self, monkeypatch):
        _clear_openai_env(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "legacy-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:4000/v1")
        monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "legacy-embedding")
        s = OpenAISettings(_env_file=None)
        assert s.api_key == "legacy-key"
        assert s.base_url == "http://localhost:4000/v1"
        assert s.embedding_model == "legacy-embedding"

    def test_loads_from_dotenv_in_current_workdir(self, monkeypatch, tmp_path: Path):
        _clear_openai_env(monkeypatch)
        env_path = tmp_path / ".env"
        env_path.write_text(
            "\n".join(
                [
                    "LLM_API_KEY=dotenv-key",
                    "LLM_BASE_URL=https://example.com/v1",
                    "EMBEDDING_MODEL=test-embedding",
                ]
            ),
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        s = OpenAISettings()
        assert s.api_key == "dotenv-key"
        assert s.base_url == "https://example.com/v1"
        assert s.embedding_model == "test-embedding"


class TestIndexingSettings:
    def test_defaults(self, monkeypatch):
        _clear_indexing_env(monkeypatch)
        s = IndexingSettings(_env_file=None)
        assert s.structured_chunking_enabled is True
        assert s.hierarchical_chunking_enabled is True
        assert s.graph_chunking_enabled is True
        assert s.chunk_size == 1000
        assert s.chunk_overlap == 200
        assert s.parent_chunk_size == 4000
        assert s.context_window_chars == 1200
        assert s.graph_skeleton_chunk_size == 3000
        assert s.graph_skeleton_chunk_max_size == 4200
        assert s.graph_peripheral_chunk_size == 1000
        assert s.graph_min_entities_per_chunk == 2
        assert s.graph_sentence_boundary_only is True
        assert s.skeleton_entity_density_weight == 0.35
        assert s.skeleton_beta == 0.25
        assert s.skeleton_beta_short_doc == 1.0
        assert s.skeleton_beta_medium_doc == 0.5
        assert s.skeleton_beta_long_doc == 0.3
        assert s.skeleton_short_doc_max_chunks == 8
        assert s.skeleton_medium_doc_max_chunks == 24
        assert s.knn_k == 5
        assert s.pagerank_damping == 0.85
        assert s.pagerank_damping_technical == 0.8
        assert s.pagerank_damping_paper == 0.9

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("INDEXING_SKELETON_BETA", "0.3")
        monkeypatch.setenv("INDEXING_SKELETON_BETA_SHORT_DOC", "0.9")
        monkeypatch.setenv("INDEXING_PAGERANK_DAMPING_PAPER", "0.92")
        s = IndexingSettings(_env_file=None)
        assert s.skeleton_beta == 0.3
        assert s.skeleton_beta_short_doc == 0.9
        assert s.pagerank_damping_paper == 0.92


class TestRetrievalSettings:
    def test_defaults(self, monkeypatch):
        _clear_retrieval_env(monkeypatch)
        s = RetrievalSettings(_env_file=None)
        assert s.top_k_vector == 10
        assert s.top_k_bm25 == 10
        assert s.top_k_final == 10
        assert s.prompt_max_chunks == 10
        assert s.prompt_max_chars == 18000
        assert s.vector_threshold == 0.5
        assert s.max_hops == 2
        assert s.graph_entry_top_k == 5
        assert s.graph_cooccurrence_limit == 32
        assert s.graph_passage_limit == 12
        assert s.ppr_alpha == 0.15
        assert s.rrf_k == 60
        assert s.fanout_max_workers == 3
        assert s.fanout_timeout_ms == 15000
        assert s.fulltext_index_name == "passage_text_index"
        assert s.reranker_backend == "lexical_semantic"
        assert s.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert s.empty_channel_penalty == 0.35
        assert s.sparse_channel_penalty == 0.75
        assert s.weak_channel_min_results == 2
        assert s.bm25_lexical_boost == 1.2
        assert s.graph_evidence_boost == 1.1
        assert s.lexical_overlap_threshold == 0.5


class TestAgentSettings:
    def test_defaults(self, monkeypatch):
        for key in (
            "AGENT_MAX_RETRIES",
            "AGENT_MAX_RERANKS",
            "AGENT_MAX_QUERY_REWRITES",
            "AGENT_REQUEST_TIME_BUDGET_MS",
            "AGENT_RELEVANCE_THRESHOLD",
        ):
            monkeypatch.delenv(key, raising=False)
        s = AgentSettings(_env_file=None)
        assert s.max_retries == 1
        assert s.max_reranks == 1
        assert s.max_query_rewrites == 0
        assert s.request_time_budget_ms == 1500
        assert s.relevance_threshold == 2.0


class TestBenchmarkSettings:
    def test_defaults(self):
        s = BenchmarkSettings(_env_file=None)
        assert s.llm_max_retries == 2
        assert s.llm_connect_timeout_seconds == 5.0
        assert s.llm_read_timeout_seconds == 30.0
        assert s.llm_write_timeout_seconds == 30.0
        assert s.llm_pool_timeout_seconds == 30.0
        assert s.llm_initial_backoff_seconds == 2.0
        assert s.llm_max_backoff_seconds == 12.0
        assert s.llm_jitter_seconds == 0.25
        assert s.llm_max_parallel_requests == 1
        assert s.llm_min_request_interval_ms == 750


class TestSettings:
    def test_nested_settings(self):
        s = Settings()
        assert isinstance(s.neo4j, Neo4jSettings)
        assert isinstance(s.openai, OpenAISettings)
        assert isinstance(s.indexing, IndexingSettings)
        assert isinstance(s.retrieval, RetrievalSettings)
        assert isinstance(s.agent, AgentSettings)
        assert isinstance(s.ingest, IngestSettings)
        assert isinstance(s.benchmark, BenchmarkSettings)
        assert s.log_level == "INFO"

    def test_get_settings_returns_instance(self):
        s = get_settings()
        assert isinstance(s, Settings)


class TestMakeOpenaiClient:
    def test_raises_when_no_key_and_no_base_url(self):
        cfg = Settings()
        cfg.openai = OpenAISettings(api_key="", base_url="")
        with pytest.raises(ValueError, match="LLM_API_KEY or LLM_BASE_URL"):
            make_openai_client(cfg)

    def test_works_with_api_key(self):
        cfg = Settings()
        cfg.openai = OpenAISettings(api_key="sk-test-key", base_url="")
        client = make_openai_client(cfg)
        assert client is not None

    def test_works_with_base_url_only(self):
        cfg = Settings()
        cfg.openai = OpenAISettings(api_key="", base_url="http://localhost:4000/v1")
        client = make_openai_client(cfg)
        assert client is not None

    def test_uses_separate_embedding_provider_when_configured(self):
        cfg = Settings()
        cfg.openai = OpenAISettings(
            api_key="llm-key",
            base_url="https://api.deepseek.com/v1",
            embedding_api_key="embed-key",
            embedding_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        client = make_openai_client(cfg)
        assert client is not None

    def test_ingest_profile_uses_shorter_timeouts_and_sdk_retries_disabled(self, monkeypatch):
        cfg = Settings()
        cfg.openai = OpenAISettings(
            api_key="llm-key",
            base_url="https://proxy.example/v1",
            embedding_api_key="embed-key",
            embedding_base_url="https://embed.example/v1",
        )
        cfg.ingest = IngestSettings(
            llm_max_retries=5,
            llm_connect_timeout_seconds=5.0,
            llm_read_timeout_seconds=45.0,
            llm_write_timeout_seconds=45.0,
            llm_pool_timeout_seconds=45.0,
        )

        build_calls = []

        def fake_build_openai_client(**kwargs):
            build_calls.append(kwargs)
            client = MagicMock()
            client.chat = MagicMock()
            client.embeddings = MagicMock()
            return client

        monkeypatch.setattr("rag_core.config._build_openai_client", fake_build_openai_client)
        make_openai_client(cfg, profile="ingest")

        assert len(build_calls) == 2
        assert build_calls[0]["max_retries"] == 0
        assert build_calls[1]["max_retries"] == 0
        timeout = build_calls[0]["timeout"]
        assert timeout.connect == 5.0
        assert timeout.read == 45.0
        assert timeout.write == 45.0
        assert timeout.pool == 45.0

    def test_benchmark_profile_uses_benchmark_timeouts_and_sdk_retries_disabled(self, monkeypatch):
        cfg = Settings()
        cfg.openai = OpenAISettings(
            api_key="llm-key",
            base_url="https://proxy.example/v1",
            embedding_api_key="embed-key",
            embedding_base_url="https://embed.example/v1",
        )
        cfg.benchmark = BenchmarkSettings(
            llm_max_retries=1,
            llm_connect_timeout_seconds=4.0,
            llm_read_timeout_seconds=25.0,
            llm_write_timeout_seconds=26.0,
            llm_pool_timeout_seconds=27.0,
        )

        build_calls = []

        def fake_build_openai_client(**kwargs):
            build_calls.append(kwargs)
            client = MagicMock()
            client.chat = MagicMock()
            client.embeddings = MagicMock()
            return client

        monkeypatch.setattr("rag_core.config._build_openai_client", fake_build_openai_client)
        make_openai_client(cfg, profile="benchmark")

        assert len(build_calls) == 2
        assert build_calls[0]["max_retries"] == 0
        assert build_calls[1]["max_retries"] == 0
        timeout = build_calls[0]["timeout"]
        assert timeout.connect == 4.0
        assert timeout.read == 25.0
        assert timeout.write == 26.0
        assert timeout.pool == 27.0
