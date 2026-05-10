"""Agentic Graph RAG configuration via Pydantic Settings."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from httpx import Timeout
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings

_ENV_FILE_CONFIG = {
    "env_file": ".env",
    "env_file_encoding": "utf-8",
    "extra": "ignore",
}


class Neo4jSettings(BaseSettings):
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "neo4j"
    database: str = "neo4j"

    model_config = {**_ENV_FILE_CONFIG, "env_prefix": "NEO4J_"}

    @property
    def resolved_database(self) -> str | None:
        """Return the configured database name, or None to use server default."""
        value = self.database.strip()
        return value or None


class OpenAISettings(BaseSettings):
    api_key: str = Field(
        default="",
        validation_alias=AliasChoices("LLM_API_KEY", "OPENAI_API_KEY"),
    )
    base_url: str = Field(
        default="",
        validation_alias=AliasChoices("LLM_BASE_URL", "OPENAI_BASE_URL"),
    )
    embedding_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("EMBEDDING_API_KEY"),
    )
    embedding_base_url: str = Field(
        default="",
        validation_alias=AliasChoices("EMBEDDING_BASE_URL"),
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        validation_alias=AliasChoices("EMBEDDING_MODEL", "OPENAI_EMBEDDING_MODEL"),
    )
    embedding_dimensions: int = Field(
        default=1536,
        validation_alias=AliasChoices(
            "EMBEDDING_DIMENSIONS",
            "OPENAI_EMBEDDING_DIMENSIONS",
        ),
    )
    llm_model: str = Field(
        default="gpt-4o",
        validation_alias=AliasChoices("LLM_MODEL", "OPENAI_LLM_MODEL"),
    )
    llm_model_mini: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("LLM_MODEL_MINI", "OPENAI_LLM_MODEL_MINI"),
    )
    llm_temperature: float = Field(
        default=0.0,
        validation_alias=AliasChoices("LLM_TEMPERATURE", "OPENAI_LLM_TEMPERATURE"),
    )

    model_config = {
        **_ENV_FILE_CONFIG,
        "populate_by_name": True,
    }

    @property
    def resolved_embedding_api_key(self) -> str:
        """Use embedding-specific key when configured, otherwise fall back to LLM."""
        return self.embedding_api_key or self.api_key

    @property
    def resolved_embedding_base_url(self) -> str:
        """Use embedding-specific endpoint when configured, otherwise fall back to LLM."""
        return self.embedding_base_url or self.base_url


class IndexingSettings(BaseSettings):
    structured_chunking_enabled: bool = True
    hierarchical_chunking_enabled: bool = True
    graph_chunking_enabled: bool = True
    chunk_size: int = 1000
    chunk_overlap: int = 200
    parent_chunk_size: int = 4000
    context_window_chars: int = 1200
    graph_skeleton_chunk_size: int = 3000
    graph_skeleton_chunk_max_size: int = 4200
    graph_peripheral_chunk_size: int = 1000
    graph_min_entities_per_chunk: int = 2
    graph_sentence_boundary_only: bool = True
    skeleton_entity_density_weight: float = 0.35
    skeleton_beta: float = 0.25
    skeleton_beta_short_doc: float = 1.0
    skeleton_beta_medium_doc: float = 0.5
    skeleton_beta_long_doc: float = 0.3
    skeleton_short_doc_max_chunks: int = 8
    skeleton_medium_doc_max_chunks: int = 24
    knn_k: int = 5
    pagerank_damping: float = 0.85
    pagerank_damping_technical: float = 0.8
    pagerank_damping_paper: float = 0.9
    tfidf_low_idf_threshold: float = 1.2
    tfidf_low_info_chunk_score_threshold: float = 0.6
    tfidf_max_keywords: int = 8

    model_config = {**_ENV_FILE_CONFIG, "env_prefix": "INDEXING_"}


class RetrievalSettings(BaseSettings):
    top_k_vector: int = 10
    top_k_bm25: int = 10
    top_k_final: int = 10
    prompt_max_chunks: int = 10
    prompt_max_chars: int = 18000
    vector_threshold: float = 0.5
    max_hops: int = 2
    graph_entry_top_k: int = 5
    graph_cooccurrence_limit: int = 32
    graph_passage_limit: int = 12
    ppr_alpha: float = 0.15
    rrf_k: int = 60
    fanout_max_workers: int = 3
    fanout_timeout_ms: int = 15000
    fulltext_index_name: str = "passage_text_index"
    reranker_backend: str = "lexical_semantic"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    empty_channel_penalty: float = 0.35
    sparse_channel_penalty: float = 0.75
    weak_channel_min_results: int = 2
    bm25_lexical_boost: float = 1.2
    graph_evidence_boost: float = 1.1
    lexical_overlap_threshold: float = 0.5
    tfidf_query_min_idf: float = 1.2
    tfidf_query_max_keywords: int = 6
    confidence_min: float = 0.1

    model_config = {**_ENV_FILE_CONFIG, "env_prefix": "RETRIEVAL_"}


class AgentSettings(BaseSettings):
    max_retries: int = 1
    max_reranks: int = 1
    max_query_rewrites: int = 0
    request_time_budget_ms: int = 1500
    relevance_threshold: float = 2.0
    reflection_skip_score_threshold: float = 0.85

    model_config = {**_ENV_FILE_CONFIG, "env_prefix": "AGENT_"}


class BenchmarkSettings(BaseSettings):
    llm_max_retries: int = 2
    llm_connect_timeout_seconds: float = 5.0
    llm_read_timeout_seconds: float = 30.0
    llm_write_timeout_seconds: float = 30.0
    llm_pool_timeout_seconds: float = 30.0
    llm_initial_backoff_seconds: float = 2.0
    llm_max_backoff_seconds: float = 12.0
    llm_jitter_seconds: float = 0.25
    llm_max_parallel_requests: int = 1
    llm_min_request_interval_ms: int = 750

    model_config = {**_ENV_FILE_CONFIG, "env_prefix": "BENCHMARK_"}


class IngestSettings(BaseSettings):
    llm_max_retries: int = 5
    llm_connect_timeout_seconds: float = 5.0
    llm_read_timeout_seconds: float = 45.0
    llm_write_timeout_seconds: float = 45.0
    llm_pool_timeout_seconds: float = 45.0
    llm_initial_backoff_seconds: float = 2.0
    llm_max_backoff_seconds: float = 8.0
    llm_jitter_seconds: float = 0.25
    llm_max_consecutive_failures: int = 3
    llm_total_budget_seconds: float = 600.0
    embedding_batch_size: int = 64

    model_config = {**_ENV_FILE_CONFIG, "env_prefix": "INGEST_"}


class Settings(BaseSettings):
    neo4j: Neo4jSettings = Neo4jSettings()
    openai: OpenAISettings = OpenAISettings()
    indexing: IndexingSettings = IndexingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    agent: AgentSettings = AgentSettings()
    benchmark: BenchmarkSettings = BenchmarkSettings()
    ingest: IngestSettings = IngestSettings()

    log_level: str = "INFO"

    model_config = _ENV_FILE_CONFIG


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Create and cache settings instance loading from environment."""
    return Settings()


class OpenAIClientBundle:
    """Expose chat and embeddings from different OpenAI-compatible providers."""

    def __init__(self, llm_client: Any, embedding_client: Any) -> None:
        self._llm_client = llm_client
        self._embedding_client = embedding_client
        self.chat = llm_client.chat
        self.embeddings = embedding_client.embeddings

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm_client, name)


def _build_openai_client(
    *,
    api_key: str,
    base_url: str,
    missing_error: str,
    timeout: Timeout | None = None,
    max_retries: int | None = None,
):
    """Create an OpenAI-compatible client for one provider endpoint."""
    from openai import OpenAI

    if not api_key and not base_url:
        raise ValueError(missing_error)

    kwargs: dict[str, Any] = {}
    if api_key:
        kwargs["api_key"] = api_key
    elif base_url:
        kwargs["api_key"] = "none"
    if base_url:
        kwargs["base_url"] = base_url
    if timeout is not None:
        kwargs["timeout"] = timeout
    if max_retries is not None:
        kwargs["max_retries"] = max(0, int(max_retries))
    return OpenAI(**kwargs)


def _profile_timeout(cfg: Settings, profile: str) -> Timeout | None:
    if profile == "ingest":
        return Timeout(
            connect=max(0.0, float(cfg.ingest.llm_connect_timeout_seconds)),
            read=max(0.0, float(cfg.ingest.llm_read_timeout_seconds)),
            write=max(0.0, float(cfg.ingest.llm_write_timeout_seconds)),
            pool=max(0.0, float(cfg.ingest.llm_pool_timeout_seconds)),
        )
    if profile == "benchmark":
        return Timeout(
            connect=max(0.0, float(cfg.benchmark.llm_connect_timeout_seconds)),
            read=max(0.0, float(cfg.benchmark.llm_read_timeout_seconds)),
            write=max(0.0, float(cfg.benchmark.llm_write_timeout_seconds)),
            pool=max(0.0, float(cfg.benchmark.llm_pool_timeout_seconds)),
        )
    return None


def _profile_max_retries(cfg: Settings, profile: str) -> int | None:
    # Disable the OpenAI SDK's built-in retries so all retry behavior flows
    # through our explicit resilience controller with predictable backoff.
    if profile in {"ingest", "benchmark"}:
        return 0
    return 0


def make_openai_client(settings: Settings | None = None, *, profile: str = "default"):
    """Create OpenAI-compatible clients for LLM and embeddings.

    `LLM_*` controls chat/completions.
    `EMBEDDING_*` can point to a separate provider; if omitted it falls back to `LLM_*`.
    Legacy `OPENAI_*` names are still accepted for compatibility.
    """
    cfg = settings or get_settings()
    timeout = _profile_timeout(cfg, profile)
    max_retries = _profile_max_retries(cfg, profile)
    llm_client = _build_openai_client(
        api_key=cfg.openai.api_key,
        base_url=cfg.openai.base_url,
        missing_error=(
            "LLM_API_KEY or LLM_BASE_URL must be set. "
            "Legacy OPENAI_API_KEY / OPENAI_BASE_URL are also supported."
        ),
        timeout=timeout,
        max_retries=max_retries,
    )
    embedding_client = _build_openai_client(
        api_key=cfg.openai.resolved_embedding_api_key,
        base_url=cfg.openai.resolved_embedding_base_url,
        missing_error=(
            "EMBEDDING_API_KEY / EMBEDDING_BASE_URL (or LLM_* fallback) must be set. "
            "Legacy OPENAI_* variables are also supported."
        ),
        timeout=timeout,
        max_retries=max_retries,
    )
    return OpenAIClientBundle(llm_client, embedding_client)
