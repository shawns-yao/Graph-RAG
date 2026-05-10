"""Unified data models for Agentic Graph RAG."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field

# ---------------------------------------------------------------------------
# Ingestion models (from RAG 2.0)
# ---------------------------------------------------------------------------

class DocumentBlock(BaseModel):
    """Structured document block preserved from parsed source content."""

    block_type: str
    text: str
    heading_path: list[str] = Field(default_factory=list)
    order_index: int = 0
    page: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A text chunk with optional contextual enrichment and embedding."""

    id: str = ""
    content: str
    context: str = ""
    embedding: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def enriched_content(self) -> str:
        if self.context:
            return f"{self.context}\n\n{self.content}"
        return self.content


# ---------------------------------------------------------------------------
# Knowledge Graph models (from TKB)
# ---------------------------------------------------------------------------

class Entity(BaseModel):
    """An entity extracted from text.

    `entity_confidence` is the LLM's self-reported confidence when extracting
    the entity. Known to be poorly calibrated — used here only as a filter
    threshold (<0.7 entities skip phrase-link building) to drop obvious junk.
    Not a probability; do not aggregate arithmetically.
    """

    id: str = ""
    name: str
    entity_type: str = ""
    description: str = ""
    entity_confidence: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class Relationship(BaseModel):
    """A relationship between two entities."""

    id: str = ""
    source: str
    target: str
    relation_type: str
    description: str = ""
    weight: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class TemporalEvent(BaseModel):
    """A temporal event from the knowledge graph."""

    id: str = ""
    content: str
    valid_from: str = ""
    valid_to: str = ""
    entity_type: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Graph RAG models (NEW — KET-RAG / HippoRAG 2)
# ---------------------------------------------------------------------------

class PhraseNode(BaseModel):
    """Entity-level node for graph navigation (HippoRAG 2).

    `pagerank_score` comes from NetworkX PageRank over the KNN graph and has
    rigorous mathematical meaning. `confidence` inherits from Entity's
    extraction-time LLM self-report and is treated as a quality filter, not
    a probability.
    """

    id: str = ""
    name: str
    entity_type: str = ""
    pagerank_score: float = 0.0
    confidence: float = 0.0
    passage_ids: list[str] = Field(default_factory=list)


class PassageNode(BaseModel):
    """Full-text passage node for context preservation (HippoRAG 2)."""

    id: str = ""
    text: str
    chunk_id: str = ""
    embedding: list[float] = Field(default_factory=list)
    phrase_ids: list[str] = Field(default_factory=list)


class GraphContext(BaseModel):
    """Assembled context from graph traversal."""

    triplets: list[dict[str, str]] = Field(default_factory=list)
    passages: list[str] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Router + retrieval models (NEW)
# ---------------------------------------------------------------------------

class QueryType(str, Enum):
    """Query complexity categories for the agentic router."""

    SIMPLE = "simple"
    RELATION = "relation"
    MULTI_HOP = "multi_hop"
    GLOBAL = "global"
    TEMPORAL = "temporal"


class RouterDecision(BaseModel):
    """Output of the query router.

    Note on `confidence`: this is a heuristic rule-match strength (0-1), not a
    calibrated probability. It is used only for logging / trace display and
    does not drive any retrieval decisions. Hard-rule matches report 0.84-0.99
    based on how specific the matched pattern is.
    """

    query_type: QueryType
    confidence: float = 0.0
    reasoning: str = ""
    suggested_tool: str = ""


class SearchResult(BaseModel):
    """A single search result from vector store or graph."""

    chunk: Chunk
    score: float = 0.0
    score_normalized: float | None = None
    rank: int = 0
    source: str = "vector"  # "vector", "graph", "hybrid"


# ---------------------------------------------------------------------------
# Provenance models (v6 — pipeline trace)
# ---------------------------------------------------------------------------

class ProviderDiagnostic(BaseModel):
    """Provider-level retrieval diagnostics for a tool execution."""

    source: str
    results_count: int = 0
    top_score: float = 0.0
    average_score: float = 0.0
    reused: bool = False
    executed: bool = False
    top_chunk_ids: list[str] = Field(default_factory=list)


class ToolStep(BaseModel):
    """One tool execution step in the pipeline."""

    tool_name: str
    results_count: int = 0
    relevance_score: float = 0.0
    duration_ms: int = 0
    query_used: str = ""
    cache_hit: bool = False
    reused_sources: list[str] = Field(default_factory=list)
    executed_sources: list[str] = Field(default_factory=list)
    provider_diagnostics: list[ProviderDiagnostic] = Field(default_factory=list)


class ReflectionStep(BaseModel):
    """Structured reflection result for one retrieval attempt.

    Note: Reflection is a policy classifier (answer/retry/stop), not a numeric
    scorer. Decisions are driven by `verdict`, `evidence_status`, `action`, and
    `failure_type` enums — not by continuous scores.
    """

    attempt: int = 0
    tool_name: str = ""
    query_used: str = ""
    evidence_status: str = ""
    gap_type: str = ""
    action: str = ""
    required_tool: str = ""
    verdict: str = ""
    missing_information: list[str] = Field(default_factory=list)
    missing_entities: list[str] = Field(default_factory=list)
    missing_relationships: list[str] = Field(default_factory=list)
    coverage_gap_sources: list[str] = Field(default_factory=list)
    candidate_fix_paths: list[str] = Field(default_factory=list)
    preferred_tools: list[str] = Field(default_factory=list)
    preferred_providers: list[str] = Field(default_factory=list)
    retry_scope: str = ""
    reasoning: str = ""
    failure_type: str = ""
    recommended_action: str = ""
    should_retry: bool = True
    should_rewrite_query: bool = False
    should_rerank_again: bool = False
    comparison_to_previous: str = ""


class WorkflowMemoryEntry(BaseModel):
    """Structured workflow memory captured across routing, retrieval, and retries."""

    stage: str
    message: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class EscalationStep(BaseModel):
    """Tool-to-tool escalation record."""

    from_tool: str
    to_tool: str
    reason: str = ""
    rephrased_query: str = ""
    duration_ms: int = 0
    cached_sources_reused: list[str] = Field(default_factory=list)


class RouterStep(BaseModel):
    """Router classification result with timing."""

    method: str  # "pattern", "llm", "mangle"
    decision: RouterDecision
    duration_ms: int = 0
    rules_fired: list[str] = Field(default_factory=list)


ConfidenceLevel = Literal["high", "medium", "low"]


class GeneratorStep(BaseModel):
    """Answer generation metadata.

    `evidence_score` is the retrieval-layer heuristic (avg score_normalized).
    `confidence_level` is the end-to-end confidence classification derived from
    evidence_score + reflection verdict + answer guard status. These two
    signals are intentionally separate: evidence_score measures retrieval
    quality, confidence_level measures whether the answer can be trusted.
    """

    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    evidence_score: float = 0.0
    confidence_level: ConfidenceLevel = "medium"
    completeness_check: bool | None = None
    duration_ms: int = 0


class VerifiedClaim(BaseModel):
    """One factual claim checked against the knowledge graph."""

    text: str
    key_terms: list[str] = Field(default_factory=list)
    supported: bool
    top_chunk_id: str = ""


class ClaimVerificationStep(BaseModel):
    """Chain-of-Verification style post-generation verification.

    Records the outcome of extracting factual claims from the answer and
    checking each one against graph retrieval evidence. Failed verification
    does not trigger retry — it only attaches a caveat and may downgrade
    the confidence level.
    """

    claims_total: int = 0
    claims_supported: int = 0
    verified_claims: list[VerifiedClaim] = Field(default_factory=list)
    unsupported_claims: list[VerifiedClaim] = Field(default_factory=list)
    skipped_reason: str = ""
    duration_ms: int = 0

    @property
    def support_rate(self) -> float:
        """Fraction of claims that found supporting graph evidence."""
        if self.claims_total == 0:
            return 0.0
        return self.claims_supported / self.claims_total


class PipelineTrace(BaseModel):
    """Full pipeline provenance artifact."""

    trace_id: str
    timestamp: str
    query: str
    expanded_query: str = ""
    final_answer: str = ""
    session_id: str = ""
    router_step: RouterStep | None = None
    tool_steps: list[ToolStep] = Field(default_factory=list)
    reflection_steps: list[ReflectionStep] = Field(default_factory=list)
    workflow_memory: list[WorkflowMemoryEntry] = Field(default_factory=list)
    escalation_steps: list[EscalationStep] = Field(default_factory=list)
    generator_step: GeneratorStep | None = None
    verification_step: ClaimVerificationStep | None = None
    total_duration_ms: int = 0


class QAResult(BaseModel):
    """Final Q&A result with answer, sources, and dual confidence signals.

    - `evidence_score`: retrieval evidence strength (0-1), derived from the
      retrieval layer's score_normalized heuristics. This only measures
      whether the retrieved chunks match the query well — it does NOT
      measure whether the final answer is correct.
    - `confidence_level`: end-to-end answer trust level ("high"/"medium"/"low"),
      derived from evidence_score + reflection verdict + answer guard status.
      This is the user-facing trust signal.
    """

    answer: str
    sources: list[SearchResult] = Field(default_factory=list)
    evidence_score: float = 0.0
    confidence_level: ConfidenceLevel = "medium"
    query: str = ""
    expanded_query: str = ""
    retries: int = 0
    router_decision: RouterDecision | None = None
    graph_context: GraphContext | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    trace: PipelineTrace | None = None  # v6 provenance

    @computed_field  # type: ignore[prop-decorator]
    @property
    def confidence(self) -> float:
        """Legacy compatibility: map confidence_level to a single number.

        Deprecated: prefer `evidence_score` and `confidence_level` directly.
        """
        return {"high": 0.85, "medium": 0.55, "low": 0.25}[self.confidence_level]
