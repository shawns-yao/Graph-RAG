# Agentic Graph RAG

**Skeleton Indexing + VectorCypher + Agentic Router with Self-Correction + Mangle Reasoning + Typed API**

A production-ready Graph RAG system combining four cutting-edge techniques from recent research into a unified retrieval pipeline with declarative reasoning, full pipeline provenance, and a typed API contract (FastAPI REST + MCP).

## Benchmark Results (v14)

Evaluated on **30 bilingual questions** (15 Doc1 Russian + 15 Doc2 English) across 6 retrieval modes = **180 evaluations**:

| Mode | Total | Delta vs v12 | Description |
|------|-------|-------------|-------------|
| **Vector** | **30/30 (100%)** | +2 | Embedding similarity search |
| **Cypher** | **27/30 (90%)** | -1 | Graph traversal via VectorCypher (3-hop) |
| **Hybrid** | **30/30 (100%)** | +2 | Vector + Graph with cosine re-ranking |
| **Agent (pattern)** | **28/30 (93%)** | 0 | Auto-routing via regex patterns + self-correction |
| **Agent (LLM)** | **29/30 (96%)** | +1 | Auto-routing via GPT-4o-mini |
| **Agent (Mangle)** | **30/30 (100%)** | +2 | Declarative Datalog rule routing + self-correction |
| **Overall** | **174/180 (96.7%)** | **+6 (+3.7pp)** | |

Accuracy by query type (across all modes):

| Type | Accuracy | Delta vs v12 |
|------|----------|-------------|
| relation | 42/42 (100%) | — |
| simple | 41/42 (97%) | -1 |
| temporal | 23/24 (95%) | -1 |
| multi_hop | 34/36 (94%) | +3 (was 86%) |
| global | 34/36 (94%) | +5 (was 80%) |

**3 modes at 100%** (vector, hybrid, agent_mangle). **Zero persistent failures** — every question passes in at least 4/6 modes.

<details>
<summary>Benchmark history (v3 → v14)</summary>

| Version | Questions | Overall | Key Changes |
|---------|-----------|---------|-------------|
| v3 | 15 | 34/90 (38%) | Baseline (lang=en, pre-improvements) |
| v4 | 15 | 60/90 (67%) | lang=ru, cosine ranking, synthesis prompt, temporal boost |
| v5 | 15 | 66/90 (73%) | comprehensive_search, completeness check, retry query, max_hops=3 |
| v7 | 20 | 87/120 (73%) | Dual-document (Doc1 RU + Doc2 EN), RELATED_TO edges |
| v9 | 20 | 84/120 (70%) | Hybrid cosine re-ranking (replaced RRF) |
| v10 | 30 | 118/180 (65%) | 15 new Doc2 questions, co-occurrence expansion restored |
| v11 | 30 | 144/180 (80%) | Enumeration prompt, global query detection, judge 2K |
| v12 | 30 | 168/180 (93%) | Hybrid judge, smart mention routing, cross-doc detection, LiteLLM |
| v14 | 30 | 174/180 (96.7%) | Semantic judge, cross-language retrieval routing, Q27 fixed |

</details>

## Key Techniques

| Technique | Source | What It Does |
|-----------|--------|-------------|
| **Skeleton Indexing** | KET-RAG (KDD 2025) | KNN graph -> PageRank -> selective entity extraction (10x cost savings) |
| **Dual Node Structure** | HippoRAG 2 (ICML 2025) | Phrase nodes + passage nodes + Personalized PageRank |
| **VectorCypher** | Neo4j / GraphRAG | Vector entry points -> Cypher traversal -> context assembly |
| **Agentic Router** | Custom | Pattern/LLM/Mangle classification -> tool selection -> self-correction loop |
| **Mangle Reasoning** | PyMangle (Datalog) | Declarative routing rules with bilingual keyword matching (65 keywords) |

## Architecture

```
Ingestion:
  Document -> Docling -> Chunker -> Enricher -> Embedder
           -> Skeleton Indexer (PageRank top-B)
           -> Dual Node Builder (phrase + passage nodes)
           -> Neo4j (Vector Index + Knowledge Graph)

Retrieval:
  Query -> Router (simple/relation/multi_hop/global/temporal)
           Router cascade: Mangle -> LLM -> Pattern fallback
        -> Tool Selection (vector/cypher/hybrid/comprehensive/full_read/temporal)
        -> Self-Correction Loop (reflect -> rerank or retry with targeted tool/provider upgrades)
        -> Graph Verifier (contradiction detection)
        -> Generator (GPT-4o synthesis + dynamic confidence + citations)
```

## Project Structure

```
agentic-graph-rag/
├── packages/rag-core/        # Shared pip package (models, config, ingestion, retrieval)
│   └── rag_core/
│       ├── models.py          # Chunk, Entity, SearchResult, QAResult, QueryType
│       ├── config.py          # Pydantic Settings (nested: Neo4j, OpenAI, Indexing, Agent)
│       ├── loader.py          # Docling: PDF/DOCX/PPTX + GPU
│       ├── chunker.py         # Table-aware chunking
│       ├── enricher.py        # Contextual enrichment (OpenAI)
│       ├── embedder.py        # text-embedding-3-small batch processing
│       ├── vector_store.py    # Neo4j Vector Index CRUD
│       ├── kg_client.py       # Graphiti wrapper + Cypher
│       ├── generator.py       # LLM answer synthesis + dynamic confidence
│       ├── reflector.py       # Relevance evaluation (1-5 scale) + completeness check
│       └── i18n.py            # RU/EN localization (~50 keys)
│
├── agentic_graph_rag/         # Graph RAG components
│   ├── indexing/
│   │   ├── skeleton.py        # KET-RAG: KNN -> PageRank -> skeletal extraction
│   │   └── dual_node.py       # HippoRAG 2: phrase + passage nodes + PPR
│   ├── retrieval/
│   │   └── vector_cypher.py   # Vector entry -> Cypher traversal -> context
│   ├── agent/
│   │   ├── router.py          # Query classifier (pattern + LLM + Mangle)
│   │   ├── retrieval_agent.py # Orchestrator + self-correction loop + provenance
│   │   └── tools.py           # 7 tools: vector, cypher, community, hybrid, temporal, full_read, comprehensive
│   ├── generation/
│   │   └── claim_verifier.py  # Chain-of-Verification: extract claims + graph-based verification
│   ├── reasoning/
│   │   ├── reasoning_engine.py # PyMangle Datalog engine wrapper
│   │   └── rules/             # Mangle rules: routing.mg, access.mg, graph.mg
│   ├── optimization/
│   │   ├── cache.py           # LRU SubgraphCache + CommunityCache
│   │   └── monitor.py         # QueryMonitor + PageRank tuning suggestions
│   └── service.py             # PipelineService — typed internal contract
│
├── api/                       # v6: Typed API contract
│   ├── app.py                 # FastAPI factory with lifespan
│   ├── routes.py              # REST endpoints (query, trace, health, graph_stats)
│   ├── deps.py                # Dependency injection (PipelineService singleton)
│   └── mcp_server.py          # MCP tools (resolve_intent, search_graph, explain_trace)
│
├── pymangle/                  # PyMangle Datalog engine (~5K lines)
│
├── benchmark/
│   ├── questions.json         # 30 test questions (5 types, EN/RU, 2 documents)
│   ├── runner.py              # 6-mode benchmark runner
│   └── compare.py             # Comparison table generator
│
├── scripts/
│   └── ingest.py              # CLI ingestion: python scripts/ingest.py <file>
│
├── data/
│   ├── sample_graph_rag.txt                # Doc1 (RU benchmark Q1-Q15): Graph RAG overview
│   └── sample_semantic_companion_layer.txt # Doc2 (EN benchmark Q16-Q30): SCL / MeaningHub architecture
│
├── docker-compose.yml         # Neo4j 5.x (docker compose up -d)
├── run_api.py                 # API launcher (uvicorn, port 8507)
└── tests/                     # 454 unit tests (346 core + 108 pymangle)
```

## Quick Start

### Prerequisites

- Python 3.12+
- Docker — [Docker Desktop](https://www.docker.com/products/docker-desktop/) for Windows/Mac, or `docker-ce` for Linux ([install guide](https://docs.docker.com/engine/install/))
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### 1. Install

```bash
git clone https://github.com/vpakspace/agentic-graph-rag.git
cd agentic-graph-rag

pip install -e packages/rag-core --no-deps
pip install -r requirements.txt
```

> **Note on Docling**: The first time you load a PDF or DOCX document, Docling will download
> its ML models (~1-2 GB). This is a one-time download. If you only plan to ingest `.txt` files,
> no model download is needed.

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` — at minimum set these two values:
- `OPENAI_API_KEY` — your OpenAI API key
- `NEO4J_PASSWORD` — password you want for the Neo4j database

### 3. Start Neo4j

Using Docker Compose (recommended):

```bash
docker compose up -d
```

Or manually:

```bash
docker run -d \
  --name agentic-graph-rag-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  neo4j:5
```

### 4. Ingest a Document

Two sample documents are included for testing (both needed for the full 30-question benchmark):

```bash
# Ingest both documents (required for benchmark reproducibility)
PYTHONPATH=.:pymangle python scripts/ingest.py data/
```

Or ingest individually:

```bash
PYTHONPATH=.:pymangle python scripts/ingest.py data/sample_graph_rag.txt                # Doc1: Graph RAG (RU, Q1-Q15)
PYTHONPATH=.:pymangle python scripts/ingest.py data/sample_semantic_companion_layer.txt # Doc2: SCL (EN, Q16-Q30)
```

This runs the full pipeline: load → chunk → enrich → embed → store → skeleton indexing → dual-node graph.

Options:
- `--skip-enrichment` — skip LLM contextual enrichment (faster, fewer OpenAI calls)
- `--skip-skeleton` — skip skeleton indexing (vector store only, no knowledge graph)
- `--use-gpu` — enable GPU acceleration for Docling

You can also ingest your own documents (PDF, DOCX, TXT):

```bash
PYTHONPATH=.:pymangle python scripts/ingest.py /path/to/your/document.pdf
PYTHONPATH=.:pymangle python scripts/ingest.py /path/to/documents/  # entire directory
```

### 5. Run the System

Start the API server:

```bash
PYTHONPATH=.:pymangle python run_api.py  # http://localhost:8507
```

The API is the supported runtime for demo and programmatic access.
Use the HTTP endpoints or MCP mount on port `8507`.

> **Why PYTHONPATH?** The project uses two source roots: the main project directory and
> `pymangle/` (the Datalog engine). Setting `PYTHONPATH=.:pymangle` makes both importable.
> Alternatively, you can `export PYTHONPATH=.:pymangle` in your shell profile.

### Run Tests

```bash
PYTHONPATH=.:pymangle pytest tests/ packages/rag-core/tests/ pymangle/tests/ -x -q  # 586 tests, ~4 seconds
```

### API Endpoints

REST (port 8507):
- `POST /api/v1/query` — Query the pipeline (returns answer + trace)
- `GET /api/v1/trace/{id}` — Retrieve a pipeline trace by ID
- `GET /api/v1/health` — Health check (Neo4j connectivity)
- `GET /api/v1/graph/stats` — Graph node/edge statistics

MCP tools (SSE at `/mcp`):
- `resolve_intent` — Classify query type and select tool
- `search_graph` — Execute full pipeline search
- `explain_trace` — Explain a pipeline trace

### Run Benchmark

```python
from benchmark.runner import run_benchmark
from benchmark.compare import compare_modes

# Requires Neo4j running + documents ingested
results = run_benchmark(driver, openai_client)  # lang="ru" by default
print(compare_modes(results))
```

## Retrieval Modes

| Mode | Description | Best For |
|------|-------------|----------|
| **Vector** | Cosine similarity on embeddings | Simple factual queries |
| **Cypher** | Graph traversal via VectorCypher | Relationship queries |
| **Hybrid** | Vector + Graph with RRF fusion | Multi-hop queries |
| **Agent (pattern)** | Auto-routing via regex patterns | General use (fast) |
| **Agent (LLM)** | Auto-routing via GPT-4o-mini | General use (accurate) |
| **Agent (Mangle)** | Auto-routing via Datalog rules | General use (deterministic) |

## Pipeline Provenance (v6)

Every query produces a `PipelineTrace` — a structured record of the full pipeline execution:

```json
{
  "trace_id": "tr_abc123def456",
  "timestamp": "2026-02-17T12:00:00Z",
  "query": "Какие методы используются?",
  "router_step": {"method": "mangle", "decision": {"query_type": "simple", "suggested_tool": "vector_search"}},
  "tool_steps": [{"tool_name": "vector_search", "results_count": 10, "relevance_score": 3.2, "duration_ms": 150}],
  "escalation_steps": [],
  "generator_step": {"model": "gpt-4o-mini", "prompt_tokens": 1200, "completion_tokens": 350, "confidence": 0.82},
  "total_duration_ms": 1800
}
```

Traces are cached (LRU, 100 entries) and retrievable via `GET /api/v1/trace/{id}` or the MCP `explain_trace` tool.

## Self-Correction Loop

The agent evaluates retrieval quality on a 1-5 scale and then chooses among three actions:

``` 
answer | rerank | retry
```

Retries are not a fixed linear escalation chain. The workflow combines:

- deterministic query heuristics
- reflection-recommended tools/providers
- a fallback matrix per current tool

The loop can refresh only part of `hybrid_search`, rewrite the query when broader recall is justified, or stop early when reflection is repeating covered gaps. For `GLOBAL` queries, the top-level workflow may run one additional `comprehensive_search` pass when low-confidence evidence suggests the first answer is incomplete. Cross-language internal alias queries can still route directly to `full_document_read`, but that is a router choice, not the default top-level retry path.

## Mangle Reasoning

Declarative routing via PyMangle (Python Datalog engine):

- **65 bilingual keywords** (Russian + English) in `routing.mg`
- **Router cascade**: Mangle (confidence 0.7) -> LLM (0.85) -> Pattern (0.5)
- **UI**: "Mangle Router" checkbox in sidebar, Reasoning tab for rule testing
- **Rules**: `agentic_graph_rag/reasoning/rules/` — routing, access control, graph traversal

## Configuration

All settings via `.env` or environment variables:

| Setting | Default | Description |
|---------|---------|-------------|
| `OPENAI_API_KEY` | — | OpenAI API key |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection |
| `INDEXING_SKELETON_BETA` | `0.25` | Fraction of chunks for full extraction |
| `INDEXING_KNN_K` | `10` | KNN graph neighbors |
| `INDEXING_PAGERANK_DAMPING` | `0.85` | PageRank damping factor |
| `RETRIEVAL_TOP_K_VECTOR` | `10` | Vector search results count |
| `RETRIEVAL_TOP_K_FINAL` | `10` | Final results after fusion |
| `RETRIEVAL_VECTOR_THRESHOLD` | `0.5` | Minimum similarity score |
| `RETRIEVAL_MAX_HOPS` | `3` | Max graph traversal depth |
| `AGENT_MAX_RETRIES` | `2` | Self-correction retries |
| `AGENT_RELEVANCE_THRESHOLD` | `2.0` | Minimum relevance score (1-5) |

## Tech Stack

- **LLM**: OpenAI GPT-4o / GPT-4o-mini
- **Embeddings**: text-embedding-3-small (1536 dim)
- **Graph DB**: Neo4j 5.x (Vector Index + Cypher)
- **Reasoning**: PyMangle (Datalog engine)
- **Doc Parsing**: Docling (PDF/DOCX/PPTX + GPU)
- **Graph Algorithms**: NetworkX (PageRank, KNN, PPR)
- **API**: FastAPI (REST + MCP via FastMCP)
- **Testing**: pytest (586 tests) + ruff

## References

- [KET-RAG: Cost-Efficient Graph RAG](https://arxiv.org/abs/2502.09304) (KDD 2025)
- [HippoRAG 2: Agentic Retrieval](https://arxiv.org/abs/2502.14802) (ICML 2025)
- [VectorCypher: Neo4j Graph Retrieval](https://neo4j.com/docs/)
- [Agentic RAG: Self-Correcting Retrieval](https://arxiv.org/abs/2401.15884)
- [Mangle: Datalog Dialect](https://github.com/google/mangle) (Google Research)

## License

MIT
