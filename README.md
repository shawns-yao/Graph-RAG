# Agentic Graph RAG

Agentic Graph RAG is a medical-oriented Graph RAG project that combines skeleton graph extraction, dual-layer graph modeling, multi-channel retrieval, and self-correcting answer verification.

The system is designed for document-scale medical knowledge retrieval where plain vector search is not enough: relation queries, multi-hop questions, evidence tracing, and hallucination control all need explicit graph structure and a verifiable retrieval loop.

## Highlights

- **Skeleton graph extraction**: ranks document chunks with KNN + PageRank, then applies selective LLM extraction to reduce full-document extraction cost.
- **Dual-layer graph model**: stores `PhraseNode` entity nodes and `PassageNode` evidence passages with `MENTIONED_IN` and `RELATED_TO` relationships.
- **Multi-channel retrieval**: combines vector search, BM25, graph traversal, hybrid search, and query-type routing.
- **Self-correction loop**: uses LangGraph workflow state to evaluate retrieval quality, retry with a better tool, rewrite queries, and verify final claims against evidence.
- **Typed API contract**: exposes FastAPI REST endpoints and MCP tools for querying, trace inspection, graph stats, and intent resolution.
- **Evaluation assets**: includes medical benchmark questions, trace evaluation outputs, assertion-classifier metrics, and unit tests.

## Architecture

```text
Ingestion:
  Document
    -> Loader / Chunker
    -> Context enrichment
    -> Embedding
    -> Skeleton indexer: KNN -> PageRank -> selective extraction
    -> Dual node builder: PhraseNode + PassageNode
    -> Neo4j vector index + knowledge graph

Retrieval:
  Query
    -> Query signal extraction
    -> Router: deterministic / pattern / optional LLM
    -> Tool selection: vector / BM25 / graph / hybrid / comprehensive
    -> Reflection and retry planning
    -> Evidence contract generation
    -> Claim verification
    -> Answer with citations and trace
```

## Project Structure

```text
agentic-graph-rag/
├── agentic_graph_rag/
│   ├── agent/                 # Router, LangGraph workflow, retry planner, tools
│   ├── generation/            # Claim verification
│   ├── indexing/              # Skeleton extraction and dual-node graph modeling
│   ├── optimization/          # Cache and monitoring utilities
│   ├── retrieval/             # VectorCypher and fusion retrieval
│   ├── service.py             # PipelineService
│   └── trace_*.py             # Trace storage and explanation
├── api/                       # FastAPI REST + MCP server
├── benchmark/                 # Benchmark runner and metric adapters
├── data/                      # Small sample data and assertion evaluation assets
├── docs/                      # Design notes and benchmark documentation
├── packages/rag-core/         # Shared RAG core package
├── scripts/                   # Ingestion, benchmark, export, and evaluation scripts
├── test/medical_benchmark/    # Medical benchmark questions and result files
├── tests/                     # Project-level tests
├── docker-compose.yml         # Neo4j service
├── pyproject.toml
├── requirements.txt
└── run_api.py
```

## Current Evaluation Snapshot

These numbers come from the committed gold set in `test/medical_benchmark/eval_gold/`, committed benchmark outputs in `test/medical_benchmark/results/`, and the canonical live evaluator in `scripts/evaluate_resume_metrics_live.py`. They are useful as a reproducible local benchmark snapshot, not as a universal claim across all medical corpora. Metrics with tiny denominators, proxy-only values, or 100% results should not be used as headline claims.

### Resume-Oriented Metrics

Source: `docs/resume_metrics_fill.md` and `scripts/evaluate_resume_metrics_live.py`.

| Metric | Value | Denominator | Resume use |
|---|---:|---:|---|
| Live LLM prompt token cost reduction | 69.68% | 16 chunks | Yes, local benchmark only |
| Entity graph coverage gain vs skeleton-only baseline | +4.17 pp | 48 entities | Yes |
| Skeleton-only LLM entity accuracy | 65.38% | 78 positive/negative entities | No: lower than full-document LLM |
| Positive relation recall | 95.00% | 20 relations | Yes |
| Relation false-positive rate | 33.33% | 30 hard negatives | Yes |
| Relation false-positive reduction vs co-occurrence baseline | +33.34 pp | 30 hard negatives | Yes |
| 3-hop graph accuracy | 75.00% | 24 multi-hop tasks | Yes |
| 3-hop gain vs 1-hop baseline | +29.17 pp | 24 multi-hop tasks | Yes |
| Fusion evidence answerability gain vs best single channel | +6.66 pp | 30 questions | Yes |

Run it locally:

```bash
python scripts/evaluate_resume_metrics_live.py --sections extraction,qa --modes vector_search,cypher_traverse,hybrid_search,agent_pattern --output test/medical_benchmark/results/resume_live_metrics.json
```

The skeleton chunks are not manually selected. The evaluator calls the project selection code: KNN graph -> PageRank -> blended score using PageRank, entity density, medical section prior, and hard-fact signal -> greedy diversity selection. QA and hallucination metrics require live LLM judge completion; do not fill them from proxy or cooldown-interrupted runs.

### Retrieval Benchmark

Source: `test/medical_benchmark/results/benchmark_results.json`

| Metric | Vector | Cypher |
|---|---:|---:|
| Questions | 30 | 30 |
| Overall accuracy | 73.33% | 80.00% |
| Gain vs vector | - | +6.67 pp |
| Multi-hop accuracy | 50.00% | 50.00% |
| Average latency | 39,512 ms | 29,771 ms |

### Self-Correction And Verification

Source: `test/medical_benchmark/results/trace_eval_fixed_questions.json`

| Metric | Value |
|---|---:|
| QA cases | 10 |
| Cases with retrieval retry/tool switch | 4 / 10 |
| Retry rate | 40.00% |
| Atomic claims | 22 |
| Supported claims | 22 / 22 |

### Assertion Guard

Source: `data/assertion/dialogue_eval_metrics.json` and `data/assertion/dialogue_eval_metrics_after_guard.json`

| Dataset | Before guard | After guard | Absolute gain | Relative error reduction |
|---|---:|---:|---:|---:|
| dialogue assertion | 90.17% | 99.58% | +9.41 pp | 95.73% |

## Requirements

- Python 3.12+
- Neo4j 5.x
- Docker or a reachable Neo4j instance
- LLM API credentials compatible with the project's `.env` configuration

The current project declares Python 3.12+ in `pyproject.toml`. Some local test runs may fail on Python 3.10 because dependencies such as LangGraph are required by the runtime workflow.

## Quick Start

### 1. Clone

```bash
git clone https://github.com/shawns-yao/Graph-RAG.git
cd Graph-RAG
```

### 2. Create Environment

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
```

Set at least:

```text
LLM_API_KEY=...
LLM_BASE_URL=...
LLM_MODEL=...
EMBEDDING_API_KEY=...
EMBEDDING_BASE_URL=...
EMBEDDING_MODEL=...
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
```

Do not commit `.env`.

### 5. Start Neo4j

```bash
docker compose up -d
```

Check connectivity:

```bash
python -c "from dotenv import load_dotenv; load_dotenv(); from rag_core.config import get_settings; print(get_settings().neo4j.uri)"
```

### 6. Ingest Documents

```bash
python scripts/ingest.py data/
```

For a single document:

```bash
python scripts/ingest.py path/to/document.pdf
python scripts/ingest.py path/to/document.docx
python scripts/ingest.py path/to/document.txt
```

Useful options:

- `--skip-enrichment`: skip LLM contextual enrichment.
- `--skip-skeleton`: skip skeleton graph extraction.
- `--use-gpu`: enable GPU acceleration when supported by the document loader stack.

### 7. Run API

```bash
python run_api.py
```

Default local API:

```text
http://localhost:8507
```

Main REST endpoints:

- `POST /api/v1/query`: run a RAG query.
- `GET /api/v1/trace/{id}`: inspect a pipeline trace.
- `GET /api/v1/health`: service and Neo4j health check.
- `GET /api/v1/graph/stats`: graph statistics.

MCP tools are mounted at `/mcp`:

- `resolve_intent`
- `search_graph`
- `explain_trace`

## Running Tests

```bash
pytest -q
```

Focused tests:

```bash
pytest -q tests
pytest -q packages/rag-core/tests
```

Lint, when `ruff` is installed:

```bash
ruff check .
```

## Running Benchmarks

The benchmark path depends on Neo4j, ingested documents, and configured LLM/embedding providers.

Three-mode medical benchmark:

```bash
python test/medical_benchmark/run_benchmark_3modes.py
```

General benchmark runner:

```bash
python -m benchmark.runner \
  --questions test/medical_benchmark/questions_master.json \
  --modes vector_only,graph_only,hybrid_rerank,vector_chain,graph_chain
```

Existing result files:

- `test/medical_benchmark/results/benchmark_results.json`
- `test/medical_benchmark/results/trace_eval_fixed_questions.json`
- `data/assertion/*metrics*.json`

## Core Concepts

### Skeleton Extraction

Full-document LLM extraction is expensive and noisy. This project first builds a chunk similarity graph, scores chunks with PageRank, and runs deeper extraction only on high-value chunks. Long-tail chunks still stay searchable through passage storage and vector/BM25 retrieval.

### Dual-Layer Graph

The graph separates entity-level structure from passage-level evidence:

- `PhraseNode`: entity or phrase node.
- `PassageNode`: source text passage.
- `MENTIONED_IN`: connects an entity to the passage where it appears.
- `RELATED_TO`: connects entities with extracted semantic relationships.

This keeps graph traversal grounded in evidence and makes answer citations traceable.

### Retrieval Fusion

Different query types need different retrieval behavior:

- factual queries: vector or BM25 can be enough.
- relation queries: graph traversal is preferred.
- multi-hop queries: hybrid retrieval and rerank are more useful.
- global questions: broader recall and comprehensive search are needed.

The agent can retry with another tool when reflection detects missing entities, missing relationships, weak evidence, or off-topic context.

### Verification Loop

The generation layer creates an evidence contract and verifies answer claims. Unsupported or incorrect claims can be blocked, downgraded, or trigger additional retrieval depending on the workflow state and budget.

## Documentation

Design notes are under `docs/`:

- `docs/01-骨架图谱抽取.md`
- `docs/02-双节点图谱建模.md`
- `docs/03-三路融合检索.md`
- `docs/04-检索自纠循环.md`
- `docs/08-Benchmark与评估体系.md`
- `docs/09-项目难点与踩坑.md`

## Security Notes

- `.env` is ignored and must not be committed.
- Do not commit local model checkpoints, raw private medical data, or generated large corpora unless they are intentionally sanitized.
- Treat LLM provider keys, Neo4j credentials, and benchmark patient data as sensitive.

## License

MIT
