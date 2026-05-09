"""Agentic Graph RAG — Streamlit UI.

7 tabs: Ingest, Search & Q&A, Graph Explorer, Agent Trace, Benchmark, Reasoning, Settings.
Port: 8506
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import httpx
import streamlit as st
from agentic_graph_rag.trace_explain import explain_trace_payload
from rag_core.config import get_settings
from rag_core.i18n import get_translator
from rag_core.models import Chunk

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Agentic Graph RAG",
    page_icon="🔗",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

lang = st.sidebar.radio("Language / Язык", ["en", "ru"], index=0)
t = get_translator(lang)

st.sidebar.title(t("app_title"))
st.sidebar.caption(t("app_subtitle"))

use_gpu = st.sidebar.checkbox(t("ingest_gpu"), value=False)
use_llm_router = st.sidebar.checkbox(
    "LLM Router" if lang == "en" else "LLM Роутер",
    value=False,
)
use_mangle_router = st.sidebar.checkbox(
    "Mangle Router" if lang == "en" else "Mangle Роутер",
    value=False,
)

API_URL = os.environ.get("AGR_API_URL", "http://localhost:8507")


# ---------------------------------------------------------------------------
# Lazy-loaded resources
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_neo4j_driver():
    from neo4j import GraphDatabase
    cfg = get_settings()
    driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password))
    try:
        with driver.session() as session:
            session.run("RETURN 1").single()
    except Exception as exc:
        st.error(f"Cannot connect to Neo4j at {cfg.neo4j.uri}. Run: docker-compose up -d")
        raise SystemExit(1) from exc
    return driver


@st.cache_resource
def _get_openai_client():
    from rag_core.config import make_openai_client
    cfg = get_settings()
    return make_openai_client(cfg)


@st.cache_resource
def _get_vector_store():
    from rag_core.vector_store import VectorStore
    store = VectorStore()
    store.init_index()
    return store


@st.cache_resource
def _get_reasoning_engine():
    from agentic_graph_rag.reasoning.reasoning_engine import ReasoningEngine
    rules_dir = str(Path(__file__).resolve().parent.parent / "agentic_graph_rag" / "reasoning" / "rules")
    return ReasoningEngine(rules_dir)


@st.cache_resource
def _get_cache():
    from agentic_graph_rag.optimization.cache import SubgraphCache
    return SubgraphCache()


@st.cache_resource
def _get_monitor():
    from agentic_graph_rag.optimization.monitor import QueryMonitor
    return QueryMonitor()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "last_qa" not in st.session_state:
    st.session_state.last_qa = None
if "last_trace" not in st.session_state:
    st.session_state.last_trace = None
if "last_trace_payload" not in st.session_state:
    st.session_state.last_trace_payload = None


def _build_local_trace_payload(trace_data: dict[str, Any] | None) -> dict[str, Any] | None:
    """Build explain payload locally when API explain endpoint is unavailable."""
    if not trace_data:
        return None
    try:
        from rag_core.models import PipelineTrace

        trace = PipelineTrace.model_validate(trace_data)
        return explain_trace_payload(trace)
    except Exception:
        return {"trace": trace_data, "explain": {}}


def _fetch_trace_payload_from_api(trace_id: str) -> dict[str, Any] | None:
    """Fetch structured trace explain payload from API."""
    try:
        response = httpx.get(
            f"{API_URL}/api/v1/trace/{trace_id}/explain",
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_ingest, tab_search, tab_graph, tab_trace, tab_bench, tab_reasoning, tab_settings = st.tabs([
    t("tab_ingest"),
    t("tab_search"),
    t("tab_graph_explorer"),
    t("tab_agent_trace"),
    t("tab_benchmark"),
    t("tab_reasoning"),
    t("tab_settings"),
])


# ===================== TAB 1: INGEST ======================================

with tab_ingest:
    st.header(t("ingest_header"))
    st.caption(t("ingest_supported"))

    source = st.radio(
        t("ingest_upload"),
        [t("ingest_source_upload"), t("ingest_source_path")],
        horizontal=True,
    )

    file_path: str | None = None
    if source == t("ingest_source_upload"):
        uploaded = st.file_uploader(
            t("ingest_upload"),
            type=["txt", "pdf", "docx", "pptx", "xlsx", "html"],
            label_visibility="collapsed",
        )
        if uploaded:
            suffix = Path(uploaded.name).suffix
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded.read())
            tmp.flush()
            file_path = tmp.name
    else:
        file_path = st.text_input(
            t("ingest_path_input"),
            placeholder=t("ingest_path_placeholder"),
        )
        if file_path and not Path(file_path).exists():
            st.warning(t("ingest_path_not_found", path=file_path))
            file_path = None

    skip_enrichment = st.checkbox(t("ingest_skip_enrichment"), value=False)

    build_graph = st.checkbox(
        "Build Knowledge Graph" if lang == "en" else "Построить граф знаний",
        value=True,
    )

    if st.button(t("ingest_button"), disabled=not file_path):
        try:
            driver = _get_neo4j_driver()
            client = _get_openai_client()
            store = _get_vector_store()

            from rag_core.chunker import chunk_document, chunk_document_for_graph
            from rag_core.embedder import embed_chunks
            from rag_core.enricher import enrich_chunks
            from rag_core.loader import load_document

            progress = st.progress(0, text=t("ingest_loading"))
            document = load_document(file_path, use_gpu=use_gpu)
            text = document.markdown
            st.info(t("ingest_chars_loaded", chars=len(text)))
            progress.progress(15, text=t("ingest_chunking"))

            cfg = get_settings()
            chunks = chunk_document(
                document,
                parent_chunk_size=cfg.indexing.parent_chunk_size,
                child_chunk_size=cfg.indexing.chunk_size,
                child_chunk_overlap=cfg.indexing.chunk_overlap,
                context_chars=cfg.indexing.context_window_chars,
            )
            graph_chunks = chunk_document_for_graph(document)
            st.info(t("ingest_chunks_created", count=len(chunks)))
            progress.progress(30, text=t("ingest_enriching"))

            if not skip_enrichment:
                chunks = enrich_chunks(chunks, text)
            progress.progress(45, text=t("ingest_embedding"))

            chunks = embed_chunks(chunks)
            progress.progress(60, text=t("ingest_storing"))

            store.add_chunks(chunks)
            total = store.count()

            # --- Build Knowledge Graph (skeleton + dual-node) ---
            if build_graph:
                from agentic_graph_rag.indexing.dual_node import (
                    build_dual_graph,
                    embed_phrase_nodes,
                    init_phrase_index,
                )
                from agentic_graph_rag.indexing.skeleton import build_skeleton_index

                graph_label = "Building graph..." if lang == "en" else "Построение графа..."
                progress.progress(70, text=graph_label)

                graph_embed_inputs = [
                    Chunk.model_validate(chunk.model_dump())
                    for chunk in graph_chunks
                ]
                if graph_embed_inputs:
                    graph_embed_inputs = embed_chunks(graph_embed_inputs)
                embeddings = [c.embedding for c in graph_embed_inputs]
                entities, relationships, skeletal, peripheral = build_skeleton_index(
                    graph_embed_inputs, embeddings, openai_client=client,
                )

                ent_label = (
                    f"Extracted {len(entities)} entities, {len(relationships)} relationships"
                    if lang == "en"
                    else f"Извлечено {len(entities)} сущностей, {len(relationships)} связей"
                )
                st.info(ent_label)
                progress.progress(80, text=graph_label)

                phrase_nodes, passage_nodes, link_count = build_dual_graph(
                    entities, chunks, driver,
                    relationships=relationships,
                )

                progress.progress(90, text=graph_label)

                # Embed PhraseNodes and create vector index
                if phrase_nodes:
                    embed_phrase_nodes(phrase_nodes, driver, openai_client=client)
                    init_phrase_index(driver)

            progress.progress(100, text="Done")

            graph_msg = ""
            if build_graph:
                graph_msg = (
                    f" | Graph: {len(entities)} entities, {len(relationships)} rels"
                    if lang == "en"
                    else f" | Граф: {len(entities)} сущностей, {len(relationships)} связей"
                )
            st.success(t("ingest_success", chunks=len(chunks), total=total) + graph_msg)
        except Exception as e:
            st.error(t("error", msg=str(e)))


# ===================== TAB 2: SEARCH & Q&A ================================

with tab_search:
    st.header(t("search_header"))

    mode = st.radio(
        t("search_mode"),
        [
            t("search_mode_vector"),
            t("search_mode_hybrid"),
            t("search_mode_agent"),
        ],
        horizontal=True,
    )

    query = st.text_input(t("search_input"), placeholder=t("search_placeholder"))

    if st.button(t("search_button"), disabled=not query):
        # Map UI mode to API mode
        mode_map = {
            t("search_mode_vector"): "vector",
            t("search_mode_hybrid"): "hybrid",
            t("search_mode_agent"): "agent_pattern",
        }
        api_mode = mode_map.get(mode, "agent_pattern")
        if use_mangle_router:
            api_mode = "agent_mangle"
        elif use_llm_router:
            api_mode = "agent_llm"

        with st.spinner(t("search_thinking")):
            try:
                # Try API first (thin client)
                resp = httpx.post(
                    f"{API_URL}/api/v1/query",
                    json={"text": query, "mode": api_mode},
                    timeout=120.0,
                )
                resp.raise_for_status()
                data = resp.json()

                from rag_core.models import QAResult
                qa = QAResult.model_validate(data)
                st.session_state.last_qa = qa
                st.session_state.last_trace = data.get("trace")
                trace_id = data.get("trace", {}).get("trace_id")
                st.session_state.last_trace_payload = (
                    _fetch_trace_payload_from_api(trace_id) if trace_id else None
                ) or _build_local_trace_payload(data.get("trace"))

            except (httpx.ConnectError, httpx.HTTPStatusError):
                # Fallback to direct Python if API not available
                st.caption("API not available — using direct mode")
                driver = _get_neo4j_driver()
                client = _get_openai_client()

                if api_mode in ("agent_pattern", "agent_llm", "agent_mangle"):
                    from agentic_graph_rag.agent.retrieval_agent import run as agent_run
                    reasoning = _get_reasoning_engine() if use_mangle_router else None
                    qa = agent_run(
                        query, driver, openai_client=client,
                        use_llm_router=use_llm_router, reasoning=reasoning,
                    )
                elif api_mode == "hybrid":
                    from rag_core.generator import generate_answer

                    from agentic_graph_rag.agent.tools import hybrid_search
                    results = hybrid_search(query, driver, client)
                    qa = generate_answer(query, results, client)
                else:
                    from rag_core.generator import generate_answer

                    from agentic_graph_rag.agent.tools import vector_search
                    results = vector_search(query, driver, client)
                    qa = generate_answer(query, results, client)

                st.session_state.last_qa = qa
                # Build trace from qa.trace if available
                if qa.trace:
                    st.session_state.last_trace = qa.trace.model_dump()
                    st.session_state.last_trace_payload = explain_trace_payload(qa.trace)
                else:
                    trace_dict: dict[str, Any] = {"query": query, "mode": mode}
                    if qa.router_decision:
                        trace_dict["router_step"] = {
                            "method": "direct",
                            "decision": qa.router_decision.model_dump(),
                        }
                    st.session_state.last_trace = trace_dict
                    st.session_state.last_trace_payload = _build_local_trace_payload(trace_dict)

        # Display answer
        qa = st.session_state.last_qa
        if qa:
            st.subheader(t("search_answer"))
            st.write(qa.answer)

            col1, col2 = st.columns(2)
            with col1:
                st.metric(t("search_confidence"), f"{qa.confidence:.0%}")
            with col2:
                st.metric(t("search_retries", count=qa.retries), qa.retries)

            if qa.router_decision:
                st.caption(t("search_query_type", qtype=qa.router_decision.query_type.value))
                st.caption(t("search_router_confidence", conf=qa.router_decision.confidence))

            if qa.sources:
                with st.expander(t("search_sources", count=len(qa.sources))):
                    for i, src in enumerate(qa.sources, 1):
                        st.markdown(f"**{i}.** {src.chunk.content[:200]}...")
                        st.caption(t("search_source_score", score=src.score))


# ===================== TAB 3: GRAPH EXPLORER ==============================

with tab_graph:
    st.header(t("graph_header"))

    max_nodes = st.slider(t("graph_max_nodes"), 10, 200, 50)

    try:
        driver = _get_neo4j_driver()
        with driver.session() as session:
            # Count phrase nodes
            phrase_count = session.run(
                "MATCH (n:PhraseNode) RETURN count(n) AS cnt"
            ).single()["cnt"]

            # Count passage nodes
            passage_count = session.run(
                "MATCH (n:PassageNode) RETURN count(n) AS cnt"
            ).single()["cnt"]

            st.metric(t("graph_phrase_nodes"), phrase_count)
            st.metric(t("graph_passage_nodes"), passage_count)

            if phrase_count == 0 and passage_count == 0:
                st.info(t("graph_no_data"))
            else:
                # Build graphviz dot string
                result = session.run(
                    """MATCH (a:PhraseNode)-[r]->(b)
                    RETURN a.name AS src, type(r) AS rel, b.name AS tgt
                    LIMIT $limit""",
                    limit=max_nodes,
                )
                edges = [(rec["src"], rec["rel"], rec["tgt"]) for rec in result]

                if edges:
                    dot_lines = [
                        "digraph G {",
                        "  rankdir=LR;",
                        '  node [shape=box, style=filled, fillcolor="#E8F4FD"];',
                    ]
                    for src, rel, tgt in edges:
                        safe_src = (src or "?").replace('"', '\\"')
                        safe_tgt = (tgt or "?").replace('"', '\\"')
                        safe_rel = (rel or "").replace('"', '\\"')
                        dot_lines.append(f'  "{safe_src}" -> "{safe_tgt}" [label="{safe_rel}"];')
                    dot_lines.append("}")
                    st.graphviz_chart("\n".join(dot_lines))
                else:
                    st.info(t("graph_no_data"))

    except Exception as e:
        st.warning(t("error", msg=str(e)))


# ===================== TAB 4: AGENT TRACE =================================

with tab_trace:
    st.header(t("trace_header"))

    trace_payload = st.session_state.get("last_trace_payload")
    trace_data = st.session_state.get("last_trace")
    trace_raw = trace_payload.get("trace", trace_data) if trace_payload else trace_data
    trace_explain = trace_payload.get("explain", {}) if trace_payload else {}
    if trace_raw is None:
        st.info(t("trace_no_data"))
    else:
        # Router decision
        router_view = trace_explain.get("router")
        router_step = trace_raw.get("router_step", {})
        if router_view or router_step:
            st.subheader(t("trace_routing"))
            rs = router_view or {
                "method": router_step.get("method", "—"),
                "duration_ms": router_step.get("duration_ms", 0),
                "query_type": router_step.get("decision", {}).get("query_type", "—"),
                "confidence": router_step.get("decision", {}).get("confidence", 0),
                "suggested_tool": router_step.get("decision", {}).get("suggested_tool", "—"),
                "reasoning": router_step.get("decision", {}).get("reasoning", ""),
                "rules_fired": router_step.get("rules_fired", []),
            }
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Method", rs.get("method", "—"))
            with col2:
                st.metric(t("trace_query_type"), rs.get("query_type", "—"))
            with col3:
                st.metric(t("trace_confidence"), f"{rs.get('confidence', 0):.0%}")
            with col4:
                st.metric(t("trace_tool"), rs.get("suggested_tool", "—"))
            if rs.get("rules_fired"):
                st.caption(f"Rules: {', '.join(rs['rules_fired'])}")
            st.caption(f"{t('trace_reasoning')}: {rs.get('reasoning', '')}")
            st.caption(f"Duration: {rs.get('duration_ms', 0)}ms")

        # Tool steps
        retrieval_view = trace_explain.get("retrieval", {})
        if trace_raw.get("tool_steps"):
            st.divider()
            st.subheader("Tool Steps")
            explained_steps = retrieval_view.get("steps", [])
            for i, step in enumerate(trace_raw["tool_steps"], 1):
                explained = explained_steps[i - 1] if i - 1 < len(explained_steps) else {}
                with st.container(border=True):
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric(f"Step {i}", step["tool_name"])
                    with c2:
                        st.metric("Results", step.get("results_count", 0))
                    with c3:
                        score = step.get("relevance_score", 0)
                        st.metric("Relevance", f"{score:.1f}/5.0")
                    with c4:
                        st.metric("Duration", f"{step.get('duration_ms', 0)}ms")
                    if explained.get("providers"):
                        provider_rows = []
                        for provider in explained["providers"]:
                            provider_rows.append({
                                "Provider": provider.get("source", "—"),
                                "Status": provider.get("status", "idle"),
                                "Results": provider.get("results_count", 0),
                                "Top Score": provider.get("top_score", 0.0),
                                "Avg Score": provider.get("average_score", 0.0),
                                "Top Chunks": ", ".join(provider.get("top_chunk_ids", [])),
                            })
                        st.dataframe(provider_rows, use_container_width=True, hide_index=True)
                    if explained.get("empty_sources"):
                        st.caption(f"Empty sources: {', '.join(explained['empty_sources'])}")

        provider_summary = retrieval_view.get("provider_summary", [])
        if provider_summary:
            st.divider()
            st.subheader("Provider Summary")
            summary_rows = []
            for item in provider_summary:
                summary_rows.append({
                    "Provider": item.get("source", "—"),
                    "Executed": item.get("executed_steps", 0),
                    "Reused": item.get("reused_steps", 0),
                    "Empty": item.get("empty_steps", 0),
                    "Total Results": item.get("total_results", 0),
                    "Max Top Score": item.get("max_top_score", 0.0),
                    "Top Chunks": ", ".join(item.get("top_chunk_ids", [])),
                })
            st.dataframe(summary_rows, use_container_width=True, hide_index=True)
            likely_gaps = retrieval_view.get("likely_gaps", [])
            if likely_gaps:
                st.warning(f"Likely gaps: {', '.join(likely_gaps)}")

        # Escalation steps
        escalation_steps = trace_explain.get("escalations") or trace_raw.get("escalation_steps")
        if escalation_steps:
            st.divider()
            st.subheader("Escalations")
            for esc in escalation_steps:
                dur = f" ({esc['duration_ms']}ms)" if esc.get("duration_ms") else ""
                st.warning(
                    f"**{esc['from_tool']}** → **{esc['to_tool']}**: {esc.get('reason', '')}{dur}"
                )

        # Generator step
        generator_step = trace_explain.get("generation") or trace_raw.get("generator_step")
        if generator_step:
            st.divider()
            st.subheader("Generator")
            gs = generator_step
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Model", gs.get("model", "—"))
            with c2:
                tokens = gs.get("prompt_tokens", 0) + gs.get("completion_tokens", 0)
                st.metric("Tokens", tokens)
            with c3:
                st.metric("Confidence", f"{gs.get('confidence', 0):.0%}")
            with c4:
                st.metric("Duration", f"{gs.get('duration_ms', 0)}ms")

        # Summary — latency breakdown
        st.divider()
        total_ms = trace_raw.get("total_duration_ms", 0)
        router_ms = router_step.get("duration_ms", 0) if router_step else 0
        tool_ms = sum(s.get("duration_ms", 0) for s in trace_raw.get("tool_steps", []))
        gen_ms = generator_step.get("duration_ms", 0) if generator_step else 0
        other_ms = max(0, total_ms - router_ms - tool_ms - gen_ms)

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Total", f"{total_ms}ms")
        with c2:
            st.metric("Router", f"{router_ms}ms")
        with c3:
            st.metric("Tools", f"{tool_ms}ms")
        with c4:
            st.metric("Generator", f"{gen_ms}ms")
        with c5:
            st.metric("Other", f"{other_ms}ms")

        st.caption(f"Trace ID: {trace_raw.get('trace_id', '—')}")

        # Raw JSON (expandable)
        with st.expander("Raw Trace JSON"):
            st.json(trace_payload or trace_raw)


# ===================== TAB 5: BENCHMARK ===================================

with tab_bench:
    st.header(t("bench_header"))

    bench_modes = st.multiselect(
        t("bench_mode"),
        ["vector_only", "bm25_only", "graph_only", "hybrid", "hybrid_rerank"],
        default=["vector_only", "hybrid", "hybrid_rerank"],
    )

    if st.button(t("bench_run"), disabled=not bench_modes):
        try:
            driver = _get_neo4j_driver()
            client = _get_openai_client()

            from benchmark.compare import compare_modes, compute_metrics
            from benchmark.runner import load_questions, run_benchmark

            questions = load_questions()

            with st.spinner(t("bench_running", current=0, total=len(questions))):
                all_results = run_benchmark(
                    driver, client, modes=bench_modes, questions=questions, lang=lang,
                )

            # Comparison table
            st.subheader(t("bench_results"))
            comparison = compare_modes(all_results)
            st.dataframe(comparison, use_container_width=True)

            # Per-mode details
            for mode_name, results in all_results.items():
                m = compute_metrics(results)
                with st.expander(f"{mode_name}: {m['correct']}/{m['total']} ({m['accuracy']:.0%})"):
                    rows = []
                    for r in results:
                        rows.append({
                            t("bench_col_q"): r["id"],
                            t("bench_col_status"): "PASS" if r["passed"] else "FAIL",
                            t("bench_col_confidence"): f"{r['confidence']:.2f}",
                            t("bench_col_retries"): r["retries"],
                            "P@5": f"{r.get('precision_at_k', 0.0):.2f}",
                            "R@5": f"{r.get('recall_at_k', 0.0):.2f}",
                            "MRR": f"{r.get('reciprocal_rank', 0.0):.2f}",
                            t("bench_col_question"): r["question"][:80],
                        })
                    st.dataframe(rows, use_container_width=True)

        except Exception as e:
            st.error(t("error", msg=str(e)))


# ===================== TAB 6: REASONING ===================================

with tab_reasoning:
    st.header(t("reasoning_header"))

    from pathlib import Path as _Path

    _default_rules_dir = _Path(__file__).resolve().parent.parent / "agentic_graph_rag" / "reasoning" / "rules"

    # Load default rules from .mg files
    _default_sources: dict[str, str] = {}
    if _default_rules_dir.exists():
        for _p in sorted(_default_rules_dir.glob("*.mg")):
            _default_sources[_p.stem] = _p.read_text()

    # Rule selector
    _source_names = list(_default_sources.keys()) if _default_sources else ["routing"]
    _selected_source = st.selectbox(
        t("reasoning_rules_label"),
        _source_names,
        index=0,
    )

    # Editable text area
    _default_text = _default_sources.get(_selected_source, "% Write Mangle rules here\n")
    rules_text = st.text_area(
        t("reasoning_rules_help"),
        value=_default_text,
        height=300,
        key=f"rules_{_selected_source}",
    )

    st.divider()

    # --- Routing test ---
    st.subheader(t("reasoning_routing_header"))
    test_query = st.text_input(
        t("reasoning_query_label"),
        placeholder=t("reasoning_query_placeholder"),
        key="reasoning_query",
    )

    if st.button(t("reasoning_run"), disabled=not test_query):
        try:
            from agentic_graph_rag.reasoning.reasoning_engine import ReasoningEngine

            engine = ReasoningEngine.from_sources({_selected_source: rules_text})
            result = engine.classify_query(test_query)

            if result is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(t("reasoning_tool"), result["tool"])
                with col2:
                    # Determine category from route
                    st.metric(t("reasoning_category"), _selected_source)
                st.success(f"route_to(\"{result['tool']}\", \"{test_query[:60]}...\")")
            else:
                st.warning(t("reasoning_no_match"))
        except Exception as e:
            st.error(t("reasoning_error", msg=str(e)))

    st.divider()

    # --- Access control test ---
    if "access" in _default_sources or _selected_source == "access":
        st.subheader(t("reasoning_access_header"))
        col_role, col_action = st.columns(2)
        with col_role:
            access_role = st.text_input(t("reasoning_role_label"), value="/viewer", key="access_role")
        with col_action:
            access_action = st.text_input(t("reasoning_action_label"), value="/read", key="access_action")

        if st.button(t("reasoning_access_run")):
            try:
                from agentic_graph_rag.reasoning.reasoning_engine import ReasoningEngine

                access_src = _default_sources.get("access", rules_text) if _selected_source != "access" else rules_text
                engine = ReasoningEngine.from_sources({"access": access_src})
                allowed = engine.check_access("test_user", access_role, access_action)
                if allowed:
                    st.success(t("reasoning_access_allowed"))
                else:
                    st.error(t("reasoning_access_denied"))
            except Exception as e:
                st.error(t("reasoning_error", msg=str(e)))

    st.divider()

    # --- Stratification visualization ---
    st.subheader(t("reasoning_strata_header"))
    try:
        from agentic_graph_rag.reasoning.reasoning_engine import ReasoningEngine

        engine = ReasoningEngine.from_sources({_selected_source: rules_text})
        strata = engine.get_strata(_selected_source)
        if strata:
            for idx, predicates in enumerate(strata):
                st.text(t("reasoning_strata_text", idx=idx, predicates=", ".join(predicates)))
            # Mermaid diagram
            mermaid_lines = ["graph TD"]
            for idx, predicates in enumerate(strata):
                node_id = f"S{idx}"
                label = f"Stratum {idx}\\n{', '.join(predicates[:5])}"
                if len(predicates) > 5:
                    label += f"\\n+{len(predicates) - 5} more"
                mermaid_lines.append(f"    {node_id}[\"{label}\"]")
                if idx > 0:
                    mermaid_lines.append(f"    S{idx - 1} --> {node_id}")
            st.markdown("```mermaid\n" + "\n".join(mermaid_lines) + "\n```")
        else:
            st.info(t("reasoning_no_match"))
    except Exception as e:
        st.error(t("reasoning_error", msg=str(e)))


# ===================== TAB 7: SETTINGS ====================================

with tab_settings:
    st.header(t("settings_header"))

    # Current config
    st.subheader(t("settings_current"))
    cfg = get_settings()
    config_dict = {
        "Neo4j URI": cfg.neo4j.uri,
        "Embedding Model": cfg.openai.embedding_model,
        "LLM Model": cfg.openai.llm_model,
        "Chunk Size": cfg.indexing.chunk_size,
        "Skeleton Beta": cfg.indexing.skeleton_beta,
        "KNN K": cfg.indexing.knn_k,
        "PageRank Damping": cfg.indexing.pagerank_damping,
        "Top K Vector": cfg.retrieval.top_k_vector,
        "Top K Final": cfg.retrieval.top_k_final,
        "Max Hops": cfg.retrieval.max_hops,
        "Max Retries": cfg.agent.max_retries,
        "Relevance Threshold": cfg.agent.relevance_threshold,
    }
    st.json(config_dict)

    # Vector store stats
    st.subheader(t("settings_store_stats"))
    try:
        store = _get_vector_store()
        st.write(t("settings_total_chunks", count=store.count()))
    except Exception as e:
        st.caption(t("error", msg=str(e)))

    # Cache stats
    st.subheader(t("settings_cache_header"))
    cache = _get_cache()
    cs = cache.stats()
    st.write(t("settings_cache_size", size=cs["size"], max_size=cs["max_size"]))
    st.write(t("settings_cache_hit_rate", rate=cs["hit_rate"]))

    # Monitor stats
    st.subheader(t("settings_monitor_header"))
    monitor = _get_monitor()
    ms = monitor.get_stats()
    st.write(t("settings_monitor_total", count=ms["total_queries"]))
    if ms["total_queries"] > 0:
        st.json(ms)
        suggestions = monitor.suggest_pagerank_weights()
        if suggestions.get("adjustments"):
            st.subheader(t("settings_suggestions"))
            st.json(suggestions)

    # Clear DB
    st.subheader(t("settings_clear_db"))
    confirm = st.text_input(t("settings_clear_confirm"), key="clear_confirm")
    if st.button(t("settings_clear_button"), disabled=confirm != "DELETE"):
        try:
            store = _get_vector_store()
            count = store.count()
            store.delete_all()
            st.success(t("settings_cleared", count=count))
        except Exception as e:
            st.error(t("error", msg=str(e)))
