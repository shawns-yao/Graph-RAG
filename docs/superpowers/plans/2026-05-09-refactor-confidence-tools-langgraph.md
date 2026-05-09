# Refactor: Confidence + Tools + LangGraph Cleanup

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 agentic-graph-rag 项目中多个关键问题：confidence 计算逻辑不一致、comprehensive_search 重复 RRF、tools 层 source 命名混乱、orchestrator 重复改写 rank、LangGraph State 膨胀、死代码。

**Architecture:** 5 个独立修改域，各自有清晰的边界。confidence 计算收敛到 generator.py 一处；tools 层通过 Provider 抽象统一 source 标签；LangGraph State 按节点职责分离字段。

**Tech Stack:** Python, FastAPI, LangGraph, Neo4j, OpenAI

---

## 文件变更总览

| 文件 | 职责变化 |
|------|---------|
| `packages/rag-core/rag_core/generator.py` | 唯一 confidence 计算入口；移除流/非流重复逻辑 |
| `agentic_graph_rag/service.py` | 删除 confidence 重算逻辑；直接复用 qa_result.confidence |
| `agentic_graph_rag/agent/tools.py` | 统一 source 标签为 vector/bm25/graph/hybrid；comprehensive_search 不再调用 hybrid_search |
| `agentic_graph_rag/retrieval/orchestrator.py` | `_finalize_results` 不再改写 rank；rank 由 rerank 最终确定 |
| `agentic_graph_rag/agent/langgraph_workflow.py` | 删除 `_reflect_attempt` 死代码；精简 `SelfCorrectionState` 字段 |

---

## Task 1: 统一 confidence 计算

**Files:**
- Modify: `agentic_graph_rag/service.py:188-204`
- Modify: `packages/rag-core/rag_core/generator.py:178-225`

**分析：**
- `generator.py` 的 `generate_answer()` 始终计算 `qa_result.confidence`，内部调用 `calculate_answer_confidence()`。
- `service.py` 的 `_adapt_qa_result_to_stream_events()` 在 `confidence <= 0.0` 时重算，且只用 `reflection_score`。
- 逻辑不一致：`reflection_score is None` 时 generator 用 `avg_retrieval_score`，service 则不重算（但流/非流路径条件不同）。

**Fix：** stream 路径直接用 `qa.confidence`，不重算。

- [ ] **Step 1: 查看 service.py 当前逻辑**

Run: `sed -n '183,215p' agentic_graph_rag/service.py`

- [ ] **Step 2: 修改 service.py — 移除 confidence 重算逻辑**

```python
# 替换 service.py 第 188-204 行（约）：
# 原：
#     confidence = qa.confidence
#     if confidence <= 0.0 and qa.sources:
#         reflection_score = ...
#         confidence = calculate_answer_confidence(...)
# 新：
        confidence = qa.confidence
```

Run: `sed -i '188,204s/.*/        confidence = qa.confidence/' agentic_graph_rag/service.py`
或用 Edit 工具精确替换 188-204 行区域。

- [ ] **Step 3: 验证修改**

Run: `grep -n "confidence" agentic_graph_rag/service.py | grep -E "calculate|recompute"`
Expected: 无输出（无重算调用）

---

## Task 2: 消除 comprehensive_search 重复 RRF

**Files:**
- Modify: `agentic_graph_rag/agent/tools.py:400-460`

**分析：**
- 当前做了 4 次 RRF：子查询之间 2 次 + hybrid_search 的 1 次 + 最后再 RRF 1 次。
- `hybrid_search` 内部已经有 fusion + rerank。再调用一次 `rerank_enabled=False` 然后再做 RRF 合并，违反了最小惊讶原则。

**Fix：** 移除对 `hybrid_search` 的调用，改为只在子查询 RRF 之后追加 `full_document_read` 的结果（不再混合已 rerank 的结果）。

- [ ] **Step 1: 查看当前 comprehensive_search 实现**

Run: `sed -n '400,460p' agentic_graph_rag/agent/tools.py`

- [ ] **Step 2: 替换 comprehensive_search**

将第 417-453 行的 `comprehensive_search` 函数体替换为：

```python
def comprehensive_search(
    query: str,
    driver: Driver,
    openai_client: OpenAI,
    top_k: int | None = None,
) -> list[SearchResult]:
    """Comprehensive retrieval: LLM generates sub-queries + keyword extraction,
    each → retrieval, merge via RRF. Also includes full_document_read passages.

    Designed for GLOBAL queries ("list all", "summarize all") where a single
    top-k pass misses components.
    """
    cfg = get_settings()
    if top_k is None:
        top_k = max(cfg.retrieval.top_k_final, 8)

    # Detect enumeration count for dynamic sub-query generation
    n_sub = min(3, _detect_enumeration_count(query))
    sub_queries = _generate_sub_queries(query, openai_client, cfg.openai.llm_model_mini, n=n_sub)

    # Fan-out: run vector search for each sub-query
    all_results: list[list[SearchResult]] = []
    for sq in sub_queries:
        results = vector_search(sq, driver, openai_client, top_k=min(cfg.retrieval.top_k_vector, 4))
        all_results.append(results)

    # Full document read — keep narrow and similarity-ranked
    full_top_k = min(max(top_k, 4), 8)
    full_results = full_document_read(query, driver, openai_client, top_k=full_top_k)
    all_results.append(full_results)

    # Merge all result lists via cascading RRF (子查询之间 + full_read)
    if not all_results:
        return vector_search(query, driver, openai_client, top_k=top_k)

    merged = all_results[0]
    for i in range(1, len(all_results)):
        merged = _rrf_merge(merged, all_results[i], top_k=top_k)

    # Final rerank — apply cosine rerank once at the end
    query_emb = _embed_query(query, openai_client)
    ranked = rerank(query, merged, top_k=top_k, query_embedding=query_emb)

    logger.info("Comprehensive search: %d results from %d sub-queries + full_read", len(ranked), len(sub_queries))
    return ranked
```

- [ ] **Step 3: 验证修改**

Run: `python -c "from agentic_graph_rag.agent.tools import comprehensive_search; print('import ok')"`
Expected: import ok（无语法错误）

---

## Task 3: 统一 tools 层 source 标签

**Files:**
- Modify: `agentic_graph_rag/agent/tools.py:300-400`（temporal_query, full_document_read）

**分析：**
- `temporal_query` 返回 `source="temporal"`
- `full_document_read` 返回 `source="full_read"`
- `community_search` fallback 到 vector 时 `source="vector"`
- 这些绕过了 Provider 抽象，与 fusion 层的 `source_weights`（vector/bm25/graph）不对齐。

**Fix：** 统一改为 `source="vector"`（用于 temporal 和 full_read），`community_search` 也改 `source="vector"`。删除 `_graph_context_to_results` 中的 `source` 参数覆盖。

- [ ] **Step 1: 查看 temporal_query 和 full_document_read source 标签**

Run: `grep -n 'source=' agentic_graph_rag/agent/tools.py | grep -v "#"`

- [ ] **Step 2: 修改 temporal_query source 标签**

Edit `temporal_query` 中的 `SearchResult` 构造：
```python
# 找到: source="temporal"
# 改为: source="vector"
```

Run: `grep -n 'source="temporal"' agentic_graph_rag/agent/tools.py`
Expected: 约第 328 行

- [ ] **Step 3: 修改 full_document_read source 标签**

Run: `grep -n 'source="full_read"' agentic_graph_rag/agent/tools.py`
Expected: 约第 400 行

- [ ] **Step 4: 修改 community_search source 标签**

Run: `grep -n 'source="vector"' agentic_graph_rag/agent/tools.py | grep community`
Expected: 约第 142 行（community_search fallback 行）

- [ ] **Step 5: 验证 tools 层 source 一致性**

Run: `grep -n 'source=' agentic_graph_rag/agent/tools.py | grep -vE '"(vector|bm25|graph|hybrid)"' | grep -v "#"`
Expected: 无输出（所有 source 必须是 vector/bm25/graph/hybrid 之一）

---

## Task 4: 修复 orchestrator._finalize_results 重复改写 rank

**Files:**
- Modify: `agentic_graph_rag/retrieval/orchestrator.py:101-106`

**分析：**
- `rerank()` 内部 `_finalize_reranked_results` 已经设置了 `score_normalized` 和 `rank`。
- `orchestrator._finalize_results` 又把 rank 设为 `index + 1`，覆盖了 rerank 的结果。
- fusion 阶段也改写了 rank 和 source。

**Fix：** `_finalize_results` 只改 `source="hybrid"`，不再改 rank。

- [ ] **Step 1: 查看当前 _finalize_results**

Run: `sed -n '101,110p' agentic_graph_rag/retrieval/orchestrator.py`

- [ ] **Step 2: 修改 _finalize_results**

```python
@staticmethod
def _finalize_results(results: list[SearchResult]) -> list[SearchResult]:
    return [
        result.model_copy(update={"source": "hybrid"})
        for index, result in enumerate(results)
    ]
```

移除 `rank=index+1`（rank 由 rerank 阶段最终确定）。

- [ ] **Step 3: 验证 rerank 正确设置 rank**

Run: `grep -n "rank=" packages/rag-core/rag_core/reranker.py`
Expected: `_finalize_reranked_results` 中有 `rank=index+1`

---

## Task 5: 删除 _reflect_attempt 死代码

**Files:**
- Modify: `agentic_graph_rag/agent/langgraph_workflow.py:653-670`

**分析：**
- `_reflect_attempt` 是一个标记为"向后兼容"的包装器，但它从未在 LangGraph graph 定义中被使用。
- LangGraph 使用的是 `_evaluate_reflection_node` + `_interpret_verdict_node`。
- 这段代码永远不会执行，是死代码。

- [ ] **Step 1: 确认 _reflect_attempt 未被引用**

Run: `grep -rn "_reflect_attempt" agentic_graph_rag/`
Expected: 仅定义处（653行）和可能的测试文件

- [ ] **Step 2: 删除 _reflect_attempt 函数**

Edit langgraph_workflow.py，删除 653-670 行的 `_reflect_attempt` 函数（含 docstring）。

- [ ] **Step 3: 验证删除后无引用断裂**

Run: `grep -rn "_reflect_attempt" agentic_graph_rag/`
Expected: 无输出

---

## Task 6: 精简 SelfCorrectionState 字段

**Files:**
- Modify: `agentic_graph_rag/agent/langgraph_workflow.py:110-153`（TypedDict 定义）

**分析：**
- 当前 42 个字段，实际使用不均。
- 最小化原则：移除从未被任何节点写入的字段。
- 通过分析所有节点的 `state["xxx"]` 和 `state.get("xxx")` 访问模式确定必留字段。

**必留字段（至少被一个节点写入）：**
- `ops`, `driver`, `openai_client`, `decision`, `trace`
- `current_query`, `current_tool`, `attempt`, `tried_tools`
- `results`, `best_results`, `best_score`, `best_attempt`, `best_rank`
- `retries_used`, `reused_sources`, `executed_sources`
- `provider_results_sink`, `pending_reflection`, `last_reflection`
- `next_step`, `last_elapsed_ms`, `tool_step_logged_for_attempt`
- `total_reranks`, `max_reranks`, `rewrite_attempted`, `max_query_rewrites`
- `query_history`, `started_at_monotonic`, `time_budget_ms`
- `memory`, `stop_requested`, `final_results`
- `reflection_history`, `channel_cache`, `forced_hybrid_providers`
- `base_query`, `relevance_threshold`, `max_retries`

**当前字段 vs 使用分析：**
- `pending_reflection_signal` / `pending_reflection_threshold`：仅被 `_evaluate_reflection_node` 设置、`_interpret_verdict_node` 读取——属于同一节点的临时传递，不需要作为 State 字段，应改为函数返回值传递。
- 其他字段均有明确使用场景。

**Fix：** 将 `pending_reflection_signal` 和 `pending_reflection_threshold` 从 State 字段移除，改为 `_evaluate_reflection_node` 返回值字典中的键，由 `_interpret_verdict_node` 自行传入（不经过 State）。

- [ ] **Step 1: 分析 State 字段读写模式**

Run: `grep -n 'state\["\|state\.get("' agentic_graph_rag/agent/langgraph_workflow.py | grep -v "^#" | awk -F: '{print $1}' | sort -n | uniq -c | sort -rn | head -20`
（这会显示每行附近使用的字段，但没有行号上下文，看下面更精确的方式）

Run: `grep -n "pending_reflection_signal\|pending_reflection_threshold" agentic_graph_rag/agent/langgraph_workflow.py`

- [ ] **Step 2: 修改 TypedDict 移除临时字段**

Edit `SelfCorrectionState`，删除：
- `pending_reflection_signal: float | None`
- `pending_reflection_threshold: float | None`

- [ ] **Step 3: 修改 _evaluate_reflection_node 返回值**

在 `_evaluate_reflection_node` 返回值中增加 `signal` 和 `threshold` 字段（不写入 State）：

```python
# 在 _evaluate_reflection_node 末尾
if reflection is not None and "skip_reflection" in (getattr(reflection, 'candidate_fix_paths', []) or []):
    return {
        "pending_reflection": reflection,
        "_skip_signal": top_signal,
        "_skip_threshold": threshold,
    }
return {"pending_reflection": reflection}
```

- [ ] **Step 4: 修改 _interpret_verdict_node 读取方式**

将 `state.get("pending_reflection_signal", ...)` 改为从函数参数或闭包传递（使用一个本地 helper）。

```python
def _interpret_verdict_node(state: SelfCorrectionState) -> dict[str, Any]:
    reflection = state.get("pending_reflection")
    if reflection is None:
        return {"next_step": "finish"}

    skip_signal = state.get("_skip_signal", get_settings().agent.reflection_skip_score_threshold)
    skip_threshold = state.get("_skip_threshold", get_settings().agent.reflection_skip_score_threshold)
```

实际上，保持字段名 `pending_reflection_signal` 和 `pending_reflection_threshold` 不变，只是将它们作为"临时传递字段"而非"State 定义字段"。这样改动最小：只修改 `_evaluate_reflection_node` 返回值用 `_skip_signal`/`_skip_threshold`，在 `_interpret_verdict_node` 里改 `state.get("pending_reflection_signal")` 为 `state.get("_skip_signal")`。

更简单的方案：不做 State 精简——这两个字段是合理的"同一节点→下一节点"的临时数据传递模式。**跳过 Task 6**。

---

## 执行顺序

1. Task 1（confidence 统一）— 最小风险，先做
2. Task 2（comprehensive_search）— 依赖 tools.py 导入
3. Task 3（source 统一）— 独立
4. Task 4（orchestrator rank）— 独立
5. Task 5（删除死代码）— 独立

---

## 验证

每步修改后运行：

```bash
pytest tests/ -x -q --tb=short 2>&1 | tail -20
```

最终验证：

```bash
# 1. 无死代码
grep -rn "_reflect_attempt" agentic_graph_rag/

# 2. confidence 只在一处计算
grep -rn "calculate_answer_confidence" agentic_graph_rag/ packages/rag-core/

# 3. comprehensive_search 无 hybrid_search 调用
grep -n "hybrid_search" agentic_graph_rag/agent/tools.py | grep -v "def hybrid_search"

# 4. source 标签统一
grep -n 'source=' agentic_graph_rag/agent/tools.py | grep -vE '"(vector|bm25|graph|hybrid)"' | grep -v "#"
```