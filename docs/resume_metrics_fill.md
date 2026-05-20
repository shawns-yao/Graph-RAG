# 简历指标填充与实测口径

以下数据分为两类：

- 已可填：已有 Neo4j 实图、gold set、benchmark 或 live smoke 支撑。
- 待 live 全量：必须完整调用 Neo4j + LLM 跑完 30 题后才能填，当前 LLM provider 返回 `model_cooldown`，不能伪造。

## 可直接填入版本

**设计骨架图谱抽取流程**：针对全文档 LLM 抽取成本高、共现式关系构建假阳性率达 **66.67%**、无法处理万级以上医疗文档的问题，采用核心 - 长尾分层抽取策略，结合 KNN + PageRank 文档块权重排序、LLM 实体关系验证与轻量级实体对齐算法；在保持 **100.00%** gold entity 覆盖度（48/48，仅作小样本实测，不作为 headline）的前提下，离线 chunk 成本代理指标降低 **68.75%**，live LLM prompt token 成本降低 **69.68%**；实体抽取判定准确率当前为 **74.36%**，live skeleton-only LLM 抽取准确率为 **65.38%**，未证明“提升”，因此简历不建议写“实体抽取准确率提升至 xx%”。

**创新双层图谱建模结构**：针对传统单层图谱假阳性高、多跳推理“关系爆炸”、无法证据溯源的问题，设计 PhraseNode（实体层）+ PassageNode（段落层）双层图结构，通过 `MENTIONED_IN` 证据关联边、`RELATED_TO` 实体关系边与低信息实体过滤机制；在 30 条关系硬负例上，关系假阳性率由共现 baseline 的 **66.67%** 降至 **33.33%**，降低 **33.34 个百分点**；24 条多跳任务中，3 跳图遍历准确率由 1-hop baseline 的 **45.83%** 提升至 **75.00%**，提升 **29.17 个百分点**。

**实现三路融合检索与动态路由**：针对单通道向量检索无法适配不同类型查询、复杂多跳问题准确率仅 **25.00%**（vector-like evidence answerability，8 条 multi-hop 问题）的问题，实现多通道智能检索架构，支持基于 query type 的智能路由、vector / BM25 / graph 三路检索引擎与增量通道刷新统一 rerank；30 题医疗检索证据可回答率由最佳单通道 **56.67%** 提升至融合检索 **63.33%**，提升 **6.66 个百分点**。复杂多跳问题当前融合 evidence answerability 仍为 **25.00%**，未验证提升，不建议写“复杂多跳问题准确率提升 xx%”。

**构建自循环纠错与验证机制**：针对传统 RAG 幻觉率高、答案不可靠的核心痛点，基于 LangGraph 构建检索自循环，实现 LLM 召回质量评估、查询改写与工具切换、CoVe 答案事实逐条图谱验证；10 条固定 trace eval 中，**40.00%** 的问题自动触发补充召回或工具切换，CoVe 产出 22 条 atomic claims 且全部有证据支持。由于该固定集出现 **100.00% verified / 0.00% hallucination proxy**，不建议写入简历 headline；真实“幻觉率从 xx% 降至 xx%、答案可信度提升 xx%”必须跑 live LLM judge 全量实验后再填。

## 推荐简历写法

**设计骨架图谱抽取流程**：针对全文档 LLM 抽取成本高、共现式关系构建噪声高、难以扩展到大规模医疗文档的问题，采用核心 - 长尾分层抽取策略，结合 KNN + PageRank 文档块权重排序、LLM 实体关系验证与轻量级实体对齐算法；在本地医疗 benchmark 上，深度抽取 chunk 由 16 个降至 5 个，live LLM prompt token 成本降低 **69.68%**，实体图谱覆盖较 skeleton-only baseline 提升 **4.17 个百分点**。

**创新双层图谱建模结构**：针对传统单层图谱假阳性高、多跳推理关系爆炸、无法证据溯源的问题，设计 PhraseNode（实体层）+ PassageNode（段落层）双层图结构，通过 `MENTIONED_IN` 证据关联边、`RELATED_TO` 实体关系边与低信息实体过滤机制；关系假阳性率由共现 baseline 的 **66.67%** 降至 **33.33%**，24 条多跳任务中 3 跳图遍历准确率达到 **75.00%**，较 1-hop baseline 提升 **29.17 个百分点**。

**实现三路融合检索与动态路由**：针对单通道向量检索无法适配不同类型查询的问题，实现基于 query type 的智能路由、vector / BM25 / graph 三路检索引擎与增量通道刷新统一 rerank；30 题医疗检索证据可回答率由最佳单通道 **56.67%** 提升至 **63.33%**，提升 **6.66 个百分点**。

**构建自循环纠错与验证机制**：基于 LangGraph 构建检索自循环，实现召回质量评估、查询改写、工具切换与 CoVe 答案事实逐条图谱验证；10 条固定 trace eval 中 **40.00%** 的问题自动触发补充召回或工具切换，22 条 atomic claims 均完成证据级验证。幻觉率与答案可信度已设计 live LLM judge 实验，待 provider 恢复后填入全量实测值。

## 待补全的真实实验

脚本：

```powershell
.venv\Scripts\python.exe scripts\evaluate_resume_metrics_live.py `
  --sections extraction,qa `
  --modes vector_search,cypher_traverse,hybrid_search,agent_pattern `
  --output test\medical_benchmark\results\resume_live_metrics.json
```

实验会真实调用：

- Neo4j：读取 `RagChunk / PhraseNode / PassageNode`，执行 vector / BM25 / graph / hybrid 检索。
- LLM extraction：对 full-document chunks 与 skeleton chunks 分别抽取实体，计算 prompt token 成本、实体覆盖、实体准确率。
- LLM generation：对 30 题生成答案。
- LLM judge：对照 `questions_master.json` 的 gold answer，输出 `answer_score / hallucination / confidence_score`。
- CoVe：从 pipeline trace 读取 `claims_total / claims_supported / claims_incorrect`，计算事实支持率与错误声明率。

当前阻塞：

```text
LLM provider 返回 model_cooldown，gemini-3-flash-preview 需要约 8 小时恢复。
因此不填“幻觉率从 xx% 降至 xx%、答案可信度提升 xx%”。
```
