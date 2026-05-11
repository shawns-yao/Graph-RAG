# Retrieval Planning Redesign

## 背景

当前链路中有几类规则会把 query 的表面形态直接映射到工具或 prompt 策略：

```text
短 query -> vector_search
长 query -> comprehensive_search
temporal regex -> temporal_query
global/relation/multihop regex -> 固定 query_type / 工具
current_tool + failure_type -> retry matrix next_tool
query 长度 -> prompt chunk 上限
```

这些规则的问题不是有 `if/else`，而是它们把语义判断写死成了字符串形态判断。短问题也可能需要精确数值证据，长问题也可能只是带背景的单事实查询。继续堆规则会让系统变成补丁池。

目标是把链路从：

```text
query shape -> tool
current tool -> next tool
query length -> prompt size
```

改成：

```text
objective query signals -> retrieval plan
evidence gap -> correction action
required evidence -> evidence pack
```

## 设计原则

1. Signal Extractor 只观察，不决策。
2. Router 暂时保留主通道，不被 signals 覆盖。
3. Signals 只决定是否加伴随通道。
4. 不打分，不用长度阈值模拟语义。
5. 不维护 generic term 列表来判断重要性。
6. Query anchors 和 evidence anchors 分开。
7. Verification planner 只在固定工具集合中选补证据工具，不判断事实真假。
8. Regenerate 必须有门槛：有新证据 + core claim gap。
9. 先改执行层，再改决策层。

## 当前问题

### 1. Router 层硬规则过多

当前 router 同时承担：

```text
query type 判断
初始工具选择
部分 query shape 规则兜底
```

问题：

```text
短/长 query 不是检索策略
regex 覆盖不了开放问法
query_type 和 tool 绑定过死
后续 self-correction 被迫纠偏前面的工具选择
```

目标：Router 先保留现状，但逐步降级为主通道建议。未来 Router 不直接决定最终工具集合。

### 2. Self-correction 层矩阵味重

当前主要由这些结构决定下一步工具：

```text
_RETRY_TOOL_MATRIX
_REFLECTION_RULES
_QUERY_TYPE_TOOL_HINTS
_get_contextual_fallback_tools
_get_next_tool
```

问题：

```text
工具选择逻辑分散
failure_type 到 tool 的映射写死
current_tool -> next_tool 有线性链味道
provider diagnostics 没有成为主要依据
```

目标：后续改成 EvidenceState -> RetryPlan，按缺什么证据选工具，不按当前工具线性跳转。

### 3. Verification planner 方向正确但还可增强

当前 verification planner 已经比较干净：

```text
verify_answer 发现 unsupported claims
build_gap_report 生成 gap
plan_correction 让 LLM 在 allowlist 工具中选下一步
execute_correction_tool 执行补证据
```

保留原则：

```text
LLM 只选工具，不判断事实真假
工具必须 allowlist
非法 JSON / 非法工具走 deterministic fallback
最多 retry 一次
```

后续增强方向：从 single-tool correction plan 升级为 multi-tool correction plan，但仍受预算限制。

### 4. 生成层不应按 query 长短裁剪证据

当前生成层中存在类似：

```text
短 query -> 少放 chunks
enumeration query -> 另一套 chunks/char 上限
```

问题：短 query 也可能需要多个关键证据。例如：

```text
eGFR<30怎么办？
GOLD 2级怎么治？
18 μg是什么药的剂量？
FEV1/FVC<0.70说明什么？
```

目标：用 Evidence Pack Builder 替代 query-length prompt 裁剪。先 pin 必要证据，再按预算裁剪。

## Anchor 设计

### Query anchors 和 evidence anchors 分开

不能把 query 里出现的检索锚点和答案/证据里出现的事实锚点混在一起。

```text
query_anchors:
  从用户 query 提取，用于初始检索计划和 focus query。

evidence_anchors:
  从 retrieved evidence / generated answer / verification claims 提取，
  用于 verification gap 和二次补证据。
```

例子：

```text
query: 噻托溴铵剂量是多少？
query_anchors: 噻托溴铵

generated answer: 噻托溴铵 18 μg 每日1次
evidence_anchors / claim_anchors: 噻托溴铵, 18 μg, 每日1次
```

`18 μg` 和 `每日1次` 不应该强塞进 Query Understanding，因为 query 里没有。它们应由 verifier / gap report 产生。

### Signal Extractor 不判断语义重要性

不做：

```text
specificity_score
score >= 1.0
len(text) >= 4
len(text) >= 6
generic term 扣分列表
```

原因：这些都是用数字或手写列表模拟语义判断，会变成新补丁。

Signal Extractor 只提取客观形态：

```text
numeric
threshold
symbolic
quoted
phrase
```

实现方式用 tokenization + 字符类型分类，不用领域词典和泛化正则池。

推荐流程：

```text
1. 对 query 做轻量 tokenization，保留中文片段、拉丁缩写、数字、单位、符号。
2. 对 token 或相邻 token 组合做字符构成判断。
3. 含比较符号和数值的组合归为 threshold。
4. 含医学缩写斜杠、加号、希腊字母、单位符号的组合归为 symbolic。
5. 含数字和单位/频率结构的组合归为 numeric。
6. 其余连续文本片段归为 phrase。
```

注意：`每日1次` 这种表达虽然含汉字，但在检索和 pin 场景里应归为
numeric/frequency，而不是普通 phrase。P1 测试必须覆盖每个验证例，确认 kind
归类和预期一致。

### Anchor kind

建议结构：

```python
class LexicalAnchor(BaseModel):
    text: str
    kind: Literal[
        "numeric",
        "threshold",
        "symbolic",
        "quoted",
        "phrase",
    ]
    source: Literal["query", "evidence", "claim"]
```

不包含：

```text
score
required
entity_type
confidence
```

### 强形态 anchor

以下 kind 是强形态 anchor：

```text
numeric
threshold
symbolic
quoted
```

它们只表示“字面形态强”，不是语义重要性高。

例子：

```text
18 μg                  -> numeric
每日1次                -> numeric
FEV1/FVC < 0.70        -> threshold + symbolic
eGFR < 30              -> threshold
LABA+LAMA+ICS          -> symbolic
"18 μg每日1次"         -> quoted
```

### Phrase anchor

普通中文短语统一作为 phrase：

```text
噻托溴铵
嗜酸性粒细胞
诊断标准
给药频率
用药原则
治疗目标
```

P1 不判断它们 required / not required。Phrase anchor 可以用于：

```text
focus query
verification gap query building
normal rerank feature
```

但不能单独触发 initial bm25 companion。

P1 不区分实体型 phrase 和功能型 phrase。这个判断信息不足，过早做会退回到
stopword / generic term 补丁。

生产级处理方式是把 phrase 的重要性判断后移：

```text
检索阶段保证实体覆盖
Evidence Pack 不对普通 phrase 做强 pin
普通 phrase 对应 evidence 走正常 rerank / diversity
强 pin 只给 strong-form anchors 和 correction retrieval 新证据
```

phrase anchor 用于 focus query 时，只能拼进检索 query 文本，不作为过滤条件。
实现上要避免把功能词大量拼入 focus query；P1 只保留 tokenization 后的候选片段，
P3 依赖 rerank 和 provider 结果决定是否进入 prompt。

也就是说，pin 机制不是用来补救检索遗漏的。它只防止已经召回的关键数值、
阈值、符号组合或纠错证据被 prompt budget 裁掉。

## P2 Initial Retrieval 并行策略

### 核心约束

Router 决定主通道，signals 只决定是否加伴随通道。

```text
router.suggested_tool 永远保留
signals 不覆盖 router
max_initial_tools = 2
```

### Companion tool 规则

P2 只让强形态 anchor 触发 `bm25_search`：

```text
if query_anchors contains numeric / threshold / symbolic / quoted:
    add bm25_search
```

Phrase anchor 不触发 bm25：

```text
噻托溴铵剂量是多少？
-> phrase anchor: 噻托溴铵
-> 不因为 phrase 加 bm25
```

建议伪逻辑：

```python
def build_initial_tools(router_tool, signals):
    tools = [router_tool]

    if signals.has_strong_form_anchor:
        add_if_not_present("bm25_search")

    elif signals.has_multiple_constraints and router_tool != "cypher_traverse":
        add_if_not_present("cypher_traverse")

    elif router_tool in {"bm25_search", "cypher_traverse"}:
        add_if_not_present("vector_search")

    return tools[:2]
```

注意：这里没有让 signals 覆盖 router。P4 之前不改变主决策。

### 示例

```text
FEV1/FVC < 0.70
router: vector_search
signals: threshold/symbolic
plan: vector_search + bm25_search
```

```text
噻托溴铵剂量是多少？
router: vector_search
signals: phrase
plan: vector_search
```

```text
噻托溴铵 18 μg 每日1次 是否正确？
router: vector_search
signals: phrase + numeric
plan: vector_search + bm25_search
```

```text
GOLD 3-4级且嗜酸性粒细胞≥100/μL推荐什么？
router: cypher_traverse
signals: threshold/symbolic/multiple_constraints
plan: cypher_traverse + bm25_search
```

```text
列出所有入院诊断
router: comprehensive_search
signals: enumeration only
plan: comprehensive_search
```

## Evidence Pack Builder

目标：替代“按 query 长短裁剪 prompt”的逻辑。

证据选择顺序：

```text
1. pin 包含强形态 query anchors 的 evidence
2. pin correction retrieval 新增 evidence
3. pin verifier 标记的 core claim evidence
4. 其余按 rerank / score 填充
5. 最后按 token budget 裁剪
```

原则：

```text
先保证关键证据不丢，再考虑 prompt budget。
```

短 query 不应天然少放 evidence。长 query 也不应天然多放 evidence。

## Evidence anchors 生成位置

Evidence anchors 不在 Signal Extractor 里生成。

它们应在 Verifier / Gap Report 阶段生成：

```text
generated answer
  -> claim extraction
  -> claim anchors
  -> verify against evidence
  -> EvidenceGap
  -> verification planner
```

建议结构：

```python
class EvidenceGap(BaseModel):
    claim_text: str
    claim_role: Literal["core", "supporting", "supplemental"]
    gap_type: str
    anchors: list[LexicalAnchor]  # source="claim"
    missing_anchor_texts: list[str]
    can_change_answer: bool
```

当前可以先把 anchors 挂在 correction planner 的 gap report 上；未来再迁到统一 EvidenceGap 模型。

## Claim role 判定

Regenerate gate 依赖 `core claim`，所以 claim role 不能让 LLM 自由发挥。

生产级做法是结构启发式 + query 对齐，必要时让 LLM 在窄标准内辅助标注：

```text
core:
  直接回答用户问题的事实声明。
  通常来自答案第一段、结论句、列表项主干。
  与 query anchor / 用户问题目标直接对齐。

supporting:
  支撑 core 的条件、来源、适用范围、限定说明。
  错了会影响严谨性，但通常不改变主答案。

supplemental:
  背景解释、泛化科普、非用户直接询问的扩展信息。
```

推荐判定顺序：

```text
1. 结构位置：答案首句、直接结论、列表主项优先 core。
2. Query 对齐：包含 query anchors 且回答 query 目标的 claim 才能是 core。
3. 强形态保护：包含 numeric / threshold / symbolic 的 claim，若与 query 目标相关，
   至少是 supporting；直接回答问题时是 core。
4. LLM 只在 claim extraction prompt 里按上述标准输出 role，不单独调用 role classifier。
5. 无法判断时降级为 supporting，不升级为 core。
```

claim extraction prompt 必须包含反例，防止 LLM 把所有 claim 都标成 core：

```text
背景解释不是 core。
泛化科普不是 core。
患者一般信息不是 core，除非用户问题直接询问该信息。
来源、注意事项、适用范围通常是 supporting，不是 core。
```

fallback 必须 deterministic：

```text
role 不在枚举值内 -> supporting
JSON 解析失败 -> 使用原有 claim text，role=supporting
claim text 为空 -> 丢弃该 claim
```

这不是语义打分，而是可解释的离散规则。role 错误会影响 regenerate gate，
因此宁可少 regenerate，也不要把 supplemental 错判成 core 导致反复重写。

## Regenerate Gate

不能 verification gap 就 regenerate。

Regenerate 触发条件：

```text
1. 有 correction retrieval 新证据
2. gap 属于 core claim
3. 新证据命中 gap.missing_anchor_texts 中至少一个
```

不要单独调用 LLM 判断“新证据是否可能改变答案”。这个判断成本接近直接
regenerate，而且会引入新的不确定性。用 missing anchor 命中作为 deterministic
条件即可。

不触发 regenerate：

```text
planner retry 返回 0 results
answer 已明确说证据不存在
supplemental claim 缺证据
只是解释性句子 unsupported
预算不足
```

示例：

```text
eGFR < 30 题：
planner 选择 bm25_search，但返回 0 results。
=> 不 regenerate，返回 partial。
```

```text
噻托溴铵剂量答错：
答案说 12 μg，每日2次；补证据找到 18 μg每日1次。
=> regenerate。
```

## 延迟预算

目标：

```text
普通问题 P95 < 15s
复杂/纠错问题 P95 < 35s
hard timeout 45s
```

约束：

```text
max_initial_tools = 2
max_correction_tools = 2
max_regenerations = 1
max_llm_calls = 4
correction planner 最多触发一次
```

P1 就要加入显式 BudgetTracker，贯穿整个 workflow。每次 LLM 调用前先检查预算，
超限直接返回 partial / stop reason，不进入补救链路。

LLM 调用计数：

```text
generate answer: 1
claim extraction / verification: 1
correction planner: 1
regenerate: 1
```

planner JSON 解析失败后的 deterministic fallback 不增加 LLM 计数。预算计数必须按
“实际发起 LLM 请求”记录，不能按节点是否执行记录。

Fast path 默认：

```text
Signal Extractor: 0 LLM
Initial Retrieval: 1-2 tools parallel
Generate: 1 LLM
Verifier: claim extraction + hard checks
```

Worst path 必须受预算截断：

```text
Understanding / planner 不得无限串联
没有新证据不得 regenerate
超预算返回 partial + trace reason
```

## 实施状态

```text
Verification Correction Planner:
  status: completed
  completed_at: 2026-05-11
  commit: 324dca2 Add verification correction planner

P1 Query Signal Extractor:
  status: completed
  completed_at: 2026-05-11
  commit: f61fbdb Add query signal extraction

P2/P3 Initial Retrieval + Evidence Pack Builder:
  status: completed
  completed_at: 2026-05-11
  commit: ac16640 Add companion retrieval and evidence packing

Workflow LLM BudgetTracker:
  status: completed
  completed_at: 2026-05-11
  commit: 93a41ab Add workflow LLM budget tracker

Budget Enforcement In Agent Nodes:
  status: completed
  completed_at: 2026-05-11
  commit: a0193cc Enforce workflow LLM budget in agent nodes

P6 Regenerate Gate - New Evidence Required:
  status: completed
  completed_at: 2026-05-11
  commit: 604b4d2 Gate verification regeneration on new evidence

Claim Role Extraction + Fallback:
  status: completed
  completed_at: 2026-05-11
  commit: bca1b70 Add claim role extraction fallback

P6 Regenerate Gate - Core Claims Only:
  status: completed
  completed_at: 2026-05-11
  commit: eb1d988 Gate verification correction on core claims

P5 Gap-Based RetryPlan:
  status: completed
  completed_at: 2026-05-11
  commit: d21dc34 Add gap-based retrieval retry plan
```

## 落地顺序

### P1: Query Signal Extractor

新增：

```text
agentic_graph_rag/agent/query_signals.py
```

只做 query-local 形态提取：

```text
numeric
threshold
symbolic
quoted
phrase
```

不做：

```text
score
required
IDF
corpus hit count
LLM extraction
```

验证点：

```text
FEV1/FVC < 0.70 -> threshold/symbolic
18 μg 每日1次 -> numeric
eGFR < 30 -> threshold
噻托溴铵 -> phrase
诊断标准 -> phrase, 不触发 bm25
```

### P2/P3: Initial Retrieval 并行 + Evidence Pack Builder

P2 和 P3 作为一个迭代交付，不拆成两个可上线中间态。

原因：并行检索拿到的新证据如果仍被旧 prompt 裁剪丢掉，测试会误判为
“并行检索无效”。执行层和 pack 层要一起闭环。

保留 router 主工具，增加 companion tools。

```text
router_tool + bm25_search only when strong form anchors exist
max_initial_tools = 2
```

同时替换 query-length prompt 裁剪。

验证点：

```text
FEV1/FVC 查询初始工具包含 vector_search + bm25_search
噻托溴铵普通剂量查询不因 phrase 自动加 bm25
eGFR < 30 查询初始工具包含 router_tool + bm25_search
强形态 anchor 对应证据不会因短 query 被裁掉
correction retrieval 新证据会被 pin 到 prompt
```

### P4: Router 去 tool 化

把 Router 从工具决策逐步降级成 query intent /主通道建议。

验证点：

```text
旧 suggested_tool 兼容
新 RetrievalPlan 成为实际执行依据
```

### P5: Self-correction 矩阵收敛

把：

```text
current_tool -> next_tool
```

改成：

```text
evidence_gap -> retrieval_need -> retry_plan
```

验证点：

```text
_RETRY_TOOL_MATRIX 不再主导工具切换
provider diagnostics 进入 RetryPlan 输入
```

### P6: Regenerate Gate

加入：

```text
new evidence + core claim gap 才 regenerate
```

验证点：

```text
retry 返回 0 results 不 regenerate
core numeric contradiction + 新证据 regenerate
supplemental unsupported 不 regenerate
```

## 明确不做

P1 阶段不做：

```text
LLM query understanding
IDF required 判断
长度阈值 required 判断
generic term 列表
phrase anchor 触发 bm25
```

P2 阶段不做：

```text
signals 覆盖 router 主工具
fanout 超过 2 个初始工具
```

P3 阶段不做：

```text
用 query 长短决定 evidence 数量
```

## 最终结论

本方案保留现有系统的稳定部分：

```text
确定性主流程
LLM reflection
CoVe verification
verification correction planner
```

同时逐步替换不稳定部分：

```text
query shape -> tool
current_tool -> next_tool matrix
query length -> prompt size
```

最终目标是：

```text
客观形态信号决定检索覆盖约束
证据缺口决定纠错动作
证据重要性决定 prompt pack
LLM 只在窄输入、窄输出的位置做规划
```
