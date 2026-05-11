"""Externalized routing and retrieval keyword rules.

Keep deterministic rule vocabularies in one place so they can be reviewed,
tested, and expanded without scattering hard-coded strings across workflow code.
"""

from __future__ import annotations

import re

# TODO: Replace part of these manual keyword lists with a small local semantic
# router model once we have enough labeled routing traces for safe rollout.

RELATION_PATTERNS = [
    r"\brelat\w*\b", r"\bconnect\w*\b", r"\blink\w*\b",
    r"\bbetween\b",
    r"区别", r"差异", r"关联", r"关系", r"依赖", r"影响",
]

MULTI_HOP_PATTERNS = [
    r"\bchain\b", r"\bpath\b", r"\bcompar\w*\b", r"\bthrough\b",
    r"\bhow .+ affect\b",
    # Chinese multi-hop cues: causal chain / decision justification
    r"为什么.+不用",
    r"为什么.+要停",
    r"为什么.+选择",
    r"判断过程", r"说明判断过程", r"说明原因", r"变化趋势",
    # Medical composite queries: two sub-questions chained with "?"
    # e.g. "目标剂量是多少？滴定周期是多久？" / "A 如何？B 怎样？"
    r"[是为][^？]{0,20}？[^？]{1,30}[是为何][^？]{0,20}？",
]

GLOBAL_PATTERNS = [
    r"\ball\b", r"\bevery\b", r"\boverview\b", r"\blist\b", r"\bsummar\w*\b",
    r"\bshow all\b",
    r"需要哪些", r"有哪些", r"分类列出", r"完整", r"汇总", r"列出", r"整体",
]

TEMPORAL_PATTERNS = [
    r"\bwhen\b", r"\bdate\b", r"\btime\w*\b", r"\bhistor\w*\b",
    r"\bbefore\b", r"\bafter\b",
    r"\b\d{4}[-/]\d{2}\b",
]

RELATION_QUERY_KEYWORDS = (
    # Generic relation cues
    "关系",
    "区别",
    "差异",
    "影响",
    "依赖",
    "关联",
    "compare",
    "difference",
    "impact",
    "relation",
    "relationship",
    # Medical-domain relation cues (causal / interaction / selection / alternative)
    "导致",
    "引起",
    "引发",
    "产生",
    "相互作用",
    "禁忌",
    "禁用",
    "适应症",
    "副作用",
    "不良反应",
    "替代",
    "改用",
    "换用",
    "如何调整",
    "如何处理",
    "如何选择",
    "首选",
    "过敏",
    "耐药",
    "联用",
    "合用",
)

GLOBAL_QUERY_KEYWORDS = (
    "总结",
    "概览",
    "总览",
    "整体",
    "全部",
    "所有",
    "汇总",
    "列出",
    "哪些",
    "分类",
    "完整",
    "overview",
    "summary",
    "summarize",
    "list all",
    "show all",
)

INTERNAL_ALIAS_CONCEPT_PATTERN = re.compile(
    r"\b(semantic\s+c(ore|ompanion)|SCL|companion\s+layer)\b",
    re.IGNORECASE,
)

INTERNAL_ALIAS_GLOBAL_PATTERN = re.compile(
    r"\b(list all|describe all|all\s+\w+\s+decisions)\b",
    re.IGNORECASE,
)

# Backward-compatible aliases for existing imports. Keep internal naming neutral
# because this rule is not a product-level "bilingual mode", only an alias hint.
CROSS_LANGUAGE_DOC2_PATTERN = INTERNAL_ALIAS_CONCEPT_PATTERN
CROSS_LANGUAGE_GLOBAL_PATTERN = INTERNAL_ALIAS_GLOBAL_PATTERN
