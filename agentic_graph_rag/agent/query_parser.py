"""Query structure parsing for graph-aware retrieval.

Decomposes a natural-language query into structured slots that downstream
graph traversal can use as constraints. This prevents the graph from
doing unconstrained BFS from generic entry points.

Design:
- One LLM call per query (uses mini model for speed).
- Falls back to regex heuristics if LLM fails.
- Output is a frozen dataclass consumed by vector_cypher and providers.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rag_core.config import get_settings, make_openai_client

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)

_PARSE_PROMPT = """You are a medical query structure parser for a Graph RAG system.
Extract structured slots from the query. Return ONLY valid JSON.

Slots:
- background_entities: diseases/conditions that set the clinical context (e.g. "糖尿病", "高血压")
- focus_entities: the specific drugs/tests/procedures the question is ABOUT (e.g. "二甲双胍", "ACEI")
- constraints: numeric thresholds, conditions, or qualifiers (e.g. "eGFR<30", "≥3分", "青霉素过敏")
- relation_intent: what relationship the user wants (one of: 诊断/推荐/禁忌/替代/疗程/剂量/副作用/比较/目标值/处理/机制/预后)
- target_slot: what type of answer is expected (e.g. "药物名", "数值阈值", "时长", "步骤列表", "原因")

Example:
Query: "eGFR小于30时二甲双胍应该如何调整？"
Output:
{
  "background_entities": ["eGFR"],
  "focus_entities": ["二甲双胍"],
  "constraints": ["eGFR<30"],
  "relation_intent": "禁忌",
  "target_slot": "用药建议"
}

Example:
Query: "PCI术后使用药物洗脱支架的患者，双抗治疗应该持续多久？"
Output:
{
  "background_entities": ["PCI", "药物洗脱支架"],
  "focus_entities": ["双抗治疗", "DAPT"],
  "constraints": ["PCI术后", "药物洗脱支架"],
  "relation_intent": "疗程",
  "target_slot": "时长"
}
"""

# Relation intent keywords for regex fallback.
_INTENT_PATTERNS: list[tuple[str, str]] = [
    (r"禁忌|禁用|不能用|不宜", "禁忌"),
    (r"首选|推荐|应该用|选择", "推荐"),
    (r"替代|改用|换用|过敏", "替代"),
    (r"疗程|持续|多久|多长", "疗程"),
    (r"剂量|目标剂量|滴定|多少mg", "剂量"),
    (r"副作用|不良反应|导致|引起", "副作用"),
    (r"比较|优势|劣势|区别|差异", "比较"),
    (r"目标|目标值|控制目标|范围", "目标值"),
    (r"如何处理|怎么办|处理", "处理"),
    (r"诊断|标准|诊断标准", "诊断"),
    (r"机制|原理|为什么", "机制"),
]

# CJK + Latin token extraction for entity candidates.
_ENTITY_TOKEN_RE = re.compile(
    r"[\u4e00-\u9fff]{2,}|[A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][a-z]+)*|[A-Z]{2,}[-]?\d*"
)
_CONSTRAINT_RE = re.compile(
    r"(?:[<>≤≥]=?\s*\d+(?:\.\d+)?)|(?:\d+(?:\.\d+)?\s*(?:mg|ml|mmol|pg|g/L|%|次|分|期|级))"
)
_BACKGROUND_STOPWORDS = {
    "患者", "治疗", "使用", "应该", "如何", "什么", "多少", "哪些",
}


@dataclass(frozen=True, slots=True)
class QueryStructure:
    """Parsed query structure for constrained graph retrieval."""

    background_entities: tuple[str, ...] = ()
    focus_entities: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    relation_intent: str = ""
    target_slot: str = ""

    @property
    def all_entities(self) -> list[str]:
        """All entities (focus first, then background) for entry-point lookup."""
        seen: set[str] = set()
        result: list[str] = []
        for entity in [*self.focus_entities, *self.background_entities]:
            lower = entity.lower()
            if lower not in seen:
                seen.add(lower)
                result.append(entity)
        return result

    @property
    def has_constraints(self) -> bool:
        return bool(self.constraints)

    @property
    def expansion_terms(self) -> tuple[str, ...]:
        """Terms that graph expansion is allowed to stay close to."""
        terms: list[str] = []
        seen: set[str] = set()
        for term in [*self.focus_entities, *self.background_entities, *self.constraints]:
            normalized = term.strip().lower()
            if len(normalized) < 2 or normalized in seen:
                continue
            seen.add(normalized)
            terms.append(term.strip())
        return tuple(terms)

    @property
    def seed_terms(self) -> tuple[str, ...]:
        """High-priority terms for entry-point discovery."""
        terms: list[str] = []
        seen: set[str] = set()
        for term in [*self.focus_entities, *self.constraints, *self.background_entities]:
            normalized = term.strip().lower()
            if len(normalized) < 2 or normalized in seen:
                continue
            seen.add(normalized)
            terms.append(term.strip())
        return tuple(terms)


def parse_query_structure(
    query: str,
    openai_client: OpenAI | None = None,
) -> QueryStructure:
    """Parse query into structured slots via LLM, with regex fallback."""
    cfg = get_settings()
    if openai_client is None:
        openai_client = make_openai_client(cfg)

    # Try LLM parsing first.
    try:
        result = _parse_via_llm(query, openai_client, cfg.openai.llm_model_mini)
        if result is not None:
            return result
    except Exception as exc:
        logger.warning("LLM query parsing failed (%s); using regex fallback", exc)

    # Regex fallback.
    return _parse_via_regex(query)


def _parse_via_llm(query: str, client: OpenAI, model: str) -> QueryStructure | None:
    """One LLM call to extract query structure."""
    prompt = f"{_PARSE_PROMPT}\n\nQuery: {query}"
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    raw = (response.choices[0].message.content or "").strip()
    payload = _extract_json(raw)
    if not payload:
        return None

    return QueryStructure(
        background_entities=tuple(_str_list(payload.get("background_entities"))),
        focus_entities=tuple(_str_list(payload.get("focus_entities"))),
        constraints=tuple(_str_list(payload.get("constraints"))),
        relation_intent=str(payload.get("relation_intent") or "").strip(),
        target_slot=str(payload.get("target_slot") or "").strip(),
    )


def _parse_via_regex(query: str) -> QueryStructure:
    """Deterministic fallback: extract entities, constraints, intent from query."""
    # Extract entity candidates.
    tokens = _ENTITY_TOKEN_RE.findall(query)
    entities = [
        t.strip() for t in tokens
        if t.strip() and t.strip() not in _BACKGROUND_STOPWORDS and len(t.strip()) >= 2
    ]

    # Extract constraints.
    constraints = _CONSTRAINT_RE.findall(query)

    # Detect relation intent.
    relation_intent = ""
    for pattern, intent in _INTENT_PATTERNS:
        if re.search(pattern, query):
            relation_intent = intent
            break

    # Heuristic: first entity is usually focus, rest are background.
    focus = entities[:2] if entities else []
    background = entities[2:] if len(entities) > 2 else []

    return QueryStructure(
        background_entities=tuple(background),
        focus_entities=tuple(focus),
        constraints=tuple(constraints),
        relation_intent=relation_intent,
        target_slot="",
    )


def _extract_json(text: str) -> dict | None:
    """Tolerant JSON extraction."""
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _str_list(value: object) -> list[str]:
    """Coerce to list of non-empty strings."""
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
