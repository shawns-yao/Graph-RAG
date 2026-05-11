"""Query-local lexical signal extraction for retrieval planning.

This module extracts objective text forms only. It does not score anchors,
infer semantic importance, consult corpus statistics, or classify entities.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

AnchorKind = Literal["numeric", "threshold", "symbolic", "quoted", "phrase"]
AnchorSource = Literal["query", "evidence", "claim"]

_COMPARISON = r"(?:<=|>=|<|>|≤|≥|=)"
_NUMBER = r"\d+(?:\.\d+)?"
_UNIT = r"(?:μg|mcg|mg|g|ml|mL|l|L|%|次/年|次|/μL|/uL|ml/min|mL/min)"
_LATIN_TOKEN = r"[A-Za-z][A-Za-z0-9]*"
_SYMBOLIC_RE = re.compile(
    rf"{_LATIN_TOKEN}(?:[+/_.-]{_LATIN_TOKEN}|/[A-Za-z0-9]+)+",
    re.IGNORECASE,
)
_THRESHOLD_RE = re.compile(
    rf"(?P<left>{_SYMBOLIC_RE.pattern}|{_LATIN_TOKEN})\s*{_COMPARISON}\s*{_NUMBER}",
    re.IGNORECASE,
)
_CJK_COMPARISON_RE = re.compile(
    rf"(?P<left>{_SYMBOLIC_RE.pattern}|{_LATIN_TOKEN})\s*"
    rf"(?P<operator>小于等于|大于等于|不超过|不低于|至少|小于|低于|大于|高于|超过)\s*"
    rf"(?P<number>{_NUMBER})",
    re.IGNORECASE,
)
_CJK_COMPARISON_OPERATORS = {
    "小于等于": "≤",
    "不超过": "≤",
    "大于等于": "≥",
    "不低于": "≥",
    "至少": "≥",
    "小于": "<",
    "低于": "<",
    "大于": ">",
    "高于": ">",
    "超过": ">",
}
_NUMERIC_WITH_UNIT_RE = re.compile(rf"{_NUMBER}\s*{_UNIT}", re.IGNORECASE)
_FREQUENCY_RE = re.compile(rf"(?:每日|每周|每月|每年|每天)?\s*{_NUMBER}\s*次")
_CODE_ANCHOR_RE = re.compile(
    r"\b(?:[A-Z]{2,}[-_]?\d+[A-Z0-9_-]*|[A-Za-z]+-\d+[A-Za-z0-9_-]*|"
    r"\d+(?:\.\d+){1,3}|ERR[-_ ]?[A-Za-z0-9_-]+)\b",
    re.IGNORECASE,
)
_QUOTED_RE = re.compile(r"[\"“”'‘’]([^\"“”'‘’]+)[\"“”'‘’]")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
_QUESTION_SUFFIX_RE = re.compile(
    r"(是什么|是多少|有哪些|是什么|怎么办|如何|怎么|是否正确|是否|吗|呢)$"
)


class LexicalAnchor(BaseModel):
    text: str
    kind: AnchorKind
    source: AnchorSource = "query"


class QuerySignals(BaseModel):
    anchors: list[LexicalAnchor] = Field(default_factory=list)


def _normalize_space(text: str) -> str:
    return " ".join(text.strip().split())


def _add_anchor(
    anchors: list[LexicalAnchor],
    seen: set[tuple[str, str]],
    text: str,
    kind: AnchorKind,
) -> None:
    normalized = _normalize_space(text)
    if not normalized:
        return
    key = (normalized.casefold(), kind)
    if key in seen:
        return
    anchors.append(LexicalAnchor(text=normalized, kind=kind))
    seen.add(key)


def _covered_spans(spans: list[tuple[int, int]], start: int, end: int) -> bool:
    return any(start >= span_start and end <= span_end for span_start, span_end in spans)


def _strip_question_suffix(text: str) -> str:
    previous = text
    current = _QUESTION_SUFFIX_RE.sub("", previous)
    while current != previous:
        previous = current
        current = _QUESTION_SUFFIX_RE.sub("", previous)
    return current.strip(" ，,。？?:：")


def _normalized_cjk_threshold(match: re.Match[str]) -> str:
    operator = _CJK_COMPARISON_OPERATORS[match.group("operator")]
    return f"{match.group('left')} {operator} {match.group('number')}"


def extract_query_signals(query: str) -> QuerySignals:
    """Extract query-local anchor forms without semantic scoring."""
    anchors: list[LexicalAnchor] = []
    seen: set[tuple[str, str]] = set()
    protected_spans: list[tuple[int, int]] = []

    for match in _QUOTED_RE.finditer(query):
        _add_anchor(anchors, seen, match.group(1), "quoted")
        protected_spans.append(match.span())

    for match in _THRESHOLD_RE.finditer(query):
        _add_anchor(anchors, seen, match.group(0), "threshold")
        protected_spans.append(match.span())

    for match in _CJK_COMPARISON_RE.finditer(query):
        _add_anchor(anchors, seen, _normalized_cjk_threshold(match), "threshold")
        protected_spans.append(match.span())

    for match in _SYMBOLIC_RE.finditer(query):
        if _covered_spans(protected_spans, *match.span()):
            _add_anchor(anchors, seen, match.group(0), "symbolic")
            continue
        _add_anchor(anchors, seen, match.group(0), "symbolic")
        protected_spans.append(match.span())

    for pattern in (_NUMERIC_WITH_UNIT_RE, _FREQUENCY_RE):
        for match in pattern.finditer(query):
            _add_anchor(anchors, seen, match.group(0), "numeric")
            protected_spans.append(match.span())

    for match in _CODE_ANCHOR_RE.finditer(query):
        if _covered_spans(protected_spans, *match.span()):
            continue
        _add_anchor(anchors, seen, match.group(0), "symbolic")
        protected_spans.append(match.span())

    for match in _CJK_RUN_RE.finditer(query):
        if _covered_spans(protected_spans, *match.span()):
            continue
        phrase = _strip_question_suffix(match.group(0))
        if phrase:
            _add_anchor(anchors, seen, phrase, "phrase")

    return QuerySignals(anchors=anchors)


def has_strong_form_anchor(signals: QuerySignals) -> bool:
    return any(
        anchor.kind in {"numeric", "threshold", "symbolic", "quoted"}
        for anchor in signals.anchors
    )
