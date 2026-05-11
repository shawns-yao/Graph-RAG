"""Corpus-driven phrase candidates for graph entity extraction.

This module generates candidates only. It does not decide final entity truth;
LLM validation and conservative alias merge remain downstream responsibilities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag_core.models import Chunk

from agentic_graph_rag.text_signals import build_tfidf_profile, rank_keywords

_PAREN_ALIAS_RE = re.compile(
    r"(?P<label>[\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9+\-/. ]{1,40})"
    r"[（(](?P<alias>[A-Za-z][A-Za-z0-9+\-/.]{1,30})(?:[，,、][^）)]*)?[）)]"
)
_HEADING_LINE_RE = re.compile(r"^(?P<label>[^：:\n]{2,40})[：:]$")
_LIST_HEAD_RE = re.compile(r"^[-*•]\s*(?P<label>[^：:\n]{2,40})[：:]")
_LOW_VALUE_TERMS = {
    "治疗",
    "患者",
    "使用",
    "剂量",
    "检查",
    "诊断",
    "标准",
    "风险",
    "目标",
    "管理",
    "方案",
}


@dataclass(frozen=True, slots=True)
class PhraseCandidate:
    phrase: str
    source_chunk: str
    sources: tuple[str, ...] = field(default_factory=tuple)
    aliases: tuple[str, ...] = field(default_factory=tuple)


def mine_phrase_candidates(chunks: list[Chunk], *, max_per_chunk: int = 12) -> list[PhraseCandidate]:
    """Generate entity candidates from structure and TF-IDF signals."""
    if not chunks:
        return []

    profile = build_tfidf_profile([chunk.enriched_content for chunk in chunks])
    merged: dict[tuple[str, str], PhraseCandidate] = {}

    def add(chunk: Chunk, phrase: str, source: str, aliases: tuple[str, ...] = ()) -> None:
        cleaned = _clean_phrase(phrase)
        if not _valid_candidate(cleaned):
            return
        key = (chunk.id or "", cleaned.casefold())
        existing = merged.get(key)
        if existing is None:
            merged[key] = PhraseCandidate(
                phrase=cleaned,
                source_chunk=chunk.id or "",
                sources=(source,),
                aliases=tuple(alias for alias in aliases if alias),
            )
            return
        merged[key] = PhraseCandidate(
            phrase=existing.phrase,
            source_chunk=existing.source_chunk,
            sources=tuple(dict.fromkeys([*existing.sources, source])),
            aliases=tuple(dict.fromkeys([*existing.aliases, *aliases])),
        )

    for chunk in chunks:
        text = chunk.enriched_content
        for phrase, alias in _paren_aliases(text):
            add(chunk, phrase, "parenthetical_alias", (alias,))
            add(chunk, alias, "parenthetical_alias")

        for line in text.splitlines():
            stripped = line.strip()
            heading = _HEADING_LINE_RE.match(stripped)
            if heading:
                add(chunk, heading.group("label"), "heading")
            list_head = _LIST_HEAD_RE.match(stripped)
            if list_head:
                add(chunk, list_head.group("label"), "list_head")

        for keyword in rank_keywords(
            text,
            profile,
            min_idf=1.2,
            max_keywords=max_per_chunk,
            max_cjk_ngram=8,
        ):
            add(chunk, keyword.term, "tfidf")

    return list(merged.values())


def _paren_aliases(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for match in _PAREN_ALIAS_RE.finditer(text):
        label = _clean_phrase(match.group("label"))
        alias = _clean_phrase(match.group("alias"))
        if label and alias:
            pairs.append((label, alias))
    return pairs


def _clean_phrase(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip(" ,.;:()[]{}，。；：、")
    cleaned = re.sub(r"^(?:[-*•]+|[—–]+|>)+\s*", "", cleaned)
    cleaned = re.sub(r"\s*(?:[-*•]+|[—–]+|>)+$", "", cleaned)
    cleaned = re.sub(r"^(?:--+|[-=]+>)\s*", "", cleaned)
    cleaned = re.sub(r"\s*(?:--+|[-=]+>)$", "", cleaned)
    cleaned = re.sub(r"^(?:和|与|及|或|在|对|为|是|应|需)+", "", cleaned)
    cleaned = re.sub(r"(?:的|时|后|前|中|者)$", "", cleaned)
    return cleaned.strip(" ,.;:()[]{}，。；：、")


def _valid_candidate(phrase: str) -> bool:
    if len(phrase) < 2 or len(phrase) > 40:
        return False
    if phrase.casefold() in _LOW_VALUE_TERMS:
        return False
    if phrase.startswith(("--", "->", "-->", "- ")) or phrase.endswith(("--", "->", "-->", " -")):
        return False
    if re.fullmatch(r"\d+(?:\.\d+)?", phrase):
        return False
    if len(set(phrase)) <= 1:
        return False
    return True
