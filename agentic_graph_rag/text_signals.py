"""Lightweight TF-IDF text signals shared by indexing and retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

_CJK_SEGMENT_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]{2,}")
_LATIN_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{1,}")


@dataclass(slots=True)
class TfidfKeyword:
    term: str
    score: float
    idf: float
    tf: int


@dataclass(slots=True)
class TfidfProfile:
    document_count: int
    document_frequency: dict[str, int]

    def df(self, term: str) -> int:
        normalized = normalize_term(term)
        if not normalized:
            return 0
        return self.document_frequency.get(normalized, 0)

    def idf(self, term: str) -> float:
        normalized = normalize_term(term)
        if not normalized:
            return 0.0
        df = self.df(normalized)
        if df <= 0:
            return 0.0
        return math.log((1.0 + self.document_count) / (1.0 + df)) + 1.0


def normalize_term(term: str) -> str:
    return re.sub(r"\s+", " ", term.strip().casefold())


def extract_terms(
    text: str,
    *,
    max_cjk_ngram: int = 8,
) -> list[str]:
    """Extract normalized lexical terms from mixed Chinese/Latin text."""
    if not text.strip():
        return []

    terms: list[str] = []
    seen: set[str] = set()

    for token in _LATIN_TOKEN_RE.findall(text):
        normalized = normalize_term(token)
        if len(normalized) < 2 or normalized in seen:
            continue
        seen.add(normalized)
        terms.append(normalized)

    for segment in _CJK_SEGMENT_RE.findall(text):
        max_len = min(max_cjk_ngram, len(segment))
        for ngram_len in range(2, max_len + 1):
            for start in range(0, len(segment) - ngram_len + 1):
                token = normalize_term(segment[start : start + ngram_len])
                if token in seen or _is_low_value_cjk_token(token):
                    continue
                seen.add(token)
                terms.append(token)

    return terms


def build_tfidf_profile(
    texts: list[str],
    *,
    max_cjk_ngram: int = 8,
) -> TfidfProfile:
    """Build document-frequency statistics from a text corpus."""
    document_frequency: dict[str, int] = {}
    document_count = 0

    for text in texts:
        if not text.strip():
            continue
        document_count += 1
        unique_terms = set(extract_terms(text, max_cjk_ngram=max_cjk_ngram))
        for term in unique_terms:
            document_frequency[term] = document_frequency.get(term, 0) + 1

    return TfidfProfile(
        document_count=max(1, document_count),
        document_frequency=document_frequency,
    )


def rank_keywords(
    text: str,
    profile: TfidfProfile,
    *,
    min_idf: float,
    max_keywords: int = 8,
    max_cjk_ngram: int = 8,
) -> list[TfidfKeyword]:
    """Rank the most informative lexical terms in a text span."""
    if not text.strip():
        return []

    term_counts = Counter(extract_terms(text, max_cjk_ngram=max_cjk_ngram))
    ranked: list[TfidfKeyword] = []
    for term, tf in term_counts.items():
        idf = profile.idf(term)
        if idf < min_idf:
            continue
        length_boost = 1.0 + min(len(term), 8) / 8.0
        ranked.append(
            TfidfKeyword(
                term=term,
                score=tf * idf * length_boost,
                idf=idf,
                tf=tf,
            )
        )

    ranked.sort(key=lambda item: (-item.score, -item.idf, -len(item.term), item.term))
    return ranked[:max_keywords]


def text_signal_score(
    text: str,
    profile: TfidfProfile,
    *,
    min_idf: float,
    max_keywords: int = 8,
    max_cjk_ngram: int = 8,
) -> float:
    """Compute a normalized TF-IDF signal score for a text span."""
    ranked = rank_keywords(
        text,
        profile,
        min_idf=min_idf,
        max_keywords=max_keywords,
        max_cjk_ngram=max_cjk_ngram,
    )
    if not ranked:
        return 0.0

    raw_score = sum(item.score for item in ranked)
    total_terms = max(1, len(extract_terms(text, max_cjk_ngram=max_cjk_ngram)))
    return raw_score / math.sqrt(total_terms)


def best_term_idf(
    text: str,
    profile: TfidfProfile,
    *,
    max_cjk_ngram: int = 8,
) -> float:
    """Return the strongest IDF signal seen in the text span."""
    terms = extract_terms(text, max_cjk_ngram=max_cjk_ngram)
    if not terms:
        return 0.0
    return max(profile.idf(term) for term in terms)


def _is_low_value_cjk_token(token: str) -> bool:
    return len(set(token)) <= 1
