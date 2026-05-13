"""Rule-based assertion status labeling for Chinese medical entities."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Literal

AssertionLabel = Literal[
    "affirmed",
    "negated",
    "speculated",
    "conditional",
    "historical",
    "family_history",
]

ASSERTION_LABELS: tuple[AssertionLabel, ...] = (
    "affirmed",
    "negated",
    "speculated",
    "conditional",
    "historical",
    "family_history",
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])\s*|\n+")
_TERMINATION_RE = re.compile(r"[。！？!?；;]|(?:，|,)(?:但|但是|然而|仍|可|却)")
_ENTITY_RE = re.compile(
    r"(?:"
    r"[A-Za-z][A-Za-z0-9/+.-]{1,20}|"
    r"[\u4e00-\u9fffA-Za-z0-9/+.-]{1,24}"
    r"(?:病|炎|癌|瘤|症|综合征|感染|梗死|衰竭|损伤|狭窄|阻塞|栓塞|"
    r"高血压|糖尿病|冠心病|肺炎|肾病|肝病|支架|禁忌|药|片|胶囊|注射液)"
    r")"
)
_ACTION_OBJECT_RE = re.compile(
    r"(?:禁用|慎用|使用|服用|口服|给予|应用|予以|改用)"
    r"(?P<entity>[\u4e00-\u9fffA-Za-z0-9/+.-]{2,20})"
)

_NEGATED_CUES = (
    "未见",
    "未发现",
    "未提示",
    "未触及",
    "无",
    "否认",
    "排除",
    "没有",
    "阴性",
    "不支持",
    "不考虑",
)
_SPECULATED_CUES = (
    "不能排除",
    "不除外",
    "疑似",
    "考虑",
    "可能",
    "待排",
    "倾向于",
    "拟诊",
)
_CONDITIONAL_CUES = (
    "如果",
    "若",
    "当",
    "满足",
    "出现",
    "时",
    "则",
    "禁用",
    "慎用",
)
_HISTORICAL_CUES = (
    "既往",
    "病史",
    "曾患",
    "曾因",
    "术后",
    "既往诊断",
    "既往使用",
)
_FAMILY_CUES = (
    "家族史",
    "父亲",
    "母亲",
    "兄弟",
    "姐妹",
    "一级亲属",
    "遗传史",
)

_SCOPE_CHARS = 24
_ENTITY_PREFIX_NOISE = (
    "患者",
    "病人",
    "本人",
    "我",
    "男性",
    "既往",
    "目前",
    "不能排除",
    "不除外",
    "考虑",
    "疑似",
    "可能",
    "无",
    "没有",
    "否认",
    "未见",
    "未发现",
    "排除",
    "比如像",
    "比如",
    "或是",
    "如",
    "像",
    "患上",
    "患有",
    "一些",
    "建议",
)


@dataclass(frozen=True)
class AssertionDecision:
    label: AssertionLabel
    cue: str = ""
    confidence: float = 0.6


@dataclass(frozen=True)
class AssertionExample:
    text: str
    entity: str
    label: AssertionLabel
    start: int
    end: int
    cue: str
    source: str = "weak_rule"
    confidence: float = 0.6
    difficulty: str = "medium"
    domain: str = "medical"

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def split_sentences(text: str) -> list[str]:
    """Split Chinese medical text into non-empty sentence-like spans."""
    return [
        sentence.strip()
        for sentence in _SENTENCE_SPLIT_RE.split(text)
        if sentence and sentence.strip()
    ]


def iter_sentence_spans(text: str) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    cursor = 0
    for sentence in split_sentences(text):
        start = text.find(sentence, cursor)
        if start < 0:
            start = text.find(sentence)
        if start < 0:
            continue
        end = start + len(sentence)
        spans.append((sentence, start, end))
        cursor = end
    return spans


def find_entity_offsets(text: str, entity: str) -> tuple[int, int] | None:
    if not entity:
        return None
    start = text.find(entity)
    if start < 0:
        return None
    return start, start + len(entity)


def classify_assertion_by_rules(sentence: str, entity: str) -> AssertionDecision:
    offset = find_entity_offsets(sentence, entity)
    if offset is None:
        return AssertionDecision("affirmed", confidence=0.5)
    start, end = offset

    family = _first_scoped_cue(sentence, start, end, _FAMILY_CUES, sentence_level=True)
    if family:
        return AssertionDecision("family_history", family, 0.9)

    speculated = _first_scoped_cue(sentence, start, end, _SPECULATED_CUES)
    if speculated:
        return AssertionDecision("speculated", speculated, 0.85)

    negated = _first_scoped_cue(sentence, start, end, _NEGATED_CUES)
    if negated:
        return AssertionDecision("negated", negated, 0.95)

    conditional = _first_scoped_cue(
        sentence,
        start,
        end,
        _CONDITIONAL_CUES,
        sentence_level=_has_threshold_condition(sentence),
    )
    if conditional:
        return AssertionDecision("conditional", conditional, 0.8)

    historical = _first_scoped_cue(sentence, start, end, _HISTORICAL_CUES)
    if historical:
        return AssertionDecision("historical", historical, 0.75)

    return AssertionDecision("affirmed", confidence=0.7)


def extract_assertion_candidates(text: str) -> list[AssertionExample]:
    examples: list[AssertionExample] = []
    seen: set[tuple[int, int, str]] = set()
    for sentence, sentence_start, _ in iter_sentence_spans(text):
        for entity, match_start in _iter_candidate_entities(sentence):
            if len(entity) < 2:
                continue
            if _is_noisy_entity(entity):
                continue
            entity_start = sentence.find(entity, match_start)
            if entity_start < 0:
                continue
            start = sentence_start + entity_start
            end = start + len(entity)
            key = (start, end, entity)
            if key in seen:
                continue
            seen.add(key)
            decision = classify_assertion_by_rules(sentence, entity)
            examples.append(
                AssertionExample(
                    text=sentence,
                    entity=entity,
                    label=decision.label,
                    start=entity_start,
                    end=entity_start + len(entity),
                    cue=decision.cue,
                    confidence=decision.confidence,
                    difficulty=_difficulty_for(decision.label, sentence),
                )
            )
    return examples


def _iter_candidate_entities(sentence: str) -> Iterable[tuple[str, int]]:
    for match in _ENTITY_RE.finditer(sentence):
        yield _clean_entity(match.group(0)), match.start()
    for match in _ACTION_OBJECT_RE.finditer(sentence):
        entity = _clean_entity(match.group("entity"))
        yield entity, match.start("entity")


def _clean_entity(text: str) -> str:
    candidate = text.strip(" \t\r\n，,。；;：:（）()[]【】")
    for marker in ("如果", "比如像", "比如", "像"):
        marker_index = candidate.find(marker)
        if marker_index >= 0 and len(candidate) > marker_index + len(marker) + 1:
            candidate = candidate[marker_index + len(marker):]
    changed = True
    while changed:
        changed = False
        for prefix in _ENTITY_PREFIX_NOISE:
            if candidate.startswith(prefix) and len(candidate) > len(prefix) + 1:
                candidate = candidate[len(prefix):]
                changed = True
                break
    return candidate


def _is_noisy_entity(entity: str) -> bool:
    if len(entity) > 16:
        return True
    noisy_prefixes = (
        "引发",
        "导致",
        "造成",
        "包括",
        "了解",
        "进行",
        "做好",
        "注意",
        "避免",
        "保持",
        "采取",
        "接受",
    )
    noisy_fragments = (
        "一些",
        "主要病",
        "发生其实",
        "的发生",
        "和",
        "以及",
        "或者",
    )
    generic_entities = {"疾病", "男科疾病", "心血管疾病", "男科病", "得病"}
    return (
        entity.startswith(noisy_prefixes)
        or any(fragment in entity for fragment in noisy_fragments)
        or entity in generic_entities
    )


def _first_scoped_cue(
    sentence: str,
    entity_start: int,
    entity_end: int,
    cues: tuple[str, ...],
    *,
    sentence_level: bool = False,
) -> str:
    if sentence_level:
        for cue in cues:
            if cue in sentence:
                return cue

    prefix = sentence[max(0, entity_start - _SCOPE_CHARS):entity_start]
    suffix = sentence[entity_end:entity_end + 8]
    prefix_after_stop = _after_last_termination(prefix)
    suffix_before_stop = _before_first_termination(suffix)
    for cue in cues:
        if cue in prefix_after_stop or cue in suffix_before_stop:
            return cue
    return ""


def _after_last_termination(text: str) -> str:
    matches = list(_TERMINATION_RE.finditer(text))
    if not matches:
        return text
    return text[matches[-1].end():]


def _before_first_termination(text: str) -> str:
    match = _TERMINATION_RE.search(text)
    if not match:
        return text
    return text[:match.start()]


def _has_threshold_condition(sentence: str) -> bool:
    return bool(
        re.search(r"(?:<|>|<=|>=|≤|≥)\s*\d+(?:\.\d+)?", sentence)
        or re.search(r"(?:低于|高于|不低于|不超过|至少)\s*\d+", sentence)
    )


def _difficulty_for(label: AssertionLabel, sentence: str) -> str:
    if label == "affirmed":
        return "easy"
    if len(sentence) > 40 or "但" in sentence or "但是" in sentence:
        return "hard"
    return "medium"
