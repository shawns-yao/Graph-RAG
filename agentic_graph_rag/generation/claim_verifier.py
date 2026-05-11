"""Chain-of-Verification inspired answer verification.

Reference: Dhuliawala et al., "Chain-of-Verification Reduces Hallucination
in Large Language Models" (ACL 2024).

After the generator produces an answer, we extract atomic factual claims
and verify each one against the knowledge graph via `cypher_traverse`.
Claims are assigned a discrete verification level: correct, possible_correct,
or incorrect.

Design choices for this project:
- Verification is only triggered for relation / multi_hop / global queries
  where cross-fact consistency matters. Verifiable simple answers also run it.
- Extraction uses ONE LLM call (structured JSON output). Verification itself is
  deterministic: entities and canonical numeric facts are checked against
  evidence; non-numeric relations only pass when directly supported.
- Missing evidence means possible_correct, not incorrect. Only explicit
  canonical numeric contradiction is marked incorrect.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rag_core.config import get_settings, make_openai_client
from rag_core.models import ClaimVerificationStep, SearchResult, VerifiedClaim

if TYPE_CHECKING:
    from neo4j import Driver
    from openai import OpenAI

logger = logging.getLogger(__name__)

_EXTRACTION_PROMPT = """You extract atomic factual claims from an answer for verification.

Rules:
- Each claim must be a standalone factual statement that can be independently checked.
- Use the question to restore omitted subjects or conditions from short answers.
- Break compound sentences into separate claims.
- Skip conversational filler, hedges, and interpretive commentary.
- Keep claims concise (under 30 words each).
- At most 6 claims. Pick the most load-bearing ones if there are more.
- For each claim, separately identify:
  - role: core, supporting, or supplemental
  - entities: drug names, disease names, test names, anatomical terms (exact surface forms)
  - numeric_constraints: numeric thresholds, percentages, doses, durations
  - relation_actions: action/relationship phrases (e.g. 禁用, 每日1次, 推荐)

Role rules:
- core: directly answers the user's question.
- supporting: condition, source, applicability, or limitation that supports the core answer.
- supplemental: background explanation or extra information not directly asked.
- Background explanation is not core.
- Generic education is not core.
- Patient general information is not core unless directly asked.
- Sources, caveats, and applicability ranges are usually supporting, not core.
- When unsure, use supporting, not core.

Return JSON:
{
  "claims": [
    {
      "text": "<concise factual claim>",
      "role": "core|supporting|supplemental",
      "entities": ["entity1", "entity2"],
      "numeric_constraints": ["30", "12个月"],
      "relation_actions": ["禁用"]
    }
  ]
}

Example:
Answer: "ACEI常见副作用为干咳（发生率15-20%），此时改用ARB。"
Output: {
  "claims": [
    {
      "text": "ACEI的常见副作用是干咳",
      "role": "core",
      "entities": ["ACEI", "干咳"],
      "numeric_constraints": [],
      "relation_actions": ["副作用"]
    },
    {
      "text": "ACEI引起干咳的发生率为15-20%",
      "role": "supporting",
      "entities": ["ACEI"],
      "numeric_constraints": ["15-20%"],
      "relation_actions": ["发生率"]
    },
    {
      "text": "ACEI干咳后应改用ARB",
      "role": "core",
      "entities": ["ACEI", "ARB"],
      "numeric_constraints": [],
      "relation_actions": ["改用"]
    }
  ]
}
"""

_MAX_CLAIMS = 6
_MAX_CLAIM_CHARS = 200
_VALID_CLAIM_ROLES = {"core", "supporting", "supplemental"}


@dataclass(frozen=True, slots=True)
class ClaimExtractionResult:
    claims: list["ExtractedClaim"]


@dataclass(frozen=True, slots=True)
class ExtractedClaim:
    text: str
    role: str = "supporting"
    entities: tuple[str, ...] = ()
    numeric_constraints: tuple[str, ...] = ()
    relation_actions: tuple[str, ...] = ()

    @property
    def key_terms(self) -> tuple[str, ...]:
        return (*self.entities, *self.numeric_constraints, *self.relation_actions)


@dataclass(frozen=True, slots=True)
class QuantitativeFact:
    kind: str
    value: str
    unit: str = ""
    operator: str = ""

    def key(self) -> str:
        parts = [self.kind, self.value]
        if self.unit:
            parts.append(self.unit)
        if self.operator:
            parts.append(self.operator)
        return ":".join(parts)


def extract_claims(
    answer: str,
    *,
    query: str = "",
    openai_client: OpenAI | None = None,
) -> ClaimExtractionResult:
    """Extract atomic factual claims from a generated answer via one LLM call."""
    cfg = get_settings()
    if openai_client is None:
        openai_client = make_openai_client(cfg)

    stripped = (answer or "").strip()
    if not stripped:
        return ClaimExtractionResult(claims=[])

    question_block = f"\nQuestion:\n{query.strip()[:800]}\n" if query.strip() else ""
    prompt = f"{_EXTRACTION_PROMPT}{question_block}\nAnswer:\n{stripped[:2000]}"
    try:
        response = openai_client.chat.completions.create(
            model=cfg.openai.llm_model_mini,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        raw = response.choices[0].message.content or ""
        payload = _extract_json(raw)
        items = payload.get("claims", []) if isinstance(payload, dict) else []
    except Exception as exc:
        logger.warning("Claim extraction failed (%s); skipping verification", exc)
        return ClaimExtractionResult(claims=[])

    claims: list[ExtractedClaim] = []
    for item in items[:_MAX_CLAIMS]:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text or len(text) > _MAX_CLAIM_CHARS:
            continue
        role = str(item.get("role") or "supporting").strip().lower()
        if role not in _VALID_CLAIM_ROLES:
            role = "supporting"
        claims.append(
            ExtractedClaim(
                text=text,
                role=role,
                entities=tuple(_str_list(item.get("entities")))[:5],
                numeric_constraints=tuple(
                    _str_list(item.get("numeric_constraints") or item.get("values"))
                )[:5],
                relation_actions=tuple(
                    _str_list(item.get("relation_actions") or item.get("relation"))
                )[:4],
            )
        )
    return ClaimExtractionResult(claims=claims)


def verify_claims(
    claims: list[ExtractedClaim],
    *,
    cypher_traverse: "callable",
    driver: Driver,
    openai_client: OpenAI,
    existing_evidence: list[SearchResult] | None = None,
) -> ClaimVerificationStep:
    """Verify each claim against the knowledge graph.

    Hard checks require entities and numeric constraints to appear in evidence.
    Relation/action semantics are checked only after hard evidence is present.
    """
    if not claims:
        return ClaimVerificationStep(
            claims_total=0,
            claims_supported=0,
            claims_possible=0,
            claims_incorrect=0,
            verified_claims=[],
            unsupported_claims=[],
            status="skipped",
            skipped_reason="no_claims",
        )

    verified: list[VerifiedClaim] = []
    unsupported: list[VerifiedClaim] = []
    retrieval_failures = 0

    for claim in claims:
        checked = _verify_claim_against_evidence(
            claim,
            existing_evidence or [],
            openai_client=openai_client,
        )
        if checked.verification_level == "correct":
            verified.append(checked)
            continue
        if checked.verification_level == "incorrect":
            unsupported.append(checked)
            continue
        if checked.failure_type == "soft_fail":
            unsupported.append(checked)
            continue

        evidence, retrieval_error = _retrieve_claim_evidence(
            claim,
            cypher_traverse,
            driver,
            openai_client,
        )
        if retrieval_error:
            retrieval_failures += 1
            continue
        vc = _verify_claim_against_evidence(
            claim,
            evidence,
            openai_client=openai_client,
        )
        if vc.verification_level == "correct":
            verified.append(vc)
        else:
            unsupported.append(vc)

    possible_count = sum(
        1 for claim in unsupported if claim.verification_level == "possible_correct"
    )
    incorrect_count = sum(
        1 for claim in unsupported if claim.verification_level == "incorrect"
    )

    if retrieval_failures and retrieval_failures == len(claims):
        return ClaimVerificationStep(
            claims_total=len(claims),
            claims_supported=len(verified),
            claims_possible=possible_count,
            claims_incorrect=incorrect_count,
            verified_claims=verified,
            unsupported_claims=[],
            skipped_reason="claim_evidence_retrieval_failed",
            status="skipped",
        )

    status = "passed"
    if incorrect_count:
        status = "retry_required"
    elif possible_count:
        status = "partial"
    return ClaimVerificationStep(
        claims_total=len(claims),
        claims_supported=len(verified),
        claims_possible=possible_count,
        claims_incorrect=incorrect_count,
        verified_claims=verified,
        unsupported_claims=unsupported,
        status=status,
    )


def build_caveat(step: ClaimVerificationStep) -> str:
    """Compose a human-readable caveat for claims needing attention."""
    if not step.unsupported_claims:
        return ""
    lines = ["Note: Claim verification found statements that need attention:"]
    for vc in step.unsupported_claims[:4]:
        label = (
            "possibly correct but not fully proven"
            if vc.verification_level == "possible_correct"
            else "contradicted or incorrect"
        )
        lines.append(f"  - [{label}] {vc.text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _retrieve_claim_evidence(
    claim: ExtractedClaim,
    cypher_traverse,
    driver,
    openai_client,
) -> tuple[list[SearchResult], bool]:
    """Call cypher_traverse with the claim text to get candidate evidence."""
    query_text = claim.text
    if claim.key_terms:
        query_text = f"{claim.text} {' '.join(claim.key_terms)}"
    try:
        return cypher_traverse(query_text, driver, openai_client, top_k=3), False
    except Exception as exc:
        logger.warning("Claim verification retrieval failed for %r: %s", claim.text, exc)
        return [], True


def _verify_claim_against_evidence(
    claim: ExtractedClaim,
    evidence: list[SearchResult],
    *,
    openai_client: OpenAI | None,
) -> VerifiedClaim:
    """Verify one claim with hard evidence checks plus optional soft relation check."""
    base = {
        "text": claim.text,
        "key_terms": list(claim.key_terms),
        "entities": list(claim.entities),
        "numeric_constraints": list(claim.numeric_constraints),
        "relation_actions": list(claim.relation_actions),
        "claim_role": claim.role,
    }
    if not evidence:
        return VerifiedClaim(
            **base,
            supported=False,
            verification_level="possible_correct",
            failure_type="hard_fail",
        )

    hard_terms = [_normalize_hard_term(term) for term in claim.entities if term.strip()]
    hard_facts = _quantitative_fact_keys(
        " ".join((*claim.numeric_constraints, *claim.relation_actions))
    )
    if not hard_terms:
        hard_terms = [_normalize_hard_term(term) for term in claim.key_terms if term.strip()]

    for result in evidence:
        raw_content = result.chunk.enriched_content or result.chunk.content or ""
        content = _normalize_hard_text(raw_content)
        if hard_terms and not all(term in content for term in hard_terms):
            continue
        evidence_facts = _quantitative_fact_keys(raw_content)
        if hard_facts and not hard_facts.issubset(evidence_facts):
            if _has_conflicting_quantitative_fact(hard_facts, evidence_facts):
                return VerifiedClaim(
                    **base,
                    supported=False,
                    verification_level="incorrect",
                    failure_type="hard_fail",
                    top_chunk_id=result.chunk.id or "",
                )
            continue
        relation_actions = _relation_actions_requiring_direct_support(
            claim.relation_actions,
            has_hard_facts=bool(hard_facts),
        )
        if not relation_actions:
            return VerifiedClaim(
                **base,
                supported=True,
                verification_level="correct",
                failure_type="none",
                top_chunk_id=result.chunk.id or "",
            )
        if _locally_supports_relation(relation_actions, raw_content):
            return VerifiedClaim(
                **base,
                supported=True,
                verification_level="correct",
                failure_type="none",
                top_chunk_id=result.chunk.id or "",
            )
        return VerifiedClaim(
            **base,
            supported=False,
            verification_level="possible_correct",
            failure_type="soft_fail",
            top_chunk_id=result.chunk.id or "",
        )

    return VerifiedClaim(
        **base,
        supported=False,
        verification_level="possible_correct",
        failure_type="hard_fail",
    )


def _verification_evidence_text(text: str) -> str:
    """Prefer source evidence over graph metadata when asking the soft verifier."""
    marker = "Evidence:"
    if marker in text:
        evidence = text.split(marker, 1)[1].strip()
        if evidence:
            return evidence
    return text


_MEASUREMENT_RELATION_CUES = (
    "剂量",
    "用量",
    "用法",
    "频次",
    "频率",
    "每日",
    "每天",
    "每周",
    "发生率",
    "阈值",
    "标准",
    "时间",
    "持续",
)


def _relation_actions_requiring_direct_support(
    actions: tuple[str, ...],
    *,
    has_hard_facts: bool,
) -> tuple[str, ...]:
    candidates = _non_quantitative_actions(actions)
    if not has_hard_facts:
        return candidates
    return tuple(
        action
        for action in candidates
        if not any(cue in _normalize_hard_text(action) for cue in _MEASUREMENT_RELATION_CUES)
    )


def _locally_supports_relation(actions: tuple[str, ...], evidence_text: str) -> bool:
    """Accept direct relation evidence before asking the LLM soft verifier.

    This is intentionally conservative: hard checks already confirmed the
    entities/numeric values exist in this evidence item. It accepts exact action
    text and a narrow set of clinically equivalent prohibition verbs.
    """
    content = _normalize_hard_text(_verification_evidence_text(evidence_text))
    if not content:
        return False
    return any(
        _action_supported_by_text(action, content)
        for action in actions
        if _normalize_hard_term(action)
    )


def _action_supported_by_text(action: str, normalized_evidence: str) -> bool:
    normalized = _normalize_hard_term(action)
    if normalized in normalized_evidence:
        return True
    if _is_prohibition_action(normalized):
        return any(
            term in normalized_evidence
            for term in ("禁用", "不可使用", "不可以用", "不能使用")
        )
    if "改用" in normalized:
        return "改用" in normalized_evidence
    return False


def _is_prohibition_action(normalized_action: str) -> bool:
    return any(
        term in normalized_action
        for term in ("禁用", "不可使用", "不可以用", "不能使用")
    )


def _non_quantitative_actions(actions: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(action for action in actions if not _quantitative_fact_keys(action))


def _quantitative_fact_keys(text: str) -> set[str]:
    return {fact.key() for fact in _extract_quantitative_facts(text)}


def _has_conflicting_quantitative_fact(
    claim_facts: set[str],
    evidence_facts: set[str],
) -> bool:
    for claim_fact in claim_facts:
        claim_parts = claim_fact.split(":")
        if len(claim_parts) < 2:
            continue
        claim_kind, claim_value = claim_parts[0], claim_parts[1]
        for evidence_fact in evidence_facts:
            evidence_parts = evidence_fact.split(":")
            if len(evidence_parts) < 2 or evidence_parts[0] != claim_kind:
                continue
            if evidence_parts[1] != claim_value or evidence_parts[2:] != claim_parts[2:]:
                return True
    return False


def _extract_quantitative_facts(text: str) -> set[QuantitativeFact]:
    """Extract canonical numeric facts from arbitrary text.

    The verifier should not depend on how the LLM split fields. Numeric facts
    can appear in `numeric_constraints`, `relation_actions`, or raw evidence;
    this layer maps equivalent surface forms into stable keys before matching.
    """
    normalized = _normalize_hard_text(text)
    facts: set[QuantitativeFact] = set()

    number_pattern = r"\d+(?:\.\d+)?|[一二两三四五六七八九十]"
    for match in re.finditer(
        rf"(?:每日|每天|每)(?P<count>{number_pattern})次|(?P<count2>{number_pattern})次/(?:日|天|day|d)",
        normalized,
        re.IGNORECASE,
    ):
        facts.add(
            QuantitativeFact(
                kind="frequency",
                value=_canonical_number(match.group("count") or match.group("count2")),
                unit="day",
            )
        )

    for match in re.finditer(
        rf"(?:每周|每星期|每礼拜|每)(?P<count>{number_pattern})次|(?P<count2>{number_pattern})次/(?:周|星期|礼拜|week|w)",
        normalized,
        re.IGNORECASE,
    ):
        facts.add(
            QuantitativeFact(
                kind="frequency",
                value=_canonical_number(match.group("count") or match.group("count2")),
                unit="week",
            )
        )

    for match in re.finditer(
        r"(?P<op><=|>=|<|>|=)(?P<value>\d+(?:\.\d+)?)",
        normalized,
    ):
        facts.add(
            QuantitativeFact(
                kind="comparison",
                value=_canonical_number(match.group("value")),
                operator=match.group("op"),
            )
        )

    for match in re.finditer(
        r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>%|mg|ml|mmol|μg|ug|pg|g/l|iu|分钟|小时|个月|天|年)",
        normalized,
        re.IGNORECASE,
    ):
        facts.add(
            QuantitativeFact(
                kind="quantity",
                value=_canonical_number(match.group("value")),
                unit=_canonical_unit(match.group("unit")),
            )
        )

    return facts


def _canonical_number(value: str) -> str:
    cjk_numbers = {
        "一": "1",
        "二": "2",
        "两": "2",
        "三": "3",
        "四": "4",
        "五": "5",
        "六": "6",
        "七": "7",
        "八": "8",
        "九": "9",
        "十": "10",
    }
    if value in cjk_numbers:
        return cjk_numbers[value]
    if "." not in value:
        return value
    return value.rstrip("0").rstrip(".")


def _canonical_unit(unit: str) -> str:
    normalized = unit.casefold()
    if normalized == "ug":
        return "μg"
    if normalized == "g/l":
        return "g/l"
    return normalized


def _normalize_hard_term(text: str) -> str:
    return _normalize_hard_text(text)


def _normalize_hard_text(text: str) -> str:
    """Format-only normalization for exact entity/numeric hard checks."""
    normalized = (text or "").casefold()
    normalized = normalized.replace("（", "(").replace("）", ")")
    normalized = normalized.replace("，", ",").replace("％", "%")
    normalized = normalized.replace("≤", "<=").replace("≥", ">=")
    normalized = normalized.replace("–", "-").replace("—", "-").replace("－", "-")
    return re.sub(r"\s+", "", normalized)


def _str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _extract_json(text: str) -> dict:
    """Tolerant JSON extraction: tries direct parse, then grabs first {...}."""
    text = (text or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
