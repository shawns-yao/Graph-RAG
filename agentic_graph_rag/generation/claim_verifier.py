"""Chain-of-Verification inspired answer verification.

Reference: Dhuliawala et al., "Chain-of-Verification Reduces Hallucination
in Large Language Models" (ACL 2024).

After the generator produces an answer, we extract atomic factual claims
and verify each one against the knowledge graph via `cypher_traverse`.
Claims without graph support are flagged; the final answer is annotated
with a caveat listing unverified claims.

Design choices for this project:
- Verification is only triggered for relation / multi_hop / global queries
  where cross-fact consistency matters. Simple / temporal queries skip it.
- Extraction uses ONE LLM call (structured JSON output). Each claim is then
  verified WITHOUT LLM — we reuse the existing graph retrieval tool so
  verification cost scales with claim count, not LLM calls.
- Failed verification does NOT trigger retrieval retry (avoids amplifying
  LLM instability). It only attaches a caveat and downgrades
  confidence_level to "medium" / "low".
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
- Break compound sentences into separate claims.
- Skip conversational filler, hedges, and interpretive commentary.
- Keep claims concise (under 30 words each).
- At most 6 claims. Pick the most load-bearing ones if there are more.
- For medical / technical answers, focus on: entity relationships, numeric thresholds,
  drug-condition pairs, diagnostic criteria, treatment choices.

Return JSON:
{
  "claims": [
    {"text": "<concise factual claim>", "key_terms": ["term1", "term2"]}
  ]
}

Example:
Answer: "ACEI常见副作用为干咳（发生率15-20%），此时改用ARB。ARB不良反应较少。"
Output: {
  "claims": [
    {"text": "ACEI的常见副作用是干咳", "key_terms": ["ACEI", "干咳"]},
    {"text": "ACEI引起干咳的发生率为15-20%", "key_terms": ["ACEI", "15-20%"]},
    {"text": "ACEI干咳后应改用ARB", "key_terms": ["ACEI", "ARB"]},
    {"text": "ARB不良反应较少", "key_terms": ["ARB"]}
  ]
}
"""

_MAX_CLAIMS = 6
_MAX_CLAIM_CHARS = 200


@dataclass(frozen=True, slots=True)
class ClaimExtractionResult:
    claims: list["ExtractedClaim"]


@dataclass(frozen=True, slots=True)
class ExtractedClaim:
    text: str
    key_terms: tuple[str, ...]


def extract_claims(
    answer: str,
    *,
    openai_client: OpenAI | None = None,
) -> ClaimExtractionResult:
    """Extract atomic factual claims from a generated answer via one LLM call."""
    cfg = get_settings()
    if openai_client is None:
        openai_client = make_openai_client(cfg)

    stripped = (answer or "").strip()
    if not stripped:
        return ClaimExtractionResult(claims=[])

    prompt = f"{_EXTRACTION_PROMPT}\n\nAnswer:\n{stripped[:2000]}"
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
        terms = item.get("key_terms") or []
        if not isinstance(terms, list):
            terms = []
        normalized_terms = tuple(
            str(t).strip()
            for t in terms
            if str(t).strip()
        )[:5]
        claims.append(ExtractedClaim(text=text, key_terms=normalized_terms))
    return ClaimExtractionResult(claims=claims)


def verify_claims(
    claims: list[ExtractedClaim],
    *,
    cypher_traverse: "callable",
    driver: Driver,
    openai_client: OpenAI,
) -> ClaimVerificationStep:
    """Verify each claim against the knowledge graph.

    A claim is considered "supported" when `cypher_traverse` returns at least
    one chunk containing all of its key_terms (substring check). This is a
    cheap deterministic verification — no additional LLM call per claim.
    """
    if not claims:
        return ClaimVerificationStep(
            claims_total=0,
            claims_supported=0,
            verified_claims=[],
            unsupported_claims=[],
        )

    verified: list[VerifiedClaim] = []
    unsupported: list[VerifiedClaim] = []

    for claim in claims:
        evidence = _retrieve_claim_evidence(claim, cypher_traverse, driver, openai_client)
        supported, top_chunk_id = _check_claim_support(claim, evidence)
        vc = VerifiedClaim(
            text=claim.text,
            key_terms=list(claim.key_terms),
            supported=supported,
            top_chunk_id=top_chunk_id,
        )
        if supported:
            verified.append(vc)
        else:
            unsupported.append(vc)

    return ClaimVerificationStep(
        claims_total=len(claims),
        claims_supported=len(verified),
        verified_claims=verified,
        unsupported_claims=unsupported,
    )


def build_caveat(step: ClaimVerificationStep) -> str:
    """Compose a human-readable caveat for unsupported claims."""
    if not step.unsupported_claims:
        return ""
    lines = [
        "Note: The following statements in the answer could not be verified "
        "against the knowledge graph and should be treated with caution:",
    ]
    for vc in step.unsupported_claims[:4]:
        lines.append(f"  - {vc.text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _retrieve_claim_evidence(
    claim: ExtractedClaim,
    cypher_traverse,
    driver,
    openai_client,
) -> list[SearchResult]:
    """Call cypher_traverse with the claim text to get candidate evidence."""
    query_text = claim.text
    if claim.key_terms:
        # Anchor the query with key terms to help graph entry-point selection.
        query_text = f"{claim.text} {' '.join(claim.key_terms)}"
    try:
        return cypher_traverse(query_text, driver, openai_client, top_k=3)
    except Exception as exc:
        logger.warning("Claim verification retrieval failed for %r: %s", claim.text, exc)
        return []


def _check_claim_support(
    claim: ExtractedClaim,
    evidence: list[SearchResult],
) -> tuple[bool, str]:
    """A claim is supported if evidence contains all its key_terms (case-insensitive)."""
    if not evidence:
        return False, ""
    terms = [t.strip().lower() for t in claim.key_terms if t.strip()]
    if not terms:
        # No key terms extracted — fall back to requiring any evidence at all.
        top = evidence[0]
        return True, top.chunk.id or ""

    for result in evidence:
        content = (result.chunk.enriched_content or result.chunk.content or "").lower()
        if all(term in content for term in terms):
            return True, result.chunk.id or ""
    return False, ""


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
