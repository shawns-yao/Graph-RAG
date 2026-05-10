"""Pluggable domain-specific confidence signal boosters.

The retrieval layer is domain-agnostic. Domain-specific knowledge
(e.g., medical term co-occurrence patterns) is injected via signal
rules loaded from configuration.

Each rule specifies:
- trigger_terms: if ANY of these appear in the text, the rule fires
- co_terms: optional secondary terms that must ALSO appear for a higher boost
- boost: confidence floor when trigger fires alone
- co_boost: confidence floor when both trigger AND co_terms fire
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DomainSignalRule:
    """A single domain-specific confidence boost rule."""

    trigger_terms: tuple[str, ...]
    boost: float = 0.85
    co_terms: tuple[str, ...] = ()
    co_boost: float = 0.95


# ---------------------------------------------------------------------------
# Default rules: medical domain (COPD pulmonary function)
# These can be overridden or extended via configuration.
# ---------------------------------------------------------------------------

_DEFAULT_DOMAIN_RULES: tuple[DomainSignalRule, ...] = (
    DomainSignalRule(
        trigger_terms=("fev1/fvc", "fev1", "fvc", "肺功能检查"),
        boost=0.85,
        co_terms=("诊断",),
        co_boost=0.95,
    ),
)

_active_rules: list[DomainSignalRule] = list(_DEFAULT_DOMAIN_RULES)


def get_domain_signal_rules() -> list[DomainSignalRule]:
    """Return the currently active domain signal rules."""
    return _active_rules


def set_domain_signal_rules(rules: list[DomainSignalRule]) -> None:
    """Replace the active domain signal rules (e.g., for testing or different domains)."""
    global _active_rules  # noqa: PLW0603
    _active_rules = list(rules)


def reset_domain_signal_rules() -> None:
    """Reset to default medical domain rules."""
    global _active_rules  # noqa: PLW0603
    _active_rules = list(_DEFAULT_DOMAIN_RULES)


def apply_domain_boost(text: str, base_signal: float) -> float:
    """Apply all active domain signal rules to compute the final confidence signal.

    Returns the maximum of base_signal and any triggered rule boosts.
    """
    lowered = text.casefold()
    signal = base_signal

    for rule in _active_rules:
        if not any(term in lowered for term in rule.trigger_terms):
            continue
        # Trigger matched — apply base boost
        signal = max(signal, rule.boost)
        # Check co-occurrence for higher boost
        if rule.co_terms and any(term in lowered for term in rule.co_terms):
            signal = max(signal, rule.co_boost)

    return min(1.0, signal)
