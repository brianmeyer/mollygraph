"""Soft semantic relation gate for MollyGraph.

Evaluates each relation candidate *before* it is written to the graph database.
Returns a decision of **allow**, **quarantine**, or **skip** — never silently drops
a signal.

Decision semantics
------------------
allow      → write the relationship normally (audit_status unchanged)
quarantine → write the relationship with audit_status='quarantined' for review
skip       → do NOT write, but emit a suggestion/audit signal so the evolution
             pipeline can still learn from this candidate

Scoring
-------
1. Type-plausibility score  — how sensible is (head_type, rel_type, tail_type)?
2. Source-prior adjustment  — contacts_json boosts or down-weights certain rel types
3. gate_score               — clipped combination of (1) and (2)
4. High-confidence override — a very confident extractor can bump "skip" → "quarantine"

Feature flag
------------
MOLLYGRAPH_RELATION_SOFT_GATE_ENABLED=true   (default: true)

Threshold env vars
------------------
MOLLYGRAPH_RELATION_GATE_ALLOW_THRESHOLD        default 0.45
MOLLYGRAPH_RELATION_GATE_QUARANTINE_THRESHOLD   default 0.25
MOLLYGRAPH_RELATION_GATE_HIGH_CONF_OVERRIDE     default 0.85
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

log = logging.getLogger(__name__)

# ── Type alias ─────────────────────────────────────────────────────────────────
GateDecision = Literal["allow", "quarantine", "skip"]


@dataclass
class GateResult:
    """Result returned by :meth:`RelationSoftGate.evaluate`."""

    decision: GateDecision
    reason: str
    score: float  # 0.0 – 1.0 composite gate score


# ── Type-plausibility tables ────────────────────────────────────────────────────

# Known-plausible triples → plausibility score (0.0–1.0).
# Anything NOT listed falls through to the fallback scoring rules below.
_TRIPLE_PLAUSIBILITY: dict[tuple[str, str, str], float] = {
    # Professional / social
    ("Person", "WORKS_AT",         "Organization"): 1.00,
    ("Person", "WORKS_AT",         "Place"):        0.80,
    ("Person", "CONTACT_OF",       "Person"):       1.00,
    ("Person", "KNOWS",            "Person"):       1.00,
    ("Person", "DISCUSSED_WITH",   "Person"):       1.00,
    ("Person", "COLLABORATES_WITH","Person"):       1.00,
    ("Person", "COLLABORATES_WITH","Organization"): 0.85,
    ("Person", "MANAGES",          "Person"):       0.90,
    ("Person", "MANAGES",          "Project"):      0.95,
    ("Person", "MANAGES",          "Organization"): 0.85,
    ("Person", "REPORTS_TO",       "Person"):       0.85,
    # Education
    ("Person", "STUDIED_AT",       "Organization"): 1.00,
    ("Person", "ALUMNI_OF",        "Organization"): 1.00,
    ("Person", "CLASSMATE_OF",     "Person"):       1.00,
    ("Person", "ATTENDS",          "Event"):        0.90,
    ("Person", "ATTENDS",          "Organization"): 0.80,
    # Family
    ("Person", "CHILD_OF",         "Person"):       0.80,
    ("Person", "PARENT_OF",        "Person"):       0.80,
    # Mentorship
    ("Person", "MENTORED_BY",      "Person"):       0.85,
    ("Person", "MENTORS",          "Person"):       0.85,
    # Location
    ("Person",       "LOCATED_IN", "Place"):        1.00,
    ("Organization", "LOCATED_IN", "Place"):        1.00,
    ("Place",        "LOCATED_IN", "Place"):        0.90,
    # Affiliation
    ("Person",       "MEMBER_OF",  "Organization"): 1.00,
    ("Person",       "MEMBER_OF",  "Concept"):      0.75,
    # Technology / creation
    ("Person",       "USES",       "Technology"):   0.95,
    ("Organization", "USES",       "Technology"):   0.90,
    ("Person",       "CREATED",    "Technology"):   0.95,
    ("Person",       "CREATED",    "Project"):      0.95,
    ("Organization", "CREATED",    "Technology"):   0.90,
    ("Person",       "WORKS_ON",   "Project"):      1.00,
    ("Person",       "INTERESTED_IN","Concept"):    0.90,
    ("Person",       "INTERESTED_IN","Technology"): 0.90,
    # Org-to-org
    ("Organization", "RELATED_TO", "Organization"): 0.70,
    ("Person",       "CUSTOMER_OF","Organization"): 0.85,
    ("Person",       "RECEIVED_FROM","Person"):     0.75,
    ("Person",       "RECEIVED_FROM","Organization"):0.80,
    # Teaching
    ("Person",       "TEACHES_AT", "Organization"): 1.00,
    # Generic mentions
    ("Person",       "MENTIONS",   "Person"):       0.70,
    ("Person",       "MENTIONS",   "Organization"): 0.70,
    ("Person",       "MENTIONS",   "Place"):        0.70,
    ("Person",       "MENTIONS",   "Technology"):   0.70,
}

# Known-implausible triples → low score (still written if confidence high enough)
_TRIPLE_IMPLAUSIBLE: dict[tuple[str, str, str], float] = {
    ("Organization", "CHILD_OF",    "Person"):       0.20,
    ("Organization", "PARENT_OF",   "Person"):       0.20,
    ("Technology",   "REPORTS_TO",  "Person"):       0.15,
    ("Place",        "REPORTS_TO",  "Person"):       0.15,
    ("Technology",   "WORKS_AT",    "Organization"): 0.20,
    ("Event",        "WORKS_AT",    "Organization"): 0.20,
    ("Organization", "MENTORED_BY", "Person"):       0.25,
    ("Organization", "MENTORS",     "Person"):       0.25,
}

# Relation types where the head/tail types carry little signal → neutral scoring
_GENERIC_REL_TYPES: frozenset[str] = frozenset({
    "RELATED_TO",
    "MENTIONS",
    "DISCUSSED_WITH",
})

# ── Source-prior tables (contacts_json) ───────────────────────────────────────

# Relations commonly found in contact card data — boost plausibility
_CONTACTS_BOOSTED: frozenset[str] = frozenset({
    "CONTACT_OF",
    "WORKS_AT",
    "LOCATED_IN",
    "MEMBER_OF",
    "CLASSMATE_OF",
    "ALUMNI_OF",
    "STUDIED_AT",
})

# Relations unusual in contact card data unless extracted with very high confidence
_CONTACTS_DOWNWEIGHTED: frozenset[str] = frozenset({
    "CHILD_OF",
    "PARENT_OF",
    "REPORTS_TO",
    "MENTORED_BY",
    "MENTORS",
    "MANAGES",
})

_CONTACTS_BOOST_FACTOR:       float = 1.30
_CONTACTS_DOWNWEIGHT_FACTOR:  float = 0.55


# ── Gate singleton ─────────────────────────────────────────────────────────────

_gate_instance: "RelationSoftGate | None" = None


def get_gate() -> "RelationSoftGate":
    """Return the process-level :class:`RelationSoftGate` singleton.

    Reads environment / config on first call; subsequent calls return the
    cached instance.  Call :func:`reset_gate` in tests to force re-read.
    """
    global _gate_instance
    if _gate_instance is None:
        _gate_instance = RelationSoftGate()
    return _gate_instance


def reset_gate() -> None:
    """Discard the cached gate singleton (useful in tests)."""
    global _gate_instance
    _gate_instance = None


# ── Gate implementation ────────────────────────────────────────────────────────

class RelationSoftGate:
    """Soft semantic gate evaluated before every relationship upsert.

    Instantiate once per process via :func:`get_gate`.
    """

    def __init__(self) -> None:
        import config as _cfg

        self.enabled: bool = getattr(
            _cfg, "RELATION_SOFT_GATE_ENABLED", True
        )
        self.allow_threshold: float = float(getattr(
            _cfg, "RELATION_GATE_ALLOW_THRESHOLD", 0.45
        ))
        self.quarantine_threshold: float = float(getattr(
            _cfg, "RELATION_GATE_QUARANTINE_THRESHOLD", 0.25
        ))
        self.high_conf_override: float = float(getattr(
            _cfg, "RELATION_GATE_HIGH_CONF_OVERRIDE", 0.85
        ))

        log.debug(
            "RelationSoftGate initialised: enabled=%s allow=%.2f quarantine=%.2f high_conf=%.2f",
            self.enabled,
            self.allow_threshold,
            self.quarantine_threshold,
            self.high_conf_override,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def evaluate(
        self,
        *,
        head_type: str,
        rel_type: str,
        tail_type: str,
        source: str,
        confidence: float,
        context: str = "",
    ) -> GateResult:
        """Evaluate a relation candidate.

        Parameters
        ----------
        head_type:  entity_type of the head node (Person / Organization / …)
        rel_type:   normalised relation type (UPPER_UNDERSCORE)
        tail_type:  entity_type of the tail node
        source:     ingest source (contacts_json / whatsapp / session / …)
        confidence: extractor confidence (0.0 – 1.0)
        context:    optional context snippet for logging

        Returns
        -------
        GateResult with decision, reason, and composite gate score.
        """
        if not self.enabled:
            return GateResult(
                decision="allow",
                reason="gate disabled (MOLLYGRAPH_RELATION_SOFT_GATE_ENABLED=false)",
                score=1.0,
            )

        rel_norm  = rel_type.strip().upper().replace(" ", "_")
        head_norm = head_type.strip() if head_type else "Concept"
        tail_norm = tail_type.strip() if tail_type else "Concept"

        plausibility = self._type_plausibility(head_norm, rel_norm, tail_norm)
        gate_score   = self._apply_source_prior(plausibility, rel_norm, source)
        gate_score   = max(0.0, min(1.0, gate_score))

        return self._decide(
            gate_score=gate_score,
            confidence=confidence,
            head_type=head_norm,
            rel_type=rel_norm,
            tail_type=tail_norm,
            source=source,
        )

    # ── Scoring helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _type_plausibility(head_type: str, rel_type: str, tail_type: str) -> float:
        """Return a plausibility score for the (head_type, rel_type, tail_type) triple."""
        # 1. Exact known-plausible triple
        score = _TRIPLE_PLAUSIBILITY.get((head_type, rel_type, tail_type))
        if score is not None:
            return score

        # 2. Known-implausible triple
        score = _TRIPLE_IMPLAUSIBLE.get((head_type, rel_type, tail_type))
        if score is not None:
            return score

        # 3. Generic rel types — hard to assess; return neutral
        if rel_type in _GENERIC_REL_TYPES:
            return 0.65

        # 4. Either entity type is Concept (unknown from extractor) — can't assess
        if head_type == "Concept" or tail_type == "Concept":
            return 0.60

        # 5. Unlisted triple — mildly sceptical but not blocked
        return 0.55

    @staticmethod
    def _apply_source_prior(plausibility: float, rel_type: str, source: str) -> float:
        """Adjust plausibility score based on source-specific priors."""
        if source != "contacts_json":
            return plausibility

        if rel_type in _CONTACTS_BOOSTED:
            return plausibility * _CONTACTS_BOOST_FACTOR
        if rel_type in _CONTACTS_DOWNWEIGHTED:
            return plausibility * _CONTACTS_DOWNWEIGHT_FACTOR
        return plausibility

    def _decide(
        self,
        gate_score: float,
        confidence: float,
        head_type: str,
        rel_type: str,
        tail_type: str,
        source: str,
    ) -> GateResult:
        """Convert a numeric gate score + confidence into a final decision."""
        triple_str = f"({head_type})-[{rel_type}]->({tail_type})"
        source_tag = f"source={source}"

        if gate_score >= self.allow_threshold:
            return GateResult(
                decision="allow",
                reason=f"plausible: {triple_str} score={gate_score:.2f} {source_tag}",
                score=gate_score,
            )

        # High extractor confidence can rescue a low-plausibility triple to
        # quarantine (not skip) so the signal is preserved for review.
        if gate_score >= self.quarantine_threshold or confidence >= self.high_conf_override:
            if confidence >= self.high_conf_override:
                reason = (
                    f"low plausibility ({gate_score:.2f}) overridden by high confidence "
                    f"({confidence:.2f}): {triple_str} {source_tag} → quarantine"
                )
            else:
                reason = (
                    f"uncertain: {triple_str} score={gate_score:.2f} conf={confidence:.2f} "
                    f"{source_tag} → quarantine for review"
                )
            return GateResult(decision="quarantine", reason=reason, score=gate_score)

        # Below both thresholds and confidence too low to rescue → skip
        return GateResult(
            decision="skip",
            reason=(
                f"implausible: {triple_str} score={gate_score:.2f} conf={confidence:.2f} "
                f"{source_tag} → skip + emit suggestion signal"
            ),
            score=gate_score,
        )


__all__ = [
    "GateDecision",
    "GateResult",
    "RelationSoftGate",
    "get_gate",
    "reset_gate",
]
