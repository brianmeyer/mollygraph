"""Silver-label validation for GLiREL → GLiNER training data generation.

This module guards against anchor-entity over-attribution bugs where every
entity mentioned in a user's messages gets tagged with the account owner's
employer context (e.g. "ultrasound → WORKS_AT → Guidehouse").

Key responsibilities:
- ``validate_silver_relation()`` — gate function that rejects bad silver labels
- ``accumulate_gliner_training_data()`` — writes validated silver to JSONL

All owner-specific data is read from environment variables at import time,
making this module user-agnostic.

Environment variables:
- MOLLYGRAPH_OWNER_NAME: Full name of the graph owner (e.g. "Brian Meyer")
- MOLLYGRAPH_OWNER_ALIASES: Comma-separated aliases (e.g. "Brian,I,me,myself")
- MOLLYGRAPH_OWNER_EMPLOYERS: Comma-separated known employers
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Owner configuration — read from environment, no hardcoded names
# ──────────────────────────────────────────────────────────────────────────────

_owner_name = os.environ.get("MOLLYGRAPH_OWNER_NAME", "").strip()
_owner_aliases_raw = os.environ.get("MOLLYGRAPH_OWNER_ALIASES", "I,me,myself").strip()
_owner_employers_raw = os.environ.get("MOLLYGRAPH_OWNER_EMPLOYERS", "").strip()

OWNER_ALIASES: frozenset[str] = frozenset(
    {a.strip().lower() for a in _owner_aliases_raw.split(",") if a.strip()}
    | ({_owner_name.lower()} if _owner_name else set())
)

OWNER_KNOWN_EMPLOYERS: frozenset[str] = frozenset(
    {e.strip().lower() for e in _owner_employers_raw.split(",") if e.strip()}
)

# ──────────────────────────────────────────────────────────────────────────────
# Concept reject list: tokens / phrases that are definitely not persons or orgs.
# These should never appear as subjects of WORKS_AT / MANAGES / etc.
# ──────────────────────────────────────────────────────────────────────────────

CONCEPT_REJECT_PATTERNS: set[str] = {
    "ultrasound", "dissections", "lottery system", "military track",
    "data science", "consulting", "fintech", "healthcare",
    "robotics", "pitchbook", "managing consultant", "project manager",
    "chief of staff", "program manager", "head of business management",
}


def _is_owner_alias(entity_text: str) -> bool:
    """Return True if *entity_text* refers to the graph owner."""
    return entity_text.strip().lower() in OWNER_ALIASES


def _is_owner_employer(org_text: str) -> bool:
    """Return True if *org_text* is a known employer of the graph owner."""
    return org_text.strip().lower() in OWNER_KNOWN_EMPLOYERS


def _is_concept_entity(entity_text: str) -> bool:
    """Return True if *entity_text* is a concept, not a person or org."""
    normalized = entity_text.strip().lower()
    if normalized in CONCEPT_REJECT_PATTERNS:
        return True
    # Reject single generic words that are clearly not entities
    if len(normalized.split()) == 1 and normalized in {
        "consulting", "healthcare", "robotics", "fintech", "data",
        "science", "management", "engineering", "technology",
    }:
        return True
    return False


def validate_silver_relation(relation: dict[str, Any]) -> tuple[bool, str]:
    """Validate a GLiREL silver-label relation before acceptance.

    Returns (is_valid, reason).  If is_valid is False, the relation should be
    discarded from training data.
    """
    head = str(relation.get("head", "")).strip()
    tail = str(relation.get("tail", "")).strip()
    label = str(relation.get("label", "") or relation.get("rel_type", "")).strip().upper()

    # ── Check 1: Reject concept entities as heads of employment relations ─────
    if _is_concept_entity(head):
        return False, f"'{head}' is a concept, not a person/org"

    # ── Check 2: Reject WORKS_AT to a known employer for non-owner entities ──
    if label in {"WORKS_AT", "EMPLOYED_BY", "WORKS_FOR"}:
        if OWNER_KNOWN_EMPLOYERS and _is_owner_employer(tail):
            if not _is_owner_alias(head):
                return False, (
                    f"'{tail}' is owner's employer but '{head}' is not owner"
                )

    # ── Check 3: Reject MANAGES/REPORTS_TO where head is a concept ───────────
    if label in {"MANAGES", "REPORTS_TO", "MENTORS", "MENTORED_BY"}:
        if _is_concept_entity(head) or _is_concept_entity(tail):
            return False, f"concept entity in hierarchy relation: {head} -> {tail}"

    return True, "ok"


def detect_fanout_pollution(
    relations: list[dict[str, Any]],
    threshold: float = 0.6,
) -> bool:
    """Return True if relations exhibit N→1 anchor-entity fan-out.

    If >threshold of relations share the same tail entity, the batch
    is likely anchor-entity pollution.
    """
    if len(relations) < 4:
        return False

    tail_counts: dict[str, int] = {}
    for rel in relations:
        tail = str(rel.get("tail", "")).strip().lower()
        tail_counts[tail] = tail_counts.get(tail, 0) + 1

    if not tail_counts:
        return False

    max_count = max(tail_counts.values())
    ratio = max_count / len(relations)

    if ratio > threshold:
        dominant_tail = max(tail_counts, key=tail_counts.get)
        log.warning(
            "glirel_silver: fan-out pollution detected — %.0f%% of %d relations "
            "share tail %r. Batch rejected.",
            ratio * 100, len(relations), dominant_tail,
        )
        return True

    return False
