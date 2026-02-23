"""Unified Neo4j graph module with bi-temporal tracking and legacy helpers."""
from __future__ import annotations

import hashlib
import logging
import math
import re
import uuid
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from neo4j import GraphDatabase
from memory.models import Entity, Relationship, Episode

log = logging.getLogger("memory.graph")

# Decay half-lives (in days) by relationship tier
DECAY_HALFLIVES = {
    "structural": 1000,   # IS_A, PART_OF, BORN_ON - permanent facts
    "professional": 180,  # WORKS_AT, SKILLED_IN - long-term but changeable
    "social": 30,         # FRIEND_OF, KNOWS - personal relationships
    "ephemeral": 7,       # DISCUSSED, MENTIONED - high-churn interactions
}

# Map relation types to tiers
RELATION_TIERS = {
    # Structural (rarely changes)
    "IS_A": "structural",
    "PART_OF": "structural",
    "PARENT_OF": "structural",
    "CHILD_OF": "structural",
    "STUDIED_AT": "structural",
    "ALUMNI_OF": "structural",
    # Professional (changes occasionally)
    "WORKS_AT": "professional",
    "WORKS_ON": "professional",
    "SKILLED_IN": "professional",
    "LIVES_IN": "professional",
    "USES": "professional",
    "LOCATED_IN": "professional",
    "CREATED": "professional",
    "MANAGES": "professional",
    "DEPENDS_ON": "professional",
    "REPORTS_TO": "professional",
    "COLLABORATES_WITH": "professional",
    "CUSTOMER_OF": "professional",
    "ATTENDS": "professional",
    "TEACHES_AT": "professional",
    # Social (changes over time)
    "KNOWS": "social",
    "FRIEND_OF": "social",
    "CLASSMATE_OF": "social",
    "MENTORS": "social",
    "MENTORED_BY": "social",
    "CONTACT_OF": "social",
    "INTERESTED_IN": "social",
    "RECEIVED_FROM": "social",
    # Ephemeral (high churn)
    "MENTIONS": "ephemeral",
    "DISCUSSED_WITH": "ephemeral",
    "SAID": "ephemeral",
    "RELATED_TO": "ephemeral",
}

VALID_REL_TYPES = {
    "WORKS_ON", "WORKS_AT", "KNOWS", "USES", "LOCATED_IN",
    "DISCUSSED_WITH", "INTERESTED_IN", "CREATED", "MANAGES",
    "DEPENDS_ON", "RELATED_TO", "MENTIONS",
    "CLASSMATE_OF", "STUDIED_AT", "ALUMNI_OF",
    "MENTORS", "MENTORED_BY", "REPORTS_TO", "COLLABORATES_WITH",
    "CONTACT_OF", "CUSTOMER_OF", "ATTENDS", "PARENT_OF",
    "CHILD_OF", "RECEIVED_FROM",
    # Added: OpenClaw improvements
    "TEACHES_AT",
}
_REL_TYPE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_SAFE_PROPERTY_KEY = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# Entity blocklist — prevents noise words from entering the graph.
ENTITY_BLOCKLIST: set[str] = {
    # System/technical noise
    "api", "url", "cli", "ssh", "dns", "ssl", "tls", "tcp", "udp", "http",
    "https", "json", "yaml", "html", "css", "xml", "csv", "sql", "jwt",
    "oauth", "env", "pid", "uid", "cwd", "eof", "stdin", "stdout", "stderr",
    "captchas", "mentions", "tasks", "dedup", "restart", "sighup", "sigterm",
    "tier_silent", "heartbeat", "heartbeat_ok", "no_reply",
    # Generic words that aren't entities
    "education", "salary", "latency", "budget", "deadline", "update",
    "question", "answer", "problem", "solution", "issue", "error", "bug",
    "feature", "config", "settings", "options", "status", "result", "output",
    "input", "data", "file", "folder", "directory", "path", "link",
    "eliminator", "hammocks", "eco-chic", "self-modeling",
    # Code artifacts
    "agent.py", "main.py", "index.ts", "index.js", "/tools/",
    "bash allowlist", "cre deal", "commitment reminders",
    # System/agent internals
    "heartbeat check", "heartbeat", "catch-up semantics", "wynwood",
    # Too generic to be useful
    "morning", "evening", "today", "tomorrow", "yesterday", "weekend",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}


def _tier_for_rel_type(rel_type: str) -> str:
    return RELATION_TIERS.get(rel_type, "social")


def build_relationship_half_life_case(rel_var: str = "r") -> str:
    grouped: dict[str, list[str]] = {}
    for rel_type, tier in RELATION_TIERS.items():
        grouped.setdefault(tier, []).append(rel_type)

    lines = ["CASE"]
    for tier in ("structural", "professional", "ephemeral", "social"):
        rel_types = sorted(grouped.get(tier, []))
        if not rel_types:
            continue
        rel_list = ", ".join(f"'{rel}'" for rel in rel_types)
        lines.append(
            f"    WHEN type({rel_var}) IN [{rel_list}] THEN {float(DECAY_HALFLIVES[tier])}"
        )
    lines.append(f"    ELSE {float(DECAY_HALFLIVES['social'])}")
    lines.append("END")
    return "\n".join(lines)


def recency_score(days_since: float) -> float:
    """Exponential decay with configurable half-life."""
    return math.exp(-0.03 * days_since)


def calculate_strength(mention_count: int, days_since: float, tier: str = "social") -> float:
    """
    Calculate relationship strength with tiered decay.
    
    Args:
        mention_count: How many times this relationship has been observed
        days_since: Days since last mention
        tier: Decay tier (structural/professional/social/ephemeral)
    """
    half_life = DECAY_HALFLIVES.get(tier, 30)
    lambda_decay = math.log(2) / half_life
    
    # Base strength from mentions (log-scaled to prevent spam from dominating)
    base_strength = math.log(1 + mention_count)
    
    # Apply decay
    decayed = base_strength * math.exp(-lambda_decay * days_since)
    
    return round(decayed, 4)


def _to_iso_datetime(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()

    iso_format = getattr(value, "iso_format", None)
    if callable(iso_format):
        try:
            return str(iso_format())
        except Exception:
            pass

    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return str(isoformat())
        except Exception:
            pass

    text = str(value).strip()
    return text or None


