"""Graph suggestion tracking, digesting, and auto-adoption.

Real-time logging captures:
1) Relationship type fallbacks (unknown rel -> RELATED_TO)
2) RELATED_TO hotspots (3+ mentions)
3) Optional entity-type fallback hints (for future model schema expansion)

Nightly maintenance can call:
- build_suggestion_digest()
- run_auto_adoption()
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import config

log = logging.getLogger(__name__)

SUGGESTIONS_DIR = config.SUGGESTIONS_DIR
_ADOPTION_HISTORY_PATH = SUGGESTIONS_DIR / "adoption_history.json"

_MIN_DAYS = 3
_MIN_OCCURRENCES = 5
_MIN_FREQUENCY = 0.4


def _utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _append_jsonl(entry: dict[str, Any]) -> None:
    """Best-effort append a JSON line to today's suggestion file."""
    try:
        SUGGESTIONS_DIR.mkdir(parents=True, exist_ok=True)
        ts = str(entry.get("timestamp") or "")
        today = ts[:10] if len(ts) >= 10 else _utc_today()
        path = SUGGESTIONS_DIR / f"{today}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=True) + "\n")
    except OSError:
        log.debug("Failed to append graph suggestion", exc_info=True)


def log_relationship_rejection(
    head: str,
    tail: str,
    original_type: str,
    confidence: float,
    context: str,
) -> None:
    """Log rejected relationship (unknown type) for schema review."""
    _append_jsonl(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "relationship_rejection",
            "head": head,
            "tail": tail,
            "original_type": original_type,
            "rejected": True,
            "confidence": round(confidence, 3),
            "context": (context or "")[:200],
            "suggestion": (
                f"Add '{original_type.strip().upper().replace(' ', '_')}' to VALID_REL_TYPES to enable this relationship type"
            ),
        }
    )


def log_relationship_fallback(
    head: str,
    tail: str,
    original_type: str,
    confidence: float,
    context: str,
) -> None:
    """Log relationship type fallback to RELATED_TO."""
    _append_jsonl(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "relationship_fallback",
            "head": head,
            "tail": tail,
            "original_type": original_type,
            "fell_back_to": "RELATED_TO",
            "confidence": round(confidence, 3),
            "context": (context or "")[:200],
            "suggestion": (
                f"Consider adding '{original_type.strip().upper().replace(' ', '_')}' to VALID_REL_TYPES"
            ),
        }
    )


def log_entity_type_fallback(entity: str, original_type: str, confidence: float, context: str) -> None:
    """Optional hook for future unknown-entity-type tracking."""
    _append_jsonl(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "entity_type_fallback",
            "entity": entity,
            "original_type": original_type,
            "confidence": round(confidence, 3),
            "context": (context or "")[:200],
            "suggestion": f"Consider adding entity type '{original_type.strip()}' to ENTITY_SCHEMA",
        }
    )


def log_repeated_related_to(head: str, tail: str, mention_count: int) -> None:
    """Log hotspot when RELATED_TO reaches 3+ mentions."""
    _append_jsonl(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "related_to_hotspot",
            "head": head,
            "tail": tail,
            "mention_count": int(mention_count),
            "suggestion": f"{head} -> {tail}: RELATED_TO {mention_count}x — consider specific type",
        }
    )


def get_suggestions(date_str: str | None = None) -> list[dict[str, Any]]:
    """Read suggestions for a UTC date (default today)."""
    target_date = date_str or _utc_today()
    path = SUGGESTIONS_DIR / f"{target_date}.jsonl"
    if not path.exists():
        return []

    entries: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                entries.append(payload)
    except OSError:
        log.debug("Failed to read graph suggestions: %s", path, exc_info=True)

    return entries


def get_related_to_hotspots(min_mentions: int = 3) -> list[dict[str, Any]]:
    """Query RELATED_TO hotspots directly from Neo4j."""
    try:
        from runtime_graph import require_graph_instance

        driver = require_graph_instance().driver
        with driver.session() as session:
            result = session.run(
                """MATCH (h:Entity)-[r:RELATED_TO]->(t:Entity)
                   WHERE r.mention_count >= $min_mentions
                   RETURN h.name AS head,
                          t.name AS tail,
                          r.mention_count AS mentions,
                          r.context_snippets AS contexts
                   ORDER BY r.mention_count DESC
                   LIMIT 50""",
                min_mentions=int(min_mentions),
            )
            return [dict(record) for record in result]
    except Exception:
        log.debug("Failed querying RELATED_TO hotspots", exc_info=True)
        return []


def build_suggestion_digest(date_str: str | None = None) -> str:
    """Build a compact digest from JSONL events and RELATED_TO hotspots."""
    suggestions = get_suggestions(date_str=date_str)
    hotspots = get_related_to_hotspots(min_mentions=3)
    if not suggestions and not hotspots:
        return ""

    rel_fallback_counts: dict[str, int] = {}
    entity_fallback_counts: dict[str, int] = {}
    jsonl_hotspots: dict[str, int] = {}

    for entry in suggestions:
        entry_type = str(entry.get("type") or "")
        if entry_type == "relationship_fallback":
            original = str(entry.get("original_type") or "unknown")
            rel_fallback_counts[original] = rel_fallback_counts.get(original, 0) + 1
        elif entry_type == "entity_type_fallback":
            original = str(entry.get("original_type") or "unknown")
            entity_fallback_counts[original] = entity_fallback_counts.get(original, 0) + 1
        elif entry_type == "related_to_hotspot":
            key = f"{str(entry.get('head') or '?').strip().lower()} -> {str(entry.get('tail') or '?').strip().lower()}"
            count = int(entry.get("mention_count") or 0)
            jsonl_hotspots[key] = max(jsonl_hotspots.get(key, 0), count)

    neo4j_only = [
        hotspot
        for hotspot in hotspots
        if f"{str(hotspot.get('head') or '?').strip().lower()} -> {str(hotspot.get('tail') or '?').strip().lower()}" not in jsonl_hotspots
    ]

    lines: list[str] = []
    total_items = len(rel_fallback_counts) + len(entity_fallback_counts) + len(jsonl_hotspots) + len(neo4j_only)
    if total_items == 0:
        return ""

    total_events = sum(rel_fallback_counts.values()) + sum(entity_fallback_counts.values())
    total_events += len(jsonl_hotspots) + len(neo4j_only)

    lines.append(f"{total_items} graph suggestion(s) today ({total_events} events):")

    for original_type, count in sorted(rel_fallback_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        normalized = original_type.strip().upper().replace(" ", "_")
        lines.append(f"- rel type '{original_type}' fell back to RELATED_TO {count}x -> add {normalized}?")

    for original_type, count in sorted(entity_fallback_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        pretty = original_type.strip()
        lines.append(f"- entity type '{pretty}' fallback observed {count}x -> add to ENTITY_SCHEMA?")

    for key, count in sorted(jsonl_hotspots.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- {key}: RELATED_TO {count}x -> specific type?")

    for hotspot in neo4j_only:
        head = str(hotspot.get("head") or "?")
        tail = str(hotspot.get("tail") or "?")
        mentions = int(hotspot.get("mentions") or 0)
        lines.append(f"- {head} -> {tail}: RELATED_TO {mentions}x -> specific type?")

    return "\n".join(lines)


def _load_adoption_history() -> dict[str, dict[str, Any]]:
    """Load history keyed by suggestion key."""
    if not _ADOPTION_HISTORY_PATH.exists():
        return {}

    try:
        payload = json.loads(_ADOPTION_HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        log.debug("Failed loading adoption history", exc_info=True)
        return {}

    items = payload.get("items", {}) if isinstance(payload, dict) else {}
    if not isinstance(items, dict):
        return {}

    cleaned: dict[str, dict[str, Any]] = {}
    for key, entry in items.items():
        if not isinstance(key, str) or not isinstance(entry, dict):
            continue
        nights = entry.get("nights_seen", [])
        occurrences_by_day = entry.get("occurrences_by_day", {})
        if not isinstance(nights, list):
            nights = []
        if not isinstance(occurrences_by_day, dict):
            occurrences_by_day = {}

        cleaned[key] = {
            "key": key,
            "action": str(entry.get("action") or ""),
            "value": str(entry.get("value") or ""),
            "metadata": entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {},
            "first_seen": str(entry.get("first_seen") or ""),
            "last_seen": str(entry.get("last_seen") or ""),
            "nights_seen": sorted({str(n) for n in nights if str(n)}),
            "occurrences_by_day": {
                str(day): int(count)
                for day, count in occurrences_by_day.items()
                if str(day)
            },
            "total_occurrences": int(entry.get("total_occurrences") or 0),
        }

    return cleaned


def _save_adoption_history(history: dict[str, dict[str, Any]]) -> None:
    SUGGESTIONS_DIR.mkdir(parents=True, exist_ok=True)

    serializable_items: dict[str, dict[str, Any]] = {}
    for key in sorted(history):
        entry = history[key]
        by_day = entry.get("occurrences_by_day", {})
        total_occurrences = sum(int(v) for v in by_day.values())
        serializable_items[key] = {
            "key": key,
            "action": str(entry.get("action") or ""),
            "value": str(entry.get("value") or ""),
            "metadata": entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {},
            "first_seen": str(entry.get("first_seen") or ""),
            "last_seen": str(entry.get("last_seen") or ""),
            "nights_seen": sorted({str(d) for d in entry.get("nights_seen", []) if str(d)}),
            "occurrences_by_day": {
                str(day): int(count)
                for day, count in by_day.items()
                if str(day)
            },
            "total_occurrences": int(total_occurrences),
        }

    payload = {"version": 1, "items": serializable_items}
    _ADOPTION_HISTORY_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _days_inclusive(start_iso: str, end_iso: str) -> int:
    try:
        start = date.fromisoformat(start_iso)
        end = date.fromisoformat(end_iso)
    except ValueError:
        return 0
    return max(0, (end - start).days) + 1


def _build_today_counts(today: str) -> dict[str, dict[str, Any]]:
    """Build per-key counts from today's suggestions and hotspots.

    Keys are stable and action-specific:
    - rel_type:<REL>
    - entity_type:<TYPE>
    - reclassify:<head_lower>-><tail_lower>
    """
    entries = get_suggestions(today)
    hotspots = get_related_to_hotspots(min_mentions=3)

    counts: dict[str, dict[str, Any]] = {}

    for entry in entries:
        kind = str(entry.get("type") or "")
        if kind == "relationship_fallback":
            rel_type = str(entry.get("original_type") or "").strip().upper().replace(" ", "_")
            if not rel_type:
                continue
            key = f"rel_type:{rel_type}"
            item = counts.setdefault(
                key,
                {
                    "action": "add_rel_type",
                    "value": rel_type,
                    "count": 0,
                    "metadata": {"example": entry.get("suggestion", "")},
                },
            )
            item["count"] = int(item["count"]) + 1
        elif kind == "entity_type_fallback":
            entity_type = str(entry.get("original_type") or "").strip()
            if not entity_type:
                continue
            key = f"entity_type:{entity_type.lower()}"
            item = counts.setdefault(
                key,
                {
                    "action": "add_entity_type",
                    "value": entity_type,
                    "count": 0,
                    "metadata": {"example": entry.get("suggestion", "")},
                },
            )
            item["count"] = int(item["count"]) + 1

    for hotspot in hotspots:
        head = str(hotspot.get("head") or "").strip()
        tail = str(hotspot.get("tail") or "").strip()
        if not head or not tail:
            continue
        key = f"reclassify:{head.lower()}->{tail.lower()}"
        item = counts.setdefault(
            key,
            {
                "action": "reclassify_candidate",
                "value": f"{head}->{tail}",
                "count": 0,
                "metadata": {
                    "head": head,
                    "tail": tail,
                    "mentions": int(hotspot.get("mentions") or 0),
                },
            },
        )
        item["count"] = int(item["count"]) + 1

    return counts


def _adopt_rel_type(rel_type: str) -> bool:
    try:
        from memory import extractor
        from memory.graph import VALID_REL_TYPES

        normalized = rel_type.strip().upper().replace(" ", "_")
        if not normalized or normalized in VALID_REL_TYPES:
            return False

        VALID_REL_TYPES.add(normalized)

        relation_name = normalized.lower().replace("_", " ")
        if relation_name not in extractor.RELATION_SCHEMA:
            extractor.RELATION_SCHEMA[relation_name] = {
                "description": f"Auto-adopted relation type: {relation_name}",
                "threshold": 0.45,
            }

        log.info("Auto-adopted relationship type: %s", normalized)
        return True
    except Exception:
        log.error("Failed to auto-adopt relationship type: %s", rel_type, exc_info=True)
        return False


def _adopt_entity_type(entity_type: str) -> bool:
    try:
        from memory import extractor

        normalized = entity_type.strip()
        if not normalized or normalized in extractor.ENTITY_SCHEMA:
            return False

        extractor.ENTITY_SCHEMA[normalized] = {
            "description": f"Auto-adopted entity type: {normalized}",
            "threshold": 0.45,
        }
        log.info("Auto-adopted entity type: %s", normalized)
        return True
    except Exception:
        log.error("Failed to auto-adopt entity type: %s", entity_type, exc_info=True)
        return False


def run_auto_adoption(today: str | None = None) -> str:
    """Track and auto-adopt stable suggestions via frequency-ratio gates.

    Adoption gate (enhanced):
    - >= 3 distinct days
    - >= 5 total occurrences
    - frequency >= 0.4 where frequency = nights_seen / nights_since_first_seen
    """
    observed_day = today or _utc_today()
    today_counts = _build_today_counts(observed_day)
    history = _load_adoption_history()

    for key, observation in today_counts.items():
        action = str(observation.get("action") or "")
        value = str(observation.get("value") or "")
        count = int(observation.get("count") or 0)
        metadata = observation.get("metadata") if isinstance(observation.get("metadata"), dict) else {}

        existing = history.get(key)
        if existing is None:
            existing = {
                "key": key,
                "action": action,
                "value": value,
                "metadata": metadata,
                "first_seen": observed_day,
                "last_seen": observed_day,
                "nights_seen": [observed_day],
                "occurrences_by_day": {observed_day: count},
                "total_occurrences": count,
            }
            history[key] = existing
        else:
            nights_seen = set(str(d) for d in existing.get("nights_seen", []))
            nights_seen.add(observed_day)

            occurrences_by_day = existing.get("occurrences_by_day", {})
            if not isinstance(occurrences_by_day, dict):
                occurrences_by_day = {}
            occurrences_by_day[observed_day] = count

            existing["action"] = action or str(existing.get("action") or "")
            existing["value"] = value or str(existing.get("value") or "")
            if metadata:
                existing["metadata"] = metadata
            existing["last_seen"] = observed_day
            if not existing.get("first_seen"):
                existing["first_seen"] = observed_day
            existing["nights_seen"] = sorted(nights_seen)
            existing["occurrences_by_day"] = occurrences_by_day
            existing["total_occurrences"] = sum(int(v) for v in occurrences_by_day.values())

    adopted: list[str] = []

    for key in list(history.keys()):
        item = history[key]
        first_seen = str(item.get("first_seen") or "")
        nights_seen = sorted({str(d) for d in item.get("nights_seen", []) if str(d)})
        distinct_days = len(nights_seen)
        total_occurrences = int(item.get("total_occurrences") or 0)
        span_days = _days_inclusive(first_seen, observed_day)
        frequency = (distinct_days / span_days) if span_days > 0 else 0.0

        item["total_occurrences"] = total_occurrences

        if not (
            distinct_days >= _MIN_DAYS
            and total_occurrences >= _MIN_OCCURRENCES
            and frequency >= _MIN_FREQUENCY
        ):
            continue

        action = str(item.get("action") or "")
        value = str(item.get("value") or "")

        if action == "add_rel_type" and value and _adopt_rel_type(value):
            adopted.append(f"rel_type:{value}")
            del history[key]
        elif action == "add_entity_type" and value and _adopt_entity_type(value):
            adopted.append(f"entity_type:{value}")
            del history[key]

    _save_adoption_history(history)

    tracked_today = len(today_counts)
    tracked_total = len(history)
    if not adopted:
        return f"tracked {tracked_today} suggestions today ({tracked_total} active in history), adopted 0"

    adopted_text = "; ".join(adopted[:10])
    return (
        f"tracked {tracked_today} suggestions today ({tracked_total} active in history), "
        f"adopted {len(adopted)}: {adopted_text}"
    )


__all__ = [
    "build_suggestion_digest",
    "get_related_to_hotspots",
    "get_suggestions",
    "log_entity_type_fallback",
    "log_relationship_fallback",
    "log_repeated_related_to",
    "run_auto_adoption",
]
