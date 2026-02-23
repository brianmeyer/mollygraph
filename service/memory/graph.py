"""Neo4j graph layer — adapted from Molly memory/graph.py.

Changes from original:
- Import from service config (not Molly config)
- Removed track_latency decorator (no Molly utils dependency)
- All core logic, Cypher, and locking preserved verbatim
"""
from __future__ import annotations

import asyncio
import logging
import math
import re
import threading
import uuid
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any

import config

log = logging.getLogger(__name__)

_driver = None
_driver_lock = threading.Lock()
_GRAPH_WRITE_LOCK: asyncio.Lock | None = None
_GRAPH_WRITE_LOCK_LOOP_ID: int | None = None
_GRAPH_SYNC_WRITE_LOCK = threading.Lock()


class GraphUnavailableError(RuntimeError):
    """Raised when Neo4j is not reachable — callers should degrade gracefully."""


VALID_REL_TYPES = {
    "WORKS_ON", "WORKS_AT", "KNOWS", "USES", "LOCATED_IN",
    "DISCUSSED_WITH", "INTERESTED_IN", "CREATED", "MANAGES",
    "DEPENDS_ON", "RELATED_TO", "MENTIONS",
    "CLASSMATE_OF", "STUDIED_AT", "ALUMNI_OF",
    "MENTORS", "MENTORED_BY", "REPORTS_TO", "COLLABORATES_WITH",
    "CONTACT_OF",
    "CUSTOMER_OF", "ATTENDS", "PARENT_OF", "CHILD_OF", "RECEIVED_FROM",
    "TEACHES_AT",
}


def _relationship_half_life_case(rel_var: str = "r") -> str:
    """Build CASE expression using shared bitemporal tier mapping when available."""
    try:
        from memory.bitemporal_graph import build_relationship_half_life_case

        return build_relationship_half_life_case(rel_var)
    except Exception:
        # Safe fallback if bitemporal module dependencies are unavailable.
        return (
            "CASE\n"
            f"    WHEN type({rel_var}) IN ['IS_A', 'PART_OF', 'PARENT_OF', 'CHILD_OF', 'STUDIED_AT', 'ALUMNI_OF'] THEN 1000.0\n"
            f"    WHEN type({rel_var}) IN ['WORKS_AT', 'WORKS_ON', 'USES', 'LOCATED_IN', 'CREATED', 'MANAGES', "
            "'DEPENDS_ON', 'REPORTS_TO', 'COLLABORATES_WITH', 'CUSTOMER_OF', 'ATTENDS', 'TEACHES_AT'] THEN 180.0\n"
            f"    WHEN type({rel_var}) IN ['MENTIONS', 'DISCUSSED_WITH', 'RELATED_TO', 'SAID'] THEN 7.0\n"
            "    ELSE 30.0\n"
            "END"
        )


def _get_graph_write_lock() -> asyncio.Lock:
    global _GRAPH_WRITE_LOCK, _GRAPH_WRITE_LOCK_LOOP_ID
    loop = asyncio.get_running_loop()
    loop_id = id(loop)
    if _GRAPH_WRITE_LOCK is None or _GRAPH_WRITE_LOCK_LOOP_ID != loop_id:
        _GRAPH_WRITE_LOCK = asyncio.Lock()
        _GRAPH_WRITE_LOCK_LOOP_ID = loop_id
    return _GRAPH_WRITE_LOCK


# ---------------------------------------------------------------------------
# Driver management
# ---------------------------------------------------------------------------

def get_driver():
    """Lazy-initialize the Neo4j driver and create indexes (thread-safe)."""
    global _driver
    if _driver is not None:
        return _driver
    with _driver_lock:
        if _driver is None:
            try:
                from neo4j import GraphDatabase

                drv = GraphDatabase.driver(
                    config.NEO4J_URI,
                    auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
                )
                with drv.session() as session:
                    session.run(
                        "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)"
                    )
                    session.run(
                        "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)"
                    )
                    session.run(
                        "CREATE INDEX episode_id IF NOT EXISTS FOR (ep:Episode) ON (ep.id)"
                    )
                    session.run(
                        "CREATE INDEX entity_phone IF NOT EXISTS FOR (e:Entity) ON (e.phone)"
                    )
                _driver = drv
                log.info("Neo4j driver initialized at %s", config.NEO4J_URI)
            except Exception:
                log.warning("Neo4j unavailable — graph layer disabled", exc_info=True)
                raise GraphUnavailableError("Neo4j is not reachable")
    return _driver


def close():
    global _driver
    with _driver_lock:
        if _driver:
            _driver.close()
            _driver = None
            log.info("Neo4j driver closed")


# ---------------------------------------------------------------------------
# Strength / decay
# ---------------------------------------------------------------------------

def recency_score(days_since: float) -> float:
    """Exponential decay with ~23-day half-life."""
    return math.exp(-0.03 * days_since)


def strength_score(mentions: int, days_since: float, boost: float = 0) -> float:
    return (mentions * recency_score(days_since)) + boost


# ---------------------------------------------------------------------------
# Deduplication helpers
# ---------------------------------------------------------------------------

def _normalize(name: str) -> str:
    return name.strip().lower()


def _fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def find_matching_entity(
    name: str,
    entity_type: str,
    fuzzy_threshold: float = 0.85,
) -> str | None:
    """Find an existing entity by exact or fuzzy match. Returns entity name or None.

    Uses adaptive threshold: short names (<8 chars) require 0.95 similarity
    to prevent false merges (e.g. Preston→Patreon, Guidehouse→Geico).
    Exact matches always take priority over fuzzy matches.
    """
    # Skip entity matching for very short names — they're noise
    if len(name.strip()) < 3:
        return None

    driver = get_driver()
    normalized = _normalize(name)

    with driver.session() as session:
        # Priority 1: exact name match (any type)
        result = session.run(
            "MATCH (e:Entity) WHERE toLower(e.name) = $name RETURN e.name AS name",
            name=normalized,
        )
        record = result.single()
        if record:
            return record["name"]

        # Priority 2: exact alias match (any type)
        result = session.run(
            """MATCH (e:Entity)
               WHERE ANY(a IN e.aliases WHERE toLower(a) = $name)
               RETURN e.name AS name""",
            name=normalized,
        )
        record = result.single()
        if record:
            return record["name"]

        # Priority 3: fuzzy match (same entity_type only, adaptive threshold)
        # Short names are more prone to false positives, so raise the bar
        effective_threshold = fuzzy_threshold
        if len(name) < 8:
            effective_threshold = max(fuzzy_threshold, 0.95)
        elif len(name) < 12:
            effective_threshold = max(fuzzy_threshold, 0.90)

        result = session.run(
            "MATCH (e:Entity {entity_type: $etype}) RETURN e.name AS name",
            etype=entity_type,
        )
        best_match: str | None = None
        best_ratio: float = 0.0
        for record in result:
            existing_name = record["name"]
            ratio = _fuzzy_ratio(name, existing_name)
            if ratio >= effective_threshold and ratio > best_ratio:
                best_match = existing_name
                best_ratio = ratio

        if best_match:
            log.debug(
                "Fuzzy matched '%s' → '%s' (ratio=%.3f, threshold=%.2f)",
                name, best_match, best_ratio, effective_threshold,
            )
        return best_match

    return None


# ---------------------------------------------------------------------------
# Entity CRUD
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Entity blocklist — prevents noise words from entering the graph
# ---------------------------------------------------------------------------
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

def _is_blocklisted(name: str) -> bool:
    """Check if entity name is blocklisted or too short."""
    stripped = name.strip()
    if len(stripped) < 3:
        return True
    return stripped.lower() in ENTITY_BLOCKLIST


def _upsert_entity_sync(
    name: str,
    entity_type: str,
    confidence: float,
) -> str:
    """Create or merge an entity. Returns the canonical entity name."""
    if _is_blocklisted(name):
        log.debug("Blocked entity '%s' (blocklisted or too short)", name)
        return name.strip()
    driver = get_driver()
    now = datetime.now(timezone.utc).isoformat()
    existing = find_matching_entity(name, entity_type)

    with _GRAPH_SYNC_WRITE_LOCK, driver.session() as session:
        if existing:
            session.run(
                """MATCH (e:Entity) WHERE e.name = $existing_name
                   SET e.mention_count = coalesce(e.mention_count, 0) + 1,
                       e.last_mentioned = $now,
                       e.last_seen = $now,
                       e.valid_from = coalesce(e.valid_from, e.first_mentioned, $now),
                       e.valid_to = null,
                       e.confidence = CASE
                           WHEN $confidence > e.confidence THEN $confidence
                           ELSE e.confidence END,
                       e.aliases = CASE
                           WHEN NOT toLower($alias) IN [x IN e.aliases | toLower(x)]
                                AND toLower($alias) <> toLower(e.name)
                           THEN e.aliases + $alias
                           ELSE e.aliases END""",
                existing_name=existing,
                now=now,
                confidence=confidence,
                alias=name.strip(),
            )
            return existing
        else:
            session.run(
                """CREATE (e:Entity {
                       name: $name,
                       entity_type: $entity_type,
                       mention_count: 1,
                       first_mentioned: $now,
                       last_mentioned: $now,
                       valid_from: $now,
                       valid_to: null,
                       last_seen: $now,
                       strength: 1.0,
                       confidence: $confidence,
                       aliases: [],
                       summary: ''
                   })""",
                name=name.strip(),
                entity_type=entity_type,
                now=now,
                confidence=confidence,
            )
            return name.strip()


def upsert_entity_sync(name: str, entity_type: str, confidence: float) -> str:
    return _upsert_entity_sync(name, entity_type, confidence)


async def upsert_entity(name: str, entity_type: str, confidence: float) -> str:
    async with _get_graph_write_lock():
        return await asyncio.to_thread(_upsert_entity_sync, name, entity_type, confidence)


_SAFE_PROPERTY_KEY = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def set_entity_properties(name: str, properties: dict) -> None:
    if not properties:
        return
    driver = get_driver()
    for k in properties:
        if not _SAFE_PROPERTY_KEY.match(k):
            raise ValueError(f"Unsafe Cypher property key: {k!r}")
    set_clauses = ", ".join(f"e.{k} = ${k}" for k in properties)
    with _GRAPH_SYNC_WRITE_LOCK, driver.session() as session:
        session.run(
            f"MATCH (e:Entity {{name: $name}}) SET {set_clauses}",
            name=name,
            **properties,
        )


def _upsert_relationship_sync(
    head_name: str,
    tail_name: str,
    rel_type: str,
    confidence: float,
    context_snippet: str = "",
) -> None:
    if _is_blocklisted(head_name) or _is_blocklisted(tail_name):
        log.debug("Blocked relationship '%s'-[%s]->'%s' (blocklisted entity)", head_name, rel_type, tail_name)
        return
    driver = get_driver()
    now = datetime.now(timezone.utc).isoformat()

    label = rel_type.strip().upper().replace(" ", "_")
    if label not in VALID_REL_TYPES:
        # Option C: Reject unknown types, log for schema review, don't pollute graph
        try:
            from memory.graph_suggestions import log_relationship_rejection
            log_relationship_rejection(head_name, tail_name, rel_type, confidence, context_snippet)
            log.debug("Rejected relationship '%s'-[%s]->'%s' (unknown type), logged for review", head_name, rel_type, tail_name)
        except Exception:
            log.debug("graph suggestion rejection logging failed", exc_info=True)
        return

    snippet = context_snippet[:200]

    with _GRAPH_SYNC_WRITE_LOCK, driver.session() as session:
        result = session.run(
            f"""MATCH (h:Entity {{name: $head}})
                MATCH (t:Entity {{name: $tail}})
                MERGE (h)-[r:{label}]->(t)
                ON CREATE SET
                    r.strength = $confidence,
                    r.mention_count = 1,
                    r.valid_from = $now,
                    r.valid_to = null,
                    r.valid_at = $now,
                    r.valid_until = null,
                    r.observed_at = $now,
                    r.first_mentioned = $now,
                    r.last_mentioned = $now,
                    r.context_snippets = [$snippet]
                ON MATCH SET
                    r.mention_count = coalesce(r.mention_count, 0) + 1,
                    r.last_mentioned = $now,
                    r.observed_at = $now,
                    r.valid_from = coalesce(r.valid_from, r.valid_at, r.first_mentioned, $now),
                    r.valid_to = null,
                    r.valid_at = coalesce(r.valid_at, r.valid_from, $now),
                    r.valid_until = null,
                    r.strength = CASE
                        WHEN $confidence > r.strength THEN $confidence
                        ELSE r.strength END,
                    r.context_snippets = CASE
                        WHEN size(coalesce(r.context_snippets, [])) >= 3
                        THEN coalesce(r.context_snippets, [])[1..] + [$snippet]
                        ELSE coalesce(r.context_snippets, []) + [$snippet]
                    END,
                    r.audit_status = CASE WHEN r.audit_status IN ['quarantined', 'verified'] THEN r.audit_status ELSE null END
                RETURN r.mention_count AS mention_count""",
            head=head_name,
            tail=tail_name,
            now=now,
            confidence=confidence,
            snippet=snippet,
        )

        if label == "RELATED_TO":
            try:
                record = result.single()
                mention_count = record["mention_count"] if record else 0
                if mention_count == 3:
                    from memory.graph_suggestions import log_repeated_related_to
                    log_repeated_related_to(head_name, tail_name, mention_count)
            except Exception:
                log.debug("graph suggestion hotspot logging failed", exc_info=True)


def upsert_relationship_sync(
    head_name: str,
    tail_name: str,
    rel_type: str,
    confidence: float,
    context_snippet: str = "",
) -> None:
    _upsert_relationship_sync(head_name, tail_name, rel_type, confidence, context_snippet)


async def upsert_relationship(
    head_name: str,
    tail_name: str,
    rel_type: str,
    confidence: float,
    context_snippet: str = "",
) -> None:
    async with _get_graph_write_lock():
        await asyncio.to_thread(
            _upsert_relationship_sync,
            head_name, tail_name, rel_type, confidence, context_snippet,
        )


def _create_episode_sync(
    content_preview: str,
    source: str,
    entity_names: list[str],
) -> str:
    driver = get_driver()
    episode_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    with _GRAPH_SYNC_WRITE_LOCK, driver.session() as session:
        session.run(
            """CREATE (ep:Episode {
                   id: $id,
                   content_preview: $preview,
                   created_at: $now,
                   source: $source,
                   entities_extracted: $entities
               })""",
            id=episode_id,
            preview=content_preview[:300],
            now=now,
            source=source,
            entities=entity_names,
        )
        for name in entity_names:
            session.run(
                """MATCH (ep:Episode {id: $eid})
                   MATCH (e:Entity {name: $name})
                   MERGE (ep)-[:MENTIONS]->(e)
                   SET e.last_seen = $now,
                       e.last_mentioned = $now""",
                eid=episode_id,
                name=name,
                now=now,
            )
    return episode_id


def create_episode_sync(content_preview: str, source: str, entity_names: list[str]) -> str:
    return _create_episode_sync(content_preview, source, entity_names)


async def create_episode(content_preview: str, source: str, entity_names: list[str]) -> str:
    async with _get_graph_write_lock():
        return await asyncio.to_thread(_create_episode_sync, content_preview, source, entity_names)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def query_entity(name: str) -> dict[str, Any] | None:
    driver = get_driver()
    normalized = _normalize(name)

    with driver.session() as session:
        result = session.run(
            """MATCH (e:Entity)
               WHERE toLower(e.name) = $name
                  OR ANY(a IN e.aliases WHERE toLower(a) = $name)
               RETURN e""",
            name=normalized,
        )
        record = result.single()
        if not record:
            return None

        raw = dict(record["e"])
        _PII_FIELDS = {"phone", "email"}
        entity = {k: v for k, v in raw.items() if k not in _PII_FIELDS}

        rels_out = session.run(
            """MATCH (e:Entity)-[r]->(t:Entity)
               WHERE toLower(e.name) = $name
                  OR ANY(a IN e.aliases WHERE toLower(a) = $name)
               RETURN type(r) AS rel_type, properties(r) AS props, t.name AS target""",
            name=normalized,
        )
        rels_in = session.run(
            """MATCH (s:Entity)-[r]->(e:Entity)
               WHERE toLower(e.name) = $name
                  OR ANY(a IN e.aliases WHERE toLower(a) = $name)
               RETURN type(r) AS rel_type, properties(r) AS props, s.name AS source""",
            name=normalized,
        )

        relationships = []
        for rec in rels_out:
            relationships.append({
                "type": rec["rel_type"],
                "target": rec["target"],
                "direction": "outgoing",
                **rec["props"],
            })
        for rec in rels_in:
            relationships.append({
                "type": rec["rel_type"],
                "source": rec["source"],
                "direction": "incoming",
                **rec["props"],
            })

        entity["relationships"] = relationships
        return entity


def query_entities_for_context(entity_names: list[str]) -> str:
    """Query Neo4j for multiple entities and format for system prompt injection.

    Returns markdown-formatted string or empty string if nothing found.
    """
    if not entity_names:
        return ""

    driver = get_driver()
    normalized_names = [_normalize(name) for name in entity_names]

    with driver.session() as session:
        entity_result = session.run(
            """UNWIND $names AS lookup_name
               MATCH (e:Entity)
               WHERE toLower(e.name) = lookup_name
                  OR ANY(a IN e.aliases WHERE toLower(a) = lookup_name)
               RETURN DISTINCT e.name AS name, e.entity_type AS entity_type,
                      e.mention_count AS mention_count,
                      e.last_mentioned AS last_mentioned""",
            names=normalized_names,
        )
        entities_by_name: dict[str, dict] = {}
        for rec in entity_result:
            name = rec["name"]
            if name not in entities_by_name:
                entities_by_name[name] = {
                    "name": name,
                    "entity_type": rec["entity_type"],
                    "mention_count": rec["mention_count"],
                    "last_mentioned": rec["last_mentioned"] or "",
                    "relationships": [],
                }

        if not entities_by_name:
            return ""

        found_names = list(entities_by_name.keys())

        rels_out = session.run(
            """UNWIND $names AS ename
               MATCH (e:Entity {name: ename})-[r]->(t:Entity)
               RETURN e.name AS entity_name, type(r) AS rel_type, t.name AS target,
                      r.audit_status AS audit_status""",
            names=found_names,
        )
        for rec in rels_out:
            ent = entities_by_name.get(rec["entity_name"])
            if ent:
                ent["relationships"].append({
                    "type": rec["rel_type"],
                    "target": rec["target"],
                    "direction": "outgoing",
                    "audit_status": rec["audit_status"],
                })

        rels_in = session.run(
            """UNWIND $names AS ename
               MATCH (s:Entity)-[r]->(e:Entity {name: ename})
               RETURN e.name AS entity_name, type(r) AS rel_type, s.name AS source,
                      r.audit_status AS audit_status""",
            names=found_names,
        )
        for rec in rels_in:
            ent = entities_by_name.get(rec["entity_name"])
            if ent:
                ent["relationships"].append({
                    "type": rec["rel_type"],
                    "source": rec["source"],
                    "direction": "incoming",
                    "audit_status": rec["audit_status"],
                })

    results = list(entities_by_name.values())
    lines = ["<!-- Memory Context (knowledge graph) -->"]
    lines.append("Known entities and relationships:\n")

    for ent in results:
        etype = ent.get("entity_type", "Unknown")
        mentions = ent.get("mention_count", 0)
        last = str(ent.get("last_mentioned", ""))[:10]
        lines.append(f"- **{ent['name']}** ({etype}, {mentions} mentions, last: {last})")

        for rel in ent.get("relationships", []):
            rtype = rel["type"].replace("_", " ").lower()
            flag = " [unverified]" if rel.get("audit_status") == "quarantined" else ""
            if rel["direction"] == "outgoing":
                lines.append(f"  → {rtype} {rel['target']}{flag}")
            else:
                lines.append(f"  ← {rel['source']} {rtype}{flag}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------

def delete_entity(name: str) -> bool:
    driver = get_driver()
    normalized = _normalize(name)
    with _GRAPH_SYNC_WRITE_LOCK, driver.session() as session:
        result = session.run(
            """MATCH (e:Entity)
               WHERE toLower(e.name) = $name
                  OR ANY(a IN e.aliases WHERE toLower(a) = $name)
               DETACH DELETE e
               RETURN count(e) AS deleted""",
            name=normalized,
        )
        record = result.single()
        return record and record["deleted"] > 0


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------

def _run_strength_decay_sync() -> int:
    driver = get_driver()
    half_life_case = _relationship_half_life_case("r")
    with _GRAPH_SYNC_WRITE_LOCK, driver.session() as session:
        entity_result = session.run(
            """
            MATCH (e:Entity)
            WHERE coalesce(e.last_seen, e.last_mentioned) IS NOT NULL
            WITH e,
                 toFloat(coalesce(e.mention_count, 1)) AS mentions,
                 toFloat(duration.between(datetime(coalesce(e.last_seen, e.last_mentioned)), datetime()).days) AS days_since
            SET e.strength = mentions * exp(-0.03 * days_since)
            RETURN count(e) AS updated
            """
        )
        rel_result = session.run(
            f"""
            MATCH ()-[r]->()
            WITH r,
                 coalesce(r.last_mentioned, r.observed_at, r.valid_from, r.valid_at) AS last_seen,
                 toFloat(coalesce(r.mention_count, 1)) AS mentions,
                 {half_life_case} AS half_life
            WHERE last_seen IS NOT NULL
            WITH r, mentions, half_life,
                 toFloat(duration.between(datetime(last_seen), datetime()).days) AS days_since
            SET r.strength = log(1.0 + mentions) * exp(-((log(2.0) / half_life) * days_since))
            RETURN count(r) AS updated
            """
        )
        entity_updated = int(entity_result.single()["updated"])
        rel_updated = int(rel_result.single()["updated"])
        total = entity_updated + rel_updated
        log.info(
            "Strength decay: updated %d entities and %d relationships",
            entity_updated,
            rel_updated,
        )
        return total


def run_strength_decay_sync() -> int:
    return _run_strength_decay_sync()


async def run_strength_decay() -> int:
    async with _get_graph_write_lock():
        return await asyncio.to_thread(_run_strength_decay_sync)


def _backfill_temporal_properties_sync() -> dict[str, int]:
    driver = get_driver()
    now = datetime.now(timezone.utc).isoformat()
    with _GRAPH_SYNC_WRITE_LOCK, driver.session() as session:
        entity_record = session.run(
            """
            MATCH (e:Entity)
            OPTIONAL MATCH (ep_link:Episode)-[:MENTIONS]->(e)
            WITH e, min(coalesce(ep_link.created_at, ep_link.occurred_at, ep_link.ingested_at)) AS linked_episode_at
            OPTIONAL MATCH (ep_list:Episode)
            WHERE e.name IN coalesce(ep_list.entities_extracted, [])
            WITH e,
                 linked_episode_at,
                 min(coalesce(ep_list.created_at, ep_list.occurred_at, ep_list.ingested_at)) AS listed_episode_at,
                 e.valid_from IS NULL AS needs_valid_from,
                 e.last_seen IS NULL AS needs_last_seen,
                 e.mention_count IS NULL AS needs_mentions
            WITH e, needs_valid_from, needs_last_seen, needs_mentions,
                 coalesce(e.first_mentioned, e.last_mentioned, linked_episode_at, listed_episode_at, $now) AS inferred_valid_from
            SET e.valid_from = coalesce(e.valid_from, inferred_valid_from),
                e.last_seen = coalesce(e.last_seen, e.last_mentioned, inferred_valid_from, $now),
                e.mention_count = coalesce(e.mention_count, 1)
            RETURN count(e) AS scanned,
                   sum(CASE WHEN needs_valid_from THEN 1 ELSE 0 END) AS valid_from_backfilled,
                   sum(CASE WHEN needs_last_seen THEN 1 ELSE 0 END) AS last_seen_backfilled,
                   sum(CASE WHEN needs_mentions THEN 1 ELSE 0 END) AS mention_count_backfilled
            """,
            now=now,
        ).single()

        rel_record = session.run(
            """
            MATCH ()-[r]->()
            OPTIONAL MATCH (ep:Episode)
            WHERE any(ep_id IN coalesce(r.episode_ids, []) WHERE ep.id = ep_id)
            WITH r,
                 min(coalesce(ep.created_at, ep.occurred_at, ep.ingested_at)) AS episode_at,
                 r.valid_from IS NULL AS needs_valid_from,
                 r.observed_at IS NULL AS needs_observed_at,
                 r.valid_to IS NULL AS needs_valid_to
            WITH r, needs_valid_from, needs_observed_at, needs_valid_to,
                 coalesce(r.valid_at, r.first_mentioned, r.last_mentioned, r.observed_at, episode_at, $now) AS inferred_valid_from
            SET r.valid_from = coalesce(r.valid_from, inferred_valid_from),
                r.valid_at = coalesce(r.valid_at, r.valid_from, inferred_valid_from),
                r.observed_at = coalesce(r.observed_at, r.last_mentioned, r.valid_from, $now),
                r.valid_to = coalesce(r.valid_to, r.valid_until),
                r.valid_until = coalesce(r.valid_until, r.valid_to)
            RETURN count(r) AS scanned,
                   sum(CASE WHEN needs_valid_from THEN 1 ELSE 0 END) AS valid_from_backfilled,
                   sum(CASE WHEN needs_observed_at THEN 1 ELSE 0 END) AS observed_at_backfilled,
                   sum(CASE WHEN needs_valid_to THEN 1 ELSE 0 END) AS valid_to_backfilled
            """,
            now=now,
        ).single()

    return {
        "entities_scanned": int(entity_record["scanned"]) if entity_record else 0,
        "entities_valid_from_backfilled": int(entity_record["valid_from_backfilled"]) if entity_record else 0,
        "entities_last_seen_backfilled": int(entity_record["last_seen_backfilled"]) if entity_record else 0,
        "entities_mention_count_backfilled": int(entity_record["mention_count_backfilled"]) if entity_record else 0,
        "relationships_scanned": int(rel_record["scanned"]) if rel_record else 0,
        "relationships_valid_from_backfilled": int(rel_record["valid_from_backfilled"]) if rel_record else 0,
        "relationships_observed_at_backfilled": int(rel_record["observed_at_backfilled"]) if rel_record else 0,
        "relationships_valid_to_backfilled": int(rel_record["valid_to_backfilled"]) if rel_record else 0,
    }


def backfill_temporal_properties_sync() -> dict[str, int]:
    summary = _backfill_temporal_properties_sync()
    log.info("Legacy graph temporal backfill summary: %s", summary)
    return summary


async def backfill_temporal_properties() -> dict[str, int]:
    async with _get_graph_write_lock():
        return await asyncio.to_thread(_backfill_temporal_properties_sync)


def _delete_orphan_entities_sync() -> int:
    driver = get_driver()
    with _GRAPH_SYNC_WRITE_LOCK, driver.session() as session:
        result = session.run(
            """
            MATCH (e:Entity)
            WHERE NOT (e)--()
            DELETE e
            RETURN count(e) AS deleted
            """
        )
        deleted = result.single()["deleted"]
        log.info("Deleted %d orphan entities", deleted)
        return deleted


def delete_orphan_entities_sync() -> int:
    return _delete_orphan_entities_sync()


async def delete_orphan_entities() -> int:
    async with _get_graph_write_lock():
        return await asyncio.to_thread(_delete_orphan_entities_sync)


def delete_self_referencing_rels() -> int:
    driver = get_driver()
    with _GRAPH_SYNC_WRITE_LOCK, driver.session() as session:
        result = session.run(
            """
            MATCH (e:Entity)-[r]->(e)
            DELETE r
            RETURN count(r) AS deleted
            """
        )
        deleted = result.single()["deleted"]
        log.info("Deleted %d self-referencing relationships", deleted)
        return deleted


def delete_blocklisted_entities(blocklist: set[str]) -> int:
    driver = get_driver()
    names_lower = [n.lower() for n in blocklist]
    with _GRAPH_SYNC_WRITE_LOCK, driver.session() as session:
        result = session.run(
            """
            MATCH (e:Entity)
            WHERE toLower(e.name) IN $names
            DETACH DELETE e
            RETURN count(e) AS deleted
            """,
            names=names_lower,
        )
        deleted = result.single()["deleted"]
        log.info("Deleted %d blocklisted entities", deleted)
        return deleted


# ---------------------------------------------------------------------------
# Audit helpers
# ---------------------------------------------------------------------------

def get_relationships_for_audit(limit: int = 500) -> list[dict]:
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (h:Entity)-[r]->(t:Entity)
            RETURN h.name AS head, h.entity_type AS head_type,
                   t.name AS tail, t.entity_type AS tail_type,
                   type(r) AS rel_type,
                   r.strength AS strength,
                   r.mention_count AS mention_count,
                   r.context_snippets AS context_snippets,
                   r.audit_status AS audit_status,
                   r.first_mentioned AS first_mentioned,
                   r.last_mentioned AS last_mentioned
            ORDER BY r.strength ASC
            LIMIT $limit
            """,
            limit=limit,
        )
        return [dict(rec) for rec in result]


def get_relationship_type_distribution() -> dict[str, int]:
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH ()-[r]->()
            RETURN type(r) AS rel_type, count(r) AS cnt
            ORDER BY cnt DESC
            """
        )
        return {rec["rel_type"]: rec["cnt"] for rec in result}


def set_relationship_audit_status(head: str, tail: str, rel_type: str, status: str) -> None:
    driver = get_driver()
    if rel_type not in VALID_REL_TYPES:
        return
    with _GRAPH_SYNC_WRITE_LOCK, driver.session() as session:
        session.run(
            f"""MATCH (h:Entity {{name: $head}})-[r:{rel_type}]->(t:Entity {{name: $tail}})
                SET r.audit_status = $status""",
            head=head,
            tail=tail,
            status=status,
        )


def reclassify_relationship(
    head: str,
    tail: str,
    old_type: str,
    new_type: str,
    strength: float,
    mention_count: int,
    context_snippets: list[str] | None = None,
    first_mentioned: str | None = None,
) -> None:
    driver = get_driver()
    if old_type not in VALID_REL_TYPES or new_type not in VALID_REL_TYPES:
        return
    now = datetime.now(timezone.utc).isoformat()
    snippets = context_snippets or []
    first_ts = first_mentioned or now
    with _GRAPH_SYNC_WRITE_LOCK, driver.session() as session:
        with session.begin_transaction() as tx:
            tx.run(
                f"""MATCH (h:Entity {{name: $head}})-[r:{old_type}]->(t:Entity {{name: $tail}})
                    DELETE r""",
                head=head, tail=tail,
            )
            tx.run(
                f"""MATCH (h:Entity {{name: $head}})
                    MATCH (t:Entity {{name: $tail}})
                    MERGE (h)-[r:{new_type}]->(t)
                    ON CREATE SET
                        r.strength = $strength,
                        r.mention_count = $mention_count,
                        r.context_snippets = $snippets,
                        r.valid_from = $first_mentioned,
                        r.valid_to = null,
                        r.valid_at = $first_mentioned,
                        r.valid_until = null,
                        r.observed_at = $now,
                        r.first_mentioned = $first_mentioned,
                        r.last_mentioned = $now,
                        r.audit_status = 'auto_fixed'
                    ON MATCH SET
                        r.strength = CASE WHEN $strength > r.strength THEN $strength ELSE r.strength END,
                        r.mention_count = coalesce(r.mention_count, 0) + $mention_count,
                        r.context_snippets = CASE
                            WHEN size(coalesce(r.context_snippets, [])) >= 3
                            THEN coalesce(r.context_snippets, [])[1..] + $snippets
                            ELSE coalesce(r.context_snippets, []) + $snippets
                        END,
                        r.valid_from = coalesce(r.valid_from, r.valid_at, $first_mentioned),
                        r.valid_to = null,
                        r.valid_at = coalesce(r.valid_at, r.valid_from, $first_mentioned),
                        r.valid_until = null,
                        r.observed_at = coalesce(r.observed_at, $now),
                        r.last_mentioned = $now,
                        r.audit_status = 'auto_fixed'""",
                head=head, tail=tail, strength=strength,
                mention_count=mention_count, snippets=snippets,
                first_mentioned=first_ts, now=now,
            )
            tx.commit()


def delete_specific_relationship(head: str, tail: str, rel_type: str) -> bool:
    driver = get_driver()
    if rel_type not in VALID_REL_TYPES:
        return False
    with _GRAPH_SYNC_WRITE_LOCK, driver.session() as session:
        result = session.run(
            f"""MATCH (h:Entity {{name: $head}})-[r:{rel_type}]->(t:Entity {{name: $tail}})
                DELETE r
                RETURN count(r) AS deleted""",
            head=head, tail=tail,
        )
        record = result.single()
        return record is not None and record["deleted"] > 0


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def entity_count() -> int:
    driver = get_driver()
    with driver.session() as session:
        return session.run("MATCH (e:Entity) RETURN count(e) AS c").single()["c"]


def relationship_count() -> int:
    driver = get_driver()
    with driver.session() as session:
        return session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]


def episode_count() -> int:
    driver = get_driver()
    with driver.session() as session:
        return session.run("MATCH (ep:Episode) RETURN count(ep) AS c").single()["c"]


def get_graph_summary() -> dict[str, Any]:
    driver = get_driver()
    with driver.session() as session:
        e_count = session.run("MATCH (e:Entity) RETURN count(e) AS c").single()["c"]
        r_count = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        ep_count = session.run("MATCH (ep:Episode) RETURN count(ep) AS c").single()["c"]

        top_connected = session.run(
            """MATCH (e:Entity)
               OPTIONAL MATCH (e)-[r]-()
               WITH e, count(r) AS connections
               ORDER BY connections DESC
               LIMIT 10
               RETURN e.name AS name, e.entity_type AS type,
                      e.mention_count AS mentions, connections""",
        )
        top = [dict(rec) for rec in top_connected]

        recent = session.run(
            """MATCH (e:Entity)
               WHERE e.first_mentioned IS NOT NULL
               RETURN e.name AS name, e.entity_type AS type,
                      e.first_mentioned AS added
               ORDER BY e.first_mentioned DESC
               LIMIT 5""",
        )
        recent_list = [dict(rec) for rec in recent]

    return {
        "entity_count": e_count,
        "relationship_count": r_count,
        "episode_count": ep_count,
        "top_connected": top,
        "recent": recent_list,
    }


def get_top_entities(limit: int = 20) -> list[dict[str, Any]]:
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (e:Entity)
            WHERE e.strength IS NOT NULL
            RETURN e.name AS name, e.entity_type AS type,
                   e.strength AS strength, e.mention_count AS mentions
            ORDER BY e.strength DESC
            LIMIT $limit
            """,
            limit=limit,
        )
        return [dict(record) for record in result]
