"""
Bi-temporal graph operations for MollyGraph V2.
Handles valid_time vs observed_time tracking and relationship decay.
"""
import math
import re
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
import logging

from neo4j import GraphDatabase
from memory.models import Entity, Relationship, Episode

log = logging.getLogger(__name__)

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


class BiTemporalGraph:
    """Neo4j graph operations with bi-temporal tracking."""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._ensure_indexes()
    
    def _ensure_indexes(self):
        """Create necessary indexes and constraints."""
        with self.driver.session() as session:
            # Entity indexes
            session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)")
            session.run("CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)")
            session.run("CREATE INDEX entity_strength IF NOT EXISTS FOR (e:Entity) ON (e.strength)")
            
            # Episode indexes  
            session.run("CREATE INDEX episode_id IF NOT EXISTS FOR (ep:Episode) ON (ep.id)")
            session.run("CREATE INDEX episode_occurred IF NOT EXISTS FOR (ep:Episode) ON (ep.occurred_at)")
            session.run("CREATE INDEX episode_ingested IF NOT EXISTS FOR (ep:Episode) ON (ep.ingested_at)")
            
            # Relationship indexes (on properties, not types)
            # Note: Neo4j relationship index syntax requires -() at the end: FOR ()-[r:TYPE]-() ON (r.property)
            for rel_type in ["WORKS_AT", "KNOWS", "USES", "MENTIONS", "DISCUSSED_WITH", "IS_A", "PART_OF"]:
                session.run(f"CREATE INDEX rel_audit_status_{rel_type} IF NOT EXISTS FOR ()-[r:{rel_type}]-() ON (r.audit_status)")
    
    def close(self):
        self.driver.close()
    
    # ---------------------------------------------------------------------
    # Episode Operations
    # ---------------------------------------------------------------------
    
    def create_episode(self, episode: Episode) -> str:
        """Store an episode node."""
        with self.driver.session() as session:
            payload = episode.model_dump()
            payload["occurred_at"] = episode.occurred_at.isoformat()
            payload["ingested_at"] = episode.ingested_at.isoformat()
            payload["processed_at"] = episode.processed_at.isoformat() if episode.processed_at else None
            result = session.run("""
                CREATE (ep:Episode {
                    id: $id,
                    source: $source,
                    source_id: $source_id,
                    content_preview: $content_preview,
                    content_hash: $content_hash,
                    occurred_at: datetime($occurred_at),
                    ingested_at: datetime($ingested_at),
                    processed_at: datetime($processed_at),
                    processor_version: $processor_version,
                    entities_extracted: $entities_extracted,
                    status: $status
                })
                RETURN ep.id as id
            """, **payload)
            episode_id = result.single()["id"]
            # Keep entities touched by this episode temporally fresh.
            session.run(
                """
                MATCH (ep:Episode {id: $episode_id})
                UNWIND $entities AS entity_name
                MATCH (e:Entity)
                WHERE toLower(e.name) = toLower(entity_name)
                   OR any(alias IN coalesce(e.aliases, []) WHERE toLower(alias) = toLower(entity_name))
                MERGE (ep)-[:MENTIONS]->(e)
                SET e.last_seen = datetime($seen_at),
                    e.last_mentioned = datetime($seen_at)
                """,
                episode_id=episode_id,
                entities=episode.entities_extracted,
                seen_at=payload["ingested_at"],
            )
            return episode_id
    
    # ---------------------------------------------------------------------
    # Entity Operations  
    # ---------------------------------------------------------------------
    
    def upsert_entity(self, entity: Entity) -> Tuple[str, bool]:
        """
        Create or update an entity. Returns (entity_id, is_new).
        """
        with self.driver.session() as session:
            # Check for existing entity by name
            existing = session.run("""
                MATCH (e:Entity)
                WHERE toLower(e.name) = toLower($name)
                RETURN e.id as id, e.mention_count as mentions
            """, name=entity.name).single()
            
            if existing:
                # Update existing
                session.run("""
                    MATCH (e:Entity {id: $id})
                    SET e.last_mentioned = datetime($last_mentioned),
                        e.last_seen = datetime($last_seen),
                        e.valid_from = coalesce(e.valid_from, e.created_at, e.first_mentioned, datetime($valid_from)),
                        e.valid_to = NULL,
                        e.mention_count = coalesce(e.mention_count, 0) + 1,
                        e.strength = $strength,
                        e.aliases = CASE 
                            WHEN $new_alias IS NOT NULL AND NOT $new_alias IN e.aliases 
                            THEN e.aliases + $new_alias 
                            ELSE e.aliases 
                        END
                """, 
                    id=existing["id"],
                    last_mentioned=entity.last_mentioned.isoformat(),
                    last_seen=entity.last_mentioned.isoformat(),
                    valid_from=entity.first_mentioned.isoformat(),
                    strength=entity.strength,
                    new_alias=entity.name if entity.name not in [entity.name] else None
                )
                return existing["id"], False
            else:
                # Create new
                session.run("""
                    CREATE (e:Entity {
                        id: $id,
                        name: $name,
                        entity_type: $entity_type,
                        aliases: $aliases,
                        description: $description,
                        first_mentioned: datetime($first_mentioned),
                        last_mentioned: datetime($last_mentioned),
                        valid_from: datetime($valid_from),
                        valid_to: NULL,
                        last_seen: datetime($last_seen),
                        created_at: datetime(),
                        confidence: $confidence,
                        mention_count: $mention_count,
                        strength: $strength,
                        created_from_episode: $created_from_episode,
                        verified: $verified
                    })
                """,
                    **entity.model_dump(),
                    valid_from=entity.first_mentioned.isoformat(),
                    last_seen=entity.last_mentioned.isoformat(),
                )
                return entity.id, True
    
    # ---------------------------------------------------------------------
    # Bi-temporal Relationship Operations
    # ---------------------------------------------------------------------
    
    def upsert_relationship(self, rel: Relationship) -> Tuple[str, str]:
        """
        Create or update a relationship with bi-temporal logic.
        
        Returns:
            Tuple of (relationship_id, action) where action is:
            - "created": New relationship
            - "updated": Existing relationship strengthened
            - "superseded": Old relationship marked historical, new one created
        """
        with self.driver.session() as session:
            # Find existing relationships of same type between same entities
            # Note: Relationship type cannot be parameterized in Cypher pattern matching
            existing = session.run(f"""
                MATCH (a:Entity {{name: $source}})-[r:{rel.relation_type}]->(b:Entity {{name: $target}})
                WHERE coalesce(r.valid_until, r.valid_to) IS NULL
                   OR coalesce(r.valid_until, r.valid_to) > datetime($valid_at)
                RETURN r.observed_at as observed, r.valid_at as valid, 
                       r.strength as strength, r.id as id
            """, 
                source=rel.source_entity,
                target=rel.target_entity, 
                valid_at=rel.valid_at.isoformat() if rel.valid_at else datetime.utcnow().isoformat()
            ).single()
            
            if existing:
                # Check for contradiction (new fact contradicts old)
                if self._is_contradiction(existing, rel):
                    # Mark old as historical
                    session.run("""
                        MATCH ()-[r {id: $id}]->()
                        SET r.valid_until = datetime($new_valid_at),
                            r.valid_to = datetime($new_valid_at),
                            r.status = 'historical'
                    """,
                        id=existing["id"],
                        new_valid_at=(rel.valid_at or datetime.utcnow()).isoformat(),
                    )
                    
                    # Create new relationship
                    return self._create_relationship(session, rel), "superseded"
                else:
                    # Strengthen existing
                    session.run("""
                        MATCH ()-[r {id: $id}]->()
                        SET r.strength = coalesce(r.strength, 0.0) + log(1 + coalesce(r.mention_count, 1)),
                            r.mention_count = coalesce(r.mention_count, 0) + 1,
                            r.last_mentioned = datetime($now),
                            r.observed_at = datetime($now),
                            r.first_mentioned = coalesce(r.first_mentioned, r.observed_at, r.valid_at, datetime($now)),
                            r.valid_from = coalesce(r.valid_from, r.valid_at, datetime($now)),
                            r.valid_at = coalesce(r.valid_at, r.valid_from, datetime($now)),
                            r.valid_until = NULL,
                            r.valid_to = NULL,
                            r.context_snippets = CASE
                                WHEN size(coalesce(r.context_snippets, [])) >= 3
                                THEN coalesce(r.context_snippets, [])[1..] + $new_snippets
                                ELSE coalesce(r.context_snippets, []) + $new_snippets
                            END,
                            r.episode_ids = CASE
                                WHEN size($new_episode_ids) = 0 THEN coalesce(r.episode_ids, [])
                                ELSE coalesce(r.episode_ids, []) + [id IN $new_episode_ids WHERE NOT id IN coalesce(r.episode_ids, [])]
                            END
                    """,
                        id=existing["id"],
                        now=datetime.utcnow().isoformat(),
                        new_snippets=rel.context_snippets,
                        new_episode_ids=rel.episode_ids,
                    )
                    return existing["id"], "updated"
            else:
                # Create new
                return self._create_relationship(session, rel), "created"
    
    def _is_contradiction(self, existing: Dict, new: Relationship) -> bool:
        """
        Determine if new fact contradicts existing (e.g., different job).
        """
        # Simple heuristic: if valid_at is significantly newer and relation type
        # represents a state that can only have one value at a time
        if new.relation_type not in ["WORKS_AT", "LIVES_IN", "IS_A"]:
            return False
        
        # Handle both string and Neo4j DateTime objects.
        valid_val = existing["valid"]
        existing_valid: datetime | None = None
        if isinstance(valid_val, datetime):
            existing_valid = valid_val
        elif isinstance(valid_val, str):
            try:
                existing_valid = datetime.fromisoformat(valid_val.replace("Z", "+00:00"))
            except ValueError:
                existing_valid = None
        elif valid_val is not None:
            try:
                existing_valid = datetime.fromisoformat(str(valid_val).replace("Z", "+00:00"))
            except ValueError:
                existing_valid = None

        if existing_valid is None:
            return False
        if new.valid_at and (new.valid_at - existing_valid).days > 30:
            return True
        
        return False
    
    def _create_relationship(self, session, rel: Relationship) -> str:
        """Create a new relationship edge."""
        rel_id = str(uuid.uuid4())
        
        tier = _tier_for_rel_type(rel.relation_type)
        strength = calculate_strength(
            mention_count=rel.mention_count,
            days_since=0,
            tier=tier
        )
        
        # Note: Relationship type cannot be parameterized in Cypher CREATE
        session.run(f"""
            MATCH (a:Entity {{name: $source}}), (b:Entity {{name: $target}})
            CREATE (a)-[r:{rel.relation_type} {{
                id: $id,
                valid_at: datetime($valid_at),
                valid_until: $valid_until,
                valid_from: datetime($valid_at),
                valid_to: NULL,
                observed_at: datetime($observed_at),
                first_mentioned: datetime($first_mentioned),
                last_mentioned: datetime($last_mentioned),
                confidence: $confidence,
                strength: $strength,
                mention_count: $mention_count,
                context_snippets: $context_snippets,
                episode_ids: $episode_ids,
                audit_status: $audit_status
            }}]->(b)
        """, 
            source=rel.source_entity,
            target=rel.target_entity,
            type=rel.relation_type,
            id=rel_id,
            valid_at=rel.valid_at.isoformat() if rel.valid_at else datetime.utcnow().isoformat(),
            valid_until=rel.valid_until.isoformat() if rel.valid_until else None,
            observed_at=rel.observed_at.isoformat(),
            first_mentioned=rel.observed_at.isoformat(),
            last_mentioned=rel.observed_at.isoformat(),
            confidence=rel.confidence,
            strength=strength,
            mention_count=rel.mention_count,
            context_snippets=rel.context_snippets,
            episode_ids=rel.episode_ids,
            audit_status=rel.audit_status
        )
        
        return rel_id
    
    # ---------------------------------------------------------------------
    # Retrieval Operations
    # ---------------------------------------------------------------------
    
    def get_current_facts(self, entity_name: str, as_of: Optional[datetime] = None) -> List[Dict]:
        """
        Get facts about an entity that are valid at a specific time.
        
        Args:
            entity_name: Entity to query
            as_of: Point in time (default: now)
        """
        if as_of is None:
            as_of = datetime.utcnow()
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity {name: $name})-[r]->(target)
                WHERE coalesce(r.valid_at, r.valid_from) <= datetime($as_of)
                  AND (
                      coalesce(r.valid_until, r.valid_to) IS NULL
                      OR coalesce(r.valid_until, r.valid_to) > datetime($as_of)
                  )
                RETURN type(r) as rel_type, 
                       target.name as target_name,
                       target.entity_type as target_type,
                       r.strength as strength,
                       r.confidence as confidence
                ORDER BY r.strength DESC, coalesce(r.valid_at, r.valid_from) DESC
            """, name=entity_name, as_of=as_of.isoformat())
            
            return [dict(record) for record in result]
    
    def get_entity_context(self, entity_name: str, hops: int = 2, 
                          min_strength: float = 0.3) -> Dict[str, Any]:
        """
        Get multi-hop context around an entity.
        """
        with self.driver.session() as session:
            # Direct connections
            direct = session.run("""
                MATCH (e:Entity {name: $name})-[r]-(target)
                WHERE r.strength >= $min_strength
                RETURN type(r) as rel_type,
                       target.name as target_name,
                       target.entity_type as target_type,
                       r.strength as strength,
                       startNode(r).name = $name as is_outgoing
                ORDER BY r.strength DESC
                LIMIT 20
            """, name=entity_name, min_strength=min_strength)
            
            # 2-hop connections (friends of friends)
            if hops >= 2:
                two_hop = session.run("""
                    MATCH (e:Entity {name: $name})-[r1]-(mid)-[r2]-(target)
                    WHERE e <> target
                      AND r1.strength >= $min_strength
                      AND r2.strength >= $min_strength
                    RETURN DISTINCT target.name as target_name,
                           target.entity_type as target_type,
                           mid.name as via_entity,
                           (r1.strength + r2.strength) / 2 as avg_strength
                    ORDER BY avg_strength DESC
                    LIMIT 10
                """, name=entity_name, min_strength=min_strength)
            else:
                two_hop = []
            
            return {
                "entity": entity_name,
                "direct_connections": [dict(r) for r in direct],
                "two_hop_connections": [dict(r) for r in two_hop] if hops >= 2 else [],
            }

    # ---------------------------------------------------------------------
    # Maintenance / Audit Operations
    # ---------------------------------------------------------------------

    @staticmethod
    def _normalize_rel_type(rel_type: str) -> str:
        return rel_type.strip().upper().replace(" ", "_")

    @staticmethod
    def _is_valid_rel_type(rel_type: str) -> bool:
        return rel_type in VALID_REL_TYPES and bool(_REL_TYPE_RE.match(rel_type))

    def run_strength_decay_sync(self) -> int:
        """Decay entity and relationship strengths in one pass."""
        half_life_case = build_relationship_half_life_case("r")
        with self.driver.session() as session:
            entity_record = session.run(
                """
                MATCH (e:Entity)
                WHERE coalesce(e.last_seen, e.last_mentioned) IS NOT NULL
                WITH e,
                     toFloat(coalesce(e.mention_count, 1)) AS mentions,
                     toFloat(duration.between(datetime(coalesce(e.last_seen, e.last_mentioned)), datetime()).days) AS days_since
                SET e.strength = mentions * exp(-0.03 * days_since)
                RETURN count(e) AS updated
                """
            ).single()

            rel_record = session.run(
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
            ).single()

        entity_updated = int(entity_record["updated"]) if entity_record else 0
        rel_updated = int(rel_record["updated"]) if rel_record else 0
        total = entity_updated + rel_updated
        log.info(
            "Strength decay updated %d entities and %d relationships",
            entity_updated,
            rel_updated,
        )
        return total

    def backfill_temporal_properties_sync(self) -> Dict[str, int]:
        """Backfill missing temporal properties on existing entities/relationships."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self.driver.session() as session:
            entity_record = session.run(
                """
                MATCH (e:Entity)
                OPTIONAL MATCH (ep_link:Episode)-[:MENTIONS]->(e)
                WITH e, min(coalesce(ep_link.occurred_at, ep_link.ingested_at, ep_link.created_at)) AS linked_episode_at
                OPTIONAL MATCH (ep_list:Episode)
                WHERE e.name IN coalesce(ep_list.entities_extracted, [])
                WITH e,
                     linked_episode_at,
                     min(coalesce(ep_list.occurred_at, ep_list.ingested_at, ep_list.created_at)) AS listed_episode_at,
                     e.valid_from IS NULL AS needs_valid_from,
                     e.last_seen IS NULL AS needs_last_seen,
                     e.mention_count IS NULL AS needs_mentions
                WITH e,
                     needs_valid_from,
                     needs_last_seen,
                     needs_mentions,
                     coalesce(
                         e.created_at,
                         e.first_mentioned,
                         e.last_mentioned,
                         linked_episode_at,
                         listed_episode_at,
                         datetime($now)
                     ) AS inferred_valid_from
                SET e.valid_from = coalesce(e.valid_from, inferred_valid_from),
                    e.last_seen = coalesce(e.last_seen, e.last_mentioned, inferred_valid_from, datetime($now)),
                    e.mention_count = coalesce(e.mention_count, 1)
                RETURN count(e) AS scanned,
                       sum(CASE WHEN needs_valid_from THEN 1 ELSE 0 END) AS valid_from_backfilled,
                       sum(CASE WHEN needs_last_seen THEN 1 ELSE 0 END) AS last_seen_backfilled,
                       sum(CASE WHEN needs_mentions THEN 1 ELSE 0 END) AS mention_count_backfilled
                """,
                now=now_iso,
            ).single()

            rel_record = session.run(
                """
                MATCH ()-[r]->()
                OPTIONAL MATCH (ep:Episode)
                WHERE any(ep_id IN coalesce(r.episode_ids, []) WHERE ep.id = ep_id)
                WITH r,
                     min(coalesce(ep.occurred_at, ep.ingested_at, ep.created_at)) AS episode_at,
                     r.valid_from IS NULL AS needs_valid_from,
                     r.observed_at IS NULL AS needs_observed_at,
                     r.valid_to IS NULL AS needs_valid_to
                WITH r,
                     needs_valid_from,
                     needs_observed_at,
                     needs_valid_to,
                     coalesce(
                         r.valid_at,
                         r.first_mentioned,
                         r.last_mentioned,
                         r.observed_at,
                         episode_at,
                         datetime($now)
                     ) AS inferred_valid_from
                SET r.valid_from = coalesce(r.valid_from, inferred_valid_from),
                    r.valid_at = coalesce(r.valid_at, r.valid_from, inferred_valid_from),
                    r.observed_at = coalesce(r.observed_at, r.last_mentioned, r.valid_from, datetime($now)),
                    r.valid_to = coalesce(r.valid_to, r.valid_until),
                    r.valid_until = coalesce(r.valid_until, r.valid_to)
                RETURN count(r) AS scanned,
                       sum(CASE WHEN needs_valid_from THEN 1 ELSE 0 END) AS valid_from_backfilled,
                       sum(CASE WHEN needs_observed_at THEN 1 ELSE 0 END) AS observed_at_backfilled,
                       sum(CASE WHEN needs_valid_to THEN 1 ELSE 0 END) AS valid_to_backfilled
                """,
                now=now_iso,
            ).single()

        entity_stats = {
            "entities_scanned": int(entity_record["scanned"]) if entity_record else 0,
            "entities_valid_from_backfilled": int(entity_record["valid_from_backfilled"]) if entity_record else 0,
            "entities_last_seen_backfilled": int(entity_record["last_seen_backfilled"]) if entity_record else 0,
            "entities_mention_count_backfilled": int(entity_record["mention_count_backfilled"]) if entity_record else 0,
        }
        relationship_stats = {
            "relationships_scanned": int(rel_record["scanned"]) if rel_record else 0,
            "relationships_valid_from_backfilled": int(rel_record["valid_from_backfilled"]) if rel_record else 0,
            "relationships_observed_at_backfilled": int(rel_record["observed_at_backfilled"]) if rel_record else 0,
            "relationships_valid_to_backfilled": int(rel_record["valid_to_backfilled"]) if rel_record else 0,
        }
        summary = {**entity_stats, **relationship_stats}
        log.info("Temporal backfill summary: %s", summary)
        return summary

    def delete_orphan_entities_sync(self) -> int:
        with self.driver.session() as session:
            record = session.run(
                """
                MATCH (e:Entity)
                WHERE NOT (e)--()
                DELETE e
                RETURN count(e) AS deleted
                """
            ).single()
        deleted = int(record["deleted"]) if record else 0
        log.info("Deleted %d orphan entities", deleted)
        return deleted

    def delete_self_referencing_rels(self) -> int:
        with self.driver.session() as session:
            record = session.run(
                """
                MATCH (e:Entity)-[r]->(e)
                DELETE r
                RETURN count(r) AS deleted
                """
            ).single()
        deleted = int(record["deleted"]) if record else 0
        log.info("Deleted %d self-referencing relationships", deleted)
        return deleted

    def get_relationships_for_audit(self, limit: int = 500) -> List[Dict[str, Any]]:
        safe_limit = max(1, int(limit))
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (h:Entity)-[r]->(t:Entity)
                RETURN h.name AS head,
                       coalesce(h.entity_type, 'Concept') AS head_type,
                       t.name AS tail,
                       coalesce(t.entity_type, 'Concept') AS tail_type,
                       type(r) AS rel_type,
                       coalesce(r.strength, 0.0) AS strength,
                       coalesce(r.mention_count, 1) AS mention_count,
                       coalesce(r.context_snippets, []) AS context_snippets,
                       coalesce(r.audit_status, 'unverified') AS audit_status,
                       coalesce(r.first_mentioned, r.observed_at, r.valid_at) AS first_mentioned,
                       coalesce(r.last_mentioned, r.observed_at, r.valid_at) AS last_mentioned
                ORDER BY coalesce(r.strength, 0.0) ASC,
                         coalesce(r.last_mentioned, r.observed_at, r.valid_at) ASC
                LIMIT $limit
                """,
                limit=safe_limit,
            )
            return [dict(record) for record in result]

    def get_relationship_type_distribution(self) -> Dict[str, int]:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH ()-[r]->()
                RETURN type(r) AS rel_type, count(r) AS cnt
                ORDER BY cnt DESC
                """
            )
            return {str(record["rel_type"]): int(record["cnt"]) for record in result}

    def set_relationship_audit_status(self, head: str, tail: str, rel_type: str, status: str) -> None:
        normalized_type = self._normalize_rel_type(rel_type)
        if not self._is_valid_rel_type(normalized_type):
            return

        with self.driver.session() as session:
            session.run(
                f"""
                MATCH (h:Entity {{name: $head}})-[r:{normalized_type}]->(t:Entity {{name: $tail}})
                SET r.audit_status = $status
                """,
                head=head,
                tail=tail,
                status=status,
            )

    def reclassify_relationship(
        self,
        head: str,
        tail: str,
        old_type: str,
        new_type: str,
        strength: float,
        mention_count: int,
        context_snippets: List[str] | None = None,
        first_mentioned: str | None = None,
    ) -> None:
        normalized_old = self._normalize_rel_type(old_type)
        normalized_new = self._normalize_rel_type(new_type)
        if not self._is_valid_rel_type(normalized_old) or not self._is_valid_rel_type(normalized_new):
            return

        now_iso = datetime.now(timezone.utc).isoformat()
        first_iso = _to_iso_datetime(first_mentioned) or now_iso
        snippets = list(context_snippets or [])
        mention_value = max(1, int(mention_count or 1))
        strength_value = max(0.0, float(strength or 0.0))

        with self.driver.session() as session:
            with session.begin_transaction() as tx:
                existing = tx.run(
                    f"""
                    MATCH (h:Entity {{name: $head}})-[r:{normalized_old}]->(t:Entity {{name: $tail}})
                    RETURN r.valid_at AS valid_at,
                           r.observed_at AS observed_at,
                           r.episode_ids AS episode_ids,
                           r.confidence AS confidence
                    ORDER BY coalesce(r.last_mentioned, r.observed_at, r.valid_at) DESC
                    LIMIT 1
                    """,
                    head=head,
                    tail=tail,
                ).single()

                valid_iso = _to_iso_datetime(existing.get("valid_at")) if existing else None
                observed_iso = _to_iso_datetime(existing.get("observed_at")) if existing else None
                raw_episode_ids = existing.get("episode_ids") if existing else []
                if isinstance(raw_episode_ids, list):
                    episode_ids = [str(item) for item in raw_episode_ids if str(item).strip()]
                else:
                    episode_ids = []
                confidence = float(existing.get("confidence") or 0.7) if existing else 0.7

                tx.run(
                    f"""
                    MATCH (h:Entity {{name: $head}})-[r:{normalized_old}]->(t:Entity {{name: $tail}})
                    DELETE r
                    """,
                    head=head,
                    tail=tail,
                )

                tx.run(
                    f"""
                    MATCH (h:Entity {{name: $head}})
                    MATCH (t:Entity {{name: $tail}})
                    MERGE (h)-[r:{normalized_new}]->(t)
                    ON CREATE SET
                        r.id = $rel_id,
                        r.valid_at = datetime($valid_at),
                        r.valid_until = NULL,
                        r.valid_from = datetime($valid_at),
                        r.valid_to = NULL,
                        r.observed_at = datetime($observed_at),
                        r.first_mentioned = datetime($first_mentioned),
                        r.last_mentioned = datetime($now),
                        r.confidence = $confidence,
                        r.strength = $strength,
                        r.mention_count = $mention_count,
                        r.context_snippets = $snippets,
                        r.episode_ids = $episode_ids,
                        r.audit_status = 'auto_fixed'
                    ON MATCH SET
                        r.strength = CASE
                            WHEN $strength > coalesce(r.strength, 0.0) THEN $strength
                            ELSE coalesce(r.strength, $strength)
                        END,
                        r.mention_count = coalesce(r.mention_count, 0) + $mention_count,
                        r.context_snippets = CASE
                            WHEN size(coalesce(r.context_snippets, [])) >= 3
                            THEN coalesce(r.context_snippets, [])[1..] + $snippets
                            ELSE coalesce(r.context_snippets, []) + $snippets
                        END,
                        r.episode_ids = CASE
                            WHEN size($episode_ids) = 0 THEN coalesce(r.episode_ids, [])
                            ELSE coalesce(r.episode_ids, []) + [item IN $episode_ids WHERE NOT item IN coalesce(r.episode_ids, [])]
                        END,
                        r.last_mentioned = datetime($now),
                        r.audit_status = 'auto_fixed',
                        r.first_mentioned = coalesce(r.first_mentioned, datetime($first_mentioned)),
                        r.valid_at = coalesce(r.valid_at, datetime($valid_at)),
                        r.valid_from = coalesce(r.valid_from, r.valid_at, datetime($valid_at)),
                        r.valid_until = NULL,
                        r.valid_to = NULL,
                        r.observed_at = coalesce(r.observed_at, datetime($observed_at))
                    """,
                    head=head,
                    tail=tail,
                    rel_id=str(uuid.uuid4()),
                    valid_at=valid_iso or first_iso,
                    observed_at=observed_iso or now_iso,
                    first_mentioned=first_iso,
                    now=now_iso,
                    confidence=confidence,
                    strength=strength_value,
                    mention_count=mention_value,
                    snippets=snippets,
                    episode_ids=episode_ids,
                )
                tx.commit()

    def delete_specific_relationship(self, head: str, tail: str, rel_type: str) -> bool:
        normalized_type = self._normalize_rel_type(rel_type)
        if not self._is_valid_rel_type(normalized_type):
            return False

        with self.driver.session() as session:
            record = session.run(
                f"""
                MATCH (h:Entity {{name: $head}})-[r:{normalized_type}]->(t:Entity {{name: $tail}})
                DELETE r
                RETURN count(r) AS deleted
                """,
                head=head,
                tail=tail,
            ).single()
        return bool(record and int(record["deleted"]) > 0)

    def get_graph_summary(self) -> Dict[str, Any]:
        with self.driver.session() as session:
            entity_record = session.run("MATCH (e:Entity) RETURN count(e) AS c").single()
            rel_record = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()
            episode_record = session.run("MATCH (ep:Episode) RETURN count(ep) AS c").single()

            top_connected = session.run(
                """
                MATCH (e:Entity)
                OPTIONAL MATCH (e)-[r]-()
                WITH e, count(r) AS connections
                ORDER BY connections DESC
                LIMIT 10
                RETURN e.name AS name,
                       coalesce(e.entity_type, 'Concept') AS type,
                       coalesce(e.mention_count, 0) AS mentions,
                       connections
                """
            )

            recent = session.run(
                """
                MATCH (e:Entity)
                WITH e, coalesce(e.first_mentioned, e.last_mentioned, e.created_at) AS added
                WHERE added IS NOT NULL
                RETURN e.name AS name,
                       coalesce(e.entity_type, 'Concept') AS type,
                       added
                ORDER BY added DESC
                LIMIT 5
                """
            )
            top_rows = [dict(record) for record in top_connected]
            recent_rows = [dict(record) for record in recent]

        return {
            "entity_count": int(entity_record["c"]) if entity_record else 0,
            "relationship_count": int(rel_record["c"]) if rel_record else 0,
            "episode_count": int(episode_record["c"]) if episode_record else 0,
            "top_connected": top_rows,
            "recent": recent_rows,
        }

    def list_entities_for_embedding(self, limit: int = 5000) -> List[Dict[str, Any]]:
        """Return entity rows suitable for vector reindexing."""
        safe_limit = max(1, int(limit))
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity)
                RETURN coalesce(e.id, toLower(e.name)) AS entity_id,
                       e.name AS name,
                       coalesce(e.entity_type, 'Concept') AS entity_type,
                       coalesce(e.summary, e.description, '') AS content,
                       coalesce(e.confidence, 1.0) AS confidence
                ORDER BY coalesce(e.mention_count, 0) DESC, e.name ASC
                LIMIT $limit
                """,
                limit=safe_limit,
            )

            rows: List[Dict[str, Any]] = []
            for record in result:
                name = str(record.get("name") or "").strip()
                if not name:
                    continue
                rows.append(
                    {
                        "entity_id": str(record.get("entity_id") or name.lower()),
                        "name": name,
                        "entity_type": str(record.get("entity_type") or "Concept"),
                        "content": str(record.get("content") or ""),
                        "confidence": float(record.get("confidence") or 1.0),
                    }
                )
            return rows
