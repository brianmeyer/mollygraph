"""
Bi-temporal graph operations for MollyGraph V2.
Handles valid_time vs observed_time tracking and relationship decay.
"""
import asyncio
import math
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any, Tuple
import logging

from neo4j import GraphDatabase
from memory.models import Entity, Relationship, Episode
import config

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
    "WORKS_AT": "professional",
    "WORKS_ON": "professional", 
    "SKILLED_IN": "professional",
    "LIVES_IN": "professional",
    "KNOWS": "social",
    "FRIEND_OF": "social",
    "MENTIONS": "ephemeral",
    "DISCUSSED_WITH": "ephemeral",
    "SAID": "ephemeral",
    "IS_A": "structural",
    "PART_OF": "structural",
}


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
            """, **episode.model_dump())
            return result.single()["id"]
    
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
                        e.mention_count = e.mention_count + 1,
                        e.strength = $strength,
                        e.aliases = CASE 
                            WHEN $new_alias IS NOT NULL AND NOT $new_alias IN e.aliases 
                            THEN e.aliases + $new_alias 
                            ELSE e.aliases 
                        END
                """, 
                    id=existing["id"],
                    last_mentioned=entity.last_mentioned.isoformat(),
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
                        confidence: $confidence,
                        mention_count: $mention_count,
                        strength: $strength,
                        created_from_episode: $created_from_episode,
                        verified: $verified
                    })
                """, **entity.model_dump())
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
                WHERE r.valid_until IS NULL OR r.valid_until > datetime($valid_at)
                RETURN r.observed_at as observed, r.valid_at as valid, 
                       r.strength as strength, r.id as id
            """, 
                source=rel.source_entity,
                target=rel.target_entity, 
                valid_at=rel.valid_at.isoformat() if rel.valid_at else None
            ).single()
            
            if existing:
                # Check for contradiction (new fact contradicts old)
                if self._is_contradiction(existing, rel):
                    # Mark old as historical
                    session.run("""
                        MATCH ()-[r {id: $id}]->()
                        SET r.valid_until = datetime($new_valid_at),
                            r.status = 'historical'
                    """, id=existing["id"], new_valid_at=rel.valid_at.isoformat())
                    
                    # Create new relationship
                    return self._create_relationship(session, rel), "superseded"
                else:
                    # Strengthen existing
                    session.run("""
                        MATCH ()-[r {id: $id}]->()
                        SET r.strength = r.strength + log(1 + r.mention_count),
                            r.mention_count = r.mention_count + 1,
                            r.last_mentioned = datetime($now)
                    """, id=existing["id"], now=datetime.utcnow().isoformat())
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
        
        tier = RELATION_TIERS.get(rel.relation_type, "social")
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
                observed_at: datetime($observed_at),
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
                WHERE r.valid_at <= datetime($as_of)
                  AND (r.valid_until IS NULL OR r.valid_until > datetime($as_of))
                RETURN type(r) as rel_type, 
                       target.name as target_name,
                       target.entity_type as target_type,
                       r.strength as strength,
                       r.confidence as confidence
                ORDER BY r.strength DESC, r.valid_at DESC
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
