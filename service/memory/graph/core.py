"""Core graph class wiring mixins and connection lifecycle."""
from __future__ import annotations

from neo4j import GraphDatabase

from .constants import VALID_REL_TYPES, RELATION_TIERS
from .decisions import DecisionMixin
from .entities import EntityMixin
from .episodes import EpisodeMixin
from .maintenance import MaintenanceMixin
from .queries import QueryMixin
from .relationships import RelationshipMixin

# All relationship types that should have an audit_status index.
# Union of VALID_REL_TYPES (user-visible types) and RELATION_TIERS keys
# (all typed relationships that appear in the graph including IS_A, PART_OF etc.).
_ALL_INDEXED_REL_TYPES: frozenset[str] = frozenset(VALID_REL_TYPES) | frozenset(RELATION_TIERS.keys())


class BiTemporalGraph(
    EntityMixin,
    RelationshipMixin,
    EpisodeMixin,
    QueryMixin,
    DecisionMixin,
    MaintenanceMixin,
):
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

            # Decision indexes
            session.run(
                "CREATE CONSTRAINT decision_id_unique IF NOT EXISTS "
                "FOR (d:Decision) REQUIRE d.id IS UNIQUE"
            )
            session.run("CREATE INDEX decision_timestamp IF NOT EXISTS FOR (d:Decision) ON (d.timestamp)")
            session.run("CREATE INDEX decision_decided_by IF NOT EXISTS FOR (d:Decision) ON (d.decided_by)")
            session.run(
                "CREATE INDEX decision_source_episode_id IF NOT EXISTS "
                "FOR (d:Decision) ON (d.source_episode_id)"
            )

            # audit_status indexes for every known relationship type.
            # Neo4j syntax: FOR ()-[r:TYPE]-() ON (r.property)
            for rel_type in sorted(_ALL_INDEXED_REL_TYPES):
                session.run(
                    f"CREATE INDEX rel_audit_status_{rel_type} IF NOT EXISTS "
                    f"FOR ()-[r:{rel_type}]-() ON (r.audit_status)"
                )

    def close(self):
        self.driver.close()
