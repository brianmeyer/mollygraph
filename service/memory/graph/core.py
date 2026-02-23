"""Core graph class wiring mixins and connection lifecycle."""
from __future__ import annotations

from neo4j import GraphDatabase

from .entities import EntityMixin
from .episodes import EpisodeMixin
from .maintenance import MaintenanceMixin
from .queries import QueryMixin
from .relationships import RelationshipMixin


class BiTemporalGraph(EntityMixin, RelationshipMixin, EpisodeMixin, QueryMixin, MaintenanceMixin):
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
