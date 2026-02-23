"""Query/read operations for the bi-temporal graph."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional


class QueryMixin:
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

    def query_entity(self, name: str) -> Dict[str, Any] | None:
        """Return an entity with attached incoming/outgoing relationship details."""
        normalized = self._normalize_name(name)
        with self.driver.session() as session:
            record = session.run(
                """
                MATCH (e:Entity)
                WHERE toLower(e.name) = $name
                   OR ANY(a IN coalesce(e.aliases, []) WHERE toLower(a) = $name)
                RETURN e
                """,
                name=normalized,
            ).single()
            if not record:
                return None

            raw = dict(record["e"])
            pii_fields = {"phone", "email"}
            entity = {k: v for k, v in raw.items() if k not in pii_fields}

            rels_out = session.run(
                """
                MATCH (e:Entity)-[r]->(t:Entity)
                WHERE toLower(e.name) = $name
                   OR ANY(a IN coalesce(e.aliases, []) WHERE toLower(a) = $name)
                RETURN type(r) AS rel_type, properties(r) AS props, t.name AS target
                """,
                name=normalized,
            )
            rels_in = session.run(
                """
                MATCH (s:Entity)-[r]->(e:Entity)
                WHERE toLower(e.name) = $name
                   OR ANY(a IN coalesce(e.aliases, []) WHERE toLower(a) = $name)
                RETURN type(r) AS rel_type, properties(r) AS props, s.name AS source
                """,
                name=normalized,
            )

            relationships: list[dict[str, Any]] = []
            for rel in rels_out:
                relationships.append(
                    {
                        "type": rel["rel_type"],
                        "target": rel["target"],
                        "direction": "outgoing",
                        **dict(rel["props"]),
                    }
                )
            for rel in rels_in:
                relationships.append(
                    {
                        "type": rel["rel_type"],
                        "source": rel["source"],
                        "direction": "incoming",
                        **dict(rel["props"]),
                    }
                )

            entity["relationships"] = relationships
            return entity

    def query_entities_for_context(self, entity_names: list[str]) -> str:
        """Build markdown context block for a list of entity names."""
        if not entity_names:
            return ""

        normalized_names = [self._normalize_name(name) for name in entity_names if str(name).strip()]
        if not normalized_names:
            return ""

        with self.driver.session() as session:
            entity_result = session.run(
                """
                UNWIND $names AS lookup_name
                MATCH (e:Entity)
                WHERE toLower(e.name) = lookup_name
                   OR ANY(a IN coalesce(e.aliases, []) WHERE toLower(a) = lookup_name)
                RETURN DISTINCT e.name AS name,
                                coalesce(e.entity_type, 'Concept') AS entity_type,
                                coalesce(e.mention_count, 0) AS mention_count,
                                e.last_mentioned AS last_mentioned
                """,
                names=normalized_names,
            )
            entities_by_name: dict[str, dict[str, Any]] = {}
            for rec in entity_result:
                name = str(rec["name"])
                entities_by_name.setdefault(
                    name,
                    {
                        "name": name,
                        "entity_type": rec["entity_type"],
                        "mention_count": rec["mention_count"],
                        "last_mentioned": rec["last_mentioned"] or "",
                        "relationships": [],
                    },
                )

            if not entities_by_name:
                return ""

            found_names = list(entities_by_name.keys())
            rels_out = session.run(
                """
                UNWIND $names AS ename
                MATCH (e:Entity {name: ename})-[r]->(t:Entity)
                RETURN e.name AS entity_name,
                       type(r) AS rel_type,
                       t.name AS target,
                       r.audit_status AS audit_status
                """,
                names=found_names,
            )
            for rec in rels_out:
                ent = entities_by_name.get(str(rec["entity_name"]))
                if ent is not None:
                    ent["relationships"].append(
                        {
                            "type": rec["rel_type"],
                            "target": rec["target"],
                            "direction": "outgoing",
                            "audit_status": rec["audit_status"],
                        }
                    )

            rels_in = session.run(
                """
                UNWIND $names AS ename
                MATCH (s:Entity)-[r]->(e:Entity {name: ename})
                RETURN e.name AS entity_name,
                       type(r) AS rel_type,
                       s.name AS source,
                       r.audit_status AS audit_status
                """,
                names=found_names,
            )
            for rec in rels_in:
                ent = entities_by_name.get(str(rec["entity_name"]))
                if ent is not None:
                    ent["relationships"].append(
                        {
                            "type": rec["rel_type"],
                            "source": rec["source"],
                            "direction": "incoming",
                            "audit_status": rec["audit_status"],
                        }
                    )

        results = list(entities_by_name.values())
        lines = ["<!-- Memory Context (knowledge graph) -->", "Known entities and relationships:", ""]
        for ent in results:
            etype = ent.get("entity_type", "Unknown")
            mentions = ent.get("mention_count", 0)
            last = str(ent.get("last_mentioned", ""))[:10]
            lines.append(f"- **{ent['name']}** ({etype}, {mentions} mentions, last: {last})")
            for rel in ent.get("relationships", []):
                rel_type = str(rel["type"]).replace("_", " ").lower()
                flag = " [unverified]" if rel.get("audit_status") == "quarantined" else ""
                if rel["direction"] == "outgoing":
                    lines.append(f"  -> {rel_type} {rel['target']}{flag}")
                else:
                    lines.append(f"  <- {rel['source']} {rel_type}{flag}")
        return "\n".join(lines)

    def build_full_context(self, entity_names: list[str]) -> str:
        """Legacy alias used by prior function-based graph module."""
        return self.query_entities_for_context(entity_names)

    @staticmethod
    def _extract_query_entities(query: str) -> list[str]:
        words = [w.strip(" ,.!?:;()[]{}") for w in str(query).split()]
        entities = [w for w in words if len(w) > 2 and (w.istitle() or w.isupper())]
        if not entities:
            entities = [w for w in words if len(w) >= 4 and w.isalpha()]
        seen: set[str] = set()
        ordered: list[str] = []
        for ent in entities:
            key = ent.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(ent)
        return ordered

    def query_graph(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Query graph by extracting candidate entities from natural-language text."""
        rows: list[dict[str, Any]] = []
        for entity_name in self._extract_query_entities(query)[: max(1, int(limit))]:
            facts = self.get_current_facts(entity_name)
            if facts:
                rows.append({"entity": entity_name, "facts": facts[:10]})
        return rows

    def get_all_relationships(self, limit: int = 5000) -> List[Dict[str, Any]]:
        """Return relationship rows for training/debug workflows."""
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
                       coalesce(r.last_mentioned, r.observed_at, r.valid_at) AS last_mentioned,
                       coalesce(r.valid_at, r.valid_from) AS valid_at,
                       coalesce(r.valid_until, r.valid_to) AS valid_until
                ORDER BY coalesce(r.last_mentioned, r.observed_at, r.valid_at) DESC
                LIMIT $limit
                """,
                limit=safe_limit,
            )
            return [dict(record) for record in result]

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

    def entity_count(self) -> int:
        with self.driver.session() as session:
            record = session.run("MATCH (e:Entity) RETURN count(e) AS c").single()
        return int(record["c"]) if record else 0

    def relationship_count(self) -> int:
        with self.driver.session() as session:
            record = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()
        return int(record["c"]) if record else 0

    def episode_count(self) -> int:
        with self.driver.session() as session:
            record = session.run("MATCH (ep:Episode) RETURN count(ep) AS c").single()
        return int(record["c"]) if record else 0

    def get_top_entities(self, limit: int = 20) -> List[Dict[str, Any]]:
        safe_limit = max(1, int(limit))
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE e.strength IS NOT NULL
                RETURN e.name AS name,
                       coalesce(e.entity_type, 'Concept') AS type,
                       e.strength AS strength,
                       coalesce(e.mention_count, 0) AS mentions
                ORDER BY e.strength DESC
                LIMIT $limit
                """,
                limit=safe_limit,
            )
            return [dict(record) for record in result]
