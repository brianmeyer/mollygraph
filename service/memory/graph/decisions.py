"""Decision operations for the bi-temporal graph."""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from .constants import _to_iso_datetime


class DecisionMixin:
    @staticmethod
    def _clean_text_list(values: list[str] | None) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for value in values or []:
            item = str(value).strip()
            if not item:
                continue
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(item)
        return cleaned

    @staticmethod
    def _decision_row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
        confidence_val = row.get("confidence")
        confidence = float(confidence_val) if confidence_val is not None else None
        return {
            "id": str(row.get("id") or ""),
            "decision": str(row.get("decision") or ""),
            "reasoning": str(row.get("reasoning") or ""),
            "alternatives": [str(v) for v in (row.get("alternatives") or [])],
            "inputs": [str(v) for v in (row.get("inputs") or [])],
            "outcome": str(row.get("outcome") or ""),
            "decided_by": str(row.get("decided_by") or ""),
            "timestamp": _to_iso_datetime(row.get("timestamp")),
            "source_episode_id": row.get("source_episode_id"),
            "confidence": confidence,
            "related_entities": [str(v) for v in (row.get("related_entities") or [])],
            "preceded_by_decision_id": row.get("preceded_by_decision_id"),
        }

    def create_decision(
        self,
        *,
        decision: str,
        reasoning: str,
        alternatives: list[str] | None = None,
        inputs: list[str] | None = None,
        outcome: str,
        decided_by: str,
        related_entities: list[str] | None = None,
        preceded_by_decision_id: str | None = None,
        source_episode_id: str | None = None,
        confidence: float | None = None,
        timestamp: datetime | None = None,
        decision_id: str | None = None,
    ) -> dict[str, Any]:
        decision_text = str(decision).strip()
        if not decision_text:
            raise ValueError("decision is required")

        decided_by_name = str(decided_by).strip()
        if not decided_by_name:
            raise ValueError("decided_by is required")

        ts = timestamp or datetime.now(UTC)
        ts_iso = ts.isoformat()
        created_id = str(decision_id or uuid.uuid4())
        safe_confidence = (
            max(0.0, min(1.0, float(confidence))) if confidence is not None else None
        )

        clean_alternatives = self._clean_text_list(alternatives)
        clean_inputs = self._clean_text_list(inputs)
        clean_related_entities = self._clean_text_list(related_entities)

        with self.driver.session() as session:
            if preceded_by_decision_id:
                prev = session.run(
                    "MATCH (d:Decision {id: $id}) RETURN d.id AS id",
                    id=preceded_by_decision_id,
                ).single()
                if prev is None:
                    raise ValueError(
                        f"preceded_by_decision_id '{preceded_by_decision_id}' not found"
                    )

            if source_episode_id:
                episode = session.run(
                    "MATCH (ep:Episode {id: $id}) RETURN ep.id AS id",
                    id=source_episode_id,
                ).single()
                if episode is None:
                    raise ValueError(f"source_episode_id '{source_episode_id}' not found")

            record = session.run(
                """
                MERGE (d:Decision {id: $id})
                ON CREATE SET
                    d.decision = $decision,
                    d.reasoning = $reasoning,
                    d.alternatives = $alternatives,
                    d.inputs = $inputs,
                    d.outcome = $outcome,
                    d.decided_by = $decided_by,
                    d.timestamp = datetime($timestamp),
                    d.source_episode_id = $source_episode_id,
                    d.confidence = $confidence,
                    d.created_at = datetime()
                ON MATCH SET
                    d.decision = $decision,
                    d.reasoning = $reasoning,
                    d.alternatives = $alternatives,
                    d.inputs = $inputs,
                    d.outcome = $outcome,
                    d.decided_by = $decided_by,
                    d.timestamp = datetime($timestamp),
                    d.source_episode_id = $source_episode_id,
                    d.confidence = $confidence,
                    d.updated_at = datetime()
                WITH d
                MERGE (p:Entity {name: $decided_by})
                ON CREATE SET
                    p:Person,
                    p.id = randomUUID(),
                    p.entity_type = 'Person',
                    p.aliases = [],
                    p.first_mentioned = datetime($timestamp),
                    p.last_mentioned = datetime($timestamp),
                    p.valid_from = datetime($timestamp),
                    p.valid_to = NULL,
                    p.last_seen = datetime($timestamp),
                    p.confidence = 1.0,
                    p.mention_count = 1,
                    p.strength = 1.0,
                    p.embedding_stale = false
                ON MATCH SET
                    p:Person,
                    p.entity_type = coalesce(p.entity_type, 'Person'),
                    p.last_mentioned = datetime($timestamp),
                    p.last_seen = datetime($timestamp),
                    p.valid_to = NULL,
                    p.mention_count = coalesce(p.mention_count, 0) + 1
                MERGE (d)-[:DECIDED_BY]->(p)
                WITH d
                FOREACH (related_name IN $related_entities |
                    MERGE (e:Entity {name: related_name})
                    ON CREATE SET
                        e.id = randomUUID(),
                        e.entity_type = 'Concept',
                        e.aliases = [],
                        e.first_mentioned = datetime($timestamp),
                        e.last_mentioned = datetime($timestamp),
                        e.valid_from = datetime($timestamp),
                        e.valid_to = NULL,
                        e.last_seen = datetime($timestamp),
                        e.confidence = 0.5,
                        e.mention_count = 1,
                        e.strength = 1.0,
                        e.embedding_stale = false
                    ON MATCH SET
                        e.last_mentioned = datetime($timestamp),
                        e.last_seen = datetime($timestamp),
                        e.valid_to = NULL
                    MERGE (d)-[:RELATES_TO]->(e)
                )
                WITH d
                OPTIONAL MATCH (prev:Decision {id: $preceded_by_decision_id})
                FOREACH (_ IN CASE WHEN prev IS NULL THEN [] ELSE [1] END |
                    MERGE (d)-[:PRECEDED_BY]->(prev)
                )
                WITH d
                OPTIONAL MATCH (ep:Episode {id: $source_episode_id})
                FOREACH (_ IN CASE WHEN ep IS NULL THEN [] ELSE [1] END |
                    MERGE (d)-[:SOURCED_FROM]->(ep)
                )
                RETURN d.id AS id
                """,
                id=created_id,
                decision=decision_text,
                reasoning=str(reasoning or "").strip(),
                alternatives=clean_alternatives,
                inputs=clean_inputs,
                outcome=str(outcome or "").strip(),
                decided_by=decided_by_name,
                timestamp=ts_iso,
                source_episode_id=source_episode_id,
                confidence=safe_confidence,
                related_entities=clean_related_entities,
                preceded_by_decision_id=preceded_by_decision_id,
            ).single()

        if record is None or not record.get("id"):
            raise RuntimeError("failed to create decision")

        created = self.get_decision(str(record["id"]))
        if created is None:
            raise RuntimeError("created decision could not be fetched")
        return created

    def _query_decisions(
        self,
        *,
        q: str | None = None,
        decided_by: str | None = None,
        limit: int = 20,
        decision_id: str | None = None,
    ) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 200))
        clean_q = str(q).strip() if q is not None else None
        clean_decided_by = str(decided_by).strip() if decided_by is not None else None
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (d:Decision)
                WHERE ($decision_id IS NULL OR d.id = $decision_id)
                OPTIONAL MATCH (d)-[:DECIDED_BY]->(p:Entity)
                WITH d, coalesce(p.name, d.decided_by) AS decided_by_name
                WHERE ($q IS NULL
                       OR toLower(coalesce(d.decision, '')) CONTAINS toLower($q)
                       OR toLower(coalesce(d.reasoning, '')) CONTAINS toLower($q)
                       OR toLower(coalesce(d.outcome, '')) CONTAINS toLower($q))
                  AND ($decided_by IS NULL OR toLower(coalesce(decided_by_name, '')) = toLower($decided_by))
                OPTIONAL MATCH (d)-[:RELATES_TO]->(e:Entity)
                WITH d, decided_by_name, collect(DISTINCT e.name) AS related_entities
                OPTIONAL MATCH (d)-[:PRECEDED_BY]->(prev:Decision)
                WITH d, decided_by_name, related_entities, collect(DISTINCT prev.id) AS preceded_ids
                OPTIONAL MATCH (d)-[:SOURCED_FROM]->(ep:Episode)
                WITH d, decided_by_name, related_entities, preceded_ids, collect(DISTINCT ep.id) AS sourced_ids
                RETURN d.id AS id,
                       d.decision AS decision,
                       d.reasoning AS reasoning,
                       coalesce(d.alternatives, []) AS alternatives,
                       coalesce(d.inputs, []) AS inputs,
                       d.outcome AS outcome,
                       coalesce(decided_by_name, d.decided_by) AS decided_by,
                       d.timestamp AS timestamp,
                       coalesce(sourced_ids[0], d.source_episode_id) AS source_episode_id,
                       d.confidence AS confidence,
                       [name IN related_entities WHERE name IS NOT NULL] AS related_entities,
                       preceded_ids[0] AS preceded_by_decision_id
                ORDER BY d.timestamp DESC
                LIMIT $limit
                """,
                decision_id=decision_id,
                q=clean_q if clean_q else None,
                decided_by=clean_decided_by if clean_decided_by else None,
                limit=safe_limit,
            )
            return [self._decision_row_to_dict(dict(record)) for record in result]

    def list_decisions(
        self,
        *,
        q: str | None = None,
        decided_by: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        return self._query_decisions(q=q, decided_by=decided_by, limit=limit)

    def get_decision(self, decision_id: str) -> dict[str, Any] | None:
        rows = self._query_decisions(decision_id=str(decision_id).strip(), limit=1)
        return rows[0] if rows else None
