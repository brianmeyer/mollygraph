"""Relationship operations for the bi-temporal graph."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from memory.models import Relationship

from .constants import (
    VALID_REL_TYPES,
    _REL_TYPE_RE,
    _tier_for_rel_type,
    _to_iso_datetime,
    calculate_strength,
    log,
)


class RelationshipMixin:
    def upsert_relationship(
        self,
        rel: Relationship | str,
        tail_name: str | None = None,
        rel_type: str | None = None,
        confidence: float | None = None,
        context_snippet: str = "",
    ) -> Tuple[str, str] | None:
        """Upsert relationship using model-first API or legacy args."""
        if isinstance(rel, str):
            head_name = rel
            if tail_name is None or rel_type is None:
                raise ValueError(
                    "tail_name and rel_type are required for upsert_relationship(head, tail, rel_type, ...)"
                )
            if self._is_blocklisted(head_name) or self._is_blocklisted(tail_name):
                log.debug(
                    "Blocked relationship '%s'-[%s]->'%s' (blocklisted entity)",
                    head_name,
                    rel_type,
                    tail_name,
                )
                return None
            relationship = Relationship(
                source_entity=str(head_name).strip(),
                target_entity=str(tail_name).strip(),
                relation_type=str(rel_type).strip().upper().replace(" ", "_"),
                confidence=max(0.0, min(1.0, float(confidence if confidence is not None else 0.5))),
                valid_at=datetime.now(timezone.utc),
                observed_at=datetime.now(timezone.utc),
                context_snippets=[(context_snippet or "")[:200]] if context_snippet else [],
                episode_ids=[],
            )
            self.upsert_relationship(relationship)
            return None

        relation_type = self._normalize_rel_type(rel.relation_type)
        if not self._is_valid_rel_type(relation_type):
            try:
                from memory.graph_suggestions import log_relationship_rejection

                log_relationship_rejection(
                    rel.source_entity,
                    rel.target_entity,
                    rel.relation_type,
                    float(rel.confidence),
                    (rel.context_snippets[0] if rel.context_snippets else ""),
                )
            except Exception:
                log.debug("graph suggestion rejection logging failed", exc_info=True)
            return "", "rejected"

        if self._is_blocklisted(rel.source_entity) or self._is_blocklisted(rel.target_entity):
            log.debug(
                "Blocked relationship '%s'-[%s]->'%s' (blocklisted entity)",
                rel.source_entity,
                relation_type,
                rel.target_entity,
            )
            return "", "blocked"

        snippets = [str(snippet)[:200] for snippet in (rel.context_snippets or []) if str(snippet).strip()]
        if not snippets and context_snippet:
            snippets = [context_snippet[:200]]
        rel = rel.model_copy(
            update={
                "relation_type": relation_type,
                "context_snippets": snippets[:3],
            }
        )

        with self.driver.session() as session:
            existing = session.run(
                f"""
                MATCH (a:Entity {{name: $source}})-[r:{relation_type}]->(b:Entity {{name: $target}})
                WHERE coalesce(r.valid_until, r.valid_to) IS NULL
                   OR coalesce(r.valid_until, r.valid_to) > datetime($valid_at)
                RETURN r.observed_at AS observed,
                       r.valid_at AS valid,
                       r.strength AS strength,
                       r.id AS id
                """,
                source=rel.source_entity,
                target=rel.target_entity,
                valid_at=rel.valid_at.isoformat() if rel.valid_at else datetime.utcnow().isoformat(),
            ).single()

            if existing:
                if self._is_contradiction(existing, rel):
                    session.run(
                        """
                        MATCH ()-[r {id: $id}]->()
                        SET r.valid_until = datetime($new_valid_at),
                            r.valid_to = datetime($new_valid_at),
                            r.status = 'historical'
                        """,
                        id=existing["id"],
                        new_valid_at=(rel.valid_at or datetime.utcnow()).isoformat(),
                    )
                    return self._create_relationship(session, rel), "superseded"

                session.run(
                    """
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
                        END,
                        r.audit_status = CASE
                            WHEN r.audit_status IN ['quarantined', 'verified'] THEN r.audit_status
                            ELSE coalesce(r.audit_status, 'unverified')
                        END
                    """,
                    id=existing["id"],
                    now=datetime.utcnow().isoformat(),
                    new_snippets=rel.context_snippets,
                    new_episode_ids=rel.episode_ids,
                )

                if relation_type == "RELATED_TO":
                    self._maybe_log_related_to_hotspot(
                        source=rel.source_entity,
                        target=rel.target_entity,
                        relationship_id=str(existing["id"]),
                    )
                return str(existing["id"]), "updated"

            created_id = self._create_relationship(session, rel)
            if relation_type == "RELATED_TO":
                self._maybe_log_related_to_hotspot(
                    source=rel.source_entity,
                    target=rel.target_entity,
                    relationship_id=created_id,
                )
            return created_id, "created"

    def upsert_relationship_sync(
        self,
        head_name: str,
        tail_name: str,
        rel_type: str,
        confidence: float,
        context_snippet: str = "",
    ) -> None:
        """Compatibility wrapper for old function-based API."""
        self.upsert_relationship(head_name, tail_name, rel_type, confidence, context_snippet)

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

        # Normalize timezone awareness for safe subtraction
        if new.valid_at and existing_valid:
            from datetime import timezone as _tz
            if new.valid_at.tzinfo is not None and existing_valid.tzinfo is None:
                existing_valid = existing_valid.replace(tzinfo=_tz.utc)
            elif new.valid_at.tzinfo is None and existing_valid.tzinfo is not None:
                existing_valid = existing_valid.replace(tzinfo=None)

        if new.valid_at and (new.valid_at - existing_valid).days > 30:
            return True
        
        return False

    def _create_relationship(self, session, rel: Relationship) -> str:
        """Create (or idempotently upsert) a relationship edge.

        Uses MERGE instead of CREATE so that two concurrent workers racing
        on the same (head, rel_type, tail) triple will converge to a single
        edge rather than producing a duplicate.  ON CREATE initialises all
        properties; ON MATCH increments mention_count to account for the
        concurrent write (Issue 4).
        """
        rel_id = str(uuid.uuid4())

        tier = _tier_for_rel_type(rel.relation_type)
        strength = calculate_strength(
            mention_count=rel.mention_count,
            days_since=0,
            tier=tier,
        )

        now_iso = datetime.utcnow().isoformat()
        valid_at_iso = rel.valid_at.isoformat() if rel.valid_at else now_iso
        valid_until_iso = rel.valid_until.isoformat() if rel.valid_until else None
        observed_iso = rel.observed_at.isoformat()

        # Relationship type cannot be parameterised in Cypher — the f-string
        # is safe because relation_type is already validated by _is_valid_rel_type
        # (regex + allowlist) before this method is called.
        result = session.run(
            f"""
            MATCH (a:Entity {{name: $source}}), (b:Entity {{name: $target}})
            MERGE (a)-[r:{rel.relation_type}]->(b)
            ON CREATE SET
                r.id             = $id,
                r.valid_at       = datetime($valid_at),
                r.valid_until    = $valid_until,
                r.valid_from     = datetime($valid_at),
                r.valid_to       = NULL,
                r.observed_at    = datetime($observed_at),
                r.first_mentioned = datetime($first_mentioned),
                r.last_mentioned = datetime($last_mentioned),
                r.confidence     = $confidence,
                r.strength       = $strength,
                r.mention_count  = $mention_count,
                r.context_snippets = $context_snippets,
                r.episode_ids    = $episode_ids,
                r.audit_status   = $audit_status
            ON MATCH SET
                r.mention_count  = coalesce(r.mention_count, 0) + 1,
                r.strength       = coalesce(r.strength, 0.0) + log(1 + coalesce(r.mention_count, 1)),
                r.last_mentioned = datetime($last_mentioned),
                r.context_snippets = CASE
                    WHEN size(coalesce(r.context_snippets, [])) >= 3
                    THEN coalesce(r.context_snippets, [])[1..] + $context_snippets
                    ELSE coalesce(r.context_snippets, []) + $context_snippets
                END,
                r.episode_ids = coalesce(r.episode_ids, []) + [
                    eid IN $episode_ids WHERE NOT eid IN coalesce(r.episode_ids, [])
                ]
            RETURN r.id AS created_id
            """,
            source=rel.source_entity,
            target=rel.target_entity,
            id=rel_id,
            valid_at=valid_at_iso,
            valid_until=valid_until_iso,
            observed_at=observed_iso,
            first_mentioned=observed_iso,
            last_mentioned=observed_iso,
            confidence=rel.confidence,
            strength=strength,
            mention_count=rel.mention_count,
            context_snippets=rel.context_snippets,
            episode_ids=rel.episode_ids,
            audit_status=rel.audit_status,
        )

        record = result.single()
        if record is None:
            log.warning(
                "Relationship not created — missing entity: '%s' -[%s]-> '%s'",
                rel.source_entity,
                rel.relation_type,
                rel.target_entity,
            )
            return ""

        # Return the id that is actually on the edge (may be a pre-existing id
        # if a concurrent worker won the MERGE race).
        return str(record["created_id"])

    def _maybe_log_related_to_hotspot(self, source: str, target: str, relationship_id: str) -> None:
        """Emit suggestion signal when RELATED_TO reaches hotspot threshold."""
        try:
            with self.driver.session() as session:
                record = session.run(
                    """
                    MATCH ()-[r]->()
                    WHERE r.id = $id
                    RETURN coalesce(r.mention_count, 0) AS mentions
                    """,
                    id=relationship_id,
                ).single()
            mentions = int(record["mentions"]) if record else 0
            if mentions == 3:
                from memory.graph_suggestions import log_repeated_related_to

                log_repeated_related_to(source, target, mentions)
        except Exception:
            log.debug("graph suggestion hotspot logging failed", exc_info=True)

    @staticmethod
    def _normalize_rel_type(rel_type: str) -> str:
        return rel_type.strip().upper().replace(" ", "_")

    @staticmethod
    def _is_valid_rel_type(rel_type: str) -> bool:
        return rel_type in VALID_REL_TYPES and bool(_REL_TYPE_RE.match(rel_type))

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
