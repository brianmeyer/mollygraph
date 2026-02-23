"""Entity operations for the bi-temporal graph."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

from memory.models import Entity

from .constants import ENTITY_BLOCKLIST, _SAFE_PROPERTY_KEY, log


class EntityMixin:
    @staticmethod
    def _normalize_name(name: str) -> str:
        return name.strip().lower()

    @staticmethod
    def _fuzzy_ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, EntityMixin._normalize_name(a), EntityMixin._normalize_name(b)).ratio()

    @staticmethod
    def _is_blocklisted(name: str) -> bool:
        stripped = str(name).strip()
        if len(stripped) < 3:
            return True
        return stripped.lower() in ENTITY_BLOCKLIST

    def find_matching_entity(
        self,
        name: str,
        entity_type: str,
        fuzzy_threshold: float = 0.85,
    ) -> str | None:
        """Find an existing entity by exact/alias/fuzzy match."""
        if len(name.strip()) < 3:
            return None

        normalized = self._normalize_name(name)
        with self.driver.session() as session:
            exact = session.run(
                "MATCH (e:Entity) WHERE toLower(e.name) = $name RETURN e.name AS name",
                name=normalized,
            ).single()
            if exact:
                return str(exact["name"])

            alias = session.run(
                """
                MATCH (e:Entity)
                WHERE ANY(a IN coalesce(e.aliases, []) WHERE toLower(a) = $name)
                RETURN e.name AS name
                """,
                name=normalized,
            ).single()
            if alias:
                return str(alias["name"])

            effective_threshold = fuzzy_threshold
            if len(name) < 8:
                effective_threshold = max(fuzzy_threshold, 0.95)
            elif len(name) < 12:
                effective_threshold = max(fuzzy_threshold, 0.90)

            rows = session.run(
                "MATCH (e:Entity {entity_type: $etype}) RETURN e.name AS name",
                etype=entity_type,
            )
            best_match: str | None = None
            best_ratio = 0.0
            for row in rows:
                existing_name = str(row["name"])
                ratio = self._fuzzy_ratio(name, existing_name)
                if ratio >= effective_threshold and ratio > best_ratio:
                    best_match = existing_name
                    best_ratio = ratio

            if best_match:
                log.debug(
                    "Fuzzy matched '%s' -> '%s' (ratio=%.3f, threshold=%.2f)",
                    name,
                    best_match,
                    best_ratio,
                    effective_threshold,
                )
            return best_match

    def upsert_entity(
        self,
        entity: Entity | str,
        entity_type: str | None = None,
        confidence: float | None = None,
    ) -> Tuple[str, bool] | str:
        """Upsert entity using model-first API or legacy args."""
        if isinstance(entity, Entity):
            if self._is_blocklisted(entity.name):
                return entity.name.strip(), False

            matched_name = self.find_matching_entity(entity.name, entity.entity_type)
            with self.driver.session() as session:
                existing = None
                if matched_name:
                    existing = session.run(
                        """
                        MATCH (e:Entity)
                        WHERE e.name = $matched_name
                        RETURN e.id AS id, e.name AS canonical_name
                        """,
                        matched_name=matched_name,
                    ).single()
                if existing is None:
                    existing = session.run(
                        """
                        MATCH (e:Entity)
                        WHERE toLower(e.name) = toLower($name)
                        RETURN e.id AS id, e.name AS canonical_name
                        """,
                        name=entity.name,
                    ).single()

                if existing:
                    canonical_name = str(existing.get("canonical_name") or entity.name).strip()
                    alias = entity.name.strip()
                    alias_to_add = alias if alias and alias.lower() != canonical_name.lower() else None
                    if canonical_name:
                        entity.name = canonical_name
                    session.run(
                        """
                        MATCH (e:Entity {id: $id})
                        SET e.last_mentioned = datetime($last_mentioned),
                            e.last_seen = datetime($last_seen),
                            e.valid_from = coalesce(e.valid_from, e.created_at, e.first_mentioned, datetime($valid_from)),
                            e.valid_to = NULL,
                            e.mention_count = coalesce(e.mention_count, 0) + 1,
                            e.strength = $strength,
                            e.confidence = CASE
                                WHEN $confidence > coalesce(e.confidence, 0.0) THEN $confidence
                                ELSE coalesce(e.confidence, $confidence)
                            END,
                            e.aliases = CASE
                                WHEN $new_alias IS NOT NULL
                                     AND NOT toLower($new_alias) IN [x IN coalesce(e.aliases, []) | toLower(x)]
                                THEN coalesce(e.aliases, []) + $new_alias
                                ELSE coalesce(e.aliases, [])
                            END
                        """,
                        id=existing["id"],
                        last_mentioned=entity.last_mentioned.isoformat(),
                        last_seen=entity.last_mentioned.isoformat(),
                        valid_from=entity.first_mentioned.isoformat(),
                        strength=entity.strength,
                        confidence=entity.confidence,
                        new_alias=alias_to_add,
                    )
                    return str(existing["id"]), False

                session.run(
                    """
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

        # Legacy compatibility path (old function signature).
        name = str(entity).strip()
        etype = str(entity_type or "Concept")
        conf = float(confidence if confidence is not None else 0.5)
        if self._is_blocklisted(name):
            log.debug("Blocked entity '%s' (blocklisted or too short)", name)
            return name

        now = datetime.now(timezone.utc).isoformat()
        matched = self.find_matching_entity(name, etype)
        with self.driver.session() as session:
            if matched:
                session.run(
                    """
                    MATCH (e:Entity) WHERE e.name = $existing_name
                    SET e.mention_count = coalesce(e.mention_count, 0) + 1,
                        e.last_mentioned = datetime($now),
                        e.last_seen = datetime($now),
                        e.valid_from = coalesce(e.valid_from, e.first_mentioned, datetime($now)),
                        e.valid_to = NULL,
                        e.confidence = CASE
                            WHEN $confidence > coalesce(e.confidence, 0.0) THEN $confidence
                            ELSE coalesce(e.confidence, $confidence)
                        END,
                        e.aliases = CASE
                            WHEN NOT toLower($alias) IN [x IN coalesce(e.aliases, []) | toLower(x)]
                                 AND toLower($alias) <> toLower(e.name)
                            THEN coalesce(e.aliases, []) + $alias
                            ELSE coalesce(e.aliases, [])
                        END
                    """,
                    existing_name=matched,
                    now=now,
                    confidence=conf,
                    alias=name,
                )
                return matched

            session.run(
                """
                CREATE (e:Entity {
                    id: $id,
                    name: $name,
                    entity_type: $entity_type,
                    mention_count: 1,
                    first_mentioned: datetime($now),
                    last_mentioned: datetime($now),
                    valid_from: datetime($now),
                    valid_to: NULL,
                    last_seen: datetime($now),
                    strength: 1.0,
                    confidence: $confidence,
                    aliases: [],
                    summary: ''
                })
                """,
                id=str(uuid.uuid4()),
                name=name,
                entity_type=etype,
                now=now,
                confidence=conf,
            )
            return name

    def upsert_entity_sync(self, name: str, entity_type: str, confidence: float) -> str:
        """Compatibility wrapper for old function-based API."""
        result = self.upsert_entity(name, entity_type, confidence)
        return str(result)

    def set_entity_properties(self, name: str, properties: dict[str, Any]) -> None:
        if not properties:
            return
        for key in properties:
            if not _SAFE_PROPERTY_KEY.match(key):
                raise ValueError(f"Unsafe Cypher property key: {key!r}")
        set_clause = ", ".join(f"e.{key} = ${key}" for key in properties)
        with self.driver.session() as session:
            session.run(
                f"MATCH (e:Entity {{name: $name}}) SET {set_clause}",
                name=name,
                **properties,
            )

    def delete_entity(self, name: str) -> bool:
        normalized = self._normalize_name(name)
        with self.driver.session() as session:
            record = session.run(
                """
                MATCH (e:Entity)
                WHERE toLower(e.name) = $name
                   OR ANY(a IN coalesce(e.aliases, []) WHERE toLower(a) = $name)
                DETACH DELETE e
                RETURN count(e) AS deleted
                """,
                name=normalized,
            ).single()
        return bool(record and int(record["deleted"]) > 0)

    def delete_blocklisted_entities(self, blocklist: set[str] | None = None) -> int:
        names = [n.lower() for n in (blocklist or ENTITY_BLOCKLIST)]
        if not names:
            return 0
        with self.driver.session() as session:
            record = session.run(
                """
                MATCH (e:Entity)
                WHERE toLower(e.name) IN $names
                DETACH DELETE e
                RETURN count(e) AS deleted
                """,
                names=names,
            ).single()
        deleted = int(record["deleted"]) if record else 0
        log.info("Deleted %d blocklisted entities", deleted)
        return deleted

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
