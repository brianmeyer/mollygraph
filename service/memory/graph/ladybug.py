"""Ladybug-backed graph implementation for the stripped-down local runtime."""
from __future__ import annotations

from datetime import datetime, UTC, timezone
from difflib import SequenceMatcher
from pathlib import Path
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple

import real_ladybug as lb

from memory.models import Entity, Episode, Relationship

from .constants import ENTITY_BLOCKLIST, VALID_REL_TYPES, _tier_for_rel_type, calculate_strength, log
from .entities import _sanitize_doc_id


class LadybugGraph:
    """Embedded graph store backed by Ladybug."""

    ENTITY_TABLE = "Entity"
    EPISODE_TABLE = "Episode"
    ENTITY_REL_TABLES = tuple(sorted(rel for rel in VALID_REL_TYPES if rel != "MENTIONS"))
    EPISODE_REL_TABLES = ("MENTIONS",)

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = lb.Database(str(self.db_path))
        self.conn = lb.Connection(self.db)
        self._lock = threading.RLock()
        self._vector_store = None
        self._ensure_schema()

    @staticmethod
    def _normalize_name(name: str) -> str:
        return str(name or "").strip().lower()

    @staticmethod
    def _fuzzy_ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, LadybugGraph._normalize_name(a), LadybugGraph._normalize_name(b)).ratio()

    @staticmethod
    def _is_blocklisted(name: str) -> bool:
        stripped = str(name or "").strip()
        if len(stripped) < 3:
            return True
        return stripped.lower() in ENTITY_BLOCKLIST

    @staticmethod
    def _search_text(name: str, aliases: list[str]) -> str:
        parts = [str(name or "").strip()]
        parts.extend(str(alias or "").strip() for alias in aliases if str(alias or "").strip())
        return " ".join(parts).strip().lower()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def _ensure_schema(self) -> None:
        with self._lock:
            self.conn.execute(
                """
                CREATE NODE TABLE IF NOT EXISTS Entity(
                    id STRING PRIMARY KEY,
                    name STRING,
                    entity_type STRING,
                    aliases STRING[],
                    search_text STRING,
                    summary STRING,
                    description STRING,
                    first_mentioned TIMESTAMP,
                    last_mentioned TIMESTAMP,
                    valid_from TIMESTAMP,
                    valid_to TIMESTAMP,
                    last_seen TIMESTAMP,
                    created_at TIMESTAMP,
                    confidence DOUBLE,
                    mention_count INT64,
                    strength DOUBLE,
                    created_from_episode STRING,
                    verified BOOL,
                    embedding_stale BOOL,
                    first_seen_source STRING,
                    first_seen_at TIMESTAMP
                )
                """
            )
            self.conn.execute(
                """
                CREATE NODE TABLE IF NOT EXISTS Episode(
                    id STRING PRIMARY KEY,
                    source STRING,
                    source_id STRING,
                    content_preview STRING,
                    content_hash STRING,
                    occurred_at TIMESTAMP,
                    ingested_at TIMESTAMP,
                    processed_at TIMESTAMP,
                    processor_version STRING,
                    entities_extracted STRING[],
                    status STRING,
                    incomplete BOOL,
                    incomplete_reason STRING,
                    incomplete_at TIMESTAMP
                )
                """
            )
            for rel_type in self.ENTITY_REL_TABLES:
                self.conn.execute(
                    f"""
                    CREATE REL TABLE IF NOT EXISTS {rel_type}(
                        FROM Entity TO Entity,
                        id STRING,
                        confidence DOUBLE,
                        strength DOUBLE,
                        mention_count INT64,
                        valid_at TIMESTAMP,
                        valid_from TIMESTAMP,
                        valid_to TIMESTAMP,
                        observed_at TIMESTAMP,
                        last_mentioned TIMESTAMP,
                        first_mentioned TIMESTAMP,
                        audit_status STRING,
                        verified_by STRING,
                        verified_at TIMESTAMP,
                        status STRING,
                        context_snippets STRING[],
                        episode_ids STRING[]
                    )
                    """
                )
            self.conn.execute(
                """
                CREATE REL TABLE IF NOT EXISTS MENTIONS(
                    FROM Episode TO Entity
                )
                """
            )

    def _rows(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        with self._lock:
            result = self.conn.execute(query, parameters or {})
            return result.rows_as_dict().get_all()

    def _entity_rows(self) -> list[dict[str, Any]]:
        return self._rows(
            """
            MATCH (e:Entity)
            RETURN e.id AS id,
                   e.name AS name,
                   e.entity_type AS entity_type,
                   coalesce(e.aliases, []) AS aliases,
                   e.confidence AS confidence,
                   e.mention_count AS mention_count,
                   e.strength AS strength,
                   e.description AS description,
                   e.summary AS summary,
                   e.created_from_episode AS created_from_episode,
                   e.first_mentioned AS first_mentioned,
                   e.last_mentioned AS last_mentioned,
                   e.valid_from AS valid_from,
                   e.valid_to AS valid_to,
                   e.last_seen AS last_seen,
                   e.verified AS verified,
                   e.embedding_stale AS embedding_stale
            """
        )

    def get_entity_id_by_name(self, name: str) -> str | None:
        normalized = self._normalize_name(name)
        if not normalized:
            return None
        for row in self._entity_rows():
            row_name = self._normalize_name(row.get("name") or "")
            aliases = [self._normalize_name(alias) for alias in (row.get("aliases") or [])]
            if normalized == row_name or normalized in aliases:
                value = str(row.get("id") or "").strip()
                return value or None
        return None

    def _find_matching_entity_row(
        self,
        name: str,
        entity_type: str,
        fuzzy_threshold: float = 0.85,
    ) -> dict[str, Any] | None:
        normalized = self._normalize_name(name)
        if not normalized:
            return None

        candidates = [
            row for row in self._entity_rows()
            if str(row.get("entity_type") or "") == entity_type
        ]
        for row in candidates:
            row_name = self._normalize_name(row.get("name") or "")
            if normalized == row_name:
                return row
            aliases = [self._normalize_name(alias) for alias in (row.get("aliases") or [])]
            if normalized in aliases:
                return row

        effective_threshold = fuzzy_threshold
        if len(name) < 8:
            effective_threshold = max(fuzzy_threshold, 0.95)
        elif len(name) < 12:
            effective_threshold = max(fuzzy_threshold, 0.90)

        best_row: dict[str, Any] | None = None
        best_ratio = 0.0
        for row in candidates:
            existing_name = str(row.get("name") or "")
            ratio = self._fuzzy_ratio(name, existing_name)
            if ratio >= effective_threshold and ratio > best_ratio:
                best_ratio = ratio
                best_row = row
        return best_row

    def upsert_entity(
        self,
        entity: Entity | str,
        entity_type: str | None = None,
        confidence: float | None = None,
    ) -> Tuple[str, bool] | str:
        if isinstance(entity, str):
            now = datetime.now(UTC)
            temp = Entity(
                name=entity.strip(),
                entity_type=str(entity_type or "Concept"),
                first_mentioned=now,
                last_mentioned=now,
                confidence=float(confidence or 0.5),
                created_from_episode="",
            )
            return self.upsert_entity(temp)[0]

        if self._is_blocklisted(entity.name):
            return entity.name.strip(), False

        matched = self._find_matching_entity_row(entity.name, entity.entity_type)
        if matched:
            entity_id = str(matched.get("id") or entity.id).strip()
            canonical_name = str(matched.get("name") or entity.name).strip()
            aliases = [str(alias).strip() for alias in (matched.get("aliases") or []) if str(alias).strip()]
            if entity.name.strip() and self._normalize_name(entity.name) != self._normalize_name(canonical_name):
                if self._normalize_name(entity.name) not in {self._normalize_name(alias) for alias in aliases}:
                    aliases.append(entity.name.strip())

            params = {
                "id": entity_id,
                "name": canonical_name,
                "aliases": aliases,
                "search_text": self._search_text(canonical_name, aliases),
                "last_mentioned": entity.last_mentioned,
                "valid_from": matched.get("valid_from") or entity.first_mentioned,
                "last_seen": entity.last_mentioned,
                "strength": entity.strength,
                "confidence": max(float(entity.confidence), float(matched.get("confidence") or 0.0)),
                "mention_count": int(matched.get("mention_count") or 0) + 1,
                "description": entity.description or matched.get("description"),
                "summary": matched.get("summary"),
                "created_from_episode": matched.get("created_from_episode") or entity.created_from_episode,
                "verified": bool(matched.get("verified") or entity.verified),
            }
            with self._lock:
                self.conn.execute(
                    """
                    MATCH (e:Entity {id: $id})
                    SET e.name = $name,
                        e.aliases = $aliases,
                        e.search_text = $search_text,
                        e.last_mentioned = $last_mentioned,
                        e.valid_from = $valid_from,
                        e.valid_to = NULL,
                        e.last_seen = $last_seen,
                        e.strength = $strength,
                        e.confidence = $confidence,
                        e.mention_count = $mention_count,
                        e.description = $description,
                        e.summary = $summary,
                        e.created_from_episode = $created_from_episode,
                        e.verified = $verified,
                        e.embedding_stale = false
                    """,
                    params,
                )
            entity.name = canonical_name

            if aliases and self._vector_store is not None:
                alias_doc_id = _sanitize_doc_id(str(entity.name or ""))
                if alias_doc_id != entity_id:
                    try:
                        self._vector_store.remove_entity(alias_doc_id)
                    except Exception:
                        log.debug("LadybugGraph alias vector cleanup failed", exc_info=True)
            return entity_id, False

        params = {
            "id": entity.id,
            "name": entity.name.strip(),
            "entity_type": entity.entity_type,
            "aliases": [alias.strip() for alias in entity.aliases if alias.strip()],
            "search_text": self._search_text(entity.name, entity.aliases),
            "summary": None,
            "description": entity.description,
            "first_mentioned": entity.first_mentioned,
            "last_mentioned": entity.last_mentioned,
            "valid_from": entity.first_mentioned,
            "valid_to": None,
            "last_seen": entity.last_mentioned,
            "created_at": datetime.now(UTC),
            "confidence": entity.confidence,
            "mention_count": entity.mention_count,
            "strength": entity.strength,
            "created_from_episode": entity.created_from_episode,
            "verified": entity.verified,
            "embedding_stale": False,
            "first_seen_source": None,
            "first_seen_at": None,
        }
        with self._lock:
            self.conn.execute(
                """
                CREATE (e:Entity {
                    id: $id,
                    name: $name,
                    entity_type: $entity_type,
                    aliases: $aliases,
                    search_text: $search_text,
                    summary: $summary,
                    description: $description,
                    first_mentioned: $first_mentioned,
                    last_mentioned: $last_mentioned,
                    valid_from: $valid_from,
                    valid_to: $valid_to,
                    last_seen: $last_seen,
                    created_at: $created_at,
                    confidence: $confidence,
                    mention_count: $mention_count,
                    strength: $strength,
                    created_from_episode: $created_from_episode,
                    verified: $verified,
                    embedding_stale: $embedding_stale,
                    first_seen_source: $first_seen_source,
                    first_seen_at: $first_seen_at
                })
                """,
                params,
            )
        return entity.id, True

    def tag_entity_first_seen(self, entity_id: str, source: str) -> bool:
        with self._lock:
            self.conn.execute(
                """
                MATCH (e:Entity {id: $entity_id})
                SET e.first_seen_source = coalesce(e.first_seen_source, $source),
                    e.first_seen_at = coalesce(e.first_seen_at, $first_seen_at)
                """,
                {
                    "entity_id": entity_id,
                    "source": str(source or "").strip() or "manual",
                    "first_seen_at": datetime.now(UTC),
                },
            )
        return True

    def create_episode(
        self,
        episode: Episode | str,
        source: str | None = None,
        entity_names: list[str] | None = None,
    ) -> str:
        if not isinstance(episode, Episode):
            if source is None:
                raise ValueError("source is required when using create_episode(content_preview, source, entity_names)")
            episode = Episode(
                source=str(source),
                content_preview=str(episode)[:500],
                content_hash=str(uuid.uuid4()),
                occurred_at=datetime.now(UTC),
                entities_extracted=list(entity_names or []),
            )

        params = episode.model_dump()
        with self._lock:
            self.conn.execute(
                """
                CREATE (ep:Episode {
                    id: $id,
                    source: $source,
                    source_id: $source_id,
                    content_preview: $content_preview,
                    content_hash: $content_hash,
                    occurred_at: $occurred_at,
                    ingested_at: $ingested_at,
                    processed_at: $processed_at,
                    processor_version: $processor_version,
                    entities_extracted: $entities_extracted,
                    status: $status,
                    incomplete: false,
                    incomplete_reason: NULL,
                    incomplete_at: NULL
                })
                """,
                params,
            )
            for entity_name in episode.entities_extracted:
                self.conn.execute(
                    """
                    MATCH (ep:Episode {id: $episode_id}), (e:Entity)
                    WHERE e.name = $entity_name
                    MERGE (ep)-[:MENTIONS]->(e)
                    SET e.last_seen = $seen_at,
                        e.last_mentioned = $seen_at
                    """,
                    {
                        "episode_id": episode.id,
                        "entity_name": entity_name,
                        "seen_at": episode.ingested_at,
                    },
                )
        return episode.id

    def mark_episode_incomplete(self, episode_id: str, reason: str = "") -> bool:
        with self._lock:
            self.conn.execute(
                """
                MATCH (ep:Episode {id: $episode_id})
                SET ep.incomplete = true,
                    ep.incomplete_reason = $reason,
                    ep.incomplete_at = $incomplete_at
                """,
                {
                    "episode_id": episode_id,
                    "reason": str(reason or "")[:500],
                    "incomplete_at": datetime.now(UTC),
                },
            )
        return True

    def finalize_episode(self, episode_id: str, entity_names: list[str]) -> bool:
        with self._lock:
            self.conn.execute(
                """
                MATCH (ep:Episode {id: $episode_id})
                SET ep.incomplete = false,
                    ep.incomplete_reason = NULL,
                    ep.entities_extracted = $entity_names
                """,
                {"episode_id": episode_id, "entity_names": entity_names},
            )
            for entity_name in entity_names:
                self.conn.execute(
                    """
                    MATCH (ep:Episode {id: $episode_id}), (e:Entity)
                    WHERE e.name = $entity_name
                    MERGE (ep)-[:MENTIONS]->(e)
                    SET e.last_seen = $seen_at,
                        e.last_mentioned = $seen_at
                    """,
                    {
                        "episode_id": episode_id,
                        "entity_name": entity_name,
                        "seen_at": datetime.now(UTC),
                    },
                )
        return True

    def upsert_relationship(
        self,
        rel: Relationship | str,
        tail_name: str | None = None,
        rel_type: str | None = None,
        confidence: float | None = None,
        context_snippet: str = "",
    ) -> Tuple[str, str] | None:
        if isinstance(rel, str):
            if tail_name is None or rel_type is None:
                raise ValueError("tail_name and rel_type are required for upsert_relationship(head, tail, rel_type, ...)")
            rel = Relationship(
                source_entity=str(rel).strip(),
                target_entity=str(tail_name).strip(),
                relation_type=str(rel_type).strip().upper().replace(" ", "_"),
                confidence=max(0.0, min(1.0, float(confidence if confidence is not None else 0.5))),
                valid_at=datetime.now(timezone.utc),
                observed_at=datetime.now(timezone.utc),
                context_snippets=[(context_snippet or "")[:200]] if context_snippet else [],
                episode_ids=[],
            )

        relation_type = str(rel.relation_type or "").strip().upper().replace(" ", "_")
        if relation_type not in VALID_REL_TYPES or relation_type == "MENTIONS":
            return None
        if self._is_blocklisted(rel.source_entity) or self._is_blocklisted(rel.target_entity):
            return "", "blocked"

        source_id = self.get_entity_id_by_name(rel.source_entity)
        target_id = self.get_entity_id_by_name(rel.target_entity)
        if not source_id or not target_id:
            return "", "missing_entity"

        existing_rows = self._rows(
            f"""
            MATCH (a:Entity {{id: $source_id}})-[r:{relation_type}]->(b:Entity {{id: $target_id}})
            RETURN r.id AS id,
                   r.mention_count AS mention_count,
                   r.context_snippets AS context_snippets,
                   r.episode_ids AS episode_ids,
                   r.confidence AS confidence
            LIMIT 1
            """,
            {"source_id": source_id, "target_id": target_id},
        )
        snippets = [str(snippet)[:200] for snippet in (rel.context_snippets or []) if str(snippet).strip()][:3]
        params = {
            "source_id": source_id,
            "target_id": target_id,
            "id": str(uuid.uuid4()),
            "confidence": rel.confidence,
            "strength": rel.strength or calculate_strength(rel.mention_count, 0, _tier_for_rel_type(relation_type)),
            "mention_count": rel.mention_count,
            "valid_at": rel.valid_at or datetime.now(UTC),
            "valid_from": rel.valid_at or datetime.now(UTC),
            "valid_to": rel.valid_until,
            "observed_at": rel.observed_at,
            "last_mentioned": datetime.now(UTC),
            "first_mentioned": rel.observed_at,
            "audit_status": rel.audit_status,
            "verified_by": rel.verified_by,
            "verified_at": rel.verified_at,
            "status": "active",
            "context_snippets": snippets,
            "episode_ids": rel.episode_ids,
        }

        with self._lock:
            if existing_rows:
                current = existing_rows[0]
                merged_snippets = [
                    *[str(item) for item in (current.get("context_snippets") or []) if str(item).strip()],
                    *snippets,
                ][-3:]
                merged_episode_ids = list(dict.fromkeys([
                    *[str(item) for item in (current.get("episode_ids") or []) if str(item).strip()],
                    *[str(item) for item in rel.episode_ids if str(item).strip()],
                ]))
                self.conn.execute(
                    f"""
                    MATCH (a:Entity {{id: $source_id}})-[r:{relation_type}]->(b:Entity {{id: $target_id}})
                    SET r.confidence = CASE
                            WHEN $confidence > coalesce(r.confidence, 0.0) THEN $confidence
                            ELSE coalesce(r.confidence, $confidence)
                        END,
                        r.strength = $strength,
                        r.mention_count = coalesce(r.mention_count, 0) + $mention_count,
                        r.valid_at = coalesce(r.valid_at, $valid_at),
                        r.valid_from = coalesce(r.valid_from, $valid_from),
                        r.valid_to = $valid_to,
                        r.observed_at = $observed_at,
                        r.last_mentioned = $last_mentioned,
                        r.first_mentioned = coalesce(r.first_mentioned, $first_mentioned),
                        r.audit_status = coalesce(r.audit_status, $audit_status),
                        r.verified_by = coalesce(r.verified_by, $verified_by),
                        r.verified_at = coalesce(r.verified_at, $verified_at),
                        r.status = $status,
                        r.context_snippets = $context_snippets,
                        r.episode_ids = $episode_ids
                    """,
                    {
                        **params,
                        "context_snippets": merged_snippets,
                        "episode_ids": merged_episode_ids,
                    },
                )
                return str(current.get("id") or ""), "updated"

            self.conn.execute(
                f"""
                MATCH (a:Entity {{id: $source_id}}), (b:Entity {{id: $target_id}})
                CREATE (a)-[:{relation_type} {{
                    id: $id,
                    confidence: $confidence,
                    strength: $strength,
                    mention_count: $mention_count,
                    valid_at: $valid_at,
                    valid_from: $valid_from,
                    valid_to: $valid_to,
                    observed_at: $observed_at,
                    last_mentioned: $last_mentioned,
                    first_mentioned: $first_mentioned,
                    audit_status: $audit_status,
                    verified_by: $verified_by,
                    verified_at: $verified_at,
                    status: $status,
                    context_snippets: $context_snippets,
                    episode_ids: $episode_ids
                }}]->(b)
                """,
                params,
            )
        return params["id"], "created"

    def find_entities_containing(self, query: str, limit: int = 5) -> list[str]:
        normalized = self._normalize_name(query)
        if not normalized:
            return []
        rows = self._rows(
            """
            MATCH (e:Entity)
            WHERE lower(e.search_text) CONTAINS $q
            RETURN e.name AS name
            ORDER BY CASE WHEN lower(e.name) = $q THEN 0 ELSE 1 END, e.name ASC
            LIMIT $limit
            """,
            {"q": normalized, "limit": max(1, int(limit))},
        )
        return [str(row.get("name") or "").strip() for row in rows if str(row.get("name") or "").strip()]

    def get_neighborhood_stats(self, entity_name: str) -> tuple[int, float]:
        entity_id = self.get_entity_id_by_name(entity_name)
        if not entity_id:
            return 0, 0.0
        rows = self._rows(
            """
            MATCH (e:Entity {id: $entity_id})-[r]-(neighbor:Entity)
            RETURN count(DISTINCT neighbor) AS neighbor_count,
                   avg(r.strength) AS avg_strength
            """,
            {"entity_id": entity_id},
        )
        if not rows:
            return 0, 0.0
        return int(rows[0].get("neighbor_count") or 0), float(rows[0].get("avg_strength") or 0.0)

    def get_path_distance(self, from_name: str, to_name: str, max_hops: int = 2) -> int | None:
        from_id = self.get_entity_id_by_name(from_name)
        to_id = self.get_entity_id_by_name(to_name)
        if not from_id or not to_id:
            return None
        rows = self._rows(
            f"""
            MATCH p = (a:Entity {{id: $from_id}})-[* SHORTEST 1..{max(1, int(max_hops))}]-(b:Entity {{id: $to_id}})
            RETURN length(p) AS dist
            LIMIT 1
            """,
            {"from_id": from_id, "to_id": to_id},
        )
        if not rows:
            return None
        value = rows[0].get("dist")
        return int(value) if value is not None else None

    def incomplete_episode_count(self) -> int:
        rows = self._rows("MATCH (ep:Episode) WHERE ep.incomplete = true RETURN count(ep) AS n")
        return int(rows[0].get("n") or 0) if rows else 0

    def get_entity_type_distribution(self) -> dict[str, int]:
        rows = self._rows(
            """
            MATCH (e:Entity)
            RETURN coalesce(e.entity_type, 'Unknown') AS entity_type,
                   count(e) AS count
            """
        )
        return {str(row.get("entity_type") or "Unknown"): int(row.get("count") or 0) for row in rows}

    def get_current_facts(self, entity_name: str, as_of: Optional[datetime] = None) -> List[Dict]:
        entity_id = self.get_entity_id_by_name(entity_name)
        if not entity_id:
            return []
        point_in_time = as_of or datetime.now(UTC)
        return self._rows(
            """
            MATCH (e:Entity {id: $entity_id})-[r]->(target:Entity)
            WHERE (r.valid_from IS NULL OR r.valid_from <= $as_of)
              AND (r.valid_at IS NULL OR r.valid_at <= $as_of)
              AND (r.valid_to IS NULL OR r.valid_to > $as_of)
            RETURN label(r) AS rel_type,
                   target.name AS target_name,
                   target.entity_type AS target_type,
                   r.strength AS strength,
                   r.confidence AS confidence
            ORDER BY coalesce(r.strength, 0.0) DESC, coalesce(r.valid_from, r.valid_at, $as_of) DESC
            """,
            {"entity_id": entity_id, "as_of": point_in_time},
        )

    def get_entity_context(self, entity_name: str, hops: int = 2, min_strength: float = 0.3) -> Dict[str, Any]:
        entity_id = self.get_entity_id_by_name(entity_name)
        if not entity_id:
            return {
                "entity": entity_name,
                "direct_connections": [],
                "two_hop_connections": [],
            }

        outgoing = self._rows(
            """
            MATCH (e:Entity {id: $entity_id})-[r]->(target:Entity)
            WHERE coalesce(r.strength, 0.0) >= $min_strength
            RETURN label(r) AS rel_type,
                   target.name AS target_name,
                   target.entity_type AS target_type,
                   r.strength AS strength,
                   true AS is_outgoing
            ORDER BY r.strength DESC
            LIMIT 20
            """,
            {"entity_id": entity_id, "min_strength": min_strength},
        )
        incoming = self._rows(
            """
            MATCH (source:Entity)-[r]->(e:Entity {id: $entity_id})
            WHERE coalesce(r.strength, 0.0) >= $min_strength
            RETURN label(r) AS rel_type,
                   source.name AS target_name,
                   source.entity_type AS target_type,
                   r.strength AS strength,
                   false AS is_outgoing
            ORDER BY r.strength DESC
            LIMIT 20
            """,
            {"entity_id": entity_id, "min_strength": min_strength},
        )
        two_hop: list[dict[str, Any]] = []
        if hops >= 2:
            two_hop = self._rows(
                """
                MATCH (e:Entity {id: $entity_id})-[r1]-(mid:Entity)-[r2]-(target:Entity)
                WHERE e.id <> target.id
                  AND coalesce(r1.strength, 0.0) >= $min_strength
                  AND coalesce(r2.strength, 0.0) >= $min_strength
                RETURN DISTINCT target.name AS target_name,
                       target.entity_type AS target_type,
                       mid.name AS via_entity,
                       (coalesce(r1.strength, 0.0) + coalesce(r2.strength, 0.0)) / 2.0 AS avg_strength
                ORDER BY avg_strength DESC
                LIMIT 10
                """,
                {"entity_id": entity_id, "min_strength": min_strength},
            )
        return {
            "entity": entity_name,
            "direct_connections": outgoing + incoming,
            "two_hop_connections": two_hop,
        }

    def get_entity_delete_summary(self, name: str) -> Dict[str, Any] | None:
        entity_id = self.get_entity_id_by_name(name)
        if not entity_id:
            return None
        rows = self._rows(
            """
            MATCH (e:Entity {id: $entity_id})
            OPTIONAL MATCH (e)-[r]-()
            RETURN count(r) AS relationship_count
            """,
            {"entity_id": entity_id},
        )
        return {
            "entity_id": entity_id,
            "relationship_count": int(rows[0].get("relationship_count") or 0) if rows else 0,
        }

    def delete_entity(self, name: str) -> bool:
        entity_id = self.get_entity_id_by_name(name)
        if not entity_id:
            return False
        with self._lock:
            self.conn.execute("MATCH (e:Entity {id: $entity_id}) DETACH DELETE e", {"entity_id": entity_id})
        return True

    def delete_relationships_between(self, head: str, tail: str, rel_type: str | None = None) -> int:
        source_id = self.get_entity_id_by_name(head)
        target_id = self.get_entity_id_by_name(tail)
        if not source_id or not target_id:
            return 0
        if rel_type:
            normalized_type = str(rel_type).strip().upper().replace(" ", "_")
            if normalized_type not in VALID_REL_TYPES or normalized_type == "MENTIONS":
                return 0
            queries = [
                f"""
                MATCH (h:Entity {{id: $source_id}})-[r:{normalized_type}]->(t:Entity {{id: $target_id}})
                DELETE r
                RETURN count(r) AS deleted
                """,
                f"""
                MATCH (h:Entity {{id: $target_id}})-[r:{normalized_type}]->(t:Entity {{id: $source_id}})
                DELETE r
                RETURN count(r) AS deleted
                """,
            ]
        else:
            queries = [
                """
                MATCH (h:Entity {id: $source_id})-[r]-(t:Entity {id: $target_id})
                DELETE r
                RETURN count(r) AS deleted
                """.replace("-[r]-", "-[r]->"),
                """
                MATCH (h:Entity {id: $target_id})-[r]->(t:Entity {id: $source_id})
                DELETE r
                RETURN count(r) AS deleted
                """,
            ]
        deleted = 0
        for query in queries:
            rows = self._rows(query, {"source_id": source_id, "target_id": target_id})
            deleted += int(rows[0].get("deleted") or 0) if rows else 0
        return deleted

    def list_orphan_entity_names(self) -> List[str]:
        rows = self._rows(
            """
            MATCH (e:Entity)
            WHERE NOT (e)--()
            RETURN e.name AS name
            ORDER BY e.name ASC
            """
        )
        return [str(row.get("name") or "").strip() for row in rows if str(row.get("name") or "").strip()]

    def list_entities_page(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        entity_type: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        params = {"limit": max(1, int(limit)), "offset": max(0, int(offset))}
        if entity_type:
            params["entity_type"] = entity_type
            query = """
                MATCH (e:Entity)
                WHERE e.entity_type = $entity_type
                RETURN e.id AS id,
                       e.name AS name,
                       e.entity_type AS entity_type,
                       e.confidence AS confidence
                ORDER BY e.name ASC
                SKIP $offset
                LIMIT $limit
            """
            count_query = """
                MATCH (e:Entity)
                WHERE e.entity_type = $entity_type
                RETURN count(e) AS total
            """
        else:
            query = """
                MATCH (e:Entity)
                RETURN e.id AS id,
                       e.name AS name,
                       e.entity_type AS entity_type,
                       e.confidence AS confidence
                ORDER BY e.name ASC
                SKIP $offset
                LIMIT $limit
            """
            count_query = "MATCH (e:Entity) RETURN count(e) AS total"
        rows = self._rows(query, params)
        total_rows = self._rows(count_query, params if entity_type else {})
        total = int(total_rows[0].get("total") or 0) if total_rows else 0
        return rows, total

    def get_stale_embedding_entities(self, limit: int = 5000) -> List[Dict[str, Any]]:
        return self._rows(
            """
            MATCH (e:Entity)
            WHERE e.embedding_stale = true
            RETURN e.id AS entity_id,
                   e.name AS name,
                   coalesce(e.entity_type, 'Concept') AS entity_type,
                   coalesce(e.summary, e.description, '') AS content,
                   coalesce(e.confidence, 1.0) AS confidence
            ORDER BY coalesce(e.mention_count, 0) DESC, e.name ASC
            LIMIT $limit
            """,
            {"limit": max(1, int(limit))},
        )

    def clear_embedding_stale_flag(self, name: str) -> bool:
        entity_id = self.get_entity_id_by_name(name)
        if not entity_id:
            return False
        with self._lock:
            self.conn.execute(
                "MATCH (e:Entity {id: $entity_id}) SET e.embedding_stale = false",
                {"entity_id": entity_id},
            )
        return True

    def list_entities_for_embedding(self, limit: int = 5000) -> List[Dict[str, Any]]:
        rows = self._rows(
            """
            MATCH (e:Entity)
            RETURN e.id AS entity_id,
                   e.name AS name,
                   coalesce(e.entity_type, 'Concept') AS entity_type,
                   coalesce(e.summary, e.description, '') AS content,
                   coalesce(e.confidence, 1.0) AS confidence
            ORDER BY coalesce(e.mention_count, 0) DESC, e.name ASC
            LIMIT $limit
            """,
            {"limit": max(1, int(limit))},
        )
        output = []
        for row in rows:
            name = str(row.get("name") or "").strip()
            if not name:
                continue
            output.append(
                {
                    "entity_id": _sanitize_doc_id(str(row.get("entity_id") or name.lower())),
                    "name": name,
                    "entity_type": str(row.get("entity_type") or "Concept"),
                    "content": str(row.get("content") or ""),
                    "confidence": float(row.get("confidence") or 1.0),
                }
            )
        return output

    def entity_count(self) -> int:
        rows = self._rows("MATCH (e:Entity) RETURN count(e) AS total")
        return int(rows[0].get("total") or 0) if rows else 0

    def relationship_count(self) -> int:
        rows = self._rows("MATCH ()-[r]->() RETURN count(r) AS total")
        return int(rows[0].get("total") or 0) if rows else 0

    def episode_count(self) -> int:
        rows = self._rows("MATCH (ep:Episode) RETURN count(ep) AS total")
        return int(rows[0].get("total") or 0) if rows else 0

    def get_relationship_type_distribution(self) -> Dict[str, int]:
        rows = self._rows(
            """
            MATCH ()-[r]->()
            RETURN label(r) AS rel_type, count(r) AS count
            """
        )
        return {str(row.get("rel_type") or ""): int(row.get("count") or 0) for row in rows if row.get("rel_type")}

    def get_graph_summary(self) -> Dict[str, Any]:
        top_connected = self._rows(
            """
            MATCH (e:Entity)
            OPTIONAL MATCH (e)-[r]-()
            WITH e, count(r) AS degree
            RETURN e.name AS name,
                   e.entity_type AS entity_type,
                   degree
            ORDER BY degree DESC, e.name ASC
            LIMIT 10
            """
        )
        recent = self._rows(
            """
            MATCH (ep:Episode)
            RETURN ep.id AS id,
                   ep.source AS source,
                   ep.ingested_at AS ingested_at
            ORDER BY ep.ingested_at DESC
            LIMIT 10
            """
        )
        return {
            "entity_count": self.entity_count(),
            "relationship_count": self.relationship_count(),
            "episode_count": self.episode_count(),
            "top_connected": top_connected,
            "recent": recent,
        }
