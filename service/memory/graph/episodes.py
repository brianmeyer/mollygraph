"""Episode operations for the bi-temporal graph."""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone

from memory.models import Episode


class EpisodeMixin:
    def _create_legacy_episode(
        self,
        content_preview: str,
        source: str,
        entity_names: list[str],
    ) -> str:
        """Legacy create_episode variant used by older function-based callers."""
        episode_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        with self.driver.session() as session:
            session.run(
                """
                CREATE (ep:Episode {
                    id: $id,
                    source: $source,
                    source_id: NULL,
                    content_preview: $preview,
                    content_hash: $content_hash,
                    occurred_at: datetime($now),
                    ingested_at: datetime($now),
                    processed_at: NULL,
                    processor_version: 'legacy',
                    entities_extracted: $entities,
                    status: 'active',
                    created_at: datetime($now)
                })
                """,
                id=episode_id,
                source=source,
                preview=content_preview[:500],
                content_hash=hashlib.sha256(content_preview.encode("utf-8")).hexdigest(),
                entities=entity_names,
                now=now,
            )
            for name in entity_names:
                session.run(
                    """
                    MATCH (ep:Episode {id: $eid})
                    MATCH (e:Entity {name: $name})
                    MERGE (ep)-[:MENTIONS]->(e)
                    SET e.last_seen = datetime($now),
                        e.last_mentioned = datetime($now)
                    """,
                    eid=episode_id,
                    name=name,
                    now=now,
                )
        return episode_id

    def create_episode(
        self,
        episode: Episode | str,
        source: str | None = None,
        entity_names: list[str] | None = None,
    ) -> str:
        """Store an episode node (model or legacy argument signature)."""
        if isinstance(episode, Episode):
            with self.driver.session() as session:
                payload = episode.model_dump()
                payload["occurred_at"] = episode.occurred_at.isoformat()
                payload["ingested_at"] = episode.ingested_at.isoformat()
                payload["processed_at"] = episode.processed_at.isoformat() if episode.processed_at else None
                result = session.run(
                    """
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
                    """,
                    **payload,
                )
                episode_id = result.single()["id"]
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
                return str(episode_id)

        if source is None:
            raise ValueError("source is required when using create_episode(content_preview, source, entity_names)")
        return self._create_legacy_episode(
            content_preview=str(episode),
            source=str(source),
            entity_names=list(entity_names or []),
        )

    def create_episode_sync(self, content_preview: str, source: str, entity_names: list[str]) -> str:
        """Compatibility wrapper for old function-based API."""
        return self.create_episode(content_preview, source, entity_names)
