"""Maintenance and cleanup operations for the bi-temporal graph."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .constants import build_relationship_half_life_case, log


class MaintenanceMixin:
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

    def delete_orphan_entities_sync(self, vector_store: Optional[Any] = None) -> int:
        """Delete orphan entities (no relationships) from Neo4j.

        If *vector_store* is provided, also removes the corresponding vectors
        so the vector index stays consistent with the graph.
        """
        with self.driver.session() as session:
            # Collect orphan entity IDs before deletion so we can clean vectors.
            if vector_store is not None:
                orphan_rows: list[dict] = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE NOT (e)--()
                    RETURN e.id AS entity_id
                    """
                ).data()
            else:
                orphan_rows = []

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

        if vector_store is not None and orphan_rows:
            removed_vectors = 0
            for row in orphan_rows:
                entity_id = row.get("entity_id")
                if entity_id and isinstance(entity_id, str) and entity_id.strip():
                    try:
                        if vector_store.remove_entity(entity_id.strip()):
                            removed_vectors += 1
                    except Exception:
                        log.debug(
                            "Failed to remove vector for orphan entity_id=%s",
                            entity_id, exc_info=True,
                        )
            log.info("Removed %d orphan entity vectors", removed_vectors)

        return deleted

    def rebuild_graph_from_audit(self, actions: List[Dict[str, Any]], dry_run: bool = False) -> Dict[str, int]:
        """Apply audit action rows to mutate relationships in bulk."""
        summary = {"verified": 0, "reclassified": 0, "quarantined": 0, "deleted": 0, "skipped": 0}
        for action in actions:
            if not isinstance(action, dict):
                summary["skipped"] += 1
                continue

            decision = str(action.get("verdict") or action.get("action") or "").strip().lower()
            head = str(action.get("head") or "").strip()
            tail = str(action.get("tail") or "").strip()
            rel_type = str(action.get("rel_type") or action.get("type") or "").strip()
            suggested_type = str(action.get("suggested_type") or "").strip()

            if not head or not tail or not rel_type:
                summary["skipped"] += 1
                continue

            if decision == "verify":
                summary["verified"] += 1
                if not dry_run:
                    self.set_relationship_audit_status(head, tail, rel_type, "verified")
                continue

            if decision == "quarantine":
                summary["quarantined"] += 1
                if not dry_run:
                    self.set_relationship_audit_status(head, tail, rel_type, "quarantined")
                continue

            if decision == "delete":
                summary["deleted"] += 1
                if not dry_run:
                    self.delete_specific_relationship(head, tail, rel_type)
                continue

            if decision == "reclassify" and suggested_type:
                summary["reclassified"] += 1
                if not dry_run:
                    self.reclassify_relationship(
                        head=head,
                        tail=tail,
                        old_type=rel_type,
                        new_type=suggested_type,
                        strength=float(action.get("strength") or 0.5),
                        mention_count=int(action.get("mention_count") or 1),
                        context_snippets=action.get("context_snippets")
                        if isinstance(action.get("context_snippets"), list)
                        else [],
                        first_mentioned=str(action.get("first_mentioned") or "") or None,
                    )
                continue

            summary["skipped"] += 1
        return summary

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
