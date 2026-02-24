"""Extraction pipeline for MollyGraph v1.

This implementation is OpenClaw-first:
- Uses local GLiNER2 extractor (memory.extractor)
- Writes entities/relationships into bi-temporal Neo4j graph
- Emits suggestion signals for unknown relationship types
- Indexes entities in vector store (Zvec preferred)
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re
import threading
import time
import uuid
from datetime import datetime, UTC
from typing import Any

import config as service_config
from extraction.glirel_enrichment import GLiRELEnrichment
from memory.models import Entity, Episode, ExtractionJob, Relationship
from memory.graph import BiTemporalGraph
from memory.vector_store import VectorStore
from memory import extractor as gliner_extractor
from memory.graph_suggestions import log_relationship_fallback

try:
    from metrics.stats_logger import log_extraction as _log_extraction
except Exception:  # pragma: no cover
    _log_extraction = None  # type: ignore

try:
    from metrics.model_health import model_health_monitor as _model_health_monitor
except Exception:  # pragma: no cover
    _model_health_monitor = None  # type: ignore


def _get_gliner_model():
    """Return the active GLiNER2 model (hot-reload aware).

    Delegates to ``memory.extractor._get_model()`` which implements mtime-based
    hot-reload whenever the active model directory changes on disk.  The mtime
    check adds only a single O(1) stat() call per invocation.
    """
    return gliner_extractor._get_model()

try:
    import spacy  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    spacy = None

log = logging.getLogger(__name__)

_ALLOWED_ENTITY_TYPES = {
    "Person",
    "Organization",
    "Technology",
    "Place",
    "Project",
    "Concept",
    "Event",
}

# Derive the allowed relation type set from the canonical extractor schema so
# the two definitions can never silently diverge.  Keys in RELATION_SCHEMA are
# lowercase-with-spaces (e.g. "works on"); we normalise to UPPER_UNDERSCORE.
# A small set of pipeline-only types (MENTIONS, TEACHES_AT) is unioned in
# as they have no extractor counterpart but are valid graph edges.
try:
    from memory.extractor import RELATION_SCHEMA as _rel_schema
    _ALLOWED_REL_TYPES: set[str] = {
        key.upper().replace(" ", "_") for key in _rel_schema.keys()
    } | {
        "MENTIONS",       # pipeline-internal; not in extractor schema
        "TEACHES_AT",     # pipeline-internal; not in extractor schema
    }
except ImportError:
    # Fallback: hardcoded set kept in sync manually.
    _ALLOWED_REL_TYPES = {
        "WORKS_ON",
        "WORKS_AT",
        "KNOWS",
        "USES",
        "LOCATED_IN",
        "DISCUSSED_WITH",
        "INTERESTED_IN",
        "CREATED",
        "MANAGES",
        "DEPENDS_ON",
        "RELATED_TO",
        "MENTIONS",
        "ATTENDS",
        "TEACHES_AT",
        "ALUMNI_OF",
        "STUDIED_AT",
        "PARENT_OF",
        "CHILD_OF",
        "REPORTS_TO",
        "CUSTOMER_OF",
        "CLASSMATE_OF",
        "MENTORED_BY",
        "MENTORS",
        "COLLABORATES_WITH",
        "RECEIVED_FROM",
        "CONTACT_OF",
    }

_SPACY_LABEL_MAP = {
    "PERSON": "Person",
    "ORG": "Organization",
    "GPE": "Place",
    "LOC": "Place",
    "FAC": "Place",
    "EVENT": "Event",
    "PRODUCT": "Technology",
    "WORK_OF_ART": "Concept",
}


class ExtractionPipeline:
    """End-to-end extraction and graph/vector persistence."""

    # ── Class-level daily ingestion counters (reset when date changes) ────────
    # These are shared across all instances (there's only one per process).
    _counter_lock = threading.Lock()
    _counter_date: str = ""  # YYYY-MM-DD; reset when date changes
    _ingestion_counters: dict[str, Any] = {
        "jobs_processed": 0,
        "total_entities_extracted": 0,
        "total_relationships_extracted": 0,
        "total_new_entities": 0,       # entities that did NOT exist before
        "total_existing_entities": 0,  # entities that already existed (upserted)
        "total_fallback_relationships": 0,  # RELATED_TO fallback count
    }

    @classmethod
    def _get_ingestion_counters(cls) -> dict[str, Any]:
        """Return a snapshot of today's ingestion counters."""
        from datetime import date
        today = date.today().isoformat()
        with cls._counter_lock:
            if cls._counter_date != today:
                # Day rolled over — reset
                cls._counter_date = today
                cls._ingestion_counters = {
                    "jobs_processed": 0,
                    "total_entities_extracted": 0,
                    "total_relationships_extracted": 0,
                    "total_new_entities": 0,
                    "total_existing_entities": 0,
                    "total_fallback_relationships": 0,
                }
            c = dict(cls._ingestion_counters)
        jobs = c["jobs_processed"]
        total_ent = c["total_entities_extracted"]
        total_rel = c["total_relationships_extracted"]
        new_ent = c["total_new_entities"]
        unique_rate = round(new_ent / total_ent, 4) if total_ent > 0 else 0.0
        fallbacks = c["total_fallback_relationships"]
        fallback_rate = round(fallbacks / total_rel, 4) if total_rel > 0 else 0.0
        return {
            "date": today,
            "jobs_processed": jobs,
            "avg_entity_yield": round(total_ent / jobs, 2) if jobs > 0 else 0.0,
            "avg_relationship_yield": round(total_rel / jobs, 2) if jobs > 0 else 0.0,
            "unique_entity_rate": unique_rate,
            "related_to_fallback_rate": fallback_rate,
            "total_entities_extracted": total_ent,
            "total_relationships_extracted": total_rel,
            "total_new_entities": new_ent,
            "total_existing_entities": c["total_existing_entities"],
            "total_fallback_relationships": fallbacks,
        }

    @classmethod
    def _update_ingestion_counters(
        cls,
        entities_extracted: int,
        relationships_extracted: int,
        new_entities: int,
        existing_entities: int,
        fallback_count: int,
    ) -> None:
        """Update class-level daily counters after a successful job."""
        from datetime import date
        today = date.today().isoformat()
        with cls._counter_lock:
            if cls._counter_date != today:
                cls._counter_date = today
                cls._ingestion_counters = {
                    "jobs_processed": 0,
                    "total_entities_extracted": 0,
                    "total_relationships_extracted": 0,
                    "total_new_entities": 0,
                    "total_existing_entities": 0,
                    "total_fallback_relationships": 0,
                }
            cls._ingestion_counters["jobs_processed"] += 1
            cls._ingestion_counters["total_entities_extracted"] += entities_extracted
            cls._ingestion_counters["total_relationships_extracted"] += relationships_extracted
            cls._ingestion_counters["total_new_entities"] += new_entities
            cls._ingestion_counters["total_existing_entities"] += existing_entities
            cls._ingestion_counters["total_fallback_relationships"] += fallback_count

    def __init__(self, graph: BiTemporalGraph, vector_store: VectorStore):
        self.graph = graph
        self.vector_store = vector_store
        self._glirel_enrichment = GLiRELEnrichment()
        self._spacy_nlp: Any | None = None
        self._spacy_attempted = False
        # Give the graph access to the vector store so entity merge/delete
        # operations can keep vectors in sync with Neo4j.
        self.graph._vector_store = vector_store

    def vector_search(self, query: str, top_k: int = 10):
        """Proxy to vector store search."""
        embedding = self._text_embedding(query)
        return self.vector_store.similarity_search(embedding, top_k=top_k)

    def _mark_episode_incomplete(self, episode_id: str, reason: str = "") -> None:
        """Mark a Neo4j episode node as incomplete after a mid-job failure.

        This makes partial writes visible so operators can reconcile or retry
        without creating silent duplicates.
        """
        try:
            with self.graph.driver.session() as session:
                session.run(
                    """
                    MATCH (ep:Episode {id: $episode_id})
                    SET ep.incomplete = true,
                        ep.incomplete_reason = $reason,
                        ep.incomplete_at = datetime()
                    """,
                    episode_id=episode_id,
                    reason=reason[:500],
                )
            log.info("Marked episode %s as incomplete (mid-job failure)", episode_id)
        except Exception:
            log.warning(
                "Failed to mark episode %s as incomplete — manual reconciliation required",
                episode_id,
                exc_info=True,
            )

    async def process_job(self, job: ExtractionJob) -> ExtractionJob:
        _t_start = time.perf_counter()
        embedding_time_ms = 0.0
        vector_store_time_ms = 0.0
        glirel_relations_found = 0
        glirel_overrides = 0
        glirel_additions = 0
        glirel_training_examples = 0
        # Track partial writes so we can reconcile on failure (Issue 3).
        # These are initialised before the try block so the except clause can
        # reference them regardless of how far processing got.
        episode_id_written: str | None = None
        written_entity_names: list[str] = []
        try:
            job.status = "processing"
            job.started_at = datetime.now(UTC)

            extracted = await asyncio.to_thread(gliner_extractor.extract, job.content, 0.4)
            gliner_entities_raw = extracted.get("entities", []) if isinstance(extracted, dict) else []
            raw_relations_raw = extracted.get("relations", []) if isinstance(extracted, dict) else []
            gliner_entities = gliner_entities_raw if isinstance(gliner_entities_raw, list) else []
            raw_relations = raw_relations_raw if isinstance(raw_relations_raw, list) else []

            spacy_entities: list[dict[str, Any]] = []
            if service_config.SPACY_ENRICHMENT:
                spacy_entities = self._spacy_enrich_entities(
                    content=job.content,
                    gliner_entity_count=len(gliner_entities),
                )

            raw_entities = self._merge_entity_sources_for_glirel(
                content=job.content,
                gliner_entities=gliner_entities,
                spacy_entities=spacy_entities,
            )

            entities = self._build_entities(raw_entities)
            canonical_names: dict[str, str] = {}
            stored_entities: list[Entity] = []
            new_entity_count = 0
            existing_entity_count = 0

            for entity in entities:
                entity_id, is_new = self.graph.upsert_entity(entity)
                canonical_names[self._normalize(entity.name)] = entity.name
                stored_entities.append(entity)
                written_entity_names.append(entity.name)  # track for failure reconciliation
                if is_new:
                    new_entity_count += 1
                else:
                    existing_entity_count += 1

                # Keep vector index in sync with graph entities.
                embedding = None  # always bound; stays None if _text_embedding raises
                _embed_start = time.perf_counter()
                try:
                    embedding = self._text_embedding(f"{entity.name} {entity.entity_type} {job.content[:200]}")
                except Exception:
                    log.debug(
                        "Embedding failed for entity %s — skipping vector upsert",
                        entity.name,
                        exc_info=True,
                    )
                finally:
                    embedding_time_ms += (time.perf_counter() - _embed_start) * 1000
                if embedding is not None:
                    try:
                        _store_start = time.perf_counter()
                        try:
                            self.vector_store.add_entity(
                                entity_id=entity_id,
                                name=entity.name,
                                entity_type=entity.entity_type,
                                dense_embedding=embedding,
                                content=job.content[:500],
                                confidence=entity.confidence,
                            )
                        finally:
                            vector_store_time_ms += (time.perf_counter() - _store_start) * 1000
                    except Exception:
                        log.debug("Vector index upsert failed for %s", entity.name, exc_info=True)

            raw_relations_with_source: list[dict[str, Any]] = [
                {**rel, "source": str(rel.get("source") or "gliner2").strip().lower() or "gliner2"}
                for rel in raw_relations
                if isinstance(rel, dict)
            ]
            if service_config.GLIREL_ENABLED:
                try:
                    log.info("GLiREL enrichment starting: %d entities", len(raw_entities))
                    glirel_relations = await asyncio.to_thread(
                        self._glirel_enrichment.enrich_relations,
                        job.content,
                        raw_entities,
                    )
                    glirel_relations_found = len(glirel_relations)
                    (
                        raw_relations_with_source,
                        glirel_overrides,
                        glirel_additions,
                    ) = self._merge_glirel_relations(
                        base_relations=raw_relations_with_source,
                        glirel_relations=glirel_relations,
                        canonical_names=canonical_names,
                    )
                    log.info(
                        "GLiREL enrichment complete: found=%d overrides=%d additions=%d",
                        glirel_relations_found,
                        glirel_overrides,
                        glirel_additions,
                    )

                    training_examples = await asyncio.to_thread(
                        self._glirel_enrichment.generate_training_examples,
                        job.content,
                        glirel_relations,
                    )
                    glirel_training_examples = len(training_examples)
                    if training_examples:
                        await asyncio.to_thread(
                            self._glirel_enrichment.persist_training_examples,
                            training_examples,
                        )
                except Exception:
                    log.warning(
                        "GLiREL enrichment failed; continuing with GLiNER2-only relations",
                        exc_info=True,
                    )

            relationships, fallback_count = self._build_relationships(
                raw_relations=raw_relations_with_source,
                canonical_names=canonical_names,
                reference_time=job.reference_time,
                context=job.content,
            )

            episode = Episode(
                id=job.episode_id or str(uuid.uuid4()),
                source=self._normalize_source(job.source),
                content_preview=job.content[:500],
                content_hash=self._hash_content(job.content),
                occurred_at=job.reference_time,
                entities_extracted=[entity.name for entity in stored_entities],
            )
            self.graph.create_episode(episode)
            episode_id_written = episode.id  # track for failure reconciliation

            # Embed episode into vector store for semantic search
            try:
                import re as _re
                _episode_embed_start = time.perf_counter()
                try:
                    ep_embedding = self._text_embedding(job.content[:512])
                finally:
                    embedding_time_ms += (time.perf_counter() - _episode_embed_start) * 1000
                ep_slug = _re.sub(r'[^a-zA-Z0-9_-]', '_', f"ep_{episode.id}")
                _episode_store_start = time.perf_counter()
                try:
                    self.vector_store.add_entity(
                        entity_id=ep_slug,
                        name=f"Episode {episode.id[:8]}",
                        entity_type="Episode",
                        dense_embedding=ep_embedding,
                        content=episode.content_preview,
                    )
                finally:
                    vector_store_time_ms += (time.perf_counter() - _episode_store_start) * 1000
            except Exception:
                log.debug("Vector index upsert failed for episode %s", episode.id, exc_info=True)

            rels_created = 0
            rels_skipped = 0
            for rel in relationships:
                rel.episode_ids = [episode.id]
                result = self.graph.upsert_relationship(rel)
                if result and result[0]:  # (id, status) — id is empty string if entity missing
                    rels_created += 1
                else:
                    rels_skipped += 1
            if rels_skipped:
                log.info(
                    "Relationships: %d created/updated, %d skipped (missing entities)",
                    rels_created, rels_skipped,
                )

            job.extracted_entities = stored_entities
            job.extracted_relationships = relationships
            job.glirel_relations_found = glirel_relations_found
            job.glirel_overrides = glirel_overrides
            job.glirel_additions = glirel_additions
            job.glirel_training_examples = glirel_training_examples
            job.status = "completed"
            job.completed_at = datetime.now(UTC)
            job.error = None

            # ── Metrics logging ───────────────────────────────────────────────
            processing_time_ms = (time.perf_counter() - _t_start) * 1000
            scores = [
                self._safe_float(e.get("score"), default=0.5)
                for e in raw_entities
                if isinstance(e, dict)
            ]
            conf_min = min(scores) if scores else 0.0
            conf_max = max(scores) if scores else 0.0
            conf_avg = sum(scores) / len(scores) if scores else 0.0

            log.info(
                "extraction_complete",
                extra={
                    "job_id": getattr(job, "id", "unknown"),
                    "entities": len(stored_entities),
                    "relationships": len(relationships),
                    "fallbacks": fallback_count,
                    "glirel_relations_found": glirel_relations_found,
                    "glirel_overrides": glirel_overrides,
                    "glirel_additions": glirel_additions,
                    "glirel_training_examples": glirel_training_examples,
                    "conf_min": round(conf_min, 4),
                    "conf_max": round(conf_max, 4),
                    "conf_avg": round(conf_avg, 4),
                    "processing_ms": round(processing_time_ms, 2),
                },
            )

            if _log_extraction is not None:
                try:
                    _log_extraction(
                        job_id=getattr(job, "job_id", getattr(job, "id", "unknown")),
                        entities_extracted=len(stored_entities),
                        relationships_extracted=len(relationships),
                        fallback_count=fallback_count,
                        processing_time_ms=processing_time_ms,
                        content_length=len(job.content),
                        confidence_min=conf_min,
                        confidence_max=conf_max,
                        confidence_avg=conf_avg,
                        embedding_time_ms=embedding_time_ms,
                        vector_store_time_ms=vector_store_time_ms,
                    )
                except Exception:
                    log.debug("metrics log_extraction failed", exc_info=True)

            # ── Model health monitoring (auto-rollback on degradation) ────────
            if _model_health_monitor is not None:
                try:
                    _model_health_monitor.record_extraction(
                        total_relations=len(relationships),
                        fallback_count=fallback_count,
                    )
                except Exception:
                    log.debug("model_health record_extraction failed", exc_info=True)

            # ── Daily ingestion counters (class-level, exposed via /metrics/dashboard) ──
            try:
                ExtractionPipeline._update_ingestion_counters(
                    entities_extracted=len(stored_entities),
                    relationships_extracted=len(relationships),
                    new_entities=new_entity_count,
                    existing_entities=existing_entity_count,
                    fallback_count=fallback_count,
                )
            except Exception:
                log.debug("ingestion_counters update failed", exc_info=True)

            return job

        except Exception as exc:
            log.error("Extraction failed", exc_info=True)
            job.status = "failed"
            job.error = str(exc)
            job.completed_at = datetime.now(UTC)

            # Issue 3: Log partial graph writes so operators can reconcile.
            # We do not attempt Neo4j rollback (too complex); instead we mark
            # the episode as incomplete so queries and training can filter it.
            if episode_id_written is not None or written_entity_names:
                log.warning(
                    "Partial graph write — episode=%s entities=%s — marking incomplete for reconciliation",
                    episode_id_written,
                    written_entity_names,
                    extra={
                        "job_id": getattr(job, "id", "unknown"),
                        "episode_id": episode_id_written,
                        "written_entities": written_entity_names,
                        "error": str(exc),
                    },
                )
                if episode_id_written is not None:
                    self._mark_episode_incomplete(episode_id_written, reason=str(exc))

            return job

    def _merge_glirel_relations(
        self,
        base_relations: list[dict[str, Any]],
        glirel_relations: list[dict[str, Any]],
        canonical_names: dict[str, str],
    ) -> tuple[list[dict[str, Any]], int, int]:
        """Merge GLiREL second-pass output into GLiNER2 relations.

        Rules:
        - GLiREL relations with confidence > threshold override GLiNER2 for the same pair.
        - GLiREL relations above threshold for unseen pairs are added.
        """
        if not glirel_relations:
            return base_relations, 0, 0

        threshold = float(getattr(service_config, "GLIREL_CONFIDENCE_THRESHOLD", 0.5))
        base_pairs: set[tuple[str, str]] = set()
        for rel in base_relations:
            head = self._canonicalize_name(str(rel.get("head") or ""), canonical_names)
            tail = self._canonicalize_name(str(rel.get("tail") or ""), canonical_names)
            if not head or not tail or self._normalize(head) == self._normalize(tail):
                continue
            base_pairs.add((self._normalize(head), self._normalize(tail)))

        glirel_best_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
        for rel in glirel_relations:
            if not isinstance(rel, dict):
                continue
            confidence = self._safe_float(rel.get("confidence"), default=0.0)
            if confidence <= threshold:
                continue

            head = self._canonicalize_name(str(rel.get("head") or ""), canonical_names)
            tail = self._canonicalize_name(str(rel.get("tail") or ""), canonical_names)
            rel_type = str(rel.get("rel_type") or "").strip()
            if not head or not tail or not rel_type:
                continue
            if self._normalize(head) == self._normalize(tail):
                continue

            pair = (self._normalize(head), self._normalize(tail))
            existing = glirel_best_by_pair.get(pair)
            if existing and self._safe_float(existing.get("score"), default=0.0) >= confidence:
                continue

            glirel_best_by_pair[pair] = {
                "head": head,
                "tail": tail,
                "label": rel_type,
                "score": confidence,
                "source": "glirel",
            }

        if not glirel_best_by_pair:
            return base_relations, 0, 0

        merged_base: list[dict[str, Any]] = []
        for rel in base_relations:
            head = self._canonicalize_name(str(rel.get("head") or ""), canonical_names)
            tail = self._canonicalize_name(str(rel.get("tail") or ""), canonical_names)
            if head and tail:
                pair = (self._normalize(head), self._normalize(tail))
                if pair in glirel_best_by_pair:
                    continue
            merged_base.append(
                {
                    **rel,
                    "source": str(rel.get("source") or "gliner2").strip().lower() or "gliner2",
                }
            )

        overrides = sum(1 for pair in glirel_best_by_pair if pair in base_pairs)
        additions = sum(1 for pair in glirel_best_by_pair if pair not in base_pairs)
        merged = merged_base + list(glirel_best_by_pair.values())
        return merged, overrides, additions

    def _canonicalize_name(self, name: str, canonical_names: dict[str, str]) -> str:
        normalized = self._normalize(name)
        if not normalized:
            return ""
        return canonical_names.get(normalized, name.strip())

    def _build_entities(self, raw_entities: list[dict[str, Any]]) -> list[Entity]:
        entities: list[Entity] = []
        seen: set[str] = set()

        for item in raw_entities:
            if not isinstance(item, dict):
                continue

            name = str(item.get("text") or "").strip()
            if len(name) < 2:
                continue

            normalized = self._normalize(name)
            if normalized in seen:
                continue
            seen.add(normalized)

            raw_type = str(item.get("label") or "Concept").strip()
            entity_type = raw_type if raw_type in _ALLOWED_ENTITY_TYPES else "Concept"
            score = self._safe_float(item.get("score"), default=0.5)

            entities.append(
                Entity(
                    name=name,
                    entity_type=entity_type,
                    confidence=max(0.0, min(1.0, score)),
                    first_mentioned=datetime.now(UTC),
                    last_mentioned=datetime.now(UTC),
                    created_from_episode="pending",
                )
            )

        return entities

    def _build_relationships(
        self,
        raw_relations: list[dict[str, Any]],
        canonical_names: dict[str, str],
        reference_time: datetime,
        context: str,
    ) -> tuple[list[Relationship], int]:
        """Build relationships list. Returns (relationships, fallback_count)."""
        relationships: list[Relationship] = []
        seen: set[tuple[str, str, str]] = set()
        fallback_count = 0

        for item in raw_relations:
            if not isinstance(item, dict):
                continue

            source_raw = str(item.get("head") or "").strip()
            target_raw = str(item.get("tail") or "").strip()
            if not source_raw or not target_raw:
                continue

            source_entity = canonical_names.get(self._normalize(source_raw), source_raw)
            target_entity = canonical_names.get(self._normalize(target_raw), target_raw)
            if source_entity == target_entity:
                continue

            raw_rel = str(item.get("label") or "related to").strip()
            rel_type = self._normalize_rel_type(raw_rel)
            if rel_type not in _ALLOWED_REL_TYPES:
                log_relationship_fallback(
                    head=source_entity,
                    tail=target_entity,
                    original_type=raw_rel,
                    confidence=self._safe_float(item.get("score"), default=0.5),
                    context=context[:200],
                )
                rel_type = "RELATED_TO"
                fallback_count += 1

            dedupe_key = (source_entity.lower(), rel_type, target_entity.lower())
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            score = self._safe_float(item.get("score"), default=0.5)
            relation_source = str(item.get("source") or "gliner2").strip().lower() or "gliner2"
            relationships.append(
                Relationship(
                    source_entity=source_entity,
                    target_entity=target_entity,
                    relation_type=rel_type,
                    confidence=max(0.0, min(1.0, score)),
                    valid_at=reference_time,
                    context_snippets=[context[:200]],
                    source=relation_source,
                    episode_ids=[],
                )
            )

        return relationships, fallback_count

    def _spacy_enrich_entities(self, content: str, gliner_entity_count: int) -> list[dict[str, Any]]:
        """Optional NER enrichment when GLiNER yields sparse extraction."""
        if gliner_entity_count >= service_config.SPACY_MIN_GLINER_ENTITIES:
            return []

        nlp = self._get_spacy_nlp()
        if nlp is None:
            return []

        try:
            doc = nlp(content[:10000])
        except Exception:
            log.debug("spaCy enrichment parse failed", exc_info=True)
            return []

        enriched: list[dict[str, Any]] = []
        seen: set[str] = set()
        for ent in doc.ents:
            text = ent.text.strip()
            if len(text) < 2:
                continue
            mapped = _SPACY_LABEL_MAP.get(ent.label_)
            if not mapped:
                continue

            key = self._normalize(text)
            if key in seen:
                continue
            seen.add(key)

            enriched.append(
                {
                    "text": text,
                    "label": mapped,
                    "score": 0.35,  # keep lower than GLiNER confidence by default
                }
            )

        if enriched:
            log.debug("spaCy enrichment added %d entities", len(enriched))
        return enriched

    def _merge_entity_sources_for_glirel(
        self,
        *,
        content: str,
        gliner_entities: list[dict[str, Any]],
        spacy_entities: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge GLiNER + spaCy entities for GLiREL while deduplicating overlaps."""
        merged: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        seen_spans: list[tuple[tuple[int, int], str]] = []

        sources: list[list[dict[str, Any]]] = []
        if isinstance(gliner_entities, list):
            sources.append(gliner_entities)
        if isinstance(spacy_entities, list):
            sources.append(spacy_entities)

        for source_entities in sources:
            for entity in source_entities:
                if not isinstance(entity, dict):
                    continue

                raw_text = str(entity.get("text") or entity.get("name") or "").strip()
                if len(raw_text) < 2:
                    continue

                normalized_text = self._normalize(raw_text)
                if normalized_text in seen_names:
                    continue

                span = self._first_span(content, raw_text)
                overlaps_existing = False
                if span is not None:
                    for existing_span, existing_text in seen_spans:
                        if not self._spans_overlap(span, existing_span):
                            continue
                        if normalized_text in existing_text or existing_text in normalized_text:
                            overlaps_existing = True
                            break

                if overlaps_existing:
                    continue

                merged.append(entity)
                seen_names.add(normalized_text)
                if span is not None:
                    seen_spans.append((span, normalized_text))

        return merged

    @staticmethod
    def _first_span(content: str, value: str) -> tuple[int, int] | None:
        if not content or not value:
            return None
        match = re.search(re.escape(value), content, flags=re.IGNORECASE)
        if not match:
            return None
        return match.start(), match.end()

    @staticmethod
    def _spans_overlap(span_a: tuple[int, int], span_b: tuple[int, int]) -> bool:
        return span_a[0] < span_b[1] and span_b[0] < span_a[1]

    def _get_spacy_nlp(self):
        if self._spacy_attempted:
            return self._spacy_nlp
        self._spacy_attempted = True

        if spacy is None:
            log.warning("spaCy enrichment requested but `spacy` is not installed")
            return None

        try:
            self._spacy_nlp = spacy.load(service_config.SPACY_MODEL)
            return self._spacy_nlp
        except Exception:
            log.warning(
                "spaCy enrichment requested but model `%s` is unavailable",
                service_config.SPACY_MODEL,
            )
            return None

    @staticmethod
    def _normalize_source(source: str) -> str:
        normalized = (source or "manual").strip().lower()
        allowed = {"manual", "whatsapp", "voice", "email", "imessage"}
        if normalized in allowed:
            return normalized
        return "manual"

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().lower())

    @staticmethod
    def _normalize_rel_type(label: str) -> str:
        return label.strip().upper().replace(" ", "_")

    @staticmethod
    def _hash_content(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    async def refresh_stale_embeddings(self) -> dict[str, Any]:
        """Re-compute embeddings for all entities with embedding_stale=True.

        Called by the nightly maintenance cycle (after audit, before training)
        and exposed via ``POST /maintenance/refresh-embeddings``.

        For each stale entity the new embedding is derived from:
            ``{name} {entity_type} {latest context (summary/description)}``

        Returns a summary dict with counts of refreshed and failed entities.
        """
        stale = await asyncio.to_thread(self.graph.get_stale_embedding_entities)
        if not stale:
            log.info("refresh_stale_embeddings: no stale entities found")
            return {"refreshed": 0, "failed": 0, "total_stale": 0}

        refreshed = 0
        failed = 0
        for entity_row in stale:
            name = entity_row.get("name", "")
            entity_type = entity_row.get("entity_type", "Concept")
            content = entity_row.get("content", "")
            entity_id = entity_row.get("entity_id", "")
            confidence = float(entity_row.get("confidence") or 1.0)

            embed_text = f"{name} {entity_type} {content[:200]}".strip()
            try:
                embedding = await asyncio.to_thread(self._text_embedding, embed_text)
                self.vector_store.add_entity(
                    entity_id=entity_id,
                    name=name,
                    entity_type=entity_type,
                    dense_embedding=embedding,
                    content=content[:500],
                    confidence=confidence,
                )
                await asyncio.to_thread(self.graph.clear_embedding_stale_flag, name)
                refreshed += 1
            except Exception:
                log.warning(
                    "refresh_stale_embeddings: failed for entity %r", name, exc_info=True
                )
                failed += 1

        log.info(
            "refresh_stale_embeddings complete: refreshed=%d failed=%d total_stale=%d",
            refreshed, failed, len(stale),
        )
        return {"refreshed": refreshed, "failed": failed, "total_stale": len(stale)}

    # ── Embedding model (lazy tier-chain) ────────────────────────────────
    _embedding_model = None           # loaded SentenceTransformer or sentinel string
    _embedding_active_tier: "str | None" = None   # which tier is currently active
    _embedding_failed_tiers: set = set()           # tiers that have permanently failed this run

    # ── Reranker model (lazy-loaded, optional) ────────────────────────────────
    _reranker_model = None            # loaded CrossEncoder or None if not yet loaded/failed
    _reranker_loaded: bool = False    # True once load was attempted (avoids repeated attempts)

    @classmethod
    def invalidate_embedding_cache(cls) -> None:
        """Drop cached embedding state after provider/model configuration changes."""
        cls._embedding_model = None
        cls._embedding_active_tier = None
        cls._embedding_failed_tiers = set()

    @classmethod
    def _try_load_tier(cls, tier: str) -> bool:
        """Attempt to load/validate one embedding tier. Returns True on success."""
        tier = tier.strip()

        if tier == "hash":
            cls._embedding_model = "hash"
            cls._embedding_active_tier = "hash"
            return True

        if tier in ("sentence-transformers", "st"):
            model_name = (
                getattr(service_config, "EMBEDDING_ST_MODEL", "").strip()
                or getattr(service_config, "EMBEDDING_MODEL", "").strip()
                or "google/embeddinggemma-300m"
            )
            try:
                from sentence_transformers import SentenceTransformer
                cls._embedding_model = SentenceTransformer(
                    model_name,
                    trust_remote_code=True,
                    model_kwargs={"default_task": "retrieval"},
                )
                cls._embedding_active_tier = tier
                log.info("Embedding tier '%s' loaded: %s", tier, model_name)
                return True
            except Exception as exc:
                log.warning("Embedding tier '%s' failed (model=%s): %s", tier, model_name, exc)
                return False

        if tier == "ollama":
            # Validate connectivity before committing
            try:
                import urllib.request as _ur, json as _json
                ollama_model = getattr(service_config, "EMBEDDING_OLLAMA_MODEL", "nomic-embed-text")
                base_url = getattr(service_config, "OLLAMA_BASE_URL", "http://127.0.0.1:11434")
                payload = _json.dumps({"model": ollama_model, "prompt": "ping"}).encode()
                req = _ur.Request(
                    f"{base_url}/api/embeddings",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with _ur.urlopen(req, timeout=10) as resp:
                    _json.loads(resp.read())
                cls._embedding_model = "ollama"
                cls._embedding_active_tier = "ollama"
                log.info("Embedding tier 'ollama' ready (model=%s)", ollama_model)
                return True
            except Exception as exc:
                log.warning("Embedding tier 'ollama' failed: %s", exc)
                return False

        if tier == "cloud":
            provider = getattr(service_config, "EMBEDDING_CLOUD_PROVIDER", "openai").strip().lower()
            cloud_model = getattr(service_config, "EMBEDDING_CLOUD_MODEL", "text-embedding-3-small").strip()
            if provider == "openai":
                api_key = getattr(service_config, "OPENAI_API_KEY", "")
                if not api_key:
                    log.warning("Embedding tier 'cloud/openai' skipped: no API key")
                    return False
                try:
                    import urllib.request as _ur, json as _json
                    payload = _json.dumps({"model": cloud_model, "input": "ping"}).encode()
                    req = _ur.Request(
                        "https://api.openai.com/v1/embeddings",
                        data=payload,
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {api_key}",
                        },
                    )
                    with _ur.urlopen(req, timeout=15) as resp:
                        _json.loads(resp.read())
                    cls._embedding_model = f"cloud:{provider}:{cloud_model}"
                    cls._embedding_active_tier = "cloud"
                    log.info("Embedding tier 'cloud' ready (provider=%s model=%s)", provider, cloud_model)
                    return True
                except Exception as exc:
                    log.warning("Embedding tier 'cloud/%s' failed: %s", provider, exc)
                    return False
            log.warning("Embedding tier 'cloud' provider '%s' not yet supported", provider)
            return False

        log.warning("Unknown embedding tier %r — skipping", tier)
        return False

    @classmethod
    def _get_embedding_model(cls):
        """Return active embedding model, loading via tier chain when necessary."""
        if cls._embedding_model is not None:
            return cls._embedding_model

        if getattr(service_config, "TEST_MODE", False):
            cls._embedding_model = "hash"
            cls._embedding_active_tier = "hash"
            return cls._embedding_model

        # Legacy single-backend override: MOLLYGRAPH_EMBEDDING_BACKEND non-empty
        legacy_backend = getattr(service_config, "EMBEDDING_BACKEND", "").strip().lower()
        if legacy_backend:
            if cls._try_load_tier(legacy_backend):
                return cls._embedding_model
            log.warning("Legacy EMBEDDING_BACKEND=%r failed — falling back to hash", legacy_backend)
            cls._embedding_model = "hash"
            cls._embedding_active_tier = "hash"
            return cls._embedding_model

        # Tier chain: walk in order, skip already-failed tiers
        tier_order = getattr(
            service_config, "EMBEDDING_TIER_ORDER",
            ["sentence-transformers", "ollama", "cloud", "hash"],
        )
        for tier in tier_order:
            tier = tier.strip()
            if not tier or tier in cls._embedding_failed_tiers:
                continue
            if cls._try_load_tier(tier):
                return cls._embedding_model
            cls._embedding_failed_tiers.add(tier)

        # Absolute last resort
        log.warning("All embedding tiers failed — using hash")
        cls._embedding_model = "hash"
        cls._embedding_active_tier = "hash"
        return cls._embedding_model

    @classmethod
    def _text_embedding(cls, text: str, dim: int = 768) -> list[float]:
        """Embed text using the configured tier chain (sentence-transformers / ollama / cloud / hash)."""
        model = cls._get_embedding_model()

        if model == "ollama":
            try:
                import urllib.request as _ur, json as _json
                ollama_model = getattr(service_config, "EMBEDDING_OLLAMA_MODEL", "nomic-embed-text")
                base_url = getattr(service_config, "OLLAMA_BASE_URL", "http://127.0.0.1:11434")
                payload = _json.dumps({"model": ollama_model, "prompt": text}).encode()
                req = _ur.Request(
                    f"{base_url}/api/embeddings",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with _ur.urlopen(req, timeout=30) as resp:
                    result = _json.loads(resp.read())
                return result["embedding"]
            except Exception as exc:
                log.warning("Ollama embedding call failed: %s — marking tier failed, trying next", exc)
                cls._embedding_model = None
                cls._embedding_failed_tiers.add("ollama")
                return cls._text_embedding(text, dim)

        if isinstance(model, str) and model.startswith("cloud:"):
            _, provider, cloud_model = model.split(":", 2)
            try:
                if provider == "openai":
                    import urllib.request as _ur, json as _json
                    api_key = getattr(service_config, "OPENAI_API_KEY", "")
                    payload = _json.dumps({"model": cloud_model, "input": text}).encode()
                    req = _ur.Request(
                        "https://api.openai.com/v1/embeddings",
                        data=payload,
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {api_key}",
                        },
                    )
                    with _ur.urlopen(req, timeout=30) as resp:
                        result = _json.loads(resp.read())
                    return result["data"][0]["embedding"]
            except Exception as exc:
                log.warning("Cloud embedding (%s) call failed: %s — marking tier failed, trying next", provider, exc)
                cls._embedding_model = None
                cls._embedding_failed_tiers.add("cloud")
                return cls._text_embedding(text, dim)

        if model != "hash":
            # sentence-transformers model object
            try:
                vec = model.encode(text, normalize_embeddings=True).tolist()
                return vec
            except Exception as exc:
                log.warning(
                    "sentence-transformers encode failed: %s — marking tier failed, trying next", exc
                )
                cls._embedding_model = None
                cls._embedding_failed_tiers.add(cls._embedding_active_tier or "sentence-transformers")
                return cls._text_embedding(text, dim)

        # Hash fallback — deterministic, never fails
        tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        vector = [0.0] * dim
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for i in range(0, len(digest), 4):
                chunk = digest[i : i + 4]
                idx = int.from_bytes(chunk, "little", signed=False) % dim
                sign = 1.0 if (chunk[0] % 2 == 0) else -1.0
                vector[idx] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            return vector
        return [v / norm for v in vector]

    # ── Reranker (lazy-loaded cross-encoder) ──────────────────────────────────
    @classmethod
    def _get_reranker_model(cls):
        """Return the cached cross-encoder reranker, or None if disabled/unavailable.

        Loads lazily on first call; subsequent calls return the cached instance.
        Model is only loaded when ``RERANKER_ENABLED`` is True in config.
        """
        if cls._reranker_loaded:
            return cls._reranker_model

        cls._reranker_loaded = True  # Mark attempted regardless of outcome

        if not getattr(service_config, "RERANKER_ENABLED", False):
            log.debug("Reranker disabled (MOLLYGRAPH_RERANKER_ENABLED not set)")
            cls._reranker_model = None
            return None

        model_name = getattr(
            service_config, "RERANKER_MODEL", "jinaai/jina-reranker-v2-base-multilingual"
        )
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            cls._reranker_model = CrossEncoder(model_name, trust_remote_code=True)
            log.info("Reranker loaded: %s", model_name)
        except Exception as exc:
            log.warning("Reranker load failed (model=%s): %s — reranking disabled", model_name, exc)
            cls._reranker_model = None

        return cls._reranker_model
