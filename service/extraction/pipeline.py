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
import time
import uuid
from datetime import datetime
from typing import Any

import config as service_config
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

    def __init__(self, graph: BiTemporalGraph, vector_store: VectorStore):
        self.graph = graph
        self.vector_store = vector_store
        self._spacy_nlp: Any | None = None
        self._spacy_attempted = False

    def vector_search(self, query: str, top_k: int = 10):
        """Proxy to vector store search."""
        embedding = self._text_embedding(query)
        return self.vector_store.similarity_search(embedding, top_k=top_k)

    async def process_job(self, job: ExtractionJob) -> ExtractionJob:
        _t_start = time.perf_counter()
        try:
            job.status = "processing"
            job.started_at = datetime.utcnow()

            extracted = await asyncio.to_thread(gliner_extractor.extract, job.content, 0.4)
            raw_entities = extracted.get("entities", []) if isinstance(extracted, dict) else []
            raw_relations = extracted.get("relations", []) if isinstance(extracted, dict) else []

            if service_config.SPACY_ENRICHMENT:
                raw_entities.extend(
                    self._spacy_enrich_entities(
                        content=job.content,
                        gliner_entity_count=len(raw_entities),
                    )
                )

            entities = self._build_entities(raw_entities)
            canonical_names: dict[str, str] = {}
            stored_entities: list[Entity] = []

            for entity in entities:
                entity_id, _ = self.graph.upsert_entity(entity)
                canonical_names[self._normalize(entity.name)] = entity.name
                stored_entities.append(entity)

                # Keep vector index in sync with graph entities.
                embedding = self._text_embedding(f"{entity.name} {entity.entity_type} {job.content[:200]}")
                try:
                    self.vector_store.add_entity(
                        entity_id=entity_id,
                        name=entity.name,
                        entity_type=entity.entity_type,
                        dense_embedding=embedding,
                        content=job.content[:500],
                        confidence=entity.confidence,
                    )
                except Exception:
                    log.debug("Vector index upsert failed for %s", entity.name, exc_info=True)

            relationships, fallback_count = self._build_relationships(
                raw_relations=raw_relations,
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

            # Embed episode into vector store for semantic search
            try:
                import re as _re
                ep_embedding = self._text_embedding(job.content[:512])
                ep_slug = _re.sub(r'[^a-zA-Z0-9_-]', '_', f"ep_{episode.id}")
                self.vector_store.add_entity(
                    entity_id=ep_slug,
                    name=f"Episode {episode.id[:8]}",
                    entity_type="Episode",
                    dense_embedding=ep_embedding,
                    content=episode.content_preview,
                )
            except Exception:
                log.debug("Vector index upsert failed for episode %s", episode.id, exc_info=True)

            for rel in relationships:
                rel.episode_ids = [episode.id]
                self.graph.upsert_relationship(rel)

            job.extracted_entities = stored_entities
            job.extracted_relationships = relationships
            job.status = "completed"
            job.completed_at = datetime.utcnow()
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

            return job

        except Exception as exc:
            log.error("Extraction failed", exc_info=True)
            job.status = "failed"
            job.error = str(exc)
            job.completed_at = datetime.utcnow()
            return job

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
                    first_mentioned=datetime.utcnow(),
                    last_mentioned=datetime.utcnow(),
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

            source = canonical_names.get(self._normalize(source_raw), source_raw)
            target = canonical_names.get(self._normalize(target_raw), target_raw)
            if source == target:
                continue

            raw_rel = str(item.get("label") or "related to").strip()
            rel_type = self._normalize_rel_type(raw_rel)
            if rel_type not in _ALLOWED_REL_TYPES:
                log_relationship_fallback(
                    head=source,
                    tail=target,
                    original_type=raw_rel,
                    confidence=self._safe_float(item.get("score"), default=0.5),
                    context=context[:200],
                )
                rel_type = "RELATED_TO"
                fallback_count += 1

            dedupe_key = (source.lower(), rel_type, target.lower())
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            score = self._safe_float(item.get("score"), default=0.5)
            relationships.append(
                Relationship(
                    source_entity=source,
                    target_entity=target,
                    relation_type=rel_type,
                    confidence=max(0.0, min(1.0, score)),
                    valid_at=reference_time,
                    context_snippets=[context[:200]],
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

    # ── Embedding model (lazy singleton) ─────────────────────────────────
    _embedding_model = None

    @classmethod
    def _get_embedding_model(cls):
        if cls._embedding_model is None:
            if getattr(service_config, "TEST_MODE", False):
                cls._embedding_model = "hash"
                return cls._embedding_model
            try:
                from sentence_transformers import SentenceTransformer
                cls._embedding_model = SentenceTransformer("google/embeddinggemma-300m")
                log.info("Loaded embedding model: google/embeddinggemma-300m")
            except Exception as exc:
                log.warning("Failed to load embeddinggemma-300m: %s — falling back to hash", exc)
                cls._embedding_model = "hash"  # sentinel for fallback
        return cls._embedding_model

    @staticmethod
    def _text_embedding(text: str, dim: int = 768) -> list[float]:
        """Embed text using google/embeddinggemma-300m (with hash fallback)."""
        model = ExtractionPipeline._get_embedding_model()
        
        if model != "hash":
            try:
                vec = model.encode(text, normalize_embeddings=True).tolist()
                return vec
            except Exception:
                pass
        
        # Hash fallback (shouldn't normally be hit)
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
