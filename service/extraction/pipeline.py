"""Extraction pipeline for MollyGraph v1.

This implementation is local-first:
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
import uuid
from datetime import datetime
from typing import Any

import config as service_config
from memory.models import Entity, Episode, ExtractionJob, Relationship
from memory.bitemporal_graph import BiTemporalGraph
from memory.vector_store import VectorStore
from memory import extractor as gliner_extractor
from memory.graph_suggestions import log_relationship_fallback

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
    _st_model: Any | None = None
    _st_lock = threading.Lock()
    _embed_warning_emitted = False

    def __init__(self, graph: BiTemporalGraph, vector_store: VectorStore):
        self.graph = graph
        self.vector_store = vector_store
        self._spacy_nlp: Any | None = None
        self._spacy_attempted = False

    async def process_job(self, job: ExtractionJob) -> ExtractionJob:
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

            relationships = self._build_relationships(
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

            for rel in relationships:
                rel.episode_ids = [episode.id]
                self.graph.upsert_relationship(rel)

            job.extracted_entities = stored_entities
            job.extracted_relationships = relationships
            job.status = "completed"
            job.completed_at = datetime.utcnow()
            job.error = None
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
    ) -> list[Relationship]:
        relationships: list[Relationship] = []
        seen: set[tuple[str, str, str]] = set()

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

        return relationships

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
        allowed = {"manual", "whatsapp", "voice", "email", "imessage", "mcp", "agent"}
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

    @staticmethod
    def _text_embedding(text: str, dim: int = 768) -> list[float]:
        """Compute embedding using local-first backend with deterministic fallback."""
        backend = service_config.EMBEDDING_BACKEND

        if backend in {"sentence-transformers", "sentence_transformers", "st"}:
            try:
                with ExtractionPipeline._st_lock:
                    if ExtractionPipeline._st_model is None:
                        from sentence_transformers import SentenceTransformer

                        ExtractionPipeline._st_model = SentenceTransformer(service_config.EMBEDDING_MODEL)
                vector = ExtractionPipeline._st_model.encode(
                    text,
                    normalize_embeddings=True,
                )
                return ExtractionPipeline._resize_and_normalize_embedding(vector, dim)
            except Exception:
                if not ExtractionPipeline._embed_warning_emitted:
                    log.warning(
                        "sentence-transformers embedding unavailable; falling back to hash backend",
                        exc_info=True,
                    )
                    ExtractionPipeline._embed_warning_emitted = True

        if backend in {"ollama"}:
            try:
                import httpx

                url = f"{service_config.OLLAMA_BASE_URL.rstrip('/')}/api/embeddings"
                payload = {"model": service_config.OLLAMA_EMBED_MODEL, "prompt": text}
                headers: dict[str, str] = {"Content-Type": "application/json"}
                if service_config.OLLAMA_API_KEY:
                    headers["Authorization"] = f"Bearer {service_config.OLLAMA_API_KEY}"

                with httpx.Client(timeout=10.0) as client:
                    resp = client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                    vector = data.get("embedding")
                    if isinstance(vector, list):
                        return ExtractionPipeline._resize_and_normalize_embedding(vector, dim)
            except Exception:
                if not ExtractionPipeline._embed_warning_emitted:
                    log.warning("Ollama embedding unavailable; falling back to hash backend", exc_info=True)
                    ExtractionPipeline._embed_warning_emitted = True

        # Default: deterministic local hash embedding (no model server required).
        return ExtractionPipeline._hash_embedding(text, dim)

    @staticmethod
    def _hash_embedding(text: str, dim: int = 768) -> list[float]:
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

    @staticmethod
    def _resize_and_normalize_embedding(vector: Any, dim: int = 768) -> list[float]:
        values = [float(v) for v in vector]
        if len(values) < dim:
            values.extend([0.0] * (dim - len(values)))
        elif len(values) > dim:
            values = values[:dim]

        norm = math.sqrt(sum(v * v for v in values))
        if norm == 0:
            return values
        return [v / norm for v in values]
