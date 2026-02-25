"""Shared security, models, and helper functions for API route modules."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, UTC
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator

import config
from maintenance.lock import is_maintenance_locked
from memory.graph import VALID_REL_TYPES
from runtime_graph import get_graph_instance
from runtime_pipeline import get_pipeline_instance
from runtime_queue import get_queue_instance
from runtime_vector_store import get_vector_store_instance

log = logging.getLogger("mollygraph")

security = HTTPBearer(auto_error=False)

# ── Pydantic models ────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error: str
    code: str
    detail: str | None = None
    timestamp: str


class StatsResponse(BaseModel):
    queue: dict[str, Any]
    vector_store: dict[str, Any]
    graph: dict[str, Any]
    relationship_type_distribution: dict[str, int]
    gliner_training: dict[str, Any]
    timestamp: str


class EntityResponse(BaseModel):
    entity: str
    facts: list[dict[str, Any]]
    context: dict[str, Any]
    timestamp: str


class QueryResponse(BaseModel):
    query: str
    entities_found: list[str]
    results: list[dict[str, Any]]
    result_count: int
    timestamp: str
    reranked: bool = False
    graph_reranked: bool = False
    quality_metrics: dict[str, Any] = Field(default_factory=dict)
    retrieval_metadata: dict[str, Any] = Field(default_factory=dict)


class AuditRequest(BaseModel):
    limit: int = Field(default=500, ge=1, le=5000)
    dry_run: bool = False
    schedule: str = "nightly"
    model: str | None = None


class DeleteRelationshipRequest(BaseModel):
    source: str
    target: str
    rel_type: str | None = None

    @field_validator("rel_type")
    @classmethod
    def validate_rel_type(cls, v: str | None) -> str | None:
        if v is None:
            return v
        normalized = v.strip().upper().replace(" ", "_")
        if normalized not in VALID_REL_TYPES:
            raise ValueError(f"Unknown rel_type '{v}'. Must be one of: {sorted(VALID_REL_TYPES)}")
        return normalized


class PruneRequest(BaseModel):
    names: list[str] | None = None
    orphans: bool = False


class TrainRequest(BaseModel):
    force: bool = False


class EmbeddingConfigRequest(BaseModel):
    provider: str = Field(default="hash")
    model: str | None = None


class EmbeddingModelRequest(BaseModel):
    provider: str
    model: str
    activate: bool = False


class EmbeddingReindexRequest(BaseModel):
    limit: int = Field(default=5000, ge=1, le=500000)
    dry_run: bool = False


class ExtractorConfigRequest(BaseModel):
    backend: str = Field(default="gliner2")
    model: str | None = None
    relation_model: str | None = None


class ExtractorModelRequest(BaseModel):
    backend: str
    model: str
    role: str = Field(default="entity")
    activate: bool = False


class ExtractorPrefetchRequest(BaseModel):
    backend: str | None = None
    model: str | None = None
    relation_model: str | None = None


class ExtractorSchemaConfigRequest(BaseModel):
    mode: str = Field(default="default")
    preset: str | None = None


class ExtractorSchemaUploadRequest(BaseModel):
    entities: dict[str, Any] | list[str]
    relations: dict[str, Any] | list[str]
    activate: bool = True


# ── Security ───────────────────────────────────────────────────────────────────

def verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> str:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials
    if token != config.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


# ── Runtime guards ─────────────────────────────────────────────────────────────

def require_runtime_ready() -> None:
    queue = get_queue_instance()
    pipeline = get_pipeline_instance()
    graph = get_graph_instance()
    if queue is None or pipeline is None or graph is None:
        raise HTTPException(status_code=503, detail="Service not ready")


def require_no_maintenance() -> None:
    """Raise 503 if a maintenance cycle is currently running."""
    if is_maintenance_locked():
        raise HTTPException(
            status_code=503,
            detail="Maintenance cycle in progress — please retry in a moment",
        )


# ── Utility helpers ────────────────────────────────────────────────────────────

def _json_safe(value: Any) -> Any:
    """Convert Neo4j/Python temporal and custom values into JSON-safe primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    iso = getattr(value, "iso_format", None)
    if callable(iso):
        try:
            return iso()
        except Exception:
            pass
    iso = getattr(value, "isoformat", None)
    if callable(iso):
        try:
            return iso()
        except Exception:
            pass
    return str(value)


def _active_embedding_info() -> tuple[str, str]:
    """Return (provider, model) for the currently active embedding tier."""
    from extraction.pipeline import ExtractionPipeline
    tier = ExtractionPipeline._embedding_active_tier or "unknown"
    if tier in ("sentence-transformers", "st"):
        return (
            "sentence-transformers",
            config.EMBEDDING_ST_MODEL or config.EMBEDDING_MODEL or "google/embeddinggemma-300m",
        )
    if tier == "ollama":
        return ("ollama", config.EMBEDDING_OLLAMA_MODEL or config.OLLAMA_EMBED_MODEL or "nomic-embed-text")
    if tier == "cloud":
        return ("cloud", config.EMBEDDING_CLOUD_MODEL or "")
    if tier == "hash":
        return ("hash", "")
    backend = (config.EMBEDDING_BACKEND or "").strip().lower()
    if backend in ("huggingface", "sentence-transformers", "st", "hf"):
        return ("sentence-transformers", config.EMBEDDING_ST_MODEL or config.EMBEDDING_MODEL or "")
    if backend == "ollama":
        return ("ollama", config.OLLAMA_EMBED_MODEL or "")
    return ("unknown", "")


def _reindex_embeddings_sync(limit: int, dry_run: bool) -> dict[str, Any]:
    from extraction.pipeline import ExtractionPipeline
    graph = get_graph_instance()
    vector_store = get_vector_store_instance()
    if graph is None or vector_store is None:
        raise RuntimeError("Service not ready")

    rows = graph.list_entities_for_embedding(limit=limit)
    active_provider, active_model = _active_embedding_info()
    if dry_run:
        return {
            "status": "dry_run",
            "provider": active_provider,
            "model": active_model,
            "entities_found": len(rows),
            "reindexed": 0,
            "failed": 0,
            "sample": rows[:5],
        }

    reindexed = 0
    failed = 0
    errors: list[dict[str, str]] = []
    for row in rows:
        entity_id = str(row.get("entity_id") or "").strip()
        name = str(row.get("name") or "").strip()
        if not entity_id or not name:
            failed += 1
            continue
        entity_type = str(row.get("entity_type") or "Concept")
        content = str(row.get("content") or "")
        confidence = float(row.get("confidence") or 1.0)
        source_text = f"{name} {entity_type} {content[:500]}"
        embedding = ExtractionPipeline._text_embedding(source_text)
        try:
            vector_store.add_entity(
                entity_id=entity_id,
                name=name,
                entity_type=entity_type,
                dense_embedding=embedding,
                content=content[:1000] if content else name,
                confidence=confidence,
            )
            reindexed += 1
        except Exception as exc:
            failed += 1
            if len(errors) < 10:
                errors.append({"entity_id": entity_id, "name": name, "error": str(exc)})

    if reindexed > 0 and vector_store is not None:
        vector_store.flush()

    return {
        "status": "ok" if failed == 0 else "partial",
        "provider": active_provider,
        "model": active_model,
        "entities_found": len(rows),
        "reindexed": reindexed,
        "failed": failed,
        "errors": errors,
    }


async def _delete_entity_and_vector(name: str) -> tuple[bool, bool]:
    """Delete one entity from Neo4j + vector store. Returns (neo4j_deleted, vector_deleted)."""
    graph = get_graph_instance()
    vector_store = get_vector_store_instance()

    entity_id: str | None = None
    try:
        with graph.driver.session() as _session:
            rec = _session.run(
                """
                MATCH (e:Entity)
                WHERE toLower(e.name) = $name
                   OR ANY(a IN coalesce(e.aliases, []) WHERE toLower(a) = $name)
                RETURN coalesce(e.id, toLower(e.name)) AS entity_id
                """,
                name=name.lower(),
            ).single()
        if rec:
            entity_id = str(rec["entity_id"])
    except Exception:
        pass

    neo4j_deleted = await asyncio.to_thread(graph.delete_entity, name)
    vector_deleted = False
    if entity_id and vector_store is not None:
        try:
            vector_deleted = bool(await asyncio.to_thread(vector_store.remove_entity, entity_id))
        except Exception:
            log.debug("_delete_entity_and_vector: vector remove failed for %s", entity_id, exc_info=True)
    return neo4j_deleted, vector_deleted


def _refresh_embedding_runtime() -> None:
    from extraction.pipeline import ExtractionPipeline
    ExtractionPipeline.invalidate_embedding_cache()


def _refresh_extractor_runtime() -> None:
    from memory import extractor as memory_extractor
    memory_extractor.invalidate_model_cache()


def _extract_query_entities(query: str) -> list[str]:
    """Extract candidate entity names from a natural-language query."""
    words = [w.strip(" ,.!?:;()[]{}") for w in query.split()]
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


def _cooldown_remaining_hours(state: dict) -> int:
    """Compute hours remaining in the strategy-aware training cooldown."""
    from datetime import timedelta, timezone
    last_iso = str(state.get("gliner_last_finetune_at", "")).strip()
    if not last_iso:
        return 0
    try:
        last_dt = datetime.fromisoformat(last_iso)
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return 0
    strategy = str(
        state.get("gliner_last_training_strategy")
        or state.get("gliner_last_strategy")
        or "lora"
    ).strip().lower()
    if strategy not in {"lora", "full"}:
        strategy = "lora"
    cooldown_days = (
        config.GLINER_LORA_COOLDOWN_DAYS if strategy == "lora" else config.GLINER_FINETUNE_COOLDOWN_DAYS
    )
    remaining = timedelta(days=cooldown_days) - (datetime.now(timezone.utc) - last_dt)
    return max(0, int(remaining.total_seconds() // 3600))
