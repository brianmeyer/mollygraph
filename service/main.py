"""MollyGraph v1 HTTP service."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

import config
from metrics.stats_logger import get_summary, log_request
from audit.llm_audit import run_llm_audit
from embedding_registry import (
    add_embedding_model,
    get_embedding_registry,
    get_embedding_status,
    initialize_embedding_registry,
    set_active_embedding_provider,
)
from extractor_registry import (
    add_extractor_model,
    get_extractor_registry,
    get_extractor_status,
    initialize_extractor_registry,
    set_active_extractor_backend,
)
from extractor_schema_registry import (
    get_extractor_schema_presets,
    get_extractor_schema_status,
    initialize_extractor_schema_registry,
    set_active_extractor_schema,
    upload_custom_extractor_schema,
)
from evolution.gliner_training import get_gliner_stats, run_gliner_finetune_pipeline, GLiNERTrainingService
from metrics.model_health import model_health_monitor
from extraction.pipeline import ExtractionPipeline
from extraction.queue import ExtractionQueue, QueueWorker
from maintenance.auditor import run_maintenance_cycle
from memory.bitemporal_graph import BiTemporalGraph
from memory import extractor as memory_extractor
from memory.graph_suggestions import build_suggestion_digest
from memory.models import ExtractionJob
from memory.vector_store import VectorStore
from runtime_graph import set_graph_instance

class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
        })


_handler = logging.StreamHandler()
_handler.setFormatter(_JsonFormatter())
logging.root.addHandler(_handler)
logging.root.setLevel(logging.INFO)

log = logging.getLogger("mollygraph")

security = HTTPBearer(auto_error=False)

app = FastAPI(
    title="MollyGraph",
    description="Local-first graph + vector memory service",
    version="1.0.0",
)

@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
    log.info("request", extra={
        "path": request.url.path,
        "method": request.method,
        "status": response.status_code,
        "duration_ms": round(duration_ms, 2),
    })
    try:
        log_request(
            path=request.url.path,
            method=request.method,
            status=response.status_code,
            duration_ms=duration_ms,
        )
    except Exception:
        pass
    return response


# Global instances initialized during lifespan startup.
graph: BiTemporalGraph | None = None
vector_store: VectorStore | None = None
pipeline: ExtractionPipeline | None = None
queue: ExtractionQueue | None = None
queue_worker: QueueWorker | None = None
_worker_task: asyncio.Task | None = None


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

    # Handle neo4j.time.DateTime and similar objects.
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


class AuditRequest(BaseModel):
    limit: int = Field(default=500, ge=1, le=5000)
    dry_run: bool = False
    schedule: str = "nightly"
    model: str | None = None


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


def verify_api_key(credentials: HTTPAuthorizationCredentials | None = Depends(security)) -> str:
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


def require_runtime_ready() -> None:
    if queue is None or pipeline is None or graph is None:
        raise HTTPException(status_code=503, detail="Service not ready")


def _refresh_embedding_runtime() -> None:
    # Force re-load when HF model/provider changes.
    ExtractionPipeline.invalidate_embedding_cache()


def _refresh_extractor_runtime() -> None:
    # Force extractor model reload when backend/model changes.
    memory_extractor.invalidate_model_cache()


def _validate_strict_ai_startup() -> None:
    if not getattr(config, "STRICT_AI", False):
        return

    errors: list[str] = []
    embedding_status = get_embedding_status()
    extractor_status = get_extractor_status()

    for item in embedding_status.get("blocking_errors", []):
        text = str(item).strip()
        if text:
            errors.append(text)
    for item in extractor_status.get("blocking_errors", []):
        text = str(item).strip()
        if text:
            errors.append(text)

    if errors:
        raise RuntimeError("strict_ai startup validation failed: " + " | ".join(errors))


def _reindex_embeddings_sync(limit: int, dry_run: bool) -> dict[str, Any]:
    if graph is None or vector_store is None:
        raise RuntimeError("Service not ready")

    rows = graph.list_entities_for_embedding(limit=limit)
    if dry_run:
        return {
            "status": "dry_run",
            "provider": config.EMBEDDING_BACKEND,
            "model": (
                config.EMBEDDING_MODEL
                if config.EMBEDDING_BACKEND in {"huggingface", "sentence-transformers", "sentence_transformers", "st", "hf"}
                else config.OLLAMA_EMBED_MODEL if config.EMBEDDING_BACKEND == "ollama" else ""
            ),
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

    return {
        "status": "ok" if failed == 0 else "partial",
        "provider": config.EMBEDDING_BACKEND,
        "model": (
            config.EMBEDDING_MODEL
            if config.EMBEDDING_BACKEND in {"huggingface", "sentence-transformers", "sentence_transformers", "st", "hf"}
            else config.OLLAMA_EMBED_MODEL if config.EMBEDDING_BACKEND == "ollama" else ""
        ),
        "entities_found": len(rows),
        "reindexed": reindexed,
        "failed": failed,
        "errors": errors,
    }


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global graph, vector_store, pipeline, queue, queue_worker, _worker_task

    initialize_embedding_registry()
    initialize_extractor_registry()
    initialize_extractor_schema_registry()

    # Verify gliner2 is importable at startup.
    try:
        import gliner2  # type: ignore
        log.info("startup_check: gliner2 OK", extra={"path": gliner2.__file__})
    except ImportError as e:
        log.error("startup_check FAILED: %s. Python: %s", e, sys.executable)
        raise RuntimeError(f"GLiNER2 not importable: {e}")
    log.info("boot: python=%s venv=%s", sys.executable, os.environ.get("VIRTUAL_ENV", "none"))

    if config.TEST_MODE:
        set_graph_instance(None)
        log.info("Starting in test mode (external services disabled)")
        yield
        return

    _validate_strict_ai_startup()

    log.info("Starting MollyGraph service")

    graph = BiTemporalGraph(config.NEO4J_URI, config.NEO4J_USER, config.NEO4J_PASSWORD)
    set_graph_instance(graph)
    try:
        vector_store = VectorStore(backend=config.VECTOR_BACKEND)
    except Exception:
        log.warning(
            "Failed to initialize vector backend '%s', falling back to auto",
            config.VECTOR_BACKEND,
            exc_info=True,
        )
        vector_store = VectorStore(backend="auto")
    pipeline = ExtractionPipeline(graph=graph, vector_store=vector_store)
    queue = ExtractionQueue()

    async def process_job(job: ExtractionJob) -> ExtractionJob:
        return await pipeline.process_job(job)

    queue_worker = QueueWorker(queue=queue, processor=process_job, max_concurrent=3)
    _worker_task = asyncio.create_task(queue_worker.start())

    log.info("MollyGraph ready on %s:%s", config.HOST, config.PORT)

    try:
        yield
    finally:
        log.info("Shutting down MollyGraph")
        if queue_worker is not None:
            queue_worker.stop()
        if _worker_task is not None:
            _worker_task.cancel()
            try:
                await _worker_task
            except asyncio.CancelledError:
                pass
        set_graph_instance(None)
        graph = None


app.router.lifespan_context = lifespan


@app.exception_handler(Exception)
async def global_exception_handler(_request: Request, exc: Exception):
    log.error("Unhandled exception", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            code="INTERNAL_ERROR",
            detail=str(exc) if config.TEST_MODE else None,
            timestamp=datetime.utcnow().isoformat(),
        ).model_dump(),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=str(exc.detail),
            code=f"HTTP_{exc.status_code}",
            timestamp=datetime.utcnow().isoformat(),
        ).model_dump(),
    )


@app.get("/health")
async def health() -> dict[str, Any]:
    queue_pending = await asyncio.to_thread(queue.get_pending_count) if queue else 0
    queue_processing = await asyncio.to_thread(queue.get_processing_count) if queue else 0
    return {
        "status": "healthy",
        "version": "1.0.0",
        "port": config.PORT,
        "test_mode": config.TEST_MODE,
        "queue_pending": queue_pending,
        "queue_processing": queue_processing,
        "vector_stats": vector_store.get_stats() if vector_store else {},
    }


@app.get("/metrics/summary")
async def metrics_summary(_api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    """Return daily metrics summary: request counts, latencies, extraction quality."""
    return get_summary()


@app.get("/stats", response_model=StatsResponse)
async def stats(_api_key: str = Depends(verify_api_key)) -> StatsResponse:
    queue_stats = {
        "pending": await asyncio.to_thread(queue.get_pending_count) if queue else 0,
        "processing": await asyncio.to_thread(queue.get_processing_count) if queue else 0,
    }

    vector_stats = vector_store.get_stats() if vector_store else {}
    graph_summary = {
        "entity_count": 0,
        "relationship_count": 0,
        "episode_count": 0,
        "top_connected": [],
        "recent": [],
    }
    rel_distribution: dict[str, int] = {}

    if not config.TEST_MODE and graph is not None:
        try:
            graph_summary = graph.get_graph_summary()
            rel_distribution = graph.get_relationship_type_distribution()
        except Exception:
            log.debug("Graph summary unavailable", exc_info=True)

    graph_summary = _json_safe(graph_summary)
    rel_distribution = {str(k): int(v) for k, v in _json_safe(rel_distribution).items()}

    return StatsResponse(
        queue=queue_stats,
        vector_store=vector_stats,
        graph=graph_summary,
        relationship_type_distribution=rel_distribution,
        gliner_training=get_gliner_stats(),
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/ingest", operation_id="post_ingest")
@app.post(
    "/extract",
    operation_id="post_extract_legacy",
)  # Legacy alias kept for older integrations.
async def ingest(
    content: str,
    source: str = "manual",
    priority: int = 1,
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    require_runtime_ready()

    job = ExtractionJob(
        content=content,
        source=source,
        priority=priority,
        reference_time=datetime.utcnow(),
    )
    job_id = await asyncio.to_thread(queue.submit, job)
    queue_depth = await asyncio.to_thread(queue.get_pending_count)

    return {
        "job_id": job_id,
        "status": "queued",
        "queue_depth": queue_depth,
    }


@app.get("/entity/{name}", response_model=EntityResponse)
async def get_entity(name: str, _api_key: str = Depends(verify_api_key)) -> EntityResponse:
    require_runtime_ready()

    facts = graph.get_current_facts(name)
    if not facts:
        raise HTTPException(status_code=404, detail=f"Entity '{name}' not found")

    context = graph.get_entity_context(name, hops=2)
    return EntityResponse(
        entity=name,
        facts=facts,
        context=context,
        timestamp=datetime.utcnow().isoformat(),
    )


def _extract_query_entities(query: str) -> list[str]:
    """Extract candidate entity names from a natural-language query.

    Primary heuristic: title-cased or ALL-CAPS words >= 3 chars.
    Fuzzy fallback: if nothing qualifies, return every word >= 4 chars
    (allows queries like "tell me about rust" to match "Rust").
    """
    words = [w.strip(" ,.!?:;()[]{}") for w in query.split()]
    entities = [w for w in words if len(w) > 2 and (w.istitle() or w.isupper())]

    # Fuzzy fallback: lowercased words of reasonable length
    if not entities:
        entities = [w for w in words if len(w) >= 4 and w.isalpha()]

    # preserve order while deduping
    seen: set[str] = set()
    ordered: list[str] = []
    for ent in entities:
        key = ent.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(ent)
    return ordered


@app.get("/query", response_model=QueryResponse)
async def query(q: str, _api_key: str = Depends(verify_api_key)) -> QueryResponse:
    require_runtime_ready()

    entities = _extract_query_entities(q)
    results: list[dict[str, Any]] = []

    for entity_name in entities[:5]:
        facts = graph.get_current_facts(entity_name)
        if facts:
            results.append({"entity": entity_name, "facts": facts[:10]})

    # Fuzzy Neo4j CONTAINS fallback: if exact-name lookup missed, try substring match
    if not results:
        try:
            with graph.driver.session() as _session:
                for entity_name in entities[:5]:
                    _contains_q = entity_name.lower()
                    _rows = _session.run(
                        "MATCH (e:Entity) WHERE toLower(e.name) CONTAINS $q "
                        "RETURN e.name AS name LIMIT 5",
                        q=_contains_q,
                    )
                    for _row in _rows:
                        _name = str(_row.get("name") or "").strip()
                        if _name:
                            _facts = graph.get_current_facts(_name)
                            if _facts:
                                results.append({
                                    "entity": _name,
                                    "facts": _facts[:10],
                                    "match": "fuzzy_contains",
                                })
                    if results:
                        break
        except Exception:
            log.debug("Fuzzy CONTAINS fallback failed", exc_info=True)

    # Fallback: query vector index using deterministic embedding.
    if not results and vector_store is not None:
        embedding = ExtractionPipeline._text_embedding(q)
        for item in vector_store.similarity_search(embedding, top_k=5):
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            facts = graph.get_current_facts(name)
            if facts:
                results.append({"entity": name, "facts": facts[:5], "score": item.get("score", 0.0)})

    return QueryResponse(
        query=q,
        entities_found=entities,
        results=results,
        result_count=len(results),
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/audit", operation_id="post_audit")
@app.post("/audit/run", operation_id="post_audit_run_legacy")  # Legacy alias for older action-style mappings.
@app.post("/maintenance/audit", operation_id="post_maintenance_audit_legacy")  # Legacy alias from archived router naming.
async def audit(req: AuditRequest, _api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    schedule = (req.schedule or "nightly").strip().lower()
    if schedule not in {"nightly", "weekly"}:
        raise HTTPException(status_code=400, detail="schedule must be 'nightly' or 'weekly'")

    return await run_llm_audit(
        limit=req.limit,
        dry_run=req.dry_run,
        schedule=schedule,
        model_override=req.model,
    )


@app.get("/suggestions/digest", operation_id="get_suggestions_digest")
@app.get("/suggestions_digest", operation_id="get_suggestions_digest_legacy")  # Legacy underscore alias.
async def suggestions_digest(_api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    digest = build_suggestion_digest()
    return {
        "digest": digest,
        "has_suggestions": bool(digest.strip()),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/train/gliner", operation_id="post_train_gliner")
@app.post("/training/gliner", operation_id="post_training_gliner_legacy")  # Legacy alias for train namespace drift.
async def train_gliner(req: TrainRequest, _api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    return await run_gliner_finetune_pipeline(force=req.force)


@app.get("/train/status", operation_id="get_train_status")
@app.get("/training/status", operation_id="get_training_status_legacy")  # Legacy alias for train namespace drift.
async def train_status(_api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    return {"gliner": get_gliner_stats()}


@app.get("/embeddings/config")
async def embeddings_config(_api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    return get_embedding_registry()


@app.get("/embeddings/status")
async def embeddings_status(_api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    return get_embedding_status()


@app.post("/embeddings/config")
async def set_embeddings_config(
    req: EmbeddingConfigRequest,
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    try:
        payload = set_active_embedding_provider(req.provider, req.model)
        _refresh_embedding_runtime()
        return payload
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/embeddings/models")
async def add_embeddings_model(
    req: EmbeddingModelRequest,
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    try:
        payload = add_embedding_model(req.provider, req.model, activate=req.activate)
        if req.activate:
            _refresh_embedding_runtime()
        return payload
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/embeddings/reindex")
async def reindex_embeddings(
    req: EmbeddingReindexRequest,
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    require_runtime_ready()
    try:
        return await asyncio.to_thread(_reindex_embeddings_sync, req.limit, req.dry_run)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/extractors/config")
async def extractors_config(_api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    return get_extractor_registry()


@app.get("/extractors/status")
async def extractors_status(_api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    return get_extractor_status()


@app.get("/extractors/schema")
async def extractors_schema(_api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    return get_extractor_schema_status(include_schema=True)


@app.get("/extractors/schema/presets")
async def extractors_schema_presets(
    include_schema: bool = False,
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    return get_extractor_schema_presets(include_schema=include_schema)


@app.post("/extractors/schema")
async def set_extractors_schema(
    req: ExtractorSchemaConfigRequest,
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    try:
        payload = set_active_extractor_schema(req.mode, req.preset)
        _refresh_extractor_runtime()
        return payload
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/extractors/schema/upload")
async def upload_extractors_schema(
    req: ExtractorSchemaUploadRequest,
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    try:
        payload = upload_custom_extractor_schema(
            schema={"entities": req.entities, "relations": req.relations},
            activate=req.activate,
        )
        if req.activate:
            _refresh_extractor_runtime()
        return payload
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/extractors/config")
async def set_extractors_config(
    req: ExtractorConfigRequest,
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    try:
        payload = set_active_extractor_backend(req.backend, req.model, req.relation_model)
        _refresh_extractor_runtime()
        return payload
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/extractors/models")
async def add_extractors_model(
    req: ExtractorModelRequest,
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    try:
        payload = add_extractor_model(
            req.backend,
            req.model,
            activate=req.activate,
            role=req.role,
        )
        if req.activate:
            _refresh_extractor_runtime()
        return payload
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/extractors/prefetch")
async def prefetch_extractor_model(
    req: ExtractorPrefetchRequest,
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    try:
        return await asyncio.to_thread(
            memory_extractor.prefetch_model,
            req.backend,
            req.model,
            req.relation_model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/maintenance/run")
async def trigger_maintenance(
    background_tasks: BackgroundTasks,
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    background_tasks.add_task(run_maintenance_cycle)
    return {"status": "maintenance_triggered", "timestamp": datetime.utcnow().isoformat()}


@app.get("/metrics/model-health")
async def metrics_model_health(_api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    """Return the current model health monitor status (rollback guard)."""
    try:
        return model_health_monitor.status()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"model_health_error: {exc}") from exc


@app.get("/training/runs")
async def training_runs(
    limit: int = 20,
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """List recent training run audit trails."""
    try:
        svc = GLiNERTrainingService()
        runs = await asyncio.to_thread(svc.list_training_runs, limit)
        return {
            "runs": runs,
            "count": len(runs),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"training_runs_error: {exc}") from exc


@app.post("/maintenance/nightly")
async def trigger_nightly_maintenance(
    background_tasks: BackgroundTasks,
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """Run the full nightly maintenance pipeline in the background.

    Sequence:
    1. LLM audit (Kimi/Gemini) on recent relationships
    2. Model health check → if quality degraded, trigger auto-rollback
    3. If training data threshold met and cooldown elapsed, auto-trigger LoRA fine-tune
    """
    async def _nightly_bg() -> None:
        log.info("Nightly maintenance pipeline started")
        try:
            # Step 1: LLM audit
            from audit.llm_audit import run_llm_audit

            audit_result = await run_llm_audit(limit=200, dry_run=False, schedule="nightly")
            log.info("Nightly audit complete: %s", audit_result.get("status"))
        except Exception:
            log.warning("Nightly audit step failed", exc_info=True)

        try:
            # Step 2: Model health check
            health = model_health_monitor.status()
            rollback_triggered = health.get("rollback_triggered", False)
            if rollback_triggered:
                log.warning("Model health monitor triggered rollback: %s", health.get("reason"))
            else:
                log.info("Model health OK: fallback_rate=%.4f", health.get("fallback_rate", 0.0))
        except Exception:
            log.warning("Model health check step failed", exc_info=True)

        try:
            # Step 3: Auto-trigger LoRA if conditions met
            pipeline_result = await run_gliner_finetune_pipeline(force=False)
            log.info("Nightly LoRA pipeline: status=%s", pipeline_result.get("status"))
        except Exception:
            log.warning("Nightly LoRA pipeline step failed", exc_info=True)

        log.info("Nightly maintenance pipeline complete")

    background_tasks.add_task(_nightly_bg)
    return {
        "status": "nightly_maintenance_triggered",
        "timestamp": datetime.utcnow().isoformat(),
        "sequence": ["llm_audit", "model_health_check", "lora_pipeline"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.HOST, port=config.PORT)
