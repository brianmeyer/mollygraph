"""MollyGraph v1 HTTP service."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone, UTC
from typing import Any

from fastapi import BackgroundTasks, Body, Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator

import config
from metrics.stats_logger import (
    get_recent_retrieval_queries,
    get_retrieval_summary,
    get_session_retrieval_counters,
    get_summary,
    log_request,
    log_retrieval,
)
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
from evolution.gliner_training import (
    cleanup_stale_gliner_training_examples,
    get_gliner_stats,
    run_gliner_finetune_pipeline,
    GLiNERTrainingService,
)
from metrics.model_health import model_health_monitor
from extraction.pipeline import ExtractionPipeline
from extraction.queue import ExtractionQueue, QueueWorker
from maintenance.auditor import run_maintenance_cycle
from maintenance.lock import is_maintenance_locked
from memory.graph import BiTemporalGraph, VALID_REL_TYPES
from memory import extractor as memory_extractor
from memory.graph_suggestions import build_suggestion_digest, init_adopted_schema
from memory.models import ExtractionJob
from memory.vector_store import VectorStore
from runtime_graph import set_graph_instance
from runtime_pipeline import set_pipeline_instance
from runtime_vector_store import set_vector_store_instance

class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
        }
        if record.exc_info and record.exc_info[1] is not None:
            import traceback
            entry["traceback"] = "".join(traceback.format_exception(*record.exc_info))
        return json.dumps(entry)


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

# Service start time (set in lifespan, used by /metrics/dashboard)
_SERVICE_STARTED_AT: datetime | None = None
# Tracks how many times the worker has been auto-restarted since it last ran
# cleanly.  Reset to 0 whenever we observe the task running; incremented each
# time we attempt a restart so we don't spin-restart a persistently broken worker.
_worker_restart_count: int = 0


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


def require_no_maintenance() -> None:
    """Raise 503 if a maintenance cycle is currently running.

    Delete and prune operations must not run concurrently with
    run_maintenance_cycle() which iterates the entity graph; doing so risks
    state corruption (deleting nodes mid-iteration).
    """
    if is_maintenance_locked():
        raise HTTPException(
            status_code=503,
            detail="Maintenance cycle in progress — please retry in a moment",
        )


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


def _active_embedding_info() -> tuple[str, str]:
    """Return (provider, model) for the currently active embedding tier."""
    tier = ExtractionPipeline._embedding_active_tier or "unknown"
    if tier in ("sentence-transformers", "st"):
        return ("sentence-transformers", config.EMBEDDING_ST_MODEL or config.EMBEDDING_MODEL or "google/embeddinggemma-300m")
    if tier == "ollama":
        return ("ollama", config.EMBEDDING_OLLAMA_MODEL or config.OLLAMA_EMBED_MODEL or "nomic-embed-text")
    if tier == "cloud":
        return ("cloud", config.EMBEDDING_CLOUD_MODEL or "")
    if tier == "hash":
        return ("hash", "")
    # Fallback: read legacy config
    backend = (config.EMBEDDING_BACKEND or "").strip().lower()
    if backend in ("huggingface", "sentence-transformers", "st", "hf"):
        return ("sentence-transformers", config.EMBEDDING_ST_MODEL or config.EMBEDDING_MODEL or "")
    if backend == "ollama":
        return ("ollama", config.OLLAMA_EMBED_MODEL or "")
    return ("unknown", "")


def _reindex_embeddings_sync(limit: int, dry_run: bool) -> dict[str, Any]:
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

    # Flush WAL after bulk reindex so data is immediately visible
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


# ── PID file management ────────────────────────────────────────────────────────
_PID_FILE = config.GRAPH_MEMORY_DIR / "mollygraph.pid"


def _check_pid_conflict() -> None:
    """Abort startup if another MollyGraph instance is already running.

    Reads the PID file at ``~/.graph-memory/mollygraph.pid``:
    - If the file exists and the recorded PID is *alive*, log CRITICAL and exit(1).
    - If the file exists but the PID is dead (stale), remove it and continue.
    - If no file exists, continue normally.
    """
    if not _PID_FILE.exists():
        return

    try:
        saved_pid = int(_PID_FILE.read_text().strip())
    except (ValueError, OSError):
        log.warning("PID file '%s' is corrupt — removing it", _PID_FILE)
        _PID_FILE.unlink(missing_ok=True)
        return

    try:
        os.kill(saved_pid, 0)  # signal 0 = existence check, no actual signal sent
        pid_alive = True
    except (OSError, ProcessLookupError):
        pid_alive = False

    if pid_alive:
        log.critical(
            "Port conflict detected: MollyGraph PID %d is already running "
            "(PID file: %s). Exiting to prevent dual-process corruption.",
            saved_pid, _PID_FILE,
        )
        sys.exit(1)
    else:
        log.warning(
            "Stale PID file found (pid=%d is dead) — removing and continuing",
            saved_pid,
        )
        _PID_FILE.unlink(missing_ok=True)


def _write_pid_file() -> None:
    """Write the current process PID to the PID file."""
    _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PID_FILE.write_text(str(os.getpid()))
    log.info("PID file written: %s (pid=%d)", _PID_FILE, os.getpid())


def _remove_pid_file() -> None:
    """Remove the PID file on clean shutdown."""
    try:
        _PID_FILE.unlink(missing_ok=True)
        log.info("PID file removed: %s", _PID_FILE)
    except OSError as exc:
        log.warning("Failed to remove PID file %s: %s", _PID_FILE, exc)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global graph, vector_store, pipeline, queue, queue_worker, _worker_task, _SERVICE_STARTED_AT
    _SERVICE_STARTED_AT = datetime.now(UTC)

    # Check for port conflicts before initialising anything else.
    # In test mode skip this so unit tests can run multiple in-process instances.
    if not config.TEST_MODE:
        _check_pid_conflict()
        _write_pid_file()

    initialize_embedding_registry()
    initialize_extractor_registry()
    initialize_extractor_schema_registry()

    # Verify gliner2 is importable at startup (skip in test mode — CI doesn't install torch).
    if not config.TEST_MODE:
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

    # Apply the persisted adopted schema to in-memory state.  Moved here from
    # module-level execution (graph_suggestions.py) so that tests importing the
    # module do not trigger disk I/O or mutate globals at import time.
    init_adopted_schema()

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
    set_vector_store_instance(vector_store)
    set_pipeline_instance(pipeline)
    queue = ExtractionQueue()

    async def process_job(job: ExtractionJob) -> ExtractionJob:
        return await pipeline.process_job(job)

    queue_worker = QueueWorker(queue=queue, processor=process_job, max_concurrent=3)
    _worker_task = asyncio.create_task(queue_worker.start())

    # Reconcile any incomplete episodes left over from a previous crash.
    try:
        with graph.driver.session() as _s:
            _incomplete_count = _s.run(
                "MATCH (ep:Episode {incomplete: true}) RETURN count(ep) AS n"
            ).single()["n"]
        if _incomplete_count:
            log.warning(
                "Startup reconciliation: %d incomplete episode(s) found "
                "(partial writes from a prior crash). "
                "Inspect with: MATCH (ep:Episode {incomplete: true}) RETURN ep",
                _incomplete_count,
            )
        else:
            log.info("Startup reconciliation: no incomplete episodes found")
    except Exception:
        log.warning("Startup incomplete-episode reconciliation failed (non-fatal)", exc_info=True)

    # Auto-reindex vectors if Zvec is empty but Neo4j has entities.
    # This handles restarts after Zvec collection recreation/corruption.
    try:
        vs_stats = vector_store.get_stats() if vector_store else {}
        vs_count = vs_stats.get("entities", 0)
        if vs_count == 0 and graph is not None:
            neo4j_count = len(graph.list_entities_for_embedding(limit=1))
            if neo4j_count > 0:
                log.info("Zvec empty but Neo4j has entities — triggering startup reindex")
                reindex_result = _reindex_embeddings_sync(limit=5000, dry_run=False)
                log.info(
                    "Startup reindex complete: reindexed=%d failed=%d",
                    reindex_result.get("reindexed", 0),
                    reindex_result.get("failed", 0),
                )
                vector_store.flush()
    except Exception:
        log.warning("Startup vector reindex failed (non-fatal)", exc_info=True)

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
        set_pipeline_instance(None)
        set_graph_instance(None)
        graph = None
        if not config.TEST_MODE:
            _remove_pid_file()


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
            timestamp=datetime.now(UTC).isoformat(),
        ).model_dump(),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=str(exc.detail),
            code=f"HTTP_{exc.status_code}",
            timestamp=datetime.now(UTC).isoformat(),
        ).model_dump(),
    )


@app.get("/health")
async def health() -> dict[str, Any]:
    global _worker_task, _worker_restart_count

    queue_pending = await asyncio.to_thread(queue.get_pending_count) if queue else 0
    queue_processing = await asyncio.to_thread(queue.get_processing_count) if queue else 0
    queue_stuck = await asyncio.to_thread(queue.get_stuck_count) if queue else 0
    queue_dead = await asyncio.to_thread(queue.get_dead_count) if queue else 0

    # Issue 5: Check worker task liveness; auto-restart once if it has crashed.
    worker_status = "running"
    worker_error: str | None = None

    if _worker_task is None:
        worker_status = "not_started"
    elif _worker_task.done():
        # Extract crash reason (exception() raises CancelledError for cancelled tasks).
        try:
            exc = _worker_task.exception()
            worker_error = str(exc) if exc else "task exited unexpectedly"
        except asyncio.CancelledError:
            worker_error = "task was cancelled"
        except Exception as meta_exc:
            worker_error = f"could not retrieve exception: {meta_exc}"

        worker_status = "degraded"
        log.error("Worker task is no longer running: %s", worker_error)

        # Auto-restart once per death event.  Once a new task is running the
        # restart count is reset to 0 so a future crash can be retried too.
        if queue_worker is not None and _worker_restart_count < 1:
            try:
                log.warning("Attempting automatic worker restart (attempt #%d)", _worker_restart_count + 1)
                _worker_task = asyncio.create_task(queue_worker.start())
                _worker_restart_count += 1
                worker_status = "restarting"
                log.info("Worker restarted successfully")
            except Exception as restart_exc:
                log.error("Worker auto-restart failed: %s", restart_exc, exc_info=True)
                worker_status = "degraded"
        else:
            log.error(
                "Worker is degraded and restart limit reached (%d); manual intervention required",
                _worker_restart_count,
            )
    else:
        # Task is alive — reset restart counter so the next crash gets a retry.
        _worker_restart_count = 0

    vector_degraded = bool(vector_store and vector_store.is_degraded())
    if vector_degraded:
        log.error(
            "Vector store is in degraded mode — collection failed to open; "
            "writes are no-ops and searches return empty results"
        )

    overall_status = (
        "degraded"
        if worker_status == "degraded" or vector_degraded
        else "healthy"
    )

    return {
        "status": overall_status,
        "version": "1.0.0",
        "port": config.PORT,
        "test_mode": config.TEST_MODE,
        "queue_pending": queue_pending,
        "queue_processing": queue_processing,
        "queue_stuck": queue_stuck,
        "queue_dead": queue_dead,
        "vector_stats": vector_store.get_stats() if vector_store else {},
        "vector_degraded": vector_degraded,
        "worker_status": worker_status,
        "worker_error": worker_error,
        "worker_restart_count": _worker_restart_count,
    }


@app.get("/metrics/summary")
async def metrics_summary(_api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    """Return daily metrics summary: request counts, latencies, extraction quality."""
    return get_summary()


@app.get("/metrics/retrieval")
async def metrics_retrieval(_api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    """Return detailed retrieval metrics and recent query timing traces."""
    return {
        "date": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
        "retrieval": get_retrieval_summary(),
        "recent_queries": get_recent_retrieval_queries(limit=10),
        "vector_store": vector_store.get_stats() if vector_store else {},
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.get("/metrics/schema-drift")
async def metrics_schema_drift(_api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    """Track schema drift: entity types and relationship types over time.
    Alerts if relationship types grew >5% in 24h."""
    drift_file = config.GRAPH_MEMORY_DIR / "schema_drift.json"
    now = datetime.now(tz=timezone.utc)
    today = now.strftime("%Y-%m-%d")

    # Current counts
    current_rel_types: dict[str, int] = {}
    current_entity_types: dict[str, int] = {}
    if graph is not None:
        try:
            current_rel_types = graph.get_relationship_type_distribution()
            with graph.driver.session() as s:
                rows = s.run(
                    "MATCH (e:Entity) RETURN e.entity_type AS t, count(*) AS c"
                ).data()
                current_entity_types = {r["t"] or "Unknown": r["c"] for r in rows}
        except Exception:
            pass

    num_rel_types = len(current_rel_types)
    num_entity_types = len(current_entity_types)

    # Load history
    history: list[dict] = []
    if drift_file.exists():
        try:
            history = json.loads(drift_file.read_text())
        except Exception:
            history = []

    # Find yesterday's snapshot
    yesterday_snap = None
    for snap in reversed(history):
        if snap.get("date") != today:
            yesterday_snap = snap
            break

    drift_pct_rel = 0.0
    drift_pct_ent = 0.0
    alarm = False
    new_rel_types: list[str] = []
    new_entity_types: list[str] = []

    if yesterday_snap:
        prev_rel = yesterday_snap.get("num_rel_types", num_rel_types)
        prev_ent = yesterday_snap.get("num_entity_types", num_entity_types)
        if prev_rel > 0:
            drift_pct_rel = ((num_rel_types - prev_rel) / prev_rel) * 100
        if prev_ent > 0:
            drift_pct_ent = ((num_entity_types - prev_ent) / prev_ent) * 100
        alarm = (
            drift_pct_rel > config.SCHEMA_DRIFT_ALARM_REL_THRESHOLD
            or drift_pct_ent > config.SCHEMA_DRIFT_ALARM_ENT_THRESHOLD
        )
        prev_rel_set = set(yesterday_snap.get("rel_types", []))
        prev_ent_set = set(yesterday_snap.get("entity_types", []))
        new_rel_types = sorted(set(current_rel_types.keys()) - prev_rel_set)
        new_entity_types = sorted(set(current_entity_types.keys()) - prev_ent_set)

    # Save today's snapshot (upsert)
    today_snap = {
        "date": today,
        "num_rel_types": num_rel_types,
        "num_entity_types": num_entity_types,
        "rel_types": sorted(current_rel_types.keys()),
        "entity_types": sorted(current_entity_types.keys()),
    }
    history = [s for s in history if s.get("date") != today]
    history.append(today_snap)
    # Keep last 30 days
    history = history[-30:]
    try:
        drift_file.write_text(json.dumps(history, indent=2))
    except Exception:
        pass

    return {
        "date": today,
        "relationship_types": num_rel_types,
        "entity_types": num_entity_types,
        "drift_pct_rel_types": round(drift_pct_rel, 1),
        "drift_pct_entity_types": round(drift_pct_ent, 1),
        "alarm": alarm,
        "alarm_threshold": f"rel_types +{config.SCHEMA_DRIFT_ALARM_REL_THRESHOLD}% or entity_types +{config.SCHEMA_DRIFT_ALARM_ENT_THRESHOLD}%",
        "new_rel_types_today": new_rel_types,
        "new_entity_types_today": new_entity_types,
        "rel_type_distribution": {str(k): int(v) for k, v in current_rel_types.items()},
        "history_days": len(history),
        "timestamp": now.isoformat(),
    }


@app.get("/stats", response_model=StatsResponse)
async def stats(_api_key: str = Depends(verify_api_key)) -> StatsResponse:
    vector_stats = vector_store.get_stats() if vector_store else {}
    graph_summary = {
        "entity_count": 0,
        "relationship_count": 0,
        "episode_count": 0,
        "top_connected": [],
        "recent": [],
    }
    rel_distribution: dict[str, int] = {}
    incomplete_episodes = 0

    if not config.TEST_MODE and graph is not None:
        try:
            graph_summary = graph.get_graph_summary()
            rel_distribution = graph.get_relationship_type_distribution()
        except Exception:
            log.debug("Graph summary unavailable", exc_info=True)
        try:
            with graph.driver.session() as _s:
                incomplete_episodes = _s.run(
                    "MATCH (ep:Episode {incomplete: true}) RETURN count(ep) AS n"
                ).single()["n"]
        except Exception:
            log.debug("Incomplete episode count unavailable", exc_info=True)

    queue_stats = {
        "pending": await asyncio.to_thread(queue.get_pending_count) if queue else 0,
        "processing": await asyncio.to_thread(queue.get_processing_count) if queue else 0,
        "stuck": await asyncio.to_thread(queue.get_stuck_count) if queue else 0,
        "dead": await asyncio.to_thread(queue.get_dead_count) if queue else 0,
        "incomplete_episodes": incomplete_episodes,
    }
    graph_summary = _json_safe(graph_summary)
    rel_distribution = {str(k): int(v) for k, v in _json_safe(rel_distribution).items()}

    return StatsResponse(
        queue=queue_stats,
        vector_store=vector_stats,
        graph=graph_summary,
        relationship_type_distribution=rel_distribution,
        gliner_training=get_gliner_stats(),
        timestamp=datetime.now(UTC).isoformat(),
    )


class _IngestBody(BaseModel):
    content: str
    source: str = "manual"
    priority: int = 1


@app.post("/ingest", operation_id="post_ingest")
@app.post(
    "/extract",
    operation_id="post_extract_legacy",
)  # Legacy alias kept for older integrations.
async def ingest(
    content: str | None = None,
    source: str = "manual",
    priority: int = 1,
    _api_key: str = Depends(verify_api_key),
    body: _IngestBody | None = Body(None),
) -> dict[str, Any]:
    # Accept content from either query param or JSON body
    if body is not None and content is None:
        content = body.content
        source = body.source
        priority = body.priority
    if not content:
        raise HTTPException(status_code=422, detail="content is required (query param or JSON body)")
    require_runtime_ready()

    job = ExtractionJob(
        content=content,
        source=source,
        priority=priority,
        reference_time=datetime.now(UTC),
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
        timestamp=datetime.now(UTC).isoformat(),
    )


@app.delete("/entity/{name}", operation_id="delete_entity")
async def delete_entity_endpoint(name: str, _api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    """Delete an entity by name from Neo4j (DETACH DELETE) and from the vector store.

    Returns the number of relationships that were also removed (since DETACH DELETE
    removes all attached edges).  ``vector_removed`` is ``True`` when the
    corresponding vector store entry was also found and deleted.
    """
    require_runtime_ready()
    require_no_maintenance()

    # Step 1: Look up entity_id and count attached relationships before deletion.
    with graph.driver.session() as _session:
        rec = _session.run(
            """
            MATCH (e:Entity)
            WHERE toLower(e.name) = $name
               OR ANY(a IN coalesce(e.aliases, []) WHERE toLower(a) = $name)
            OPTIONAL MATCH (e)-[r]-()
            RETURN coalesce(e.id, toLower(e.name)) AS entity_id,
                   count(r) AS rels
            """,
            name=name.lower(),
        ).single()

    if not rec or rec["entity_id"] is None:
        raise HTTPException(status_code=404, detail=f"Entity '{name}' not found")

    entity_id: str = str(rec["entity_id"])
    rels_removed: int = int(rec["rels"]) if rec["rels"] is not None else 0

    # Step 2: Delete from Neo4j.
    deleted = await asyncio.to_thread(graph.delete_entity, name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Entity '{name}' not found")

    # Step 3: Remove from vector store.
    vector_removed = False
    if vector_store is not None:
        try:
            vector_removed = bool(await asyncio.to_thread(vector_store.remove_entity, entity_id))
        except Exception:
            log.debug("delete_entity: vector remove failed for %s", entity_id, exc_info=True)

    log.info("delete_entity: name=%s entity_id=%s rels_removed=%d vector_removed=%s",
             name, entity_id, rels_removed, vector_removed)
    return {
        "deleted": True,
        "entity": name,
        "relationships_removed": rels_removed,
        "vector_removed": vector_removed,
    }


@app.delete("/relationship", operation_id="delete_relationship")
async def delete_relationship_endpoint(
    req: DeleteRelationshipRequest,
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """Delete a specific relationship (or all relationships) between two entities.

    If ``rel_type`` is provided, only that relationship type is deleted.
    If ``rel_type`` is omitted, **all** relationship types between the two
    entities are deleted.  The count of actually deleted relationships is returned.
    """
    require_runtime_ready()
    require_no_maintenance()

    source = req.source.strip()
    target = req.target.strip()
    rel_type = req.rel_type.strip() if req.rel_type else None

    if not source or not target:
        raise HTTPException(status_code=422, detail="source and target are required")

    if rel_type:
        # Delete a specific typed relationship (both directions for safety).
        with graph.driver.session() as _session:
            rec = _session.run(
                f"""
                MATCH (h:Entity {{name: $source}})-[r:`{rel_type}`]-(t:Entity {{name: $target}})
                DELETE r
                RETURN count(r) AS deleted
                """,
                source=source,
                target=target,
            ).single()
        deleted_count = int(rec["deleted"]) if rec else 0
    else:
        # Delete ALL relationships between the two entities.
        with graph.driver.session() as _session:
            rec = _session.run(
                """
                MATCH (h:Entity {name: $source})-[r]-(t:Entity {name: $target})
                DELETE r
                RETURN count(r) AS deleted
                """,
                source=source,
                target=target,
            ).single()
        deleted_count = int(rec["deleted"]) if rec else 0

    log.info("delete_relationship: source=%s target=%s rel_type=%s deleted=%d",
             source, target, rel_type, deleted_count)
    return {
        "deleted": deleted_count,
        "source": source,
        "target": target,
        "rel_type": rel_type,
    }


async def _delete_entity_and_vector(name: str) -> tuple[bool, bool]:
    """Helper: delete one entity from Neo4j + vector store.

    Returns (neo4j_deleted, vector_deleted).
    """
    # Look up entity_id for vector removal.
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


@app.post("/entities/prune", operation_id="post_entities_prune")
async def prune_entities_endpoint(
    req: PruneRequest,
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """Bulk delete entities matching criteria.

    Accepts either:
    - ``{"names": ["entity1", "entity2"]}`` — delete specific entities by name
    - ``{"orphans": true}`` — delete all entities with zero relationships

    For each entity both the Neo4j node (DETACH DELETE) and the vector store
    entry are removed.  Returns the count of pruned entities and removed vectors.
    """
    require_runtime_ready()
    require_no_maintenance()

    names_to_prune: list[str] = []

    if req.names:
        names_to_prune = [n.strip() for n in req.names if n.strip()]
    elif req.orphans:
        # Find all entities with 0 relationships.
        with graph.driver.session() as _session:
            rows = _session.run(
                """
                MATCH (e:Entity)
                WHERE NOT (e)--()
                RETURN e.name AS name
                """
            ).data()
        names_to_prune = [r["name"] for r in rows if r.get("name")]
    else:
        raise HTTPException(
            status_code=422,
            detail="Provide 'names' (list of entity names) or 'orphans': true",
        )

    pruned: list[str] = []
    vectors_removed = 0
    for name in names_to_prune:
        try:
            neo4j_ok, vec_ok = await _delete_entity_and_vector(name)
            if neo4j_ok:
                pruned.append(name)
            if vec_ok:
                vectors_removed += 1
        except Exception:
            log.warning("prune_entities: failed to delete %s", name, exc_info=True)

    log.info("prune_entities: requested=%d pruned=%d vectors_removed=%d",
             len(names_to_prune), len(pruned), vectors_removed)
    return {
        "pruned": len(pruned),
        "entities": pruned,
        "vectors_removed": vectors_removed,
    }


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

    query_start = time.perf_counter()
    entity_extraction_ms = 0.0
    graph_exact_lookup_ms = 0.0
    graph_fuzzy_lookup_ms = 0.0
    embedding_ms = 0.0
    vector_search_ms = 0.0
    reranker_ms = 0.0

    _entity_extract_start = time.perf_counter()
    entities = _extract_query_entities(q)
    entity_extraction_ms = (time.perf_counter() - _entity_extract_start) * 1000

    # ── Parallel branch: graph (exact → fuzzy) ────────────────────────────────
    async def _graph_branch() -> tuple[list[dict[str, Any]], float, float, str]:
        """Run graph exact lookup; fall back to fuzzy CONTAINS if exact misses.
        Returns (results, exact_ms, fuzzy_ms, source_label).
        """
        _exact_start = time.perf_counter()
        _graph_results: list[dict[str, Any]] = []
        for entity_name in entities[:5]:
            facts = graph.get_current_facts(entity_name)
            if facts:
                _graph_results.append({
                    "entity": entity_name,
                    "facts": facts[:10],
                    "retrieval_source": "graph_exact",
                })
        _exact_ms = (time.perf_counter() - _exact_start) * 1000

        _fuzzy_ms = 0.0
        _source = "graph_exact" if _graph_results else "none"

        # Fuzzy CONTAINS secondary strategy if exact missed
        if not _graph_results:
            _fuzzy_start = time.perf_counter()
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
                                    _graph_results.append({
                                        "entity": _name,
                                        "facts": _facts[:10],
                                        "match": "fuzzy_contains",
                                        "retrieval_source": "graph_fuzzy",
                                    })
                        if _graph_results:
                            break
            except Exception:
                log.debug("Fuzzy CONTAINS fallback failed", exc_info=True)
            _fuzzy_ms = (time.perf_counter() - _fuzzy_start) * 1000
            if _graph_results:
                _source = "graph_fuzzy"

        return _graph_results, _exact_ms, _fuzzy_ms, _source

    # ── Parallel branch: vector similarity search ─────────────────────────────
    async def _vector_branch() -> tuple[list[dict[str, Any]], float, float]:
        """Embed query and run vector similarity search.
        Returns (results, embedding_ms, vector_search_ms).
        """
        if vector_store is None:
            return [], 0.0, 0.0

        _emb_start = time.perf_counter()
        try:
            embedding = await asyncio.to_thread(ExtractionPipeline._text_embedding, q)
        except Exception as exc:
            log.debug("Vector embedding failed: %s", exc)
            return [], (time.perf_counter() - _emb_start) * 1000, 0.0
        _emb_ms = (time.perf_counter() - _emb_start) * 1000

        _vec_start = time.perf_counter()
        try:
            vector_hits = await asyncio.to_thread(vector_store.similarity_search, embedding, 5)
        except Exception as exc:
            log.debug("Vector similarity_search failed: %s", exc)
            return [], _emb_ms, (time.perf_counter() - _vec_start) * 1000
        _vec_ms = (time.perf_counter() - _vec_start) * 1000

        _vec_results: list[dict[str, Any]] = []
        for item in vector_hits:
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            facts = graph.get_current_facts(name)
            if facts:
                _vec_results.append({
                    "entity": name,
                    "facts": facts[:5],
                    "score": item.get("score", 0.0),
                    "retrieval_source": "vector",
                })
        return _vec_results, _emb_ms, _vec_ms

    # ── Run both branches in parallel ─────────────────────────────────────────
    (graph_results, graph_exact_lookup_ms, graph_fuzzy_lookup_ms, graph_source), \
    (vector_results, embedding_ms, vector_search_ms) = await asyncio.gather(
        _graph_branch(),
        _vector_branch(),
    )

    # ── Merge: graph results have priority; vector fills gaps ─────────────────
    seen_entities: set[str] = set()
    results: list[dict[str, Any]] = []

    for r in graph_results:
        key = r["entity"].lower()
        if key not in seen_entities:
            seen_entities.add(key)
            results.append(r)

    for r in vector_results:
        key = r["entity"].lower()
        if key not in seen_entities:
            seen_entities.add(key)
            results.append(r)

    # Determine combined retrieval source
    has_graph = bool(graph_results)
    has_vector = bool(vector_results)
    if has_graph and has_vector:
        retrieval_source = "combined"
    elif has_graph:
        retrieval_source = graph_source  # "graph_exact" or "graph_fuzzy"
    elif has_vector:
        retrieval_source = "vector"
    else:
        retrieval_source = "none"

    # ── Optional reranker ─────────────────────────────────────────────────────
    reranked = False
    if getattr(config, "RERANKER_ENABLED", False) and len(results) > 1:
        _rerank_start = time.perf_counter()
        try:
            reranker = await asyncio.to_thread(ExtractionPipeline._get_reranker_model)
            if reranker is not None:
                pairs = [(q, " ".join(str(f) for f in r.get("facts", []))) for r in results]
                scores = await asyncio.to_thread(reranker.predict, pairs)
                scored = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
                results = [r for _, r in scored]
                reranked = True
        except Exception as exc:
            log.debug("Reranker step failed: %s", exc)
        reranker_ms = (time.perf_counter() - _rerank_start) * 1000

    total_latency_ms = (time.perf_counter() - query_start) * 1000
    try:
        log_retrieval(
            query=q,
            retrieval_source=retrieval_source,
            result_count=len(results),
            latency_ms=total_latency_ms,
            vector_search_ms=vector_search_ms,
            embedding_ms=embedding_ms,
            entity_extraction_ms=entity_extraction_ms,
            graph_exact_lookup_ms=graph_exact_lookup_ms,
            graph_fuzzy_lookup_ms=graph_fuzzy_lookup_ms,
            entities_queried=entities,
            reranker_ms=reranker_ms,
            graph_result_count=len(graph_results),
            vector_result_count=len(vector_results),
            graph_entity_names=[r["entity"] for r in graph_results],
            vector_entity_names=[r["entity"] for r in vector_results],
        )
    except Exception:
        log.debug("metrics log_retrieval failed", exc_info=True)

    return QueryResponse(
        query=q,
        entities_found=entities,
        results=results,
        result_count=len(results),
        timestamp=datetime.now(UTC).isoformat(),
        reranked=reranked,
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


@app.get("/audit/signals/stats", operation_id="get_audit_signals_stats")
async def audit_signals_stats(_api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    """Return counts of audit signals emitted since the last service restart.

    Signal types tracked:
    - ``relationship_reclassified``: rel type corrected to a valid type
    - ``relationship_quarantined``: rel flagged as suspicious or uncertain
    - ``relationship_removed``: rel deleted as wrong/spam
    - ``relationship_verified``: rel confirmed as correct
    - ``entity_reclassified``: entity type changed by audit
    - ``entity_merged``: two entities merged into one
    - ``entity_quarantined``: entity flagged for review
    """
    from audit.signals import get_signal_counts, SIGNAL_TYPES
    counts = get_signal_counts()
    # Ensure all known signal types appear in the response (even if zero)
    full_counts = {st: counts.get(st, 0) for st in sorted(SIGNAL_TYPES)}
    total = sum(full_counts.values())
    return {
        "signal_counts": full_counts,
        "total_signals": total,
        "note": "Counts reset on service restart",
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.get("/suggestions/digest", operation_id="get_suggestions_digest")
@app.get("/suggestions_digest", operation_id="get_suggestions_digest_legacy")  # Legacy underscore alias.
async def suggestions_digest(_api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    digest = build_suggestion_digest()
    return {
        "digest": digest,
        "has_suggestions": bool(digest.strip()),
        "timestamp": datetime.now(UTC).isoformat(),
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
    return {"status": "maintenance_triggered", "timestamp": datetime.now(UTC).isoformat()}


@app.post("/maintenance/backfill-temporal", operation_id="post_maintenance_backfill_temporal")
async def backfill_temporal_properties(
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    require_runtime_ready()
    if graph is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    backfill_fn = getattr(graph, "backfill_temporal_properties_sync", None)
    if not callable(backfill_fn):
        raise HTTPException(status_code=501, detail="Temporal backfill is not supported by the active graph backend")

    try:
        summary = await asyncio.to_thread(backfill_fn)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"temporal_backfill_error: {exc}") from exc

    return {
        "status": "temporal_backfill_completed",
        "summary": _json_safe(summary),
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.post("/maintenance/reconcile-vectors")
async def reconcile_vectors(
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """Remove vector store entries that no longer have a matching Neo4j entity.

    Algorithm
    ---------
    1. Fetch all entity_ids from Neo4j via ``list_entities_for_embedding``.
    2. Fetch all entity_ids stored in the vector store.
    3. For each vector id that has no Neo4j counterpart, remove it.
    4. Return counts: vectors_checked, orphans_removed.

    Note: ZvecBackend does not support listing all stored entity_ids; if the
    active backend returns ``None`` from ``list_all_entity_ids()``, the endpoint
    returns a partial result explaining the limitation.
    """
    require_runtime_ready()
    if graph is None or vector_store is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        # Step 1 – all Neo4j entity_ids (up to 500 000 rows).
        neo4j_rows = await asyncio.to_thread(
            graph.list_entities_for_embedding, 500_000
        )
        neo4j_ids: set[str] = {row["entity_id"] for row in neo4j_rows}

        # Step 2 – all vector store entity_ids.
        all_vec_ids: list[str] | None = await asyncio.to_thread(
            vector_store.list_all_entity_ids
        )

        if all_vec_ids is None:
            return {
                "status": "partial",
                "message": (
                    "Active vector backend does not support listing all entity ids "
                    "(ZvecBackend). Reconciliation skipped. "
                    "Consider switching to sqlite-vec for full reconciliation support."
                ),
                "neo4j_entities": len(neo4j_ids),
                "orphans_removed": 0,
                "timestamp": datetime.now(UTC).isoformat(),
            }

        # Step 3 – find orphan vectors (in vector store but not in Neo4j).
        orphan_ids = [vid for vid in all_vec_ids if vid not in neo4j_ids]

        # Step 4 – remove each orphan.
        removed = 0
        for eid in orphan_ids:
            try:
                if await asyncio.to_thread(vector_store.remove_entity, eid):
                    removed += 1
            except Exception:
                log.debug("reconcile-vectors: failed to remove %s", eid, exc_info=True)

        log.info(
            "reconcile-vectors: checked=%d neo4j_entities=%d orphans_found=%d orphans_removed=%d",
            len(all_vec_ids),
            len(neo4j_ids),
            len(orphan_ids),
            removed,
        )

        return {
            "status": "ok",
            "vectors_checked": len(all_vec_ids),
            "neo4j_entities": len(neo4j_ids),
            "orphans_found": len(orphan_ids),
            "orphans_removed": removed,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as exc:
        log.error("reconcile-vectors endpoint failed", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"reconcile_vectors_error: {exc}"
        ) from exc


@app.post("/maintenance/refresh-embeddings")
async def maintenance_refresh_embeddings(
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """Re-compute stale embeddings for entities whose type was reclassified.

    When the nightly audit reclassifies an entity (e.g. Person → Organization),
    the vector embedding still reflects the old type.  This endpoint:
    1. Queries Neo4j for all entities with ``embedding_stale=True``
    2. Re-computes embeddings using current name + type + latest context
    3. Updates the vector store
    4. Clears the ``embedding_stale`` flag on each updated entity

    Also invoked automatically during the nightly maintenance cycle (after audit,
    before training).
    """
    require_runtime_ready()
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        result = await pipeline.refresh_stale_embeddings()
        return {
            "status": "ok",
            **result,
            "timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as exc:
        log.error("refresh-embeddings endpoint failed", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"refresh_embeddings_error: {exc}"
        ) from exc


@app.post("/maintenance/quality-check")
async def maintenance_quality_check(
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """Run the golden-set graph quality evaluation and return results.

    Loads ``tests/golden_set.json`` relative to the service directory, runs
    each test input through the live extractor, and returns per-case and
    aggregate precision/recall metrics.

    Returns HTTP 503 if the extractor model is unavailable (e.g. ``gliner2``
    not installed or the service is in TEST_MODE).
    """
    import importlib
    import importlib.util
    from pathlib import Path as _Path

    # Reject in test mode — extractor returns empty results, metrics are meaningless.
    if config.TEST_MODE:
        raise HTTPException(
            status_code=503,
            detail="quality-check is unavailable in TEST_MODE (extractor disabled)",
        )

    # Ensure gliner2 is importable before attempting.
    try:
        importlib.import_module("gliner2")
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"quality-check requires gliner2 to be installed: {exc}",
        ) from exc

    # Load the quality check runner from the tests module.
    tests_dir = _Path(__file__).parent / "tests"
    if not (tests_dir / "golden_set.json").exists():
        raise HTTPException(
            status_code=404,
            detail=f"golden_set.json not found at {tests_dir / 'golden_set.json'}",
        )

    try:
        spec = importlib.util.spec_from_file_location(
            "test_graph_quality",
            str(tests_dir / "test_graph_quality.py"),
        )
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        results = await asyncio.to_thread(mod.run_quality_check)
    except Exception as exc:
        log.error("quality-check failed", exc_info=True)
        raise HTTPException(status_code=500, detail=f"quality_check_error: {exc}") from exc

    return {
        "status": "passed" if results.get("passed") else "failed",
        "results": results,
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.post("/maintenance/cleanup-training-data")
async def cleanup_training_data(
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """Remove stale GLiNER training examples that contain quarantined relations.

    Reads all .jsonl files in the training directory and checks each positive
    relation against Neo4j.  Relations that have since been quarantined or
    deleted are stripped; examples that have no positive relations remaining
    are dropped entirely.  Files are atomically rewritten.

    This is also run automatically at the start of the nightly maintenance
    pipeline (before new examples are accumulated).
    """
    try:
        result = await cleanup_stale_gliner_training_examples()
        return {
            "status": "ok",
            "result": result,
            "timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as exc:
        log.error("cleanup_training_data endpoint failed", exc_info=True)
        raise HTTPException(status_code=500, detail=f"cleanup_error: {exc}") from exc


@app.get("/metrics/model-health")
async def metrics_model_health(_api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    """Return the current model health monitor status (rollback guard)."""
    try:
        return model_health_monitor.check_health()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"model_health_error: {exc}") from exc


@app.get("/model-health/status")
async def model_health_status(_api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    """Return full model health status including continuous degradation detection.

    Returns rolling stats from both the short rollback-guard window and the
    longer degradation-detection window, plus the ``degradation_detected`` flag.
    The flag is set when the rolling RELATED_TO fallback rate exceeds
    baseline + 15% across the last 100 extractions.  No auto-rollback is
    triggered by this endpoint — it surfaces the signal for operator review.
    """
    try:
        return model_health_monitor.get_status()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"model_health_status_error: {exc}") from exc


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
            "timestamp": datetime.now(UTC).isoformat(),
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

            # Step 1b: Auto-delete entities suggested by audit (only if AUDIT_AUTO_DELETE=true)
            if config.AUDIT_AUTO_DELETE and graph is not None:
                suggestions = audit_result.get("suggestions", [])
                auto_deleted: list[str] = []
                auto_vectors_removed = 0
                for s in suggestions:
                    if str(s.get("action", "")).lower() == "delete":
                        entity_name = str(s.get("entity") or s.get("name") or "").strip()
                        if entity_name:
                            try:
                                neo4j_ok, vec_ok = await _delete_entity_and_vector(entity_name)
                                if neo4j_ok:
                                    auto_deleted.append(entity_name)
                                if vec_ok:
                                    auto_vectors_removed += 1
                            except Exception:
                                log.warning("Nightly auto-delete failed for %s", entity_name, exc_info=True)
                if auto_deleted:
                    log.info(
                        "Nightly auto-delete: deleted=%d vectors_removed=%d entities=%s",
                        len(auto_deleted), auto_vectors_removed, auto_deleted,
                    )
        except Exception:
            log.warning("Nightly audit step failed", exc_info=True)

        try:
            # Step 2: Model health check
            health = model_health_monitor.check_health()
            rollback_triggered = health.get("rollback_triggered", False)
            if rollback_triggered:
                log.warning("Model health monitor triggered rollback: %s", health.get("reason"))
            else:
                log.info("Model health OK: fallback_rate=%.4f", health.get("fallback_rate", 0.0))
        except Exception:
            log.warning("Model health check step failed", exc_info=True)

        try:
            # Step 3: Cleanup stale training examples BEFORE new accumulation
            cleanup_result = await cleanup_stale_gliner_training_examples()
            log.info(
                "Nightly stale training cleanup: files_modified=%d examples_removed=%d relations_stripped=%d",
                cleanup_result.get("files_modified", 0),
                cleanup_result.get("examples_removed", 0),
                cleanup_result.get("relations_stripped", 0),
            )
        except Exception:
            log.warning("Nightly stale training cleanup step failed", exc_info=True)

        try:
            # Step 4: Auto-trigger LoRA if conditions met (accumulates new data first)
            pipeline_result = await run_gliner_finetune_pipeline(force=False)
            log.info("Nightly LoRA pipeline: status=%s", pipeline_result.get("status"))
        except Exception:
            log.warning("Nightly LoRA pipeline step failed", exc_info=True)

        log.info("Nightly maintenance pipeline complete")

    background_tasks.add_task(_nightly_bg)
    return {
        "status": "nightly_maintenance_triggered",
        "timestamp": datetime.now(UTC).isoformat(),
        "sequence": ["llm_audit", "model_health_check", "stale_training_cleanup", "lora_pipeline"],
    }


def _cooldown_remaining_hours(state: dict) -> int:
    """Compute hours remaining in the strategy-aware training cooldown."""
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
    cooldown_days = config.GLINER_LORA_COOLDOWN_DAYS if strategy == "lora" else config.GLINER_FINETUNE_COOLDOWN_DAYS
    remaining = timedelta(days=cooldown_days) - (datetime.now(timezone.utc) - last_dt)
    return max(0, int(remaining.total_seconds() // 3600))


@app.get("/metrics/evolution")
async def metrics_evolution(_api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    """Comprehensive self-evolution metrics — the full story of how the graph improves itself."""
    require_runtime_ready()

    svc = GLiNERTrainingService()
    state = svc.state

    # Training history
    runs = await asyncio.to_thread(svc.list_training_runs, 50)
    successful_runs = [r for r in runs if r.get("combined_improvement") is not None]
    embedding_cfg = get_embedding_registry()

    vector_count: int | str = "unknown"
    if vector_store is not None:
        try:
            vector_stats = vector_store.get_stats()
        except Exception:
            vector_stats = {}
        if isinstance(vector_stats, dict):
            for key in ("entities", "dense_vectors", "vectors", "count"):
                value = vector_stats.get(key)
                if isinstance(value, (int, float)):
                    vector_count = int(value)
                    break

    # Graph health
    entity_count = await asyncio.to_thread(graph.entity_count)
    relationship_count = await asyncio.to_thread(graph.relationship_count)
    episode_count = await asyncio.to_thread(graph.episode_count)

    # Audit feedback stats
    try:
        from evolution.audit_feedback import load_audit_feedback_entries
        feedback = await asyncio.to_thread(load_audit_feedback_entries, 10000)
        approved = sum(1 for f in feedback if f.get("action") == "approve")
        rejected = sum(1 for f in feedback if f.get("action") == "reject")
        reclassified = sum(1 for f in feedback if f.get("action") == "reclassify")
    except Exception:
        approved = rejected = reclassified = 0

    # Model health
    try:
        model_health = model_health_monitor.check_health()
    except Exception:
        model_health = {"status": "unavailable"}

    # Relationship type distribution for quality signal
    try:
        dist = await asyncio.to_thread(graph.get_relationship_type_distribution)
        related_to_count = dist.get("RELATED_TO", 0)
        total_rels = sum(dist.values())
        fallback_rate = related_to_count / total_rels if total_rels > 0 else 0.0
    except Exception:
        related_to_count = 0
        total_rels = relationship_count
        fallback_rate = 0.0

    return {
        "graph": {
            "entities": entity_count,
            "relationships": relationship_count,
            "episodes": episode_count,
            "relationship_types": len(dist) if 'dist' in dir() else 0,
            "related_to_fallback_rate": round(fallback_rate, 4),
        },
        "training": {
            "total_examples": state.get("gliner_training_examples", 0),
            "total_runs": len(runs),
            "successful_runs": len(successful_runs),
            "last_strategy": state.get(
                "gliner_last_training_strategy",
                state.get("gliner_last_strategy", "unknown"),
            ),
            "last_status": state.get("gliner_last_cycle_status", "unknown"),
            "last_result": state.get("gliner_last_result", ""),
            "active_model": state.get("gliner_active_model_ref", "base"),
            "base_model": config.GLINER_BASE_MODEL,
            "cooldown": {
                "lora_days": config.GLINER_LORA_COOLDOWN_DAYS,
                "full_finetune_days": config.GLINER_FINETUNE_COOLDOWN_DAYS,
                "remaining_hours": _cooldown_remaining_hours(state),
            },
            "improvements": [
                {
                    "run_id": r.get("run_id"),
                    "strategy": r.get("mode"),
                    "entity_f1_delta": r.get("entity_improvement"),
                    "relation_f1_delta": r.get("relation_improvement"),
                    "combined_delta": r.get("combined_improvement"),
                }
                for r in successful_runs[:10]
            ],
        },
        "audit": {
            "feedback_total": approved + rejected + reclassified,
            "approved": approved,
            "rejected": rejected,
            "reclassified": reclassified,
            "approval_rate": round(approved / (approved + rejected + reclassified), 4) if (approved + rejected + reclassified) > 0 else 0.0,
        },
        "model_health": model_health,
        "embedding": {
            "provider": _active_embedding_info()[0],
            "model": _active_embedding_info()[1],
            "vectors": vector_count,
        },
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.get("/metrics/dashboard")
async def metrics_dashboard(_api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    """Single-call dashboard snapshot for README badges and status pages.

    Returns a comprehensive JSON object covering graph health, retrieval
    performance (with graph-vs-vector lift metrics), embedding config,
    extraction stats, training history, and uptime.
    """
    require_runtime_ready()

    # ── Graph health ──────────────────────────────────────────────────────────
    entity_count = await asyncio.to_thread(graph.entity_count)
    relationship_count = await asyncio.to_thread(graph.relationship_count)
    density = round(relationship_count / entity_count, 4) if entity_count > 0 else 0.0

    try:
        dist = await asyncio.to_thread(graph.get_relationship_type_distribution)
        type_count = len(dist)
        related_to_count = dist.get("RELATED_TO", 0)
        total_rels = sum(dist.values())
        related_to_rate = round(related_to_count / total_rels, 4) if total_rels > 0 else 0.0
    except Exception:
        type_count = 0
        related_to_rate = 0.0

    # ── Retrieval metrics ─────────────────────────────────────────────────────
    # Use session-level in-memory counters for lift/breakdown (accurate since restart)
    session_ret = get_session_retrieval_counters()
    # Also pull disk-based today summary for hit_rate/latency
    daily_ret = get_retrieval_summary()

    # ── Embedding / vector store ──────────────────────────────────────────────
    embedding_provider, embedding_model = _active_embedding_info()
    vector_count: int | str = "unknown"
    if vector_store is not None:
        try:
            vs_stats = vector_store.get_stats()
            for key in ("entities", "dense_vectors", "vectors", "count"):
                val = vs_stats.get(key)
                if isinstance(val, (int, float)):
                    vector_count = int(val)
                    break
        except Exception:
            pass

    # ── Extraction / ingestion ────────────────────────────────────────────────
    ing = ExtractionPipeline._get_ingestion_counters()
    # Supplement avg_yield from disk if class counters are empty (fresh restart)
    if ing["jobs_processed"] == 0:
        daily_summary = get_summary()
        disk_avg_ent = daily_summary.get("ingests", {}).get("avg_entities_per_ingest", 0.0)
        disk_avg_rel = 0.0
        ingest_count = daily_summary.get("ingests", {}).get("count_today", 0)
        if ingest_count > 0:
            total_rels_today = daily_summary.get("quality", {}).get("total_relationships_today", 0)
            disk_avg_rel = round(total_rels_today / ingest_count, 2)
    else:
        disk_avg_ent = ing["avg_entity_yield"]
        disk_avg_rel = ing["avg_relationship_yield"]

    # ── Training ──────────────────────────────────────────────────────────────
    svc = GLiNERTrainingService()
    state = svc.state
    runs = await asyncio.to_thread(svc.list_training_runs, 50)
    successful_runs = [r for r in runs if r.get("combined_improvement") is not None]
    last_f1_delta = 0.0
    if successful_runs:
        last_f1_delta = float(successful_runs[0].get("combined_improvement") or 0.0)
    cooldown_hours = _cooldown_remaining_hours(state)
    total_examples = state.get("gliner_training_examples", 0)

    # ── Uptime ────────────────────────────────────────────────────────────────
    started_at = _SERVICE_STARTED_AT.isoformat() if _SERVICE_STARTED_AT else None

    return {
        "graph": {
            "entities": entity_count,
            "relationships": relationship_count,
            "density": density,
            "types": type_count,
            "related_to_rate": related_to_rate,
        },
        "retrieval": {
            "total_queries": session_ret["total_queries"],
            "hit_rate": daily_ret["hit_rate"],
            "graph_only_hits": session_ret["graph_only_hits"],
            "vector_only_hits": session_ret["vector_only_hits"],
            "combined_hits": session_ret["combined_hits"],
            "graph_lift_pct": session_ret["graph_lift_pct"],
            "vector_lift_pct": session_ret["vector_lift_pct"],
            "avg_latency_ms": daily_ret["avg_latency_ms"],
            "p50_latency_ms": daily_ret["p50_latency_ms"],
            "p95_latency_ms": daily_ret["p95_latency_ms"],
        },
        "embedding": {
            "provider": embedding_provider,
            "model": embedding_model,
            "vectors": vector_count,
            "reranker_enabled": bool(getattr(config, "RERANKER_ENABLED", False)),
        },
        "extraction": {
            "backend": "gliner2",
            "avg_entity_yield": disk_avg_ent,
            "avg_relationship_yield": disk_avg_rel,
            "unique_entity_rate": ing["unique_entity_rate"],
            "related_to_fallback_rate": ing["related_to_fallback_rate"] or related_to_rate,
            "jobs_today": ing["jobs_processed"],
        },
        "training": {
            "total_examples": total_examples,
            "runs": len(runs),
            "successful_runs": len(successful_runs),
            "last_f1_delta": round(last_f1_delta, 4),
            "cooldown_hours": cooldown_hours,
            "active_model": state.get("gliner_active_model_ref", "base"),
        },
        "uptime": {
            "started_at": started_at,
            "total_queries": session_ret["total_queries"],
            "total_ingested_today": ing["jobs_processed"],
        },
        "timestamp": datetime.now(UTC).isoformat(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.HOST, port=config.PORT)
