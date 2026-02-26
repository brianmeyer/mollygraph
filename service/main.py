"""MollyGraph v1 HTTP service — app + lifespan only."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, UTC
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import config
from api.deps import ErrorResponse, verify_api_key
from embedding_registry import initialize_embedding_registry
from extractor_registry import initialize_extractor_registry
from extractor_schema_registry import initialize_extractor_schema_registry
from extraction.pipeline import ExtractionPipeline
from extraction.queue import ExtractionQueue, QueueWorker
from memory.graph import BiTemporalGraph
from memory import extractor as memory_extractor
from memory.graph_suggestions import init_adopted_schema
from memory.models import ExtractionJob
from memory.vector_store import VectorStore
from runtime_graph import set_graph_instance
from runtime_pipeline import set_pipeline_instance
from runtime_queue import set_queue_instance, get_queue_instance
from runtime_state import set_service_started_at
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

app = FastAPI(
    title="MollyGraph",
    description="Local-first graph + vector memory service",
    version="1.0.0",
)


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    from metrics.stats_logger import log_request
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


# Global worker state (used only by /health in this file).
_worker_task: asyncio.Task | None = None
_worker_restart_count: int = 0


def _validate_strict_ai_startup() -> None:
    if not getattr(config, "STRICT_AI", False):
        return
    from embedding_registry import get_embedding_status
    from extractor_registry import get_extractor_status
    errors: list[str] = []
    for item in get_embedding_status().get("blocking_errors", []):
        text = str(item).strip()
        if text:
            errors.append(text)
    for item in get_extractor_status().get("blocking_errors", []):
        text = str(item).strip()
        if text:
            errors.append(text)
    if errors:
        raise RuntimeError("strict_ai startup validation failed: " + " | ".join(errors))


# ── PID file management ────────────────────────────────────────────────────────
_PID_FILE = config.GRAPH_MEMORY_DIR / "mollygraph.pid"


def _check_pid_conflict() -> None:
    if not _PID_FILE.exists():
        return
    try:
        saved_pid = int(_PID_FILE.read_text().strip())
    except (ValueError, OSError):
        log.warning("PID file '%s' is corrupt — removing it", _PID_FILE)
        _PID_FILE.unlink(missing_ok=True)
        return
    try:
        os.kill(saved_pid, 0)
        pid_alive = True
    except (OSError, ProcessLookupError):
        pid_alive = False
    if pid_alive:
        log.critical(
            "Port conflict detected: MollyGraph PID %d is already running. Exiting.",
            saved_pid,
        )
        sys.exit(1)
    else:
        log.warning("Stale PID file (pid=%d is dead) — removing and continuing", saved_pid)
        _PID_FILE.unlink(missing_ok=True)


def _write_pid_file() -> None:
    _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PID_FILE.write_text(str(os.getpid()))
    log.info("PID file written: %s (pid=%d)", _PID_FILE, os.getpid())


def _remove_pid_file() -> None:
    try:
        _PID_FILE.unlink(missing_ok=True)
        log.info("PID file removed: %s", _PID_FILE)
    except OSError as exc:
        log.warning("Failed to remove PID file %s: %s", _PID_FILE, exc)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _worker_task, _worker_restart_count
    set_service_started_at(datetime.now(UTC))

    if not config.TEST_MODE:
        _check_pid_conflict()
        _write_pid_file()

    initialize_embedding_registry()
    initialize_extractor_registry()
    initialize_extractor_schema_registry()

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
    init_adopted_schema()

    graph = BiTemporalGraph(config.NEO4J_URI, config.NEO4J_USER, config.NEO4J_PASSWORD)
    set_graph_instance(graph)
    try:
        vector_store = VectorStore(backend=config.VECTOR_BACKEND)
    except Exception:
        log.warning("Failed to initialize vector backend '%s', falling back", config.VECTOR_BACKEND, exc_info=True)
        vector_store = VectorStore(backend="auto")
    pipeline = ExtractionPipeline(graph=graph, vector_store=vector_store)
    set_vector_store_instance(vector_store)
    set_pipeline_instance(pipeline)

    queue = ExtractionQueue()
    set_queue_instance(queue)

    async def process_job(job: ExtractionJob) -> ExtractionJob:
        return await pipeline.process_job(job)

    queue_worker = QueueWorker(queue=queue, processor=process_job, max_concurrent=3)
    _worker_task = asyncio.create_task(queue_worker.start())

    # Reconcile incomplete episodes from a previous crash.
    try:
        with graph.driver.session() as _s:
            _incomplete = _s.run(
                "MATCH (ep:Episode {incomplete: true}) RETURN count(ep) AS n"
            ).single()["n"]
        if _incomplete:
            log.warning("Startup: %d incomplete episode(s) found from prior crash", _incomplete)
        else:
            log.info("Startup: no incomplete episodes found")
    except Exception:
        log.warning("Startup incomplete-episode check failed (non-fatal)", exc_info=True)

    # Auto-reindex vectors if empty.
    try:
        vs_stats = vector_store.get_stats() if vector_store else {}
        vs_count = vs_stats.get("entities", 0)
        if vs_count == 0 and graph is not None:
            neo4j_count = len(graph.list_entities_for_embedding(limit=1))
            if neo4j_count > 0:
                log.info("Zvec empty but Neo4j has entities — triggering startup reindex")
                from api.deps import _reindex_embeddings_sync
                reindex_result = _reindex_embeddings_sync(limit=5000, dry_run=False)
                log.info(
                    "Startup reindex: reindexed=%d failed=%d",
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
        set_queue_instance(None)
        set_pipeline_instance(None)
        set_graph_instance(None)
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


@app.exception_handler(Exception)
async def http_exception_handler_compat(_request: Request, exc: Exception):
    # Import here to avoid circular import at module level
    from fastapi import HTTPException
    if not isinstance(exc, HTTPException):
        raise exc
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
    queue = get_queue_instance()

    queue_pending = await asyncio.to_thread(queue.get_pending_count) if queue else 0
    queue_processing = await asyncio.to_thread(queue.get_processing_count) if queue else 0
    queue_stuck = await asyncio.to_thread(queue.get_stuck_count) if queue else 0
    queue_dead = await asyncio.to_thread(queue.get_dead_count) if queue else 0

    from runtime_vector_store import get_vector_store_instance as _gvs
    vector_store = _gvs()
    worker_status = "running"
    worker_error: str | None = None

    if _worker_task is None:
        worker_status = "not_started"
    elif _worker_task.done():
        try:
            exc = _worker_task.exception()
            worker_error = str(exc) if exc else "task exited unexpectedly"
        except asyncio.CancelledError:
            worker_error = "task was cancelled"
        except Exception as meta_exc:
            worker_error = f"could not retrieve exception: {meta_exc}"

        worker_status = "degraded"
        log.error("Worker task is no longer running: %s", worker_error)

        from runtime_queue import get_queue_instance as _gq  # already imported
        # Re-import local to avoid shadowing outer 'queue'
        _queue_local = get_queue_instance()
        if _queue_local is not None and _worker_restart_count < 1:
            try:
                from extraction.queue import QueueWorker as _QW
                from memory.models import ExtractionJob as _EJ
                from runtime_pipeline import get_pipeline_instance as _gpi
                _pipeline = _gpi()

                async def _process(_job: _EJ) -> _EJ:
                    return await _pipeline.process_job(_job)

                _qw = _QW(queue=_queue_local, processor=_process, max_concurrent=3)
                _worker_task = asyncio.create_task(_qw.start())
                _worker_restart_count += 1
                worker_status = "restarting"
                log.info("Worker restarted successfully")
            except Exception as restart_exc:
                log.error("Worker auto-restart failed: %s", restart_exc, exc_info=True)
                worker_status = "degraded"
        else:
            log.error("Worker degraded, restart limit reached (%d)", _worker_restart_count)
    else:
        _worker_restart_count = 0

    vector_degraded = bool(vector_store and vector_store.is_degraded())
    if vector_degraded:
        log.error("Vector store is in degraded mode")

    overall_status = "degraded" if worker_status == "degraded" or vector_degraded else "healthy"

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


# ── Include route modules ──────────────────────────────────────────────────────
from api import query as _query_module
from api import ingest as _ingest_module
from api import admin as _admin_module
from api import decisions as _decisions_module

app.include_router(_query_module.router)
app.include_router(_ingest_module.router)
app.include_router(_admin_module.router)
app.include_router(_decisions_module.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)
