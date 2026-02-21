"""MollyGraph v1 HTTP service."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

import config
from audit.llm_audit import run_llm_audit
from evolution.gliner_training import get_gliner_stats, run_gliner_finetune_pipeline
from extraction.pipeline import ExtractionPipeline
from extraction.queue import ExtractionQueue, QueueWorker
from maintenance.auditor import run_maintenance_cycle
from memory.bitemporal_graph import BiTemporalGraph
from memory.graph import get_graph_summary, get_relationship_type_distribution
from memory.graph_suggestions import build_suggestion_digest
from memory.models import ExtractionJob
from memory.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp":"%(asctime)s","level":"%(levelname)s","message":"%(message)s"}',
)
log = logging.getLogger("mollygraph")

security = HTTPBearer(auto_error=False)

app = FastAPI(
    title="MollyGraph",
    description="OpenClaw-first graph + vector memory service",
    version="1.0.0",
)

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


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global graph, vector_store, pipeline, queue, queue_worker, _worker_task

    if config.TEST_MODE:
        log.info("Starting in test mode (external services disabled)")
        yield
        return

    log.info("Starting MollyGraph service")

    graph = BiTemporalGraph(config.NEO4J_URI, config.NEO4J_USER, config.NEO4J_PASSWORD)
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
        if graph is not None:
            graph.close()


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
    return {
        "status": "healthy",
        "version": "1.0.0",
        "port": config.PORT,
        "test_mode": config.TEST_MODE,
        "queue_pending": queue.get_pending_count() if queue else 0,
        "queue_processing": queue.get_processing_count() if queue else 0,
        "vector_stats": vector_store.get_stats() if vector_store else {},
    }


@app.get("/stats", response_model=StatsResponse)
async def stats(_api_key: str = Depends(verify_api_key)) -> StatsResponse:
    queue_stats = {
        "pending": queue.get_pending_count() if queue else 0,
        "processing": queue.get_processing_count() if queue else 0,
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

    if not config.TEST_MODE:
        try:
            graph_summary = get_graph_summary()
            rel_distribution = get_relationship_type_distribution()
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
)  # Legacy alias kept for older OpenClaw/Molly tool configs.
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
    job_id = queue.submit(job)

    return {
        "job_id": job_id,
        "status": "queued",
        "queue_depth": queue.get_pending_count(),
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
    words = [w.strip(" ,.!?:;()[]{}") for w in query.split()]
    entities = [w for w in words if len(w) > 2 and (w.istitle() or w.isupper())]
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


@app.post("/maintenance/run")
async def trigger_maintenance(
    background_tasks: BackgroundTasks,
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    background_tasks.add_task(run_maintenance_cycle)
    return {"status": "maintenance_triggered", "timestamp": datetime.utcnow().isoformat()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.HOST, port=config.PORT)
