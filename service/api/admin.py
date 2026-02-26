"""Admin / management API routes: metrics, audit, maintenance, embeddings, extractors, training."""
from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import logging
from datetime import datetime, timedelta, timezone, UTC
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel

import config
from audit.llm_audit import run_llm_audit
from embedding_registry import (
    add_embedding_model,
    get_embedding_registry,
    get_embedding_status,
    set_active_embedding_provider,
)
from evolution.gliner_training import (
    cleanup_stale_gliner_training_examples,
    get_gliner_stats,
    run_gliner_finetune_pipeline,
    GLiNERTrainingService,
)
from extractor_registry import (
    add_extractor_model,
    get_extractor_registry,
    get_extractor_status,
    set_active_extractor_backend,
)
from extractor_schema_registry import (
    get_extractor_schema_presets,
    get_extractor_schema_status,
    set_active_extractor_schema,
    upload_custom_extractor_schema,
)
from maintenance.auditor import run_maintenance_cycle
from maintenance.infra_health import evaluate_live_infra_health, InfraHealthDecision
from memory import extractor as memory_extractor
from memory.graph_suggestions import build_suggestion_digest
from metrics.model_health import model_health_monitor
from metrics.retrieval_quality import compute_retrieval_quality
from metrics.source_yield import get_daily_source_breakdown, get_source_stats
from metrics.stats_logger import (
    get_recent_retrieval_queries,
    get_retrieval_summary,
    get_session_retrieval_counters,
    get_summary,
)
from runtime_graph import get_graph_instance
from runtime_pipeline import get_pipeline_instance
from runtime_queue import get_queue_instance
from runtime_state import record_nightly_result, get_nightly_results, get_service_started_at
from runtime_vector_store import get_vector_store_instance

from api.deps import (
    AuditRequest,
    EmbeddingConfigRequest,
    EmbeddingModelRequest,
    EmbeddingReindexRequest,
    InfraHealthEvaluateRequest,
    ReconcileVectorsRequest,
    ExtractorConfigRequest,
    ExtractorModelRequest,
    ExtractorPrefetchRequest,
    ExtractorSchemaConfigRequest,
    ExtractorSchemaUploadRequest,
    StatsResponse,
    TrainRequest,
    _active_embedding_info,
    _cooldown_remaining_hours,
    _delete_entity_and_vector,
    _json_safe,
    _reindex_embeddings_sync,
    _refresh_embedding_runtime,
    _refresh_extractor_runtime,
    require_runtime_ready,
    verify_api_key,
)

log = logging.getLogger("mollygraph")
router = APIRouter()


# ── Stats ──────────────────────────────────────────────────────────────────────

@router.get("/stats", response_model=StatsResponse)
async def stats(_api_key: str = Depends(verify_api_key)) -> StatsResponse:
    graph = get_graph_instance()
    vector_store = get_vector_store_instance()
    queue = get_queue_instance()

    vector_stats = vector_store.get_stats() if vector_store else {}
    graph_summary: dict[str, Any] = {
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


# ── Metrics ────────────────────────────────────────────────────────────────────

@router.get("/metrics/summary")
async def metrics_summary(_api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    """Return daily metrics summary."""
    return get_summary()


@router.get("/metrics/retrieval")
async def metrics_retrieval(_api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    vector_store = get_vector_store_instance()
    return {
        "date": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
        "retrieval": get_retrieval_summary(),
        "recent_queries": get_recent_retrieval_queries(limit=10),
        "vector_store": vector_store.get_stats() if vector_store else {},
        "timestamp": datetime.now(UTC).isoformat(),
    }


@router.get("/metrics/schema-drift")
async def metrics_schema_drift(_api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    """Track schema drift: entity types and relationship types over time."""
    graph = get_graph_instance()
    drift_file = config.GRAPH_MEMORY_DIR / "schema_drift.json"
    now = datetime.now(tz=timezone.utc)
    today = now.strftime("%Y-%m-%d")

    current_rel_types: dict[str, int] = {}
    current_entity_types: dict[str, int] = {}
    if graph is not None:
        try:
            current_rel_types = graph.get_relationship_type_distribution()
            with graph.driver.session() as s:
                rows = s.run("MATCH (e:Entity) RETURN e.entity_type AS t, count(*) AS c").data()
                current_entity_types = {r["t"] or "Unknown": r["c"] for r in rows}
        except Exception:
            pass

    num_rel_types = len(current_rel_types)
    num_entity_types = len(current_entity_types)

    history: list[dict] = []
    if drift_file.exists():
        try:
            history = json.loads(drift_file.read_text())
        except Exception:
            history = []

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

    today_snap = {
        "date": today,
        "num_rel_types": num_rel_types,
        "num_entity_types": num_entity_types,
        "rel_types": sorted(current_rel_types.keys()),
        "entity_types": sorted(current_entity_types.keys()),
    }
    history = [s for s in history if s.get("date") != today]
    history.append(today_snap)
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
        "alarm_threshold": (
            f"rel_types +{config.SCHEMA_DRIFT_ALARM_REL_THRESHOLD}% "
            f"or entity_types +{config.SCHEMA_DRIFT_ALARM_ENT_THRESHOLD}%"
        ),
        "new_rel_types_today": new_rel_types,
        "new_entity_types_today": new_entity_types,
        "rel_type_distribution": {str(k): int(v) for k, v in current_rel_types.items()},
        "history_days": len(history),
        "timestamp": now.isoformat(),
    }


@router.get("/metrics/model-health")
async def metrics_model_health(_api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    try:
        return model_health_monitor.check_health()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"model_health_error: {exc}") from exc


@router.get("/metrics/nightly")
async def metrics_nightly(_api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    """Return nightly pipeline run history (success/failure per run)."""
    results = get_nightly_results()
    last = results[-1] if results else None

    # Aggregate review counts across all nightly runs
    total_reviewed = sum(r.get("relationships_reviewed", 0) for r in results)
    total_approved = sum(r.get("relationships_approved", 0) for r in results)
    total_flagged = sum(r.get("relationships_flagged", 0) for r in results)
    total_reclassified = sum(r.get("relationships_reclassified", 0) for r in results)

    return {
        "runs": results,
        "run_count": len(results),
        "last_run": last,
        "last_status": last.get("status") if last else None,
        "last_timestamp": last.get("timestamp") if last else None,
        "audit_totals": {
            "relationships_reviewed": total_reviewed,
            "relationships_approved": total_approved,
            "relationships_flagged": total_flagged,
            "relationships_reclassified": total_reclassified,
        },
        "timestamp": datetime.now(UTC).isoformat(),
    }


@router.get("/metrics/sources")
async def metrics_sources(_api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    """Return per-source ingestion yield statistics."""
    stats = await asyncio.to_thread(get_source_stats)
    today_breakdown = await asyncio.to_thread(get_daily_source_breakdown)
    return {
        **stats,
        "today": today_breakdown,
        "timestamp": datetime.now(UTC).isoformat(),
    }


@router.get("/metrics/retrieval/trend")
async def metrics_retrieval_trend(
    weeks: int = 4,
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """Return weekly retrieval quality trend (diversity, graph coverage).

    Reads retrieval log from stats_logger and buckets by ISO week.
    """
    from datetime import timedelta
    from metrics.stats_logger import get_recent_retrieval_queries

    now = datetime.now(tz=timezone.utc)
    cutoff = now - timedelta(weeks=weeks)

    try:
        recent = get_recent_retrieval_queries(limit=10000)
    except Exception:
        recent = []

    # Group by ISO week
    weekly: dict[str, list[dict[str, Any]]] = {}
    for q_record in recent:
        ts_str = q_record.get("timestamp") or ""
        try:
            ts = datetime.fromisoformat(ts_str)
        except Exception:
            continue
        if ts < cutoff:
            continue
        week_key = ts.strftime("%G-W%V")
        weekly.setdefault(week_key, []).append(q_record)

    trend: list[dict[str, Any]] = []
    for week_key in sorted(weekly.keys()):
        week_queries = weekly[week_key]
        # Build pseudo-results for quality computation from logged data
        pseudo_results = [
            {"entity": r.get("query", ""), "facts": [{}] if r.get("result_count", 0) > 0 else []}
            for r in week_queries
        ]
        quality = compute_retrieval_quality(pseudo_results)
        avg_latency = (
            sum(float(r.get("latency_ms", 0)) for r in week_queries) / len(week_queries)
            if week_queries else 0.0
        )
        avg_results = (
            sum(int(r.get("result_count", 0)) for r in week_queries) / len(week_queries)
            if week_queries else 0.0
        )
        trend.append({
            "week": week_key,
            "query_count": len(week_queries),
            "avg_latency_ms": round(avg_latency, 2),
            "avg_result_count": round(avg_results, 2),
            **quality,
        })

    return {
        "weeks_requested": weeks,
        "trend": trend,
        "timestamp": now.isoformat(),
    }


@router.get("/metrics/evolution")
async def metrics_evolution(_api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    """Comprehensive self-evolution metrics."""
    require_runtime_ready()
    graph = get_graph_instance()
    vector_store = get_vector_store_instance()

    svc = GLiNERTrainingService()
    state = svc.state
    runs = await asyncio.to_thread(svc.list_training_runs, 50)
    successful_runs = [r for r in runs if r.get("combined_improvement") is not None]

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

    entity_count = await asyncio.to_thread(graph.entity_count)
    relationship_count = await asyncio.to_thread(graph.relationship_count)
    episode_count = await asyncio.to_thread(graph.episode_count)

    dist: dict[str, int] = {}
    try:
        dist = await asyncio.to_thread(graph.get_relationship_type_distribution)
        related_to_count = dist.get("RELATED_TO", 0)
        total_rels = sum(dist.values())
        fallback_rate = related_to_count / total_rels if total_rels > 0 else 0.0
    except Exception:
        fallback_rate = 0.0

    try:
        from evolution.audit_feedback import load_audit_feedback_entries
        feedback = await asyncio.to_thread(load_audit_feedback_entries, 10000)
        approved = sum(1 for f in feedback if f.get("action") == "approve")
        rejected = sum(1 for f in feedback if f.get("action") == "reject")
        reclassified = sum(1 for f in feedback if f.get("action") == "reclassify")
    except Exception:
        approved = rejected = reclassified = 0

    try:
        model_health = model_health_monitor.check_health()
    except Exception:
        model_health = {"status": "unavailable"}

    # Aggregate nightly LLM review counts
    nightly_results = get_nightly_results()
    nightly_reviewed = sum(r.get("relationships_reviewed", 0) for r in nightly_results)
    nightly_approved = sum(r.get("relationships_approved", 0) for r in nightly_results)
    nightly_flagged = sum(r.get("relationships_flagged", 0) for r in nightly_results)
    nightly_reclassified = sum(r.get("relationships_reclassified", 0) for r in nightly_results)

    return {
        "graph": {
            "entities": entity_count,
            "relationships": relationship_count,
            "episodes": episode_count,
            "relationship_types": len(dist),
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
            "approval_rate": round(
                approved / (approved + rejected + reclassified), 4
            ) if (approved + rejected + reclassified) > 0 else 0.0,
            # Nightly LLM review counts (relationships that received verdicts)
            "llm_reviewed": nightly_reviewed,
            "llm_approved": nightly_approved,
            "llm_flagged": nightly_flagged,
            "llm_reclassified": nightly_reclassified,
        },
        "model_health": model_health,
        "embedding": {
            "provider": _active_embedding_info()[0],
            "model": _active_embedding_info()[1],
            "vectors": vector_count,
        },
        "timestamp": datetime.now(UTC).isoformat(),
    }


@router.get("/metrics/dashboard")
async def metrics_dashboard(_api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    """Single-call dashboard snapshot for README badges and status pages."""
    require_runtime_ready()
    graph = get_graph_instance()
    vector_store = get_vector_store_instance()
    from extraction.pipeline import ExtractionPipeline

    entity_count = await asyncio.to_thread(graph.entity_count)
    relationship_count = await asyncio.to_thread(graph.relationship_count)
    density = round(relationship_count / entity_count, 4) if entity_count > 0 else 0.0

    dist: dict[str, int] = {}
    type_count = 0
    related_to_rate = 0.0
    try:
        dist = await asyncio.to_thread(graph.get_relationship_type_distribution)
        type_count = len(dist)
        related_to_count = dist.get("RELATED_TO", 0)
        total_rels = sum(dist.values())
        related_to_rate = round(related_to_count / total_rels, 4) if total_rels > 0 else 0.0
    except Exception:
        pass

    session_ret = get_session_retrieval_counters()
    daily_ret = get_retrieval_summary()

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

    ing = ExtractionPipeline._get_ingestion_counters()
    if ing["jobs_processed"] == 0:
        daily_summary = get_summary()
        disk_avg_ent = daily_summary.get("ingests", {}).get("avg_entities_per_ingest", 0.0)
        ingest_count = daily_summary.get("ingests", {}).get("count_today", 0)
        disk_avg_rel = 0.0
        if ingest_count > 0:
            total_rels_today = daily_summary.get("quality", {}).get("total_relationships_today", 0)
            disk_avg_rel = round(total_rels_today / ingest_count, 2)
    else:
        disk_avg_ent = ing["avg_entity_yield"]
        disk_avg_rel = ing["avg_relationship_yield"]

    svc = GLiNERTrainingService()
    state = svc.state
    runs = await asyncio.to_thread(svc.list_training_runs, 50)
    successful_runs = [r for r in runs if r.get("combined_improvement") is not None]
    last_f1_delta = 0.0
    if successful_runs:
        last_f1_delta = float(successful_runs[0].get("combined_improvement") or 0.0)
    cooldown_hours = _cooldown_remaining_hours(state)
    total_examples = state.get("gliner_training_examples", 0)

    # Nightly pipeline stats
    last_nightly = get_nightly_results()
    last_nightly_result = last_nightly[-1] if last_nightly else None
    nightly_run_count = len(last_nightly)
    nightly_success_count = sum(1 for r in last_nightly if r.get("status") == "success")
    nightly_failure_count = sum(1 for r in last_nightly if r.get("status") == "failure")

    started_at = get_service_started_at()

    return {
        "graph": {
            "entities": entity_count,
            "relationships": relationship_count,
            "density": density,
            "types": type_count,
            "related_to_rate": related_to_rate,
        },
        "retrieval": {
            "total_queries": daily_ret["total_queries_today"],
            "hit_rate": daily_ret["hit_rate"],
            "graph_only_hits": daily_ret["graph_only_hits"],
            "vector_only_hits": daily_ret["vector_only_hits"],
            "combined_hits": daily_ret["combined_hits"],
            "graph_lift_pct": daily_ret["graph_lift_pct"],
            "vector_lift_pct": daily_ret["vector_lift_pct"],
            "avg_latency_ms": daily_ret["avg_latency_ms"],
            "p50_latency_ms": daily_ret["p50_latency_ms"],
            "p95_latency_ms": daily_ret["p95_latency_ms"],
            "reranked_query_count": daily_ret.get("reranked_query_count", 0),
            "avg_rerank_lift_pct": daily_ret.get("avg_rerank_lift_pct", 0.0),
            "avg_rank_improvement": daily_ret.get("avg_rank_improvement", 0.0),
        },
        "session_retrieval": session_ret,
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
        "nightly_pipeline": {
            "run_count": nightly_run_count,
            "success_count": nightly_success_count,
            "failure_count": nightly_failure_count,
            "last_run": last_nightly_result,
        },
        "uptime": {
            "started_at": started_at.isoformat() if started_at else None,
            "total_queries": daily_ret["total_queries_today"],
            "total_ingested_today": ing["jobs_processed"],
        },
        "timestamp": datetime.now(UTC).isoformat(),
    }


# ── Model health ───────────────────────────────────────────────────────────────

@router.get("/model-health/status")
async def model_health_status(_api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    try:
        return model_health_monitor.get_status()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"model_health_status_error: {exc}") from exc


# ── Audit ──────────────────────────────────────────────────────────────────────

@router.post("/audit", operation_id="post_audit")
@router.post("/audit/run", operation_id="post_audit_run_legacy")
@router.post("/maintenance/audit", operation_id="post_maintenance_audit_legacy")
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


@router.get("/audit/signals/stats", operation_id="get_audit_signals_stats")
async def audit_signals_stats(_api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    from audit.signals import get_signal_counts, SIGNAL_TYPES
    counts = get_signal_counts()
    full_counts = {st: counts.get(st, 0) for st in sorted(SIGNAL_TYPES)}
    total = sum(full_counts.values())
    return {
        "signal_counts": full_counts,
        "total_signals": total,
        "note": "Counts reset on service restart",
        "timestamp": datetime.now(UTC).isoformat(),
    }


# ── Suggestions ────────────────────────────────────────────────────────────────

@router.get("/suggestions/digest", operation_id="get_suggestions_digest")
@router.get("/suggestions_digest", operation_id="get_suggestions_digest_legacy")
async def suggestions_digest(_api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    digest = build_suggestion_digest()
    return {
        "digest": digest,
        "has_suggestions": bool(digest.strip()),
        "timestamp": datetime.now(UTC).isoformat(),
    }


# ── Training ───────────────────────────────────────────────────────────────────

@router.post("/train/gliner", operation_id="post_train_gliner")
@router.post("/training/gliner", operation_id="post_training_gliner_legacy")
async def train_gliner(req: TrainRequest, _api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    return await run_gliner_finetune_pipeline(force=req.force)


@router.get("/train/status", operation_id="get_train_status")
@router.get("/training/status", operation_id="get_training_status_legacy")
async def train_status(_api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    return {"gliner": get_gliner_stats()}


@router.get("/training/runs")
async def training_runs(
    limit: int = 20,
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
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


# ── Embeddings ─────────────────────────────────────────────────────────────────

@router.get("/embeddings/config")
async def embeddings_config(_api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    return get_embedding_registry()


@router.get("/embeddings/status")
async def embeddings_status(_api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    return get_embedding_status()


@router.post("/embeddings/config")
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


@router.post("/embeddings/models")
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


@router.post("/embeddings/reindex")
async def reindex_embeddings(
    req: EmbeddingReindexRequest,
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    require_runtime_ready()
    try:
        return await asyncio.to_thread(_reindex_embeddings_sync, req.limit, req.dry_run)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


# ── Extractors ─────────────────────────────────────────────────────────────────

@router.get("/extractors/config")
async def extractors_config(_api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    return get_extractor_registry()


@router.get("/extractors/status")
async def extractors_status(_api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    return get_extractor_status()


@router.get("/extractors/schema")
async def extractors_schema(_api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    return get_extractor_schema_status(include_schema=True)


@router.get("/extractors/schema/presets")
async def extractors_schema_presets(
    include_schema: bool = False,
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    return get_extractor_schema_presets(include_schema=include_schema)


@router.post("/extractors/schema")
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


@router.post("/extractors/schema/upload")
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


@router.post("/extractors/config")
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


@router.post("/extractors/models")
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


@router.post("/extractors/prefetch")
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


# ── Maintenance ────────────────────────────────────────────────────────────────

@router.post("/maintenance/run")
async def trigger_maintenance(
    background_tasks: BackgroundTasks,
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    background_tasks.add_task(run_maintenance_cycle)
    return {"status": "maintenance_triggered", "timestamp": datetime.now(UTC).isoformat()}


@router.post("/maintenance/backfill-temporal", operation_id="post_maintenance_backfill_temporal")
async def backfill_temporal_properties(_api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    require_runtime_ready()
    graph = get_graph_instance()
    if graph is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    backfill_fn = getattr(graph, "backfill_temporal_properties_sync", None)
    if not callable(backfill_fn):
        raise HTTPException(status_code=501, detail="Temporal backfill is not supported")
    try:
        summary = await asyncio.to_thread(backfill_fn)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"temporal_backfill_error: {exc}") from exc
    return {
        "status": "temporal_backfill_completed",
        "summary": _json_safe(summary),
        "timestamp": datetime.now(UTC).isoformat(),
    }


@router.post("/maintenance/infra-health/evaluate")
async def evaluate_infra_health(
    req: InfraHealthEvaluateRequest,
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """Evaluate infra health with deterministic policy (LLM advisory optional)."""
    require_runtime_ready()
    graph = get_graph_instance()
    vector_store = get_vector_store_instance()
    queue = get_queue_instance()
    if graph is None or vector_store is None or queue is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    evaluation = await evaluate_live_infra_health(
        graph,
        vector_store,
        queue,
        allow_rebuild=req.allow_rebuild,
        llm_advisory_enabled=req.enable_llm_advisory,
    )

    return {
        "status": "dry_run" if req.dry_run else "ok",
        "metrics": _json_safe(evaluation.metrics.__dict__),
        "deterministic_decision": evaluation.deterministic_decision.value,
        "reasons": evaluation.reasons,
        "llm_advisory": evaluation.llm_advisory,
        "final_action": evaluation.final_action.value,
        "timestamp": datetime.now(UTC).isoformat(),
    }


@router.post("/maintenance/reconcile-vectors")
async def reconcile_vectors(
    req: ReconcileVectorsRequest | None = None,
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """Remove vector store entries with no matching Neo4j entity.

    Notes:
    - mode=orphan_cleanup (default): existing behavior (list vector ids and remove orphans).
    - mode=rebuild: fallback for backends like zvec that cannot list all ids.
      Rebuild is gated by deterministic infra health unless force=true.
    """
    require_runtime_ready()
    graph = get_graph_instance()
    vector_store = get_vector_store_instance()
    if graph is None or vector_store is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        req = req or ReconcileVectorsRequest()
        mode = (req.mode or "orphan_cleanup").strip().lower()
        if mode not in {"orphan_cleanup", "rebuild"}:
            raise HTTPException(status_code=400, detail="mode must be orphan_cleanup or rebuild")

        neo4j_rows = await asyncio.to_thread(graph.list_entities_for_embedding, 500_000)
        neo4j_ids: set[str] = {row["entity_id"] for row in neo4j_rows}

        if mode == "rebuild":
            evaluation = await evaluate_live_infra_health(graph, vector_store, get_queue_instance(), allow_rebuild=True)
            rebuild_allowed = req.force or evaluation.final_action == InfraHealthDecision.REBUILD_VECTORS
            if not rebuild_allowed:
                return {
                    "status": "blocked",
                    "message": "rebuild mode requires deterministic infra-health rebuild decision or force=true",
                    "deterministic_decision": evaluation.final_action.value,
                    "reasons": evaluation.reasons,
                    "timestamp": datetime.now(UTC).isoformat(),
                }

            # Best-effort full re-embed from Neo4j (fallback path for zvec where full vector ID listing is unavailable)
            reindexed = 0
            failed = 0
            for row in neo4j_rows:
                entity_id = str(row.get("entity_id") or "").strip()
                name = str(row.get("name") or "").strip()
                if not entity_id or not name:
                    failed += 1
                    continue
                try:
                    await asyncio.to_thread(
                        vector_store.add_entity,
                        entity_id,
                        name,
                        str(row.get("entity_type") or "Concept"),
                        str(row.get("content") or name),
                    )
                    reindexed += 1
                except Exception:
                    failed += 1
            optimize_result = await asyncio.to_thread(vector_store.optimize)
            return {
                "status": "ok",
                "mode": "rebuild",
                "deterministic_decision": evaluation.final_action.value,
                "reasons": evaluation.reasons,
                "neo4j_entities": len(neo4j_ids),
                "reindexed": reindexed,
                "failed": failed,
                "optimize": _json_safe(optimize_result),
                "timestamp": datetime.now(UTC).isoformat(),
            }

        all_vec_ids: list[str] | None = await asyncio.to_thread(vector_store.list_all_entity_ids)

        if all_vec_ids is None:
            return {
                "status": "partial",
                "mode": "orphan_cleanup",
                "message": (
                    "Active vector backend does not support listing all entity ids (zvec limitation). "
                    "Use mode=rebuild as a fallback path when infra-health allows it."
                ),
                "neo4j_entities": len(neo4j_ids),
                "orphans_removed": 0,
                "timestamp": datetime.now(UTC).isoformat(),
            }

        orphan_ids = [vid for vid in all_vec_ids if vid not in neo4j_ids]
        removed = 0
        for eid in orphan_ids:
            try:
                if await asyncio.to_thread(vector_store.remove_entity, eid):
                    removed += 1
            except Exception:
                log.debug("reconcile-vectors: failed to remove %s", eid, exc_info=True)

        log.info(
            "reconcile-vectors: checked=%d neo4j=%d orphans=%d removed=%d",
            len(all_vec_ids), len(neo4j_ids), len(orphan_ids), removed,
        )
        return {
            "status": "ok",
            "mode": "orphan_cleanup",
            "vectors_checked": len(all_vec_ids),
            "neo4j_entities": len(neo4j_ids),
            "orphans_found": len(orphan_ids),
            "orphans_removed": removed,
            "timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as exc:
        log.error("reconcile-vectors endpoint failed", exc_info=True)
        raise HTTPException(status_code=500, detail=f"reconcile_vectors_error: {exc}") from exc


@router.post("/maintenance/refresh-embeddings")
async def maintenance_refresh_embeddings(_api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    require_runtime_ready()
    pipeline = get_pipeline_instance()
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        result = await pipeline.refresh_stale_embeddings()
        return {"status": "ok", **result, "timestamp": datetime.now(UTC).isoformat()}
    except Exception as exc:
        log.error("refresh-embeddings endpoint failed", exc_info=True)
        raise HTTPException(status_code=500, detail=f"refresh_embeddings_error: {exc}") from exc


@router.post("/maintenance/quality-check")
async def maintenance_quality_check(_api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    if config.TEST_MODE:
        raise HTTPException(status_code=503, detail="quality-check unavailable in TEST_MODE")
    try:
        importlib.import_module("gliner2")
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=f"quality-check requires gliner2: {exc}") from exc

    tests_dir = Path(__file__).parent.parent / "tests"
    if not (tests_dir / "golden_set.json").exists():
        raise HTTPException(status_code=404, detail="golden_set.json not found")

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


@router.post("/maintenance/cleanup-training-data")
async def cleanup_training_data(_api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    try:
        result = await cleanup_stale_gliner_training_examples()
        return {"status": "ok", "result": result, "timestamp": datetime.now(UTC).isoformat()}
    except Exception as exc:
        log.error("cleanup_training_data endpoint failed", exc_info=True)
        raise HTTPException(status_code=500, detail=f"cleanup_error: {exc}") from exc


@router.post("/maintenance/nightly")
async def trigger_nightly_maintenance(
    background_tasks: BackgroundTasks,
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """Run the full nightly maintenance pipeline in the background."""
    graph = get_graph_instance()

    async def _nightly_bg() -> None:
        log.info("Nightly maintenance pipeline started")
        audit_status_str: str | None = None
        lora_status_str: str | None = None
        error_str: str | None = None
        _audit_result: dict[str, Any] = {}
        infra_health_result: dict[str, Any] = {}

        try:
            # Step 0: deterministic infra health evaluation (advisory only in nightly payload)
            queue = get_queue_instance()
            vs = get_vector_store_instance()
            if graph is not None and vs is not None and queue is not None:
                eval_result = await evaluate_live_infra_health(
                    graph,
                    vs,
                    queue,
                    allow_rebuild=False,
                    llm_advisory_enabled=False,
                )
                infra_health_result = {
                    "deterministic_decision": eval_result.deterministic_decision.value,
                    "reasons": eval_result.reasons,
                    "final_action": eval_result.final_action.value,
                    "metrics": _json_safe(eval_result.metrics.__dict__),
                }
        except Exception:
            log.warning("Nightly infra health step failed", exc_info=True)
            infra_health_result = {"status": "error"}

        try:
            # Step 1: LLM audit
            _audit_result = await run_llm_audit(limit=200, dry_run=False, schedule="nightly")
            audit_result = _audit_result
            audit_status_str = str(audit_result.get("status", "unknown"))
            log.info("Nightly audit complete: %s", audit_status_str)

            # Step 1b: Auto-delete suggestions (if enabled)
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
                        "Nightly auto-delete: deleted=%d vectors_removed=%d",
                        len(auto_deleted), auto_vectors_removed,
                    )
        except Exception as exc:
            log.warning("Nightly audit step failed", exc_info=True)
            audit_status_str = "failed"
            error_str = str(exc)

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
            # Step 3: Cleanup stale training examples
            cleanup_result = await cleanup_stale_gliner_training_examples()
            log.info(
                "Nightly stale training cleanup: files_modified=%d examples_removed=%d",
                cleanup_result.get("files_modified", 0),
                cleanup_result.get("examples_removed", 0),
            )
        except Exception:
            log.warning("Nightly stale training cleanup step failed", exc_info=True)

        try:
            # Step 4: Auto-trigger LoRA if conditions met
            pipeline_result = await run_gliner_finetune_pipeline(force=False)
            lora_status_str = str(pipeline_result.get("status", "unknown"))
            log.info("Nightly LoRA pipeline: status=%s", lora_status_str)
        except Exception as exc:
            log.warning("Nightly LoRA pipeline step failed", exc_info=True)
            lora_status_str = "failed"
            if error_str is None:
                error_str = str(exc)

        overall = "failure" if error_str else "success"
        record_nightly_result(
            status=overall,
            audit_status=audit_status_str,
            lora_status=lora_status_str,
            error=error_str,
            relationships_reviewed=int(_audit_result.get("relationships_reviewed", 0)),
            relationships_approved=int(_audit_result.get("relationships_approved", 0)),
            relationships_flagged=int(_audit_result.get("relationships_flagged", 0)),
            relationships_reclassified=int(_audit_result.get("relationships_reclassified", 0)),
            infra_health=infra_health_result,
        )
        log.info("Nightly maintenance pipeline complete: %s", overall)

    background_tasks.add_task(_nightly_bg)
    return {
        "status": "nightly_maintenance_triggered",
        "timestamp": datetime.now(UTC).isoformat(),
        "sequence": ["infra_health_eval", "llm_audit", "model_health_check", "stale_training_cleanup", "lora_pipeline"],
    }
