"""Deterministic infrastructure health evaluator for vector index safety."""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from audit.llm_audit import call_audit_model

log = logging.getLogger("mollygraph")


class InfraHealthDecision(str, Enum):
    HEALTHY = "healthy"
    OPTIMIZE = "optimize"
    REFRESH_EMBEDDINGS = "refresh_embeddings"
    REINDEX_EMBEDDINGS = "reindex_embeddings"
    REBUILD_VECTORS = "rebuild_vectors"


@dataclass
class InfraHealthThresholds:
    min_index_completeness_optimize: float = float(
        os.environ.get("MOLLYGRAPH_INFRA_MIN_INDEX_COMPLETENESS_OPTIMIZE", "0.995")
    )
    min_index_completeness_reindex: float = float(
        os.environ.get("MOLLYGRAPH_INFRA_MIN_INDEX_COMPLETENESS_REINDEX", "0.97")
    )
    max_entity_coverage_drift_pct_reindex: float = float(
        os.environ.get("MOLLYGRAPH_INFRA_MAX_ENTITY_COVERAGE_DRIFT_PCT_REINDEX", "0.15")
    )
    max_entity_coverage_drift_pct_rebuild: float = float(
        os.environ.get("MOLLYGRAPH_INFRA_MAX_ENTITY_COVERAGE_DRIFT_PCT_REBUILD", "0.35")
    )
    similarity_error_rate_reindex: float = float(
        os.environ.get("MOLLYGRAPH_INFRA_SIM_ERROR_RATE_REINDEX", "0.15")
    )
    similarity_error_rate_rebuild: float = float(
        os.environ.get("MOLLYGRAPH_INFRA_SIM_ERROR_RATE_REBUILD", "0.40")
    )
    similarity_latency_ms_optimize: float = float(
        os.environ.get("MOLLYGRAPH_INFRA_SIM_LATENCY_MS_OPTIMIZE", "250")
    )
    similarity_latency_ms_reindex: float = float(
        os.environ.get("MOLLYGRAPH_INFRA_SIM_LATENCY_MS_REINDEX", "600")
    )
    queue_backlog_warn: int = int(os.environ.get("MOLLYGRAPH_INFRA_QUEUE_BACKLOG_WARN", "500"))
    enable_llm_advisory: bool = (
        os.environ.get("MOLLYGRAPH_INFRA_LLM_ADVISORY", "0").strip().lower() in {"1", "true", "yes", "on"}
    )


@dataclass
class InfraHealthMetrics:
    graph_entity_count: int
    vector_entity_count: int
    index_completeness: float
    queue_pending: int
    queue_processing: int
    queue_stuck: int
    queue_dead: int
    similarity_search_count: int
    similarity_search_error_count: int
    similarity_search_error_rate: float
    similarity_search_avg_ms: float
    similarity_search_p95_ms: float


@dataclass
class InfraHealthEvaluation:
    metrics: InfraHealthMetrics
    deterministic_decision: InfraHealthDecision
    reasons: list[str]
    llm_advisory: dict[str, Any] | None
    final_action: InfraHealthDecision


def _pct_drift(graph_count: int, vector_count: int) -> float:
    if graph_count <= 0:
        return 0.0
    return abs(graph_count - vector_count) / graph_count


def evaluate_infra_health(
    metrics: InfraHealthMetrics,
    thresholds: InfraHealthThresholds | None = None,
    *,
    allow_rebuild: bool = False,
) -> tuple[InfraHealthDecision, list[str]]:
    t = thresholds or InfraHealthThresholds()
    reasons: list[str] = []
    drift = _pct_drift(metrics.graph_entity_count, metrics.vector_entity_count)

    if metrics.queue_dead > 0:
        reasons.append(f"queue_dead={metrics.queue_dead}")

    if (
        allow_rebuild
        and (
            drift >= t.max_entity_coverage_drift_pct_rebuild
            or metrics.similarity_search_error_rate >= t.similarity_error_rate_rebuild
            or (
                metrics.vector_entity_count == 0
                and metrics.graph_entity_count > 500
                and metrics.similarity_search_error_rate > 0.2
            )
        )
    ):
        if drift >= t.max_entity_coverage_drift_pct_rebuild:
            reasons.append(f"entity_coverage_drift={drift:.3f}")
        if metrics.similarity_search_error_rate >= t.similarity_error_rate_rebuild:
            reasons.append(f"similarity_error_rate={metrics.similarity_search_error_rate:.3f}")
        return InfraHealthDecision.REBUILD_VECTORS, reasons or ["severe_vector_health_degradation"]

    if (
        drift >= t.max_entity_coverage_drift_pct_reindex
        or metrics.index_completeness < t.min_index_completeness_reindex
        or metrics.similarity_search_error_rate >= t.similarity_error_rate_reindex
        or metrics.similarity_search_p95_ms >= t.similarity_latency_ms_reindex
    ):
        if drift >= t.max_entity_coverage_drift_pct_reindex:
            reasons.append(f"entity_coverage_drift={drift:.3f}")
        if metrics.index_completeness < t.min_index_completeness_reindex:
            reasons.append(f"index_completeness={metrics.index_completeness:.3f}")
        if metrics.similarity_search_error_rate >= t.similarity_error_rate_reindex:
            reasons.append(f"similarity_error_rate={metrics.similarity_search_error_rate:.3f}")
        if metrics.similarity_search_p95_ms >= t.similarity_latency_ms_reindex:
            reasons.append(f"similarity_p95_ms={metrics.similarity_search_p95_ms:.1f}")
        return InfraHealthDecision.REINDEX_EMBEDDINGS, reasons

    if metrics.queue_stuck > 0 or metrics.queue_pending >= t.queue_backlog_warn:
        if metrics.queue_stuck > 0:
            reasons.append(f"queue_stuck={metrics.queue_stuck}")
        if metrics.queue_pending >= t.queue_backlog_warn:
            reasons.append(f"queue_pending={metrics.queue_pending}")
        return InfraHealthDecision.REFRESH_EMBEDDINGS, reasons

    if (
        metrics.index_completeness < t.min_index_completeness_optimize
        or metrics.similarity_search_avg_ms >= t.similarity_latency_ms_optimize
    ):
        if metrics.index_completeness < t.min_index_completeness_optimize:
            reasons.append(f"index_completeness={metrics.index_completeness:.3f}")
        if metrics.similarity_search_avg_ms >= t.similarity_latency_ms_optimize:
            reasons.append(f"similarity_avg_ms={metrics.similarity_search_avg_ms:.1f}")
        return InfraHealthDecision.OPTIMIZE, reasons

    return InfraHealthDecision.HEALTHY, ["no_thresholds_triggered"]


async def run_llm_advisory(metrics: InfraHealthMetrics, deterministic: InfraHealthDecision) -> dict[str, Any]:
    prompt = (
        "You are an infra health advisor for a graph+vector memory service. "
        "Given metrics, summarize anomalies and recommend one action from: "
        "healthy,optimize,refresh_embeddings,reindex_embeddings,rebuild_vectors. "
        "Return strict JSON with keys summary,recommended_action,risk_level.\n"
        f"metrics={asdict(metrics)}\n"
        f"deterministic_decision={deterministic.value}"
    )
    result = await call_audit_model(prompt=prompt, schedule="nightly")
    content = str(result.get("content") or "").strip()
    recommended = ""
    for cand in InfraHealthDecision:
        if cand.value in content:
            recommended = cand.value
            break
    return {
        "status": "ok",
        "provider": result.get("provider"),
        "model": result.get("model"),
        "recommended_action": recommended or "unknown",
        "summary": content[:1000],
    }


async def evaluate_live_infra_health(
    graph: Any,
    vector_store: Any,
    queue: Any,
    *,
    allow_rebuild: bool = False,
    llm_advisory_enabled: bool | None = None,
) -> InfraHealthEvaluation:
    graph_count = int(await asyncio.to_thread(graph.entity_count)) if graph is not None else 0
    vector_stats = vector_store.get_stats() if vector_store is not None else {}
    vector_count = 0
    for key in ("entities", "dense_vectors", "vectors", "count"):
        val = vector_stats.get(key)
        if isinstance(val, (int, float)):
            vector_count = int(val)
            break

    seg_health = vector_store.get_segment_health() if vector_store is not None else {}
    index_completeness = 1.0
    if isinstance(seg_health, dict):
        raw = seg_health.get("index_completeness", {})
        if isinstance(raw, dict) and raw:
            vals = [float(v) for v in raw.values() if isinstance(v, (int, float))]
            if vals:
                index_completeness = min(vals)

    pending = int(await asyncio.to_thread(queue.get_pending_count)) if queue is not None else 0
    processing = int(await asyncio.to_thread(queue.get_processing_count)) if queue is not None else 0
    stuck = int(await asyncio.to_thread(queue.get_stuck_count)) if queue is not None else 0
    dead = int(await asyncio.to_thread(queue.get_dead_count)) if queue is not None else 0

    sim_count = int(vector_stats.get("similarity_search_count") or 0)
    sim_errors = int(vector_stats.get("similarity_search_error_count") or 0)
    error_rate = (sim_errors / sim_count) if sim_count > 0 else 0.0

    metrics = InfraHealthMetrics(
        graph_entity_count=graph_count,
        vector_entity_count=vector_count,
        index_completeness=float(index_completeness),
        queue_pending=pending,
        queue_processing=processing,
        queue_stuck=stuck,
        queue_dead=dead,
        similarity_search_count=sim_count,
        similarity_search_error_count=sim_errors,
        similarity_search_error_rate=round(error_rate, 6),
        similarity_search_avg_ms=float(vector_stats.get("similarity_search_avg_ms") or 0.0),
        similarity_search_p95_ms=float(vector_stats.get("similarity_search_p95_ms") or 0.0),
    )

    deterministic, reasons = evaluate_infra_health(metrics, allow_rebuild=allow_rebuild)

    llm_advisory: dict[str, Any] | None = None
    enabled = InfraHealthThresholds().enable_llm_advisory if llm_advisory_enabled is None else llm_advisory_enabled
    if enabled:
        try:
            llm_advisory = await run_llm_advisory(metrics, deterministic)
        except Exception as exc:
            log.warning("infra health llm advisory failed", exc_info=True)
            llm_advisory = {"status": "error", "error": str(exc)}

    return InfraHealthEvaluation(
        metrics=metrics,
        deterministic_decision=deterministic,
        reasons=reasons,
        llm_advisory=llm_advisory,
        final_action=deterministic,
    )
