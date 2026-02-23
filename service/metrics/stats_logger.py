"""Metrics and stats logging for MollyGraph.

Provides:
- log_extraction: append per-job metrics to ~/.graph-memory/metrics/extraction_log.jsonl
- log_request: called by timing middleware to track per-request latency
- log_retrieval: append per-query retrieval metrics to retrieval_log.jsonl
- get_summary: compute daily summary stats for /metrics/summary endpoint
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
METRICS_DIR = Path.home() / ".graph-memory" / "metrics"
EXTRACTION_LOG = METRICS_DIR / "extraction_log.jsonl"
REQUEST_LOG = METRICS_DIR / "request_log.jsonl"
RETRIEVAL_LOG = METRICS_DIR / "retrieval_log.jsonl"

METRICS_DIR.mkdir(parents=True, exist_ok=True)

# ── In-memory request buffer (for fast P95 without reading disk) ──────────────
_req_lock = threading.Lock()
_req_latencies: list[float] = []  # rolling buffer, reset at midnight


def _today_str() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")


def _yesterday_str() -> str:
    from datetime import timedelta
    return (datetime.now(tz=timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")


# ── Log extraction stats ───────────────────────────────────────────────────────

def log_extraction(
    job_id: str,
    entities_extracted: int,
    relationships_extracted: int,
    fallback_count: int,
    processing_time_ms: float,
    content_length: int,
    confidence_min: float = 0.0,
    confidence_max: float = 0.0,
    confidence_avg: float = 0.0,
    embedding_time_ms: float = 0.0,
    vector_store_time_ms: float = 0.0,
) -> None:
    """Append a line to the extraction JSONL log."""
    record = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "date": _today_str(),
        "job_id": job_id,
        "entities_extracted": entities_extracted,
        "relationships_extracted": relationships_extracted,
        "fallback_count": fallback_count,
        "processing_time_ms": round(processing_time_ms, 2),
        "content_length": content_length,
        "confidence_min": round(confidence_min, 4),
        "confidence_max": round(confidence_max, 4),
        "confidence_avg": round(confidence_avg, 4),
        "embedding_time_ms": round(embedding_time_ms, 2),
        "vector_store_time_ms": round(vector_store_time_ms, 2),
    }
    try:
        with EXTRACTION_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        log.warning("Failed to write extraction log", exc_info=True)


# ── Log request latency ────────────────────────────────────────────────────────

def log_request(path: str, method: str, status: int, duration_ms: float) -> None:
    """Append a line to the request JSONL log and update in-memory buffer."""
    with _req_lock:
        _req_latencies.append(duration_ms)
        # Keep the buffer bounded (last 10k requests)
        if len(_req_latencies) > 10_000:
            del _req_latencies[:5_000]

    record = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "date": _today_str(),
        "path": path,
        "method": method,
        "status": status,
        "duration_ms": round(duration_ms, 2),
    }
    try:
        with REQUEST_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        log.warning("Failed to write request log", exc_info=True)


def log_retrieval(
    query: str,
    retrieval_source: str,
    result_count: int,
    latency_ms: float,
    vector_search_ms: float = 0.0,
    embedding_ms: float = 0.0,
    entity_extraction_ms: float = 0.0,
    graph_exact_lookup_ms: float = 0.0,
    graph_fuzzy_lookup_ms: float = 0.0,
    entities_queried: list[str] | None = None,
) -> None:
    """Append a line to the retrieval JSONL log."""
    source = retrieval_source if retrieval_source in {
        "graph_exact",
        "graph_fuzzy",
        "vector",
        "none",
    } else "none"
    record = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "date": _today_str(),
        "query": str(query)[:100],
        "retrieval_source": source,
        "result_count": int(max(0, result_count)),
        "latency_ms": round(latency_ms, 2),
        "vector_search_ms": round(vector_search_ms, 2),
        "embedding_ms": round(embedding_ms, 2),
        "entity_extraction_ms": round(entity_extraction_ms, 2),
        "graph_exact_lookup_ms": round(graph_exact_lookup_ms, 2),
        "graph_fuzzy_lookup_ms": round(graph_fuzzy_lookup_ms, 2),
        "entities_queried": [str(ent) for ent in (entities_queried or [])],
    }
    try:
        with RETRIEVAL_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        log.warning("Failed to write retrieval log", exc_info=True)


# ── Compute summary stats ──────────────────────────────────────────────────────

def _read_jsonl_today(path: Path, date_str: str) -> list[dict]:
    if not path.exists():
        return []
    results = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("date") == date_str:
                        results.append(rec)
                except json.JSONDecodeError:
                    continue
    except Exception:
        log.warning("Failed to read %s", path, exc_info=True)
    return results


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * pct / 100)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def get_retrieval_summary(date_str: str | None = None) -> dict[str, Any]:
    """Return retrieval metrics summary for a specific date (UTC), default today."""
    target_date = date_str or _today_str()
    recs = _read_jsonl_today(RETRIEVAL_LOG, target_date)

    latencies = [float(r.get("latency_ms", 0.0) or 0.0) for r in recs]
    vector_search = [float(r.get("vector_search_ms", 0.0) or 0.0) for r in recs]
    total = len(recs)

    source_breakdown = {
        "graph_exact": 0,
        "graph_fuzzy": 0,
        "vector": 0,
        "none": 0,
    }
    hit_count = 0

    for rec in recs:
        source = str(rec.get("retrieval_source") or "none")
        if source not in source_breakdown:
            source = "none"
        source_breakdown[source] += 1
        if int(rec.get("result_count", 0) or 0) > 0:
            hit_count += 1

    return {
        "total_queries_today": total,
        "avg_latency_ms": round(sum(latencies) / total, 2) if total else 0.0,
        "p95_latency_ms": round(_percentile(latencies, 95), 2),
        "avg_vector_search_ms": round(sum(vector_search) / total, 2) if total else 0.0,
        "hit_rate": round(hit_count / total, 4) if total else 0.0,
        "source_breakdown": source_breakdown,
    }


def get_recent_retrieval_queries(limit: int = 10) -> list[dict[str, Any]]:
    """Return most recent retrieval logs, newest first."""
    if limit <= 0 or not RETRIEVAL_LOG.exists():
        return []

    rows: list[dict[str, Any]] = []
    try:
        with RETRIEVAL_LOG.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        log.warning("Failed to read %s", RETRIEVAL_LOG, exc_info=True)
        return []

    trimmed = rows[-limit:]
    trimmed.reverse()

    return [{
        "timestamp": rec.get("timestamp"),
        "query": rec.get("query", ""),
        "retrieval_source": rec.get("retrieval_source", "none"),
        "result_count": int(rec.get("result_count", 0) or 0),
        "latency_ms": float(rec.get("latency_ms", 0.0) or 0.0),
        "entity_extraction_ms": float(rec.get("entity_extraction_ms", 0.0) or 0.0),
        "graph_exact_lookup_ms": float(rec.get("graph_exact_lookup_ms", 0.0) or 0.0),
        "graph_fuzzy_lookup_ms": float(rec.get("graph_fuzzy_lookup_ms", 0.0) or 0.0),
        "vector_search_ms": float(rec.get("vector_search_ms", 0.0) or 0.0),
        "embedding_ms": float(rec.get("embedding_ms", 0.0) or 0.0),
        "entities_queried": [str(v) for v in (rec.get("entities_queried") or [])],
    } for rec in trimmed]


def get_summary() -> dict[str, Any]:
    """Return daily summary stats."""
    today = _today_str()
    yesterday = _yesterday_str()

    # ── Request stats ────────────────────────────────────────────────────────
    req_today = _read_jsonl_today(REQUEST_LOG, today)
    latencies = [r["duration_ms"] for r in req_today if "duration_ms" in r]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    p95_latency = _percentile(latencies, 95)

    # ── Extraction stats ─────────────────────────────────────────────────────
    ext_today = _read_jsonl_today(EXTRACTION_LOG, today)
    ext_yesterday = _read_jsonl_today(EXTRACTION_LOG, yesterday)

    ingest_count = len(ext_today)
    entity_counts = [r.get("entities_extracted", 0) for r in ext_today]
    fallback_counts = [r.get("fallback_count", 0) for r in ext_today]
    rel_counts = [r.get("relationships_extracted", 0) for r in ext_today]
    processing_times = [float(r.get("processing_time_ms", 0.0) or 0.0) for r in ext_today]
    embedding_times = [float(r.get("embedding_time_ms", 0.0) or 0.0) for r in ext_today]
    vector_store_times = [float(r.get("vector_store_time_ms", 0.0) or 0.0) for r in ext_today]

    avg_entities = sum(entity_counts) / len(entity_counts) if entity_counts else 0.0
    total_fallbacks = sum(fallback_counts)
    total_rels = sum(rel_counts)
    fallback_rate = total_fallbacks / total_rels if total_rels > 0 else 0.0

    entity_total_today = sum(entity_counts)
    entity_total_yesterday = sum(r.get("entities_extracted", 0) for r in ext_yesterday)
    entity_growth = entity_total_today - entity_total_yesterday

    # ── In-memory latency snapshot ───────────────────────────────────────────
    with _req_lock:
        mem_latencies = list(_req_latencies)

    retrieval = get_retrieval_summary(today)

    return {
        "date": today,
        "requests": {
            "total_today": len(req_today),
            "avg_latency_ms": round(avg_latency, 2),
            "p95_latency_ms": round(p95_latency, 2),
            "in_memory_p95_ms": round(_percentile(mem_latencies, 95), 2),
        },
        "ingests": {
            "count_today": ingest_count,
            "avg_entities_per_ingest": round(avg_entities, 2),
            "total_entities_today": entity_total_today,
            "total_entities_yesterday": entity_total_yesterday,
            "entity_growth": entity_growth,
            "avg_processing_time_ms": round(sum(processing_times) / ingest_count, 2) if ingest_count else 0.0,
            "p95_processing_time_ms": round(_percentile(processing_times, 95), 2),
            "avg_embedding_time_ms": round(sum(embedding_times) / ingest_count, 2) if ingest_count else 0.0,
            "avg_vector_store_time_ms": round(sum(vector_store_times) / ingest_count, 2) if ingest_count else 0.0,
        },
        "retrieval": retrieval,
        "quality": {
            "related_to_fallback_rate": round(fallback_rate, 4),
            "total_fallbacks_today": total_fallbacks,
            "total_relationships_today": total_rels,
        },
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }
