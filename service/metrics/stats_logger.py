"""Metrics and stats logging for MollyGraph.

Provides:
- log_extraction: append per-job metrics to ~/.graph-memory/metrics/extraction_log.jsonl
- get_summary: compute daily summary stats for /metrics/summary endpoint
- log_request: called by timing middleware to track per-request latency
"""
from __future__ import annotations

import json
import logging
import os
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
METRICS_DIR = Path.home() / ".graph-memory" / "metrics"
EXTRACTION_LOG = METRICS_DIR / "extraction_log.jsonl"
REQUEST_LOG = METRICS_DIR / "request_log.jsonl"

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
        },
        "quality": {
            "related_to_fallback_rate": round(fallback_rate, 4),
            "total_fallbacks_today": total_fallbacks,
            "total_relationships_today": total_rels,
        },
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }
