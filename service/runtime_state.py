"""Shared runtime state: service start time and nightly pipeline tracking."""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, UTC
from pathlib import Path
import tempfile
from typing import Any

log = logging.getLogger(__name__)

_LOCK = threading.Lock()

# Service start time (set in lifespan).
_SERVICE_STARTED_AT: datetime | None = None

# Nightly pipeline result history (last N runs).
_NIGHTLY_RESULTS: list[dict[str, Any]] = []
_MAX_NIGHTLY_RESULTS = 20
_NIGHTLY_RESULTS_LOADED = False
_NIGHTLY_RESULTS_PATH = Path.home() / ".graph-memory" / "metrics" / "nightly_results.json"


def _coerce_nightly_results(payload: Any) -> list[dict[str, Any]]:
    raw = payload
    if isinstance(payload, dict) and isinstance(payload.get("runs"), list):
        raw = payload.get("runs")
    if not isinstance(raw, list):
        return []
    results: list[dict[str, Any]] = []
    for entry in raw:
        if isinstance(entry, dict):
            results.append(dict(entry))
    if len(results) > _MAX_NIGHTLY_RESULTS:
        results = results[-_MAX_NIGHTLY_RESULTS:]
    return results


def _load_nightly_results_locked() -> None:
    global _NIGHTLY_RESULTS_LOADED, _NIGHTLY_RESULTS
    if _NIGHTLY_RESULTS_LOADED:
        return
    _NIGHTLY_RESULTS_LOADED = True
    try:
        if not _NIGHTLY_RESULTS_PATH.exists():
            return
        payload = json.loads(_NIGHTLY_RESULTS_PATH.read_text(encoding="utf-8"))
        _NIGHTLY_RESULTS = _coerce_nightly_results(payload)
    except Exception:
        log.debug("Failed to load nightly results", exc_info=True)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")
        os.replace(tmp_name, str(path))
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def set_service_started_at(dt: datetime) -> None:
    global _SERVICE_STARTED_AT
    _SERVICE_STARTED_AT = dt


def get_service_started_at() -> datetime | None:
    return _SERVICE_STARTED_AT


def record_nightly_result(
    *,
    status: str,
    audit_status: str | None = None,
    lora_status: str | None = None,
    error: str | None = None,
    relationships_reviewed: int = 0,
    relationships_approved: int = 0,
    relationships_flagged: int = 0,
    relationships_reclassified: int = 0,
    infra_health: dict[str, Any] | None = None,
) -> None:
    """Record a nightly pipeline run result (success or failure)."""
    global _NIGHTLY_RESULTS
    entry: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "status": status,
        "audit_status": audit_status,
        "lora_status": lora_status,
        "error": error,
        "relationships_reviewed": relationships_reviewed,
        "relationships_approved": relationships_approved,
        "relationships_flagged": relationships_flagged,
        "relationships_reclassified": relationships_reclassified,
        "infra_health": infra_health or {},
    }
    with _LOCK:
        _load_nightly_results_locked()
        _NIGHTLY_RESULTS.append(entry)
        if len(_NIGHTLY_RESULTS) > _MAX_NIGHTLY_RESULTS:
            _NIGHTLY_RESULTS = _NIGHTLY_RESULTS[-_MAX_NIGHTLY_RESULTS:]
        snapshot = list(_NIGHTLY_RESULTS)
    try:
        _write_json_atomic(_NIGHTLY_RESULTS_PATH, snapshot)
    except Exception:
        log.debug("Failed to persist nightly results", exc_info=True)


def get_nightly_results() -> list[dict[str, Any]]:
    """Return a copy of recorded nightly pipeline results."""
    with _LOCK:
        _load_nightly_results_locked()
        return list(_NIGHTLY_RESULTS)


def get_last_nightly_result() -> dict[str, Any] | None:
    """Return the most recent nightly pipeline result, or None."""
    with _LOCK:
        _load_nightly_results_locked()
        return _NIGHTLY_RESULTS[-1] if _NIGHTLY_RESULTS else None
