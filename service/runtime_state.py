"""Shared runtime state: service start time and nightly pipeline tracking."""
from __future__ import annotations

import threading
from datetime import datetime, UTC
from typing import Any

_LOCK = threading.Lock()

# Service start time (set in lifespan).
_SERVICE_STARTED_AT: datetime | None = None

# Nightly pipeline result history (last N runs).
_NIGHTLY_RESULTS: list[dict[str, Any]] = []
_MAX_NIGHTLY_RESULTS = 20


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
) -> None:
    """Record a nightly pipeline run result (success or failure)."""
    global _NIGHTLY_RESULTS
    entry: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "status": status,
        "audit_status": audit_status,
        "lora_status": lora_status,
        "error": error,
    }
    with _LOCK:
        _NIGHTLY_RESULTS.append(entry)
        if len(_NIGHTLY_RESULTS) > _MAX_NIGHTLY_RESULTS:
            _NIGHTLY_RESULTS = _NIGHTLY_RESULTS[-_MAX_NIGHTLY_RESULTS:]


def get_nightly_results() -> list[dict[str, Any]]:
    """Return a copy of recorded nightly pipeline results."""
    with _LOCK:
        return list(_NIGHTLY_RESULTS)


def get_last_nightly_result() -> dict[str, Any] | None:
    """Return the most recent nightly pipeline result, or None."""
    with _LOCK:
        return _NIGHTLY_RESULTS[-1] if _NIGHTLY_RESULTS else None
