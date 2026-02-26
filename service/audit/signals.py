"""Audit signal event bus for MollyGraph.

Provides a singleton AuditSignalBus that publishes events when the LLM audit
applies actions (reclassify, quarantine, merge, relationship_removed, etc.).
Listeners subscribe to track metrics, invalidate caches, or notify other
subsystems.  All listeners are called synchronously in publish(); errors are
silently caught so one bad listener never blocks the audit pipeline.

Signal types
------------
``relationship_reclassified``   – rel type corrected to a valid type
``relationship_quarantined``    – rel flagged as suspicious / uncertain
``relationship_removed``        – rel deleted as wrong/spam
``relationship_verified``       – rel confirmed as correct
``entity_reclassified``         – entity type changed by audit
``entity_merged``               – two entities merged into one
``entity_quarantined``          – entity flagged for review
"""
from __future__ import annotations

import atexit
import json
import logging
import os
from pathlib import Path
import threading
import tempfile
from typing import Any, Callable

log = logging.getLogger(__name__)

# Canonical signal type names
SIGNAL_TYPES: frozenset[str] = frozenset(
    {
        "relationship_reclassified",
        "relationship_quarantined",
        "relationship_removed",
        "relationship_verified",
        "entity_reclassified",
        "entity_merged",
        "entity_quarantined",
    }
)


class AuditSignalBus:
    """Simple synchronous publish/subscribe event bus for audit signals."""

    def __init__(self) -> None:
        self._listeners: list[Callable[[str, dict[str, Any]], None]] = []
        self._lock = threading.Lock()

    def subscribe(self, callback: Callable[[str, dict[str, Any]], None]) -> None:
        """Register *callback* to receive all future signals.

        ``callback(signal_type: str, details: dict)`` is called synchronously
        inside :meth:`publish`.  Exceptions are silently swallowed.
        """
        with self._lock:
            self._listeners.append(callback)

    def publish(self, signal_type: str, details: dict[str, Any]) -> None:
        """Dispatch *signal_type* with *details* to all registered listeners.

        Never raises — listener exceptions are caught and logged at DEBUG level
        so the audit pipeline is never interrupted.
        """
        with self._lock:
            listeners = list(self._listeners)

        for listener in listeners:
            try:
                listener(signal_type, details)
            except Exception:
                log.debug("AuditSignalBus: listener error", exc_info=True)


# ── Singleton ─────────────────────────────────────────────────────────────────

_bus: AuditSignalBus | None = None
_bus_lock = threading.Lock()


def get_signal_bus() -> AuditSignalBus:
    """Return the global :class:`AuditSignalBus` singleton.

    Created (and the built-in metrics listener attached) on first call.
    Thread-safe.
    """
    global _bus
    if _bus is None:
        with _bus_lock:
            if _bus is None:
                _bus = AuditSignalBus()
                _bus.subscribe(_metrics_listener)
    return _bus


# ── Built-in metrics listener (in-process counters) ───────────────────────────

_signal_counts: dict[str, int] = {}
_counts_lock = threading.Lock()
_counts_loaded = False
_persist_timer: threading.Timer | None = None
_PERSIST_DEBOUNCE_SECONDS = 0.5
_AUDIT_SIGNAL_COUNTS_PATH = (
    Path.home() / ".graph-memory" / "metrics" / "audit_signal_counts.json"
)


def _coerce_signal_counts(payload: Any) -> dict[str, int]:
    """Parse persisted signal counts with backward-compatible shape handling."""
    raw_counts: Any = payload
    if isinstance(payload, dict) and isinstance(payload.get("signal_counts"), dict):
        raw_counts = payload.get("signal_counts")
    if not isinstance(raw_counts, dict):
        return {}
    parsed: dict[str, int] = {}
    for key, value in raw_counts.items():
        try:
            count = int(value)
        except Exception:
            continue
        if count < 0:
            continue
        parsed[str(key)] = count
    return parsed


def _load_signal_counts_locked() -> None:
    global _counts_loaded, _signal_counts
    if _counts_loaded:
        return
    _counts_loaded = True
    try:
        if not _AUDIT_SIGNAL_COUNTS_PATH.exists():
            return
        payload = json.loads(_AUDIT_SIGNAL_COUNTS_PATH.read_text(encoding="utf-8"))
        _signal_counts = _coerce_signal_counts(payload)
    except Exception:
        log.debug("Failed to load audit signal counts", exc_info=True)


def _write_signal_counts_atomic(counts: dict[str, int]) -> None:
    _AUDIT_SIGNAL_COUNTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        dir=_AUDIT_SIGNAL_COUNTS_PATH.parent, suffix=".tmp"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(counts, indent=2, ensure_ascii=True) + "\n")
        os.replace(tmp_name, str(_AUDIT_SIGNAL_COUNTS_PATH))
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _flush_signal_counts_to_disk() -> None:
    global _persist_timer
    with _counts_lock:
        timer = _persist_timer
        if timer is not None and timer is not threading.current_thread():
            timer.cancel()
        _persist_timer = None
        if not _counts_loaded:
            return
        snapshot = dict(_signal_counts)
    try:
        _write_signal_counts_atomic(snapshot)
    except Exception:
        log.debug("Failed to persist audit signal counts", exc_info=True)


def _schedule_signal_counts_persist_locked() -> None:
    global _persist_timer
    if _persist_timer is not None:
        _persist_timer.cancel()
    timer = threading.Timer(_PERSIST_DEBOUNCE_SECONDS, _flush_signal_counts_to_disk)
    timer.daemon = True
    _persist_timer = timer
    timer.start()


def _metrics_listener(signal_type: str, _details: dict[str, Any]) -> None:
    """Increment the in-memory counter for *signal_type*."""
    with _counts_lock:
        _load_signal_counts_locked()
        _signal_counts[signal_type] = _signal_counts.get(signal_type, 0) + 1
        _schedule_signal_counts_persist_locked()


def get_signal_counts() -> dict[str, int]:
    """Return a copy of signal counts accumulated since the last process restart."""
    with _counts_lock:
        _load_signal_counts_locked()
        return dict(_signal_counts)


atexit.register(_flush_signal_counts_to_disk)


__all__ = [
    "AuditSignalBus",
    "SIGNAL_TYPES",
    "get_signal_bus",
    "get_signal_counts",
]
