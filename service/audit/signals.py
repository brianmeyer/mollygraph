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

import logging
import threading
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


def _metrics_listener(signal_type: str, _details: dict[str, Any]) -> None:
    """Increment the in-memory counter for *signal_type*."""
    with _counts_lock:
        _signal_counts[signal_type] = _signal_counts.get(signal_type, 0) + 1


def get_signal_counts() -> dict[str, int]:
    """Return a copy of signal counts accumulated since the last process restart."""
    with _counts_lock:
        return dict(_signal_counts)


__all__ = [
    "AuditSignalBus",
    "SIGNAL_TYPES",
    "get_signal_bus",
    "get_signal_counts",
]
