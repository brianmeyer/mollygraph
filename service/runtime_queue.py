"""Shared runtime accessor for the active ExtractionQueue instance."""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from extraction.queue import ExtractionQueue

_QUEUE_INSTANCE: "ExtractionQueue | None" = None
_QUEUE_LOCK = threading.Lock()


def set_queue_instance(queue: "ExtractionQueue | None") -> None:
    """Register or clear the process-wide ExtractionQueue instance."""
    global _QUEUE_INSTANCE
    with _QUEUE_LOCK:
        _QUEUE_INSTANCE = queue


def get_queue_instance() -> "ExtractionQueue | None":
    """Return the current ExtractionQueue instance, or None if not initialised."""
    return _QUEUE_INSTANCE
