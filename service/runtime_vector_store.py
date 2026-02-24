"""Shared runtime accessor for the active VectorStore instance."""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memory.vector_store import VectorStore

_VECTOR_STORE_INSTANCE: "VectorStore | None" = None
_VECTOR_STORE_LOCK = threading.Lock()


def set_vector_store_instance(vs: "VectorStore | None") -> None:
    """Register or clear the process-wide VectorStore instance."""
    global _VECTOR_STORE_INSTANCE
    with _VECTOR_STORE_LOCK:
        _VECTOR_STORE_INSTANCE = vs


def get_vector_store_instance() -> "VectorStore | None":
    """Return the current VectorStore instance, or None if not yet initialized."""
    return _VECTOR_STORE_INSTANCE
