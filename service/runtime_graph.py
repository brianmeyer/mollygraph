"""Shared runtime accessor for the active BiTemporalGraph instance."""
from __future__ import annotations

import threading

import config
from memory.bitemporal_graph import BiTemporalGraph

_GRAPH_INSTANCE: BiTemporalGraph | None = None
_GRAPH_LOCK = threading.Lock()


def set_graph_instance(graph: BiTemporalGraph | None) -> None:
    """Register or clear the process-wide graph instance."""
    global _GRAPH_INSTANCE
    with _GRAPH_LOCK:
        previous = _GRAPH_INSTANCE
        _GRAPH_INSTANCE = graph

    if previous is not None and previous is not graph:
        try:
            previous.close()
        except Exception:
            # Best-effort close; caller already has a usable replacement.
            pass


def get_graph_instance() -> BiTemporalGraph | None:
    """Return the current graph instance when initialized."""
    return _GRAPH_INSTANCE


def require_graph_instance() -> BiTemporalGraph:
    """Return the active graph instance, lazily initializing if needed."""
    global _GRAPH_INSTANCE
    if _GRAPH_INSTANCE is not None:
        return _GRAPH_INSTANCE

    with _GRAPH_LOCK:
        if _GRAPH_INSTANCE is None:
            _GRAPH_INSTANCE = BiTemporalGraph(
                config.NEO4J_URI,
                config.NEO4J_USER,
                config.NEO4J_PASSWORD,
            )
        return _GRAPH_INSTANCE
