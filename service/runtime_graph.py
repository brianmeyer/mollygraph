"""Shared runtime accessor for the active BiTemporalGraph instance."""
from __future__ import annotations

import threading

import config
from memory.graph import BiTemporalGraph

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
    """Return the active graph instance, lazily initializing if needed.

    Uses double-checked locking to avoid creating duplicate instances.
    Any previous instance is closed (connection pool released) before the new
    one becomes visible, matching the behaviour of set_graph_instance().
    """
    global _GRAPH_INSTANCE
    if _GRAPH_INSTANCE is not None:
        return _GRAPH_INSTANCE

    with _GRAPH_LOCK:
        if _GRAPH_INSTANCE is not None:
            # Another thread beat us here.
            return _GRAPH_INSTANCE

        new_instance = BiTemporalGraph(
            config.NEO4J_URI,
            config.NEO4J_USER,
            config.NEO4J_PASSWORD,
        )
        # Capture any previous instance to close *outside* the lock (mirrors
        # set_graph_instance() logic and avoids potential re-entrancy issues).
        previous = _GRAPH_INSTANCE
        _GRAPH_INSTANCE = new_instance

    # Close the old instance after releasing the lock so no lock is held
    # during a potentially blocking network call.
    if previous is not None and previous is not new_instance:
        try:
            previous.close()
        except Exception:
            pass  # Best-effort — caller already has a usable replacement.

    return new_instance
