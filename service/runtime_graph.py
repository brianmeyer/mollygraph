"""Shared runtime accessor for the active graph backend instance."""
from __future__ import annotations

import threading

import config
from memory.graph import BiTemporalGraph, GraphBackend, LadybugGraph

_GRAPH_INSTANCE: GraphBackend | None = None
_GRAPH_LOCK = threading.Lock()

_GRAPH_CAPABILITY_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "core_memory": (
        "upsert_entity",
        "upsert_relationship",
        "create_episode",
        "get_current_facts",
        "get_entity_context",
    ),
    "decisions": ("create_decision", "list_decisions", "get_decision"),
    "audit": (
        "get_relationships_for_audit",
        "get_audit_coverage_metrics",
        "set_relationship_audit_status",
        "reclassify_relationship",
        "delete_specific_relationship",
    ),
    "maintenance": (
        "run_strength_decay_sync",
        "delete_orphan_entities_sync",
        "delete_self_referencing_rels",
    ),
}


def create_graph_instance() -> GraphBackend:
    backend = (config.GRAPH_BACKEND or "neo4j").strip().lower()
    if backend == "ladybug":
        return LadybugGraph(config.LADYBUG_GRAPH_DB_PATH)
    return BiTemporalGraph(
        config.NEO4J_URI,
        config.NEO4J_USER,
        config.NEO4J_PASSWORD,
    )


def get_graph_backend_name(graph: GraphBackend | object | None = None) -> str:
    if isinstance(graph, LadybugGraph):
        return "ladybug"
    if isinstance(graph, BiTemporalGraph):
        return "neo4j"
    return (config.GRAPH_BACKEND or "neo4j").strip().lower()


def graph_supports_capability(capability: str, graph: GraphBackend | object | None = None) -> bool:
    required = _GRAPH_CAPABILITY_REQUIREMENTS.get(capability)
    if required and graph is not None:
        return all(callable(getattr(graph, name, None)) for name in required)

    backend = get_graph_backend_name(graph)
    if capability == "training":
        return backend == "neo4j" or bool(getattr(graph, "driver", None))
    if capability in {"audit", "maintenance"}:
        return backend == "neo4j"
    if capability == "decisions":
        return backend == "neo4j" or (
            graph is not None
            and all(
                callable(getattr(graph, name, None))
                for name in _GRAPH_CAPABILITY_REQUIREMENTS["decisions"]
            )
        )
    return True


def get_graph_capabilities(graph: GraphBackend | object | None = None) -> list[str]:
    capabilities = ["core_memory"]
    for capability in ("decisions", "audit", "maintenance", "training"):
        if graph_supports_capability(capability, graph):
            capabilities.append(capability)
    return capabilities


def set_graph_instance(graph: GraphBackend | None) -> None:
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


def get_graph_instance() -> GraphBackend | None:
    """Return the current graph instance when initialized."""
    return _GRAPH_INSTANCE


def require_graph_instance() -> GraphBackend:
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

        new_instance = create_graph_instance()
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
