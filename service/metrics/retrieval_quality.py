"""Retrieval quality metric helpers for MollyGraph.

Provides functions for computing per-query diversity and graph-coverage
metrics that are logged alongside each retrieval event and aggregated in
the GET /metrics/retrieval/trend endpoint.

Metrics
-------
result_diversity
    Entity-type diversity of a result set.
    = # unique entity types across all fact target-types / result count
    Range [0, ∞); typically ≤ 1 for well-typed result sets.

graph_connected_pct
    Percentage of returned entities that have ≥1 graph relationship.
    = entities_with_facts / total_results
    Range [0, 1]; 1.0 means every result is graph-connected.
"""
from __future__ import annotations

from typing import Any


def compute_result_diversity(results: list[dict[str, Any]]) -> float:
    """Return entity-type diversity score for a result set.

    Diversity = # unique entity types in result target facts / result count.
    An empty result set returns 0.0.
    """
    if not results:
        return 0.0
    entity_types: set[str] = set()
    for r in results:
        for f in r.get("facts") or []:
            tt = str(f.get("target_type") or "").strip()
            if tt:
                entity_types.add(tt)
    return round(len(entity_types) / max(len(results), 1), 4)


def compute_graph_connected_pct(results: list[dict[str, Any]]) -> float:
    """Return fraction of results that have ≥1 graph relationship (fact).

    An empty result set returns 0.0.
    """
    if not results:
        return 0.0
    connected = sum(1 for r in results if r.get("facts"))
    return round(connected / len(results), 4)


def compute_retrieval_quality(results: list[dict[str, Any]]) -> dict[str, float]:
    """Return both diversity and graph_connected_pct in a single call."""
    return {
        "result_diversity": compute_result_diversity(results),
        "graph_connected_pct": compute_graph_connected_pct(results),
    }
