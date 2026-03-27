from __future__ import annotations

from pathlib import Path
import sys

_HERE = Path(__file__).parent
_SERVICE_ROOT = _HERE.parent
if str(_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVICE_ROOT))

import config as service_config
from query.graph_reranker import graph_rerank


class _FakeGraph:
    def __init__(self) -> None:
        self.neighborhood_calls: list[str] = []
        self.path_calls: list[tuple[str, str, int]] = []

    def get_neighborhood_stats(self, entity_name: str):
        self.neighborhood_calls.append(entity_name)
        if entity_name == "Bob":
            return 3, 0.75
        if entity_name == "Carol":
            return 1, 0.2
        return 0, 0.0

    def get_path_distance(self, from_name: str, to_name: str, max_hops: int = 2):
        self.path_calls.append((from_name, to_name, max_hops))
        if from_name == "Alice" and to_name == "Bob":
            return 1
        return None


def test_graph_rerank_uses_graph_object_contract():
    graph = _FakeGraph()
    results = [
        {
            "entity": "Bob",
            "facts": [{"relation_type": "WORKS_AT", "target": "OpenAI", "source": "Bob"}],
            "score": 0.45,
            "retrieval_source": "vector",
        },
        {
            "entity": "Bob",
            "facts": [{"relation_type": "WORKS_AT", "target": "OpenAI", "source": "Bob"}],
            "score": 0.6,
            "retrieval_source": "graph",
        },
        {
            "entity": "Carol",
            "facts": [{"relation_type": "RELATED_TO", "target": "Alice", "source": "Carol"}],
            "score": 0.58,
            "retrieval_source": "vector",
        },
    ]

    reranked = graph_rerank(results, "Tell me about Alice's network", ["Alice"], graph)

    assert len(reranked) == 2
    assert reranked[0]["entity"] == "Bob"
    assert reranked[0]["retrieval_source"] == "combined"
    assert reranked[0]["graph_score"] > reranked[1]["graph_score"]
    assert reranked[0]["graph_neighbor_count"] == 3
    assert reranked[0]["graph_name_match"] is False
    assert graph.neighborhood_calls  # proves graph contract was exercised
    assert graph.path_calls  # proves graph context lookups were exercised


def test_graph_rerank_keeps_direct_name_match_on_top(monkeypatch):
    monkeypatch.setattr(service_config, "MOLLYGRAPH_RERANK_NEIGHBOR_WEIGHT", 0.1, raising=False)
    monkeypatch.setattr(service_config, "MOLLYGRAPH_RERANK_STRENGTH_WEIGHT", 0.2, raising=False)
    monkeypatch.setattr(service_config, "MOLLYGRAPH_RERANK_PATH_BONUS", 0.15, raising=False)

    graph = _FakeGraph()
    results = [
        {
            "entity": "Carol",
            "facts": [{"relation_type": "RELATED_TO", "target": "Alice", "source": "Carol"}],
            "score": 0.9,
            "retrieval_source": "vector",
        },
        {
            "entity": "Alice",
            "facts": [{"relation_type": "WORKS_AT", "target": "Databricks", "source": "Alice"}],
            "score": 0.2,
            "retrieval_source": "graph",
        },
    ]

    reranked = graph_rerank(results, "Alice career background", ["Alice"], graph)

    assert reranked[0]["entity"] == "Alice"
    assert reranked[0]["graph_name_match"] is True
