"""Focused tests for query-path graph helpers.

These tests avoid Neo4j and validate the helper surface we now rely on from
`service/api/query.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# ── Path setup ─────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
_SERVICE_ROOT = _HERE.parent
if str(_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVICE_ROOT))

from api.query import _graph_rerank_results
from memory.graph.entities import EntityMixin
from memory.graph.queries import QueryMixin


class _FakeResult:
    def __init__(self, records: list[dict[str, Any]]):
        self._records = [dict(r) for r in records]

    def __iter__(self):
        return iter(self._records)

    def single(self):
        return self._records[0] if self._records else None


class _FakeSession:
    def __init__(self, responses: list[list[dict[str, Any]]]):
        self._responses = [_FakeResult(records) for records in responses]
        self.runs: list[dict[str, Any]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, query: str, **params):
        self.runs.append({"query": query, "params": params})
        if self._responses:
            return self._responses.pop(0)
        return _FakeResult([])


class _FakeDriver:
    def __init__(self, responses: list[list[dict[str, Any]]]):
        self.session_obj = _FakeSession(responses)

    def session(self):
        return self.session_obj


class _DummyGraph(EntityMixin, QueryMixin):
    def __init__(self, responses: list[list[dict[str, Any]]]):
        self.driver = _FakeDriver(responses)


class TestGraphHelpers:
    def test_find_entities_containing_uses_alias_search(self):
        graph = _DummyGraph([[{"name": "Alice"}, {"name": "Alicia"}]])

        names = graph.find_entities_containing("ali", limit=5)

        assert names == ["Alice", "Alicia"]
        query = graph.driver.session_obj.runs[0]["query"]
        assert "ANY(a IN coalesce(e.aliases, [])" in query

    def test_neighborhood_stats(self):
        graph = _DummyGraph([[{"neighbor_count": 3, "avg_strength": 0.75}]])

        neighbor_count, avg_strength = graph.get_neighborhood_stats("Alice")

        assert neighbor_count == 3
        assert avg_strength == 0.75

    def test_path_distance(self):
        graph = _DummyGraph([[{"dist": 2}]])

        assert graph.get_path_distance("Alice", "OpenAI") == 2

    def test_entity_id_lookup(self):
        graph = _DummyGraph([[{"id": "abc-123"}]])

        assert graph.get_entity_id_by_name("Alice") == "abc-123"


class TestQueryRerankHelper:
    def test_graph_rerank_uses_graph_methods_only(self):
        class _Graph:
            def get_neighborhood_stats(self, entity_name: str):
                return (4, 0.6) if entity_name == "Alice" else (1, 0.2)

            def get_path_distance(self, from_name: str, to_name: str):
                if from_name == "Alice" and to_name == "Alice":
                    return 0
                if from_name == "Alice" and to_name == "OpenAI":
                    return 1
                return None

        graph = _Graph()
        results = [
            {"entity": "OpenAI", "facts": [{"relation_type": "WORKS_AT", "target": "OpenAI"}], "score": 0.6},
            {"entity": "Alice", "facts": [{"relation_type": "WORKS_AT", "target": "Databricks"}], "score": 0.55},
        ]

        reranked = _graph_rerank_results(graph, results, "Tell me about Alice", ["Alice"])

        assert [item["entity"] for item in reranked][0] == "Alice"
        assert reranked[0]["graph_name_match"] is True
        assert reranked[0]["graph_score"] > reranked[1]["graph_score"]
