from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import Any

_HERE = Path(__file__).parent
_SERVICE_ROOT = _HERE.parent
if str(_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVICE_ROOT))

from memory.graph.decisions import DecisionMixin


class _FakeResult:
    def __init__(self, rows: list[dict[str, Any]]):
        self._rows = rows

    def single(self) -> dict[str, Any] | None:
        return self._rows[0] if self._rows else None

    def __iter__(self):
        return iter(self._rows)


class _FakeSession:
    def __init__(self, driver: "_FakeDriver"):
        self._driver = driver

    def __enter__(self) -> "_FakeSession":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        return None

    def run(self, query: str, **params: Any) -> _FakeResult:
        q = " ".join(query.split())
        self._driver.queries.append(q)

        if "MATCH (d:Decision {id: $id}) RETURN d.id AS id" in q:
            decision_id = str(params["id"])
            if decision_id in self._driver.state["decisions"]:
                return _FakeResult([{"id": decision_id}])
            return _FakeResult([])

        if "MATCH (ep:Episode {id: $id}) RETURN ep.id AS id" in q:
            episode_id = str(params["id"])
            if episode_id in self._driver.state["episodes"]:
                return _FakeResult([{"id": episode_id}])
            return _FakeResult([])

        if "MERGE (d:Decision {id: $id})" in q:
            decision_id = str(params["id"])
            row = self._driver.state["decisions"].setdefault(decision_id, {})
            row.update(
                {
                    "id": decision_id,
                    "decision": str(params["decision"]),
                    "reasoning": str(params["reasoning"]),
                    "alternatives": list(params["alternatives"]),
                    "inputs": list(params["inputs"]),
                    "outcome": str(params["outcome"]),
                    "decided_by": str(params["decided_by"]),
                    "timestamp": datetime.fromisoformat(str(params["timestamp"])),
                    "source_episode_id": params.get("source_episode_id"),
                    "confidence": params.get("confidence"),
                    "related_entities": list(params["related_entities"]),
                    "preceded_by_decision_id": params.get("preceded_by_decision_id"),
                }
            )
            return _FakeResult([{"id": decision_id}])

        if "MATCH (d:Decision)" in q and "RETURN d.id AS id," in q:
            rows = list(self._driver.state["decisions"].values())
            decision_id = params.get("decision_id")
            q_filter = params.get("q")
            decided_by = params.get("decided_by")
            limit = int(params.get("limit", 20))

            if decision_id is not None:
                rows = [r for r in rows if r["id"] == decision_id]

            if q_filter:
                needle = str(q_filter).lower()
                rows = [
                    r
                    for r in rows
                    if needle in r["decision"].lower()
                    or needle in r["reasoning"].lower()
                    or needle in r["outcome"].lower()
                ]

            if decided_by:
                needle = str(decided_by).lower()
                rows = [r for r in rows if r["decided_by"].lower() == needle]

            rows.sort(key=lambda r: r["timestamp"], reverse=True)
            return _FakeResult(
                [
                    {
                        "id": r["id"],
                        "decision": r["decision"],
                        "reasoning": r["reasoning"],
                        "alternatives": list(r["alternatives"]),
                        "inputs": list(r["inputs"]),
                        "outcome": r["outcome"],
                        "decided_by": r["decided_by"],
                        "timestamp": r["timestamp"],
                        "source_episode_id": r.get("source_episode_id"),
                        "confidence": r.get("confidence"),
                        "related_entities": list(r.get("related_entities", [])),
                        "preceded_by_decision_id": r.get("preceded_by_decision_id"),
                    }
                    for r in rows[:limit]
                ]
            )

        raise AssertionError(f"Unexpected query in fake session: {q}")


class _FakeDriver:
    def __init__(self):
        self.state: dict[str, Any] = {
            "decisions": {},
            "episodes": {"ep-1"},
        }
        self.queries: list[str] = []

    def session(self) -> _FakeSession:
        return _FakeSession(self)


class _GraphUnderTest(DecisionMixin):
    def __init__(self):
        self.driver = _FakeDriver()


def test_create_decision_merges_by_id_without_duplicates():
    graph = _GraphUnderTest()
    first = graph.create_decision(
        decision_id="dec-1",
        decision="Switch model",
        reasoning="Latency improved",
        alternatives=["keep old model"],
        inputs=["benchmark-a"],
        outcome="started rollout",
        decided_by="Brian",
        related_entities=["MollyGraph"],
        source_episode_id="ep-1",
        confidence=0.8,
        timestamp=datetime(2026, 2, 20, 10, 0, tzinfo=UTC),
    )
    second = graph.create_decision(
        decision_id="dec-1",
        decision="Switch model",
        reasoning="Latency improved",
        alternatives=["keep old model"],
        inputs=["benchmark-a"],
        outcome="rollout complete",
        decided_by="Brian",
        related_entities=["MollyGraph"],
        source_episode_id="ep-1",
        confidence=0.9,
        timestamp=datetime(2026, 2, 21, 10, 0, tzinfo=UTC),
    )

    assert first["id"] == "dec-1"
    assert second["id"] == "dec-1"
    assert second["outcome"] == "rollout complete"
    assert len(graph.driver.state["decisions"]) == 1
    assert any("MERGE (d:Decision {id: $id})" in q for q in graph.driver.queries)


def test_list_and_get_decisions_filters():
    graph = _GraphUnderTest()
    graph.create_decision(
        decision_id="dec-1",
        decision="Reindex vectors",
        reasoning="Quality dropped",
        alternatives=["do nothing"],
        inputs=["retrieval metrics"],
        outcome="queued",
        decided_by="Alex",
        related_entities=["VectorStore"],
        timestamp=datetime(2026, 2, 20, 10, 0, tzinfo=UTC),
    )
    graph.create_decision(
        decision_id="dec-2",
        decision="Switch embedding model",
        reasoning="Latency target missed",
        alternatives=["keep model"],
        inputs=["latency report"],
        outcome="completed",
        decided_by="Brian",
        related_entities=["MollyGraph", "Neo4j"],
        timestamp=datetime(2026, 2, 21, 10, 0, tzinfo=UTC),
    )

    listed = graph.list_decisions(limit=20)
    assert [item["id"] for item in listed] == ["dec-2", "dec-1"]

    q_filtered = graph.list_decisions(q="latency", limit=20)
    assert [item["id"] for item in q_filtered] == ["dec-2"]

    decided_by_filtered = graph.list_decisions(decided_by="brian", limit=20)
    assert [item["id"] for item in decided_by_filtered] == ["dec-2"]

    fetched = graph.get_decision("dec-1")
    assert fetched is not None
    assert fetched["decision"] == "Reindex vectors"
    assert graph.get_decision("missing") is None
