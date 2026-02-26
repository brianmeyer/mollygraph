from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

_HERE = Path(__file__).parent
_SERVICE_ROOT = _HERE.parent
if str(_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVICE_ROOT))

from api import decisions


class _DummyGraph:
    def __init__(self):
        self._rows: dict[str, dict[str, Any]] = {}
        self._counter = 0

    def create_decision(
        self,
        *,
        decision: str,
        reasoning: str,
        alternatives: list[str] | None = None,
        inputs: list[str] | None = None,
        outcome: str,
        decided_by: str,
        related_entities: list[str] | None = None,
        preceded_by_decision_id: str | None = None,
        source_episode_id: str | None = None,
        confidence: float | None = None,
        timestamp: datetime | None = None,
        decision_id: str | None = None,
    ) -> dict[str, Any]:
        if decision_id is None:
            self._counter += 1
            decision_id = f"dec-{self._counter}"

        row = {
            "id": decision_id,
            "decision": decision,
            "reasoning": reasoning,
            "alternatives": list(alternatives or []),
            "inputs": list(inputs or []),
            "outcome": outcome,
            "decided_by": decided_by,
            "timestamp": (timestamp or datetime.now(UTC)).isoformat(),
            "source_episode_id": source_episode_id,
            "confidence": confidence,
            "related_entities": list(related_entities or []),
            "preceded_by_decision_id": preceded_by_decision_id,
        }
        self._rows[decision_id] = row
        return row

    def list_decisions(
        self,
        *,
        q: str | None = None,
        decided_by: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        rows = list(self._rows.values())
        if q:
            needle = q.lower()
            rows = [
                r
                for r in rows
                if needle in r["decision"].lower()
                or needle in r["reasoning"].lower()
                or needle in r["outcome"].lower()
            ]
        if decided_by:
            who = decided_by.lower()
            rows = [r for r in rows if r["decided_by"].lower() == who]
        rows.sort(key=lambda r: r["timestamp"], reverse=True)
        return rows[:limit]

    def get_decision(self, decision_id: str) -> dict[str, Any] | None:
        return self._rows.get(decision_id)


def _client(monkeypatch) -> tuple[TestClient, _DummyGraph]:
    app = FastAPI()
    app.include_router(decisions.router)
    graph = _DummyGraph()
    monkeypatch.setattr(decisions, "require_runtime_ready", lambda: None)
    monkeypatch.setattr(decisions, "get_graph_instance", lambda: graph)
    return TestClient(app), graph


def test_decisions_create_list_get(monkeypatch):
    client, _graph = _client(monkeypatch)

    create_a = client.post(
        "/decisions",
        headers={"Authorization": "Bearer dev-key-change-in-production"},
        json={
            "decision": "Switch embedding model",
            "reasoning": "Latency dropped by 20%",
            "alternatives": ["Keep current model"],
            "inputs": ["benchmark-2026-02-25"],
            "outcome": "approved",
            "decided_by": "Brian",
            "related_entities": ["MollyGraph", "Neo4j"],
            "timestamp": "2026-02-25T10:00:00+00:00",
            "confidence": 0.95,
        },
    )
    assert create_a.status_code == 200
    a_body = create_a.json()
    assert a_body["id"] == "dec-1"
    assert a_body["decided_by"] == "Brian"

    create_b = client.post(
        "/decisions",
        headers={"Authorization": "Bearer dev-key-change-in-production"},
        json={
            "decision": "Reindex vectors",
            "reasoning": "Recall regression",
            "alternatives": [],
            "inputs": ["nightly metrics"],
            "outcome": "queued",
            "decided_by": "Alex",
            "related_entities": ["VectorStore"],
            "timestamp": "2026-02-26T10:00:00+00:00",
        },
    )
    assert create_b.status_code == 200
    b_body = create_b.json()
    assert b_body["id"] == "dec-2"

    listed = client.get(
        "/decisions",
        headers={"Authorization": "Bearer dev-key-change-in-production"},
        params={"q": "latency", "limit": 20},
    )
    assert listed.status_code == 200
    listed_body = listed.json()
    assert listed_body["count"] == 1
    assert listed_body["items"][0]["id"] == "dec-1"

    listed_by_user = client.get(
        "/decisions",
        headers={"Authorization": "Bearer dev-key-change-in-production"},
        params={"decided_by": "alex", "limit": 20},
    )
    assert listed_by_user.status_code == 200
    by_user_body = listed_by_user.json()
    assert by_user_body["count"] == 1
    assert by_user_body["items"][0]["id"] == "dec-2"

    fetched = client.get(
        "/decisions/dec-1",
        headers={"Authorization": "Bearer dev-key-change-in-production"},
    )
    assert fetched.status_code == 200
    assert fetched.json()["decision"] == "Switch embedding model"

    missing = client.get(
        "/decisions/missing",
        headers={"Authorization": "Bearer dev-key-change-in-production"},
    )
    assert missing.status_code == 404
