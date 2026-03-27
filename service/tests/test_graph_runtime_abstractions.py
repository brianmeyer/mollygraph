from __future__ import annotations

import asyncio
from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from api import admin, decisions, deps, ingest
from extraction.pipeline import ExtractionPipeline


class _NoDriverGraph:
    def __init__(self):
        self.deleted: list[str] = []
        self.deleted_relationships: list[tuple[str, str, str | None]] = []
        self.marked_incomplete: list[tuple[str, str]] = []
        self.finalized: list[tuple[str, list[str]]] = []
        self.first_seen: list[tuple[str, str]] = []

    def get_graph_summary(self):
        return {
            "entity_count": 3,
            "relationship_count": 4,
            "episode_count": 2,
            "top_connected": [],
            "recent": [],
        }

    def get_relationship_type_distribution(self):
        return {"WORKS_AT": 2}

    def incomplete_episode_count(self):
        return 7

    def get_entity_type_distribution(self):
        return {"Person": 2, "Organization": 1}

    def get_entity_id_by_name(self, name: str):
        if name.lower() == "alice":
            return "entity-alice"
        return None

    def get_entity_delete_summary(self, name: str):
        if name.lower() != "alice":
            return None
        return {"entity_id": "entity-alice", "relationship_count": 2}

    def delete_entity(self, name: str):
        self.deleted.append(name)
        return True

    def delete_relationships_between(self, source: str, target: str, rel_type: str | None = None):
        self.deleted_relationships.append((source, target, rel_type))
        return 3 if rel_type else 5

    def list_orphan_entity_names(self):
        return ["Alice", "Orphaned Org"]

    def list_entities_page(self, *, limit: int, offset: int, entity_type: str | None = None):
        rows = [
            {
                "id": "entity-alice",
                "name": "Alice",
                "entity_type": entity_type or "Person",
                "confidence": 0.9,
            }
        ]
        return rows, 1

    def mark_episode_incomplete(self, episode_id: str, reason: str = ""):
        self.marked_incomplete.append((episode_id, reason))
        return True

    def finalize_episode(self, episode_id: str, entity_names: list[str]):
        self.finalized.append((episode_id, list(entity_names)))
        return True

    def tag_entity_first_seen(self, entity_id: str, source: str):
        self.first_seen.append((entity_id, source))
        return True


class _NoDriverVector:
    def __init__(self):
        self.removed: list[str] = []

    def get_stats(self):
        return {"entities": 3}

    def remove_entity(self, entity_id: str):
        self.removed.append(entity_id)
        return True


class _NoDriverQueue:
    def get_pending_count(self):
        return 0

    def get_processing_count(self):
        return 0

    def get_stuck_count(self):
        return 0

    def get_dead_count(self):
        return 0


def test_delete_entity_and_vector_uses_graph_method(monkeypatch):
    graph = _NoDriverGraph()
    vector = _NoDriverVector()
    monkeypatch.setattr(deps, "get_graph_instance", lambda: graph)
    monkeypatch.setattr(deps, "get_vector_store_instance", lambda: vector)

    neo4j_deleted, vector_deleted = asyncio.run(deps._delete_entity_and_vector("Alice"))

    assert neo4j_deleted is True
    assert vector_deleted is True
    assert graph.deleted == ["Alice"]
    assert vector.removed == ["entity-alice"]


def test_stats_endpoint_uses_graph_abstractions(monkeypatch):
    app = FastAPI()
    app.include_router(admin.router)
    monkeypatch.setattr(admin, "get_graph_instance", lambda: _NoDriverGraph())
    monkeypatch.setattr(admin, "get_vector_store_instance", lambda: _NoDriverVector())
    monkeypatch.setattr(admin, "get_queue_instance", lambda: _NoDriverQueue())
    monkeypatch.setattr(admin, "get_gliner_stats", lambda: {})

    client = TestClient(app)
    res = client.get("/stats", headers={"Authorization": "Bearer dev-key-change-in-production"})

    assert res.status_code == 200
    body = res.json()
    assert body["queue"]["incomplete_episodes"] == 7
    assert body["relationship_type_distribution"]["WORKS_AT"] == 2


def test_schema_drift_endpoint_uses_graph_abstractions(monkeypatch, tmp_path):
    app = FastAPI()
    app.include_router(admin.router)
    monkeypatch.setattr(admin, "get_graph_instance", lambda: _NoDriverGraph())
    monkeypatch.setattr(config, "GRAPH_MEMORY_DIR", tmp_path)

    client = TestClient(app)
    res = client.get(
        "/metrics/schema-drift",
        headers={"Authorization": "Bearer dev-key-change-in-production"},
    )

    assert res.status_code == 200
    body = res.json()
    assert body["entity_types"] == 2
    assert body["relationship_types"] == 1
    assert body["rel_type_distribution"]["WORKS_AT"] == 2


def test_pipeline_episode_helpers_use_graph_methods():
    graph = _NoDriverGraph()
    pipeline = ExtractionPipeline.__new__(ExtractionPipeline)
    pipeline.graph = graph

    pipeline._mark_episode_incomplete("ep-1", reason="partial write")
    pipeline._finalize_episode("ep-1", ["Alice", "Acme"])

    assert graph.marked_incomplete == [("ep-1", "partial write")]
    assert graph.finalized == [("ep-1", ["Alice", "Acme"])]


def test_ingest_delete_entity_uses_graph_abstractions(monkeypatch):
    app = FastAPI()
    app.include_router(ingest.router)
    graph = _NoDriverGraph()
    vector = _NoDriverVector()

    monkeypatch.setattr(ingest, "require_runtime_ready", lambda: None)
    monkeypatch.setattr(ingest, "require_no_maintenance", lambda: None)
    monkeypatch.setattr(ingest, "get_graph_instance", lambda: graph)
    monkeypatch.setattr(ingest, "get_vector_store_instance", lambda: vector)

    client = TestClient(app)
    res = client.delete("/entity/Alice", headers={"Authorization": "Bearer dev-key-change-in-production"})

    assert res.status_code == 200
    body = res.json()
    assert body["relationships_removed"] == 2
    assert graph.deleted == ["Alice"]
    assert vector.removed == ["entity-alice"]


def test_ingest_delete_relationship_uses_graph_abstractions(monkeypatch):
    app = FastAPI()
    app.include_router(ingest.router)
    graph = _NoDriverGraph()

    monkeypatch.setattr(ingest, "require_runtime_ready", lambda: None)
    monkeypatch.setattr(ingest, "require_no_maintenance", lambda: None)
    monkeypatch.setattr(ingest, "get_graph_instance", lambda: graph)

    client = TestClient(app)
    res = client.request(
        "DELETE",
        "/relationship",
        json={"source": "Alice", "target": "Acme", "rel_type": "WORKS_AT"},
        headers={"Authorization": "Bearer dev-key-change-in-production"},
    )

    assert res.status_code == 200
    assert res.json()["deleted"] == 3
    assert graph.deleted_relationships == [("Alice", "Acme", "WORKS_AT")]


def test_prune_and_list_entities_use_graph_abstractions(monkeypatch):
    app = FastAPI()
    app.include_router(ingest.router)
    graph = _NoDriverGraph()
    vector = _NoDriverVector()

    monkeypatch.setattr(ingest, "require_runtime_ready", lambda: None)
    monkeypatch.setattr(ingest, "require_no_maintenance", lambda: None)
    monkeypatch.setattr(ingest, "get_graph_instance", lambda: graph)
    monkeypatch.setattr(ingest, "get_vector_store_instance", lambda: vector)
    monkeypatch.setattr(deps, "get_graph_instance", lambda: graph)
    monkeypatch.setattr(deps, "get_vector_store_instance", lambda: vector)

    client = TestClient(app)

    prune_res = client.post(
        "/entities/prune",
        json={"orphans": True},
        headers={"Authorization": "Bearer dev-key-change-in-production"},
    )
    assert prune_res.status_code == 200
    assert prune_res.json()["pruned"] == 2

    list_res = client.get(
        "/entities?limit=10&offset=0&type=Person",
        headers={"Authorization": "Bearer dev-key-change-in-production"},
    )
    assert list_res.status_code == 200
    body = list_res.json()
    assert body["total"] == 1
    assert body["entities"][0]["name"] == "Alice"
    assert body["type_filter"] == "Person"


def test_decisions_routes_return_501_without_backend_support(monkeypatch):
    app = FastAPI()
    app.include_router(decisions.router)
    graph = _NoDriverGraph()

    monkeypatch.setattr(decisions, "require_runtime_ready", lambda: None)
    monkeypatch.setattr(decisions, "get_graph_instance", lambda: graph)

    client = TestClient(app)
    res = client.post(
        "/decisions",
        headers={"Authorization": "Bearer dev-key-change-in-production"},
        json={
            "decision": "Use Ladybug",
            "reasoning": "Simplify local runtime",
            "outcome": "approved",
            "decided_by": "Brian",
        },
    )

    assert res.status_code == 501
    assert "not supported" in res.json()["detail"].lower()


def test_admin_routes_gate_neo4j_only_workflows(monkeypatch):
    app = FastAPI()
    app.include_router(admin.router)
    graph = _NoDriverGraph()

    monkeypatch.setattr(admin, "get_graph_instance", lambda: graph)

    client = TestClient(app)
    headers = {"Authorization": "Bearer dev-key-change-in-production"}

    audit_res = client.post("/audit", headers=headers, json={"limit": 10, "dry_run": True})
    train_res = client.post("/train/gliner", headers=headers, json={"force": False})
    runs_res = client.get("/training/runs", headers=headers)
    maintenance_res = client.post("/maintenance/run", headers=headers)

    assert audit_res.status_code == 501
    assert train_res.status_code == 501
    assert runs_res.status_code == 501
    assert maintenance_res.status_code == 501
