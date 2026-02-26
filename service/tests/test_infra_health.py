from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

from fastapi import FastAPI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient

from api import admin
from maintenance.infra_health import (
    InfraHealthDecision,
    InfraHealthMetrics,
    evaluate_infra_health,
)


def _base_metrics() -> InfraHealthMetrics:
    return InfraHealthMetrics(
        graph_entity_count=1000,
        vector_entity_count=1000,
        index_completeness=1.0,
        queue_pending=0,
        queue_processing=0,
        queue_stuck=0,
        queue_dead=0,
        similarity_search_count=100,
        similarity_search_error_count=0,
        similarity_search_error_rate=0.0,
        similarity_search_avg_ms=90.0,
        similarity_search_p95_ms=120.0,
    )


def test_policy_healthy():
    decision, reasons = evaluate_infra_health(_base_metrics())
    assert decision == InfraHealthDecision.HEALTHY
    assert reasons


def test_policy_optimize_on_index_completeness():
    decision, _ = evaluate_infra_health(replace(_base_metrics(), index_completeness=0.99))
    assert decision == InfraHealthDecision.OPTIMIZE


def test_policy_refresh_on_queue_stuck():
    decision, _ = evaluate_infra_health(replace(_base_metrics(), queue_stuck=3))
    assert decision == InfraHealthDecision.REFRESH_EMBEDDINGS


def test_policy_reindex_on_drift():
    decision, _ = evaluate_infra_health(replace(_base_metrics(), vector_entity_count=700))
    assert decision == InfraHealthDecision.REINDEX_EMBEDDINGS


def test_policy_rebuild_only_when_allowed():
    m = replace(_base_metrics(), vector_entity_count=300, similarity_search_error_rate=0.45)
    decision_no, _ = evaluate_infra_health(m, allow_rebuild=False)
    decision_yes, _ = evaluate_infra_health(m, allow_rebuild=True)
    assert decision_no == InfraHealthDecision.REINDEX_EMBEDDINGS
    assert decision_yes == InfraHealthDecision.REBUILD_VECTORS


class _DummyGraph:
    def entity_count(self):
        return 100

    def list_entities_for_embedding(self, _limit: int):
        return [{"entity_id": "e1", "name": "A", "entity_type": "Concept", "content": "A"}]


class _DummyQueue:
    def get_pending_count(self):
        return 0

    def get_processing_count(self):
        return 0

    def get_stuck_count(self):
        return 0

    def get_dead_count(self):
        return 0


class _DummyVector:
    def get_stats(self):
        return {
            "entities": 100,
            "similarity_search_count": 10,
            "similarity_search_error_count": 0,
            "similarity_search_avg_ms": 20,
            "similarity_search_p95_ms": 30,
        }

    def get_segment_health(self):
        return {"index_completeness": {"embedding": 1.0}}


def _client(monkeypatch) -> TestClient:
    app = FastAPI()
    app.include_router(admin.router)
    monkeypatch.setattr(admin, "require_runtime_ready", lambda: None)
    monkeypatch.setattr(admin, "get_graph_instance", lambda: _DummyGraph())
    monkeypatch.setattr(admin, "get_queue_instance", lambda: _DummyQueue())
    monkeypatch.setattr(admin, "get_vector_store_instance", lambda: _DummyVector())
    return TestClient(app)


def test_infra_health_endpoint_deterministic_only(monkeypatch):
    client = _client(monkeypatch)
    res = client.post(
        "/maintenance/infra-health/evaluate",
        headers={"Authorization": "Bearer dev-key-change-in-production"},
        json={"dry_run": True, "enable_llm_advisory": False},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["deterministic_decision"] == "healthy"
    assert body["llm_advisory"] is None


def test_infra_health_endpoint_llm_failure_fallback(monkeypatch):
    client = _client(monkeypatch)

    async def _boom(*_args, **_kwargs):
        raise RuntimeError("llm unavailable")

    monkeypatch.setattr("maintenance.infra_health.run_llm_advisory", _boom)

    res = client.post(
        "/maintenance/infra-health/evaluate",
        headers={"Authorization": "Bearer dev-key-change-in-production"},
        json={"enable_llm_advisory": True},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["deterministic_decision"] == body["final_action"]
    assert body["llm_advisory"]["status"] == "error"
