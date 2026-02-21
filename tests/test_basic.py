from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPO_ROOT / "service"
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))


class FakeQueue:
    def __init__(self):
        self.pending = 0
        self.processing = 0
        self.jobs: list[dict] = []

    def submit(self, job):
        self.jobs.append({"id": job.id, "content": job.content, "source": job.source})
        self.pending += 1
        return job.id

    def get_pending_count(self):
        return self.pending

    def get_processing_count(self):
        return self.processing


class FakeGraph:
    def get_current_facts(self, name: str):
        if name.lower() in {"brian", "databricks"}:
            return [
                {
                    "rel_type": "WORKS_AT",
                    "target_name": "Databricks",
                    "target_type": "Organization",
                    "strength": 0.9,
                    "confidence": 0.85,
                }
            ]
        return []

    def get_entity_context(self, name: str, hops: int = 2):
        return {
            "entity": name,
            "hops": hops,
            "direct_connections": [
                {
                    "rel_type": "WORKS_AT",
                    "target_name": "Databricks",
                    "target_type": "Organization",
                    "strength": 0.9,
                    "is_outgoing": True,
                }
            ],
            "two_hop_connections": [],
        }


class FakeVectorStore:
    def get_stats(self):
        return {"backend": "zvec", "entities": 2}

    def similarity_search(self, _embedding, top_k: int = 10, entity_type=None):
        return [{"name": "Brian", "score": 0.8}]


@pytest.fixture()
def client(monkeypatch):
    pytest.importorskip("fastapi")
    monkeypatch.setenv("MOLLYGRAPH_TEST_MODE", "1")

    import main

    importlib.reload(main)

    main.queue = FakeQueue()
    main.graph = FakeGraph()
    main.vector_store = FakeVectorStore()
    main.pipeline = object()

    async def fake_audit(**kwargs):
        return {
            "status": "ok",
            "relationships_scanned": kwargs.get("limit", 500),
            "verified": 3,
            "auto_fixed": 1,
        }

    async def fake_train(force: bool = False):
        return {"status": "finetune_triggered", "force": force}

    monkeypatch.setattr(main, "run_llm_audit", fake_audit)
    monkeypatch.setattr(main, "run_gliner_finetune_pipeline", fake_train)
    monkeypatch.setattr(main, "get_gliner_stats", lambda: {"examples_accumulated": 10, "last_cycle_status": "accumulated"})
    monkeypatch.setattr(main, "build_suggestion_digest", lambda: "1 suggestion")

    from fastapi.testclient import TestClient

    with TestClient(main.app) as test_client:
        yield test_client


AUTH_HEADERS = {"Authorization": "Bearer dev-key-change-in-production"}


def test_health_endpoint(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["test_mode"] is True


def test_ingest_entity_query_and_stats(client) -> None:
    ingest_resp = client.post(
        "/ingest",
        params={"content": "Brian works at Databricks.", "source": "manual", "priority": 1},
        headers=AUTH_HEADERS,
    )
    assert ingest_resp.status_code == 200
    ingest_payload = ingest_resp.json()
    assert ingest_payload["status"] == "queued"

    entity_resp = client.get("/entity/Brian", headers=AUTH_HEADERS)
    assert entity_resp.status_code == 200
    assert entity_resp.json()["entity"] == "Brian"

    query_resp = client.get("/query", params={"q": "What about Brian?"}, headers=AUTH_HEADERS)
    assert query_resp.status_code == 200
    assert query_resp.json()["result_count"] >= 1

    stats_resp = client.get("/stats", headers=AUTH_HEADERS)
    assert stats_resp.status_code == 200
    stats_payload = stats_resp.json()
    assert "queue" in stats_payload
    assert "vector_store" in stats_payload
    assert "gliner_training" in stats_payload


def test_audit_suggestions_and_training_endpoints(client) -> None:
    audit_resp = client.post(
        "/audit",
        json={"limit": 25, "dry_run": True, "schedule": "nightly"},
        headers=AUTH_HEADERS,
    )
    assert audit_resp.status_code == 200
    assert audit_resp.json()["status"] == "ok"

    suggestions_resp = client.get("/suggestions/digest", headers=AUTH_HEADERS)
    assert suggestions_resp.status_code == 200
    assert suggestions_resp.json()["has_suggestions"] is True

    train_status_resp = client.get("/train/status", headers=AUTH_HEADERS)
    assert train_status_resp.status_code == 200
    assert train_status_resp.json()["gliner"]["examples_accumulated"] == 10

    train_resp = client.post("/train/gliner", json={"force": True}, headers=AUTH_HEADERS)
    assert train_resp.status_code == 200
    assert train_resp.json()["status"] == "finetune_triggered"


def test_parse_verdicts_handles_json_fence() -> None:
    from audit.llm_audit import parse_verdicts

    raw = """```json
    [
      {"index": 1, "verdict": "verify"},
      {"index": 2, "verdict": "reclassify", "suggested_type": "works at"},
      {"index": 3, "verdict": "delete"}
    ]
    ```"""

    parsed = parse_verdicts(raw, batch_len=3)
    assert len(parsed) == 3
    assert parsed[1]["suggested_type"] == "WORKS_AT"
