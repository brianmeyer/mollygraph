from __future__ import annotations

import importlib
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

    async def fake_maintenance():
        return None

    monkeypatch.setattr(main, "run_llm_audit", fake_audit)
    monkeypatch.setattr(main, "run_gliner_finetune_pipeline", fake_train)
    monkeypatch.setattr(main, "run_maintenance_cycle", fake_maintenance)
    monkeypatch.setattr(
        main,
        "get_gliner_stats",
        lambda: {"examples_accumulated": 10, "last_cycle_status": "accumulated"},
    )
    monkeypatch.setattr(main, "build_suggestion_digest", lambda: "1 suggestion")

    from fastapi.testclient import TestClient

    with TestClient(main.app) as test_client:
        yield test_client
