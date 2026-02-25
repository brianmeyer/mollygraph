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

    def get_stuck_count(self):
        return 0

    def get_dead_count(self):
        return 0


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

    def list_entities_for_embedding(self, limit: int = 5000):
        rows = [
            {
                "entity_id": "entity-brian",
                "name": "Brian",
                "entity_type": "Person",
                "content": "Brian works at Databricks.",
                "confidence": 0.9,
            },
            {
                "entity_id": "entity-databricks",
                "name": "Databricks",
                "entity_type": "Organization",
                "content": "Databricks is a data company.",
                "confidence": 0.88,
            },
        ]
        return rows[: max(1, int(limit))]


class FakeVectorStore:
    def get_stats(self):
        return {"backend": "zvec", "entities": 2}

    def similarity_search(self, _embedding, top_k: int = 10, entity_type=None):
        return [{"name": "Brian", "score": 0.8}]

    def is_degraded(self):
        return False


@pytest.fixture()
def client(monkeypatch):
    pytest.importorskip("fastapi")
    monkeypatch.setenv("MOLLYGRAPH_TEST_MODE", "1")

    import config
    import main

    importlib.reload(config)
    importlib.reload(main)

    main.queue = FakeQueue()
    main.graph = FakeGraph()
    main.vector_store = FakeVectorStore()
    main.pipeline = object()

    # Patch runtime singletons across all modules that import them
    from api import deps as deps_module
    from api import ingest as ingest_module
    from api import query as query_module

    for mod in [deps_module, ingest_module, query_module]:
        if hasattr(mod, "require_runtime_ready"):
            monkeypatch.setattr(mod, "require_runtime_ready", lambda: None)
        if hasattr(mod, "get_queue_instance"):
            monkeypatch.setattr(mod, "get_queue_instance", lambda: main.queue)
        if hasattr(mod, "get_graph_instance"):
            monkeypatch.setattr(mod, "get_graph_instance", lambda: main.graph)
        if hasattr(mod, "get_pipeline_instance"):
            monkeypatch.setattr(mod, "get_pipeline_instance", lambda: main.pipeline)
        if hasattr(mod, "get_vector_store_instance"):
            monkeypatch.setattr(mod, "get_vector_store_instance", lambda: main.vector_store)

    # Patch admin module functions where routes actually call them
    from api import admin as admin_module
    from maintenance import auditor as auditor_module
    from memory import graph_suggestions as suggestions_module
    from evolution import gliner_training as training_module

    async def fake_audit(**kwargs):
        return {
            "status": "ok",
            "relationships_scanned": kwargs.get("limit", 500),
            "verified": 3,
            "auto_fixed": 1,
        }

    async def fake_train(force: bool = False):
        return {"status": "finetune_triggered", "force": force}

    def fake_maintenance():
        return None

    def fake_gliner_stats():
        return {"examples_accumulated": 10, "last_cycle_status": "accumulated"}

    def fake_suggestion_digest():
        return "1 suggestion"

    monkeypatch.setattr(admin_module, "run_llm_audit", fake_audit)
    monkeypatch.setattr(admin_module, "run_gliner_finetune_pipeline", fake_train)
    monkeypatch.setattr(auditor_module, "run_maintenance_cycle", fake_maintenance)
    monkeypatch.setattr(training_module, "get_gliner_stats", fake_gliner_stats)
    monkeypatch.setattr(suggestions_module, "build_suggestion_digest", fake_suggestion_digest)

    from fastapi.testclient import TestClient

    with TestClient(main.app) as test_client:
        yield test_client
