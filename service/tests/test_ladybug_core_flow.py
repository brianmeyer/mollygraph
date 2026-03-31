from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
import sys
import threading

from fastapi import FastAPI
from fastapi.testclient import TestClient

_HERE = Path(__file__).parent
_SERVICE_ROOT = _HERE.parent
if str(_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVICE_ROOT))

import config
from api import admin, ingest, query
from extraction.pipeline import ExtractionPipeline
from extraction.queue import ExtractionQueue
from memory import extractor as memory_extractor
from memory.graph import LadybugGraph
from memory.models import ExtractionJob
from memory.vector_store import VectorStore
from runtime_graph import set_graph_instance
from runtime_pipeline import set_pipeline_instance
from runtime_queue import set_queue_instance
from runtime_vector_store import set_vector_store_instance


def _fake_extract(_text: str, _threshold: float):
    return {
        "entities": [
            {"text": "Alice", "label": "Person", "score": 0.99},
            {"text": "Acme", "label": "Organization", "score": 0.97},
        ],
        "relations": [
            {"head": "Alice", "tail": "Acme", "label": "WORKS_AT", "score": 0.94},
        ],
    }


def _fake_embedding(_cls, text: str, dim: int = 384, _depth: int = 0) -> list[float]:
    seed = float((sum(ord(ch) for ch in text) % 17) + 1)
    return [seed] + [0.0] * (dim - 1)


def test_ladybug_core_flow_via_runtime_api(monkeypatch, tmp_path):
    monkeypatch.setattr(memory_extractor, "extract", _fake_extract)
    monkeypatch.setattr(ExtractionPipeline, "_text_embedding", classmethod(_fake_embedding))
    monkeypatch.setattr(config, "SPACY_ENRICHMENT", False)
    monkeypatch.setattr(config, "GLIREL_ENABLED", False)
    monkeypatch.setattr(config, "DECISION_TRACES_INGEST_ENABLED", False)
    monkeypatch.setattr(config, "MOLLYGRAPH_RERANK_ENABLED", False)
    monkeypatch.setattr(config, "GRAPH_RERANK_ENABLED", False)
    monkeypatch.setattr(admin, "get_gliner_stats", lambda: {})

    graph = LadybugGraph(tmp_path / "graph.lbug")
    vector_store = VectorStore(backend="ladybug", db_path=tmp_path / "vectors.lbug")
    pipeline = ExtractionPipeline(graph=graph, vector_store=vector_store)
    queue = ExtractionQueue(db_path=tmp_path / "queue.db")

    set_graph_instance(graph)
    set_vector_store_instance(vector_store)
    set_pipeline_instance(pipeline)
    set_queue_instance(queue)

    app = FastAPI()
    app.include_router(ingest.router)
    app.include_router(query.router)
    app.include_router(admin.router)
    client = TestClient(app)
    headers = {"Authorization": "Bearer dev-key-change-in-production"}

    try:
        ingest_res = client.post(
            "/ingest",
            headers=headers,
            json={
                "content": "Alice works at Acme.",
                "source": "session",
                "speaker": "Alice",
            },
        )
        assert ingest_res.status_code == 200
        assert ingest_res.json()["status"] == "queued"

        job = queue.claim_next()
        assert job is not None
        processed = asyncio.run(pipeline.process_job(job))
        queue.complete(job.id, success=True, result={"episode_id": processed.episode_id})

        entity_res = client.get("/entity/Alice", headers=headers)
        assert entity_res.status_code == 200
        entity_body = entity_res.json()
        assert any(
            fact.get("rel_type") == "WORKS_AT" and fact.get("target_name") == "Acme"
            for fact in entity_body["facts"]
        )

        query_res = client.get(
            "/query",
            headers=headers,
            params={"q": "Where does Alice work?", "limit": 5},
        )
        assert query_res.status_code == 200
        query_body = query_res.json()
        assert query_body["result_count"] >= 1
        assert query_body["results"][0]["entity"] == "Alice"

        stats_res = client.get("/stats", headers=headers)
        assert stats_res.status_code == 200
        stats_body = stats_res.json()
        assert stats_body["graph"]["entity_count"] >= 2
        assert stats_body["vector_store"]["entities"] >= 2
    finally:
        set_queue_instance(None)
        set_pipeline_instance(None)
        set_vector_store_instance(None)
        set_graph_instance(None)


def test_process_job_runs_embeddings_off_event_loop(monkeypatch, tmp_path):
    main_thread_id = threading.get_ident()
    embedding_thread_ids: list[int] = []

    def _thread_checked_embedding(_cls, text: str, dim: int = 384, _depth: int = 0) -> list[float]:
        del text, _depth
        embedding_thread_ids.append(threading.get_ident())
        assert threading.get_ident() != main_thread_id
        return [1.0] + [0.0] * (dim - 1)

    monkeypatch.setattr(memory_extractor, "extract", _fake_extract)
    monkeypatch.setattr(ExtractionPipeline, "_text_embedding", classmethod(_thread_checked_embedding))
    monkeypatch.setattr(config, "SPACY_ENRICHMENT", False)
    monkeypatch.setattr(config, "GLIREL_ENABLED", False)
    monkeypatch.setattr(config, "DECISION_TRACES_INGEST_ENABLED", False)

    graph = LadybugGraph(tmp_path / "graph.lbug")
    vector_store = VectorStore(backend="ladybug", db_path=tmp_path / "vectors.lbug")
    pipeline = ExtractionPipeline(graph=graph, vector_store=vector_store)
    job = ExtractionJob(
        content="Alice works at Acme.",
        source="session",
        speaker="Alice",
        priority=1,
        reference_time=datetime.now(UTC),
    )

    processed = asyncio.run(pipeline.process_job(job))

    assert processed.status == "completed"
    assert len(embedding_thread_ids) >= 3
