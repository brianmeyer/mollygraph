from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import config as service_config
from extraction.decision_traces import (
    DecisionExtractionResult,
    DecisionPayload,
    DecisionPrefilterResult,
    run_decision_prefilter,
)
from extraction.pipeline import ExtractionPipeline
import extraction.pipeline as pipeline_module
from memory.models import ExtractionJob


class _FakeVectorStore:
    pass


class _FakeGraph:
    def __init__(self) -> None:
        self.created_decisions: list[dict[str, object]] = []

    def create_decision(self, **kwargs):
        self.created_decisions.append(dict(kwargs))
        return {"id": "dec-test-1", **kwargs}


def _build_pipeline() -> tuple[ExtractionPipeline, _FakeGraph]:
    graph = _FakeGraph()
    pipeline = ExtractionPipeline(graph=graph, vector_store=_FakeVectorStore())  # type: ignore[arg-type]
    return pipeline, graph


def test_decision_prefilter_detects_decision_language(monkeypatch):
    monkeypatch.setattr(service_config, "DECISION_TRACES_PREFILTER_ENABLED", True, raising=False)
    monkeypatch.setattr(service_config, "DECISION_TRACES_PREFILTER_MIN_SCORE", 2, raising=False)
    monkeypatch.setattr(service_config, "DECISION_TRACES_MIN_CONTENT_CHARS", 10, raising=False)

    result = run_decision_prefilter(
        content="Let's go with jina-v5-nano instead of keeping the old embedding model.",
        source="conversation",
    )
    assert result.passed is True
    assert result.score >= 2


def test_decision_prefilter_blocks_low_signal_source(monkeypatch):
    monkeypatch.setattr(service_config, "DECISION_TRACES_SOURCE_BLOCKLIST", ["promo", "noise"], raising=False)
    monkeypatch.setattr(service_config, "DECISION_TRACES_MIN_CONTENT_CHARS", 10, raising=False)

    result = run_decision_prefilter(
        content="We decided to ship it.",
        source="conversation",
        raw_source="promo_newsletter",
    )
    assert result.passed is False
    assert result.reason == "blocked_source"


def test_decision_llm_not_called_when_prefilter_fails(monkeypatch):
    pipeline, graph = _build_pipeline()
    monkeypatch.setattr(service_config, "DECISION_TRACES_INGEST_ENABLED", True, raising=False)

    monkeypatch.setattr(
        pipeline_module,
        "run_decision_prefilter",
        lambda **_: DecisionPrefilterResult(passed=False, score=0, reason="insufficient_signal"),
    )

    calls = {"llm": 0}

    async def _fake_extract(**kwargs):
        calls["llm"] += 1
        return DecisionExtractionResult(
            is_decision=False,
            payload=None,
            provider="",
            model="",
            tier="",
            latency_ms=0,
            error="",
        )

    monkeypatch.setattr(pipeline_module, "extract_decision_trace", _fake_extract)

    job = ExtractionJob(
        content="This text has no concrete decision language.",
        source="conversation",
        reference_time=datetime.now(UTC),
    )
    created = asyncio.run(
        pipeline._maybe_record_decision_trace(
            job=job,
            ingest_source="conversation",
            episode_id="ep-1",
            speaker=None,
            related_entities=[],
        )
    )
    assert created is None
    assert calls["llm"] == 0
    assert graph.created_decisions == []


def test_decision_write_path_on_positive_extraction(monkeypatch):
    pipeline, graph = _build_pipeline()
    monkeypatch.setattr(service_config, "DECISION_TRACES_INGEST_ENABLED", True, raising=False)
    monkeypatch.setattr(service_config, "DECISION_TRACES_MIN_CONFIDENCE", 0.6, raising=False)
    monkeypatch.setattr(service_config, "DECISION_TRACES_MAX_RELATED_ENTITIES", 3, raising=False)

    monkeypatch.setattr(
        pipeline_module,
        "run_decision_prefilter",
        lambda **_: DecisionPrefilterResult(passed=True, score=4, reason="matched"),
    )

    async def _fake_extract(**kwargs):
        return DecisionExtractionResult(
            is_decision=True,
            payload=DecisionPayload(
                decision="Switch embedding model to jina-v5-nano",
                reasoning="Higher retrieval quality in benchmark run.",
                alternatives=["keep current model", "nomic-embed-text"],
                inputs=["benchmark report", "latency target"],
                outcome="Reindex completed",
                decided_by="Brian",
                confidence=0.93,
            ),
            provider="moonshot",
            model="kimi-k2.5",
            tier="primary",
            latency_ms=120,
            error="",
        )

    monkeypatch.setattr(pipeline_module, "extract_decision_trace", _fake_extract)

    reference_time = datetime.now(UTC)
    job = ExtractionJob(
        content="Let's go with jina-v5-nano instead of the old model.",
        source="conversation",
        reference_time=reference_time,
    )
    created = asyncio.run(
        pipeline._maybe_record_decision_trace(
            job=job,
            ingest_source="conversation",
            episode_id="ep-42",
            speaker="Brian",
            related_entities=["MollyGraph", "Embeddings", "MollyGraph", "Neo4j"],
        )
    )

    assert created is not None
    assert len(graph.created_decisions) == 1
    record = graph.created_decisions[0]
    assert record["decision"] == "Switch embedding model to jina-v5-nano"
    assert record["source_episode_id"] == "ep-42"
    assert record["decided_by"] == "Brian"
    assert record["related_entities"] == ["MollyGraph", "Embeddings", "Neo4j"]
