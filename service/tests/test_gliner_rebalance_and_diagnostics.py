from __future__ import annotations

import asyncio
import json
from pathlib import Path

import config
from evolution.gliner_training import GLiNERTrainingService


def _make_row(idx: int, label: str) -> dict:
    return {
        "episode_id": f"ep-{idx}",
        "text": f"Person {idx} relation",
        "entities": [{"text": "Alice", "label": "Person"}, {"text": "Org", "label": "Organization"}],
        "relations": [{"head": "Alice", "tail": "Org", "label": label}],
        "negative_relations": [],
    }


def test_rebalance_caps_dominant_and_upsamples_targets(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MOLLYGRAPH_TRAIN_REBALANCE_ENABLED", True)
    monkeypatch.setattr(config, "MOLLYGRAPH_TRAIN_REBALANCE_WORKSAT_CAP", 100)
    monkeypatch.setattr(config, "MOLLYGRAPH_TRAIN_REBALANCE_MAX_RATIO", 1.0)
    monkeypatch.setattr(config, "MOLLYGRAPH_TRAIN_REBALANCE_TARGET_LABELS", "CLASSMATE_OF")
    monkeypatch.setattr(config, "MOLLYGRAPH_TRAIN_REBALANCE_TARGET_MIN", 30)
    monkeypatch.setattr(config, "MOLLYGRAPH_TRAIN_REBALANCE_TARGET_MULTIPLIERS", "CLASSMATE_OF:2.0")
    monkeypatch.setattr(config, "GRAPH_MEMORY_DIR", tmp_path)

    service = GLiNERTrainingService(state_file=tmp_path / "state.json")

    rows = [_make_row(i, "WORKS_AT") for i in range(250)] + [_make_row(1000 + i, "CLASSMATE_OF") for i in range(5)]
    rebalanced, summary = service.rebalance_training_rows(rows, run_id="run1")

    dist = service.relation_distribution(rebalanced)
    assert summary["enabled"] is True
    assert dist["WORKS_AT"] <= 100
    assert dist["CLASSMATE_OF"] >= 30
    assert summary["artifact_path"] and Path(summary["artifact_path"]).exists()


def test_diagnostics_written_for_below_threshold_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "GRAPH_MEMORY_DIR", tmp_path)
    monkeypatch.setattr(config, "GLINER_FINETUNE_MIN_EXAMPLES", 2)
    monkeypatch.setattr(config, "GLINER_FINETUNE_BENCHMARK_THRESHOLD", 0.03)
    monkeypatch.setattr(config, "MOLLYGRAPH_TRAIN_REBALANCE_ENABLED", True)

    service = GLiNERTrainingService(state_file=tmp_path / "state.json")

    rows = [_make_row(1, "WORKS_AT"), _make_row(2, "CLASSMATE_OF"), _make_row(3, "MEMBER_OF")]
    service.load_accumulated_gliner_examples = lambda: rows
    service.select_gliner_training_strategy = lambda total: {"mode": "lora", "reason": "test"}

    async def _audit(_: str):
        return {"passed": True}

    service._run_pretraining_audit = _audit
    service.fine_tune_gliner_candidate = lambda train_rows, mode: {"ok": True, "candidate_model": str(tmp_path / "candidate"), "metadata": {}, "result": {}}
    service.benchmark_finetune_candidate = lambda eval_rows, model_ref, train_count: {
        "ok": True,
        "combined_improvement": -0.02,
        "entity_improvement": -0.01,
        "relation_improvement": -0.03,
        "improvement": -0.01,
        "per_type_relation_f1": {"improvement": {"works at": -0.2, "classmate of": 0.1}},
    }
    service.calibrate_confidence = lambda *args, **kwargs: {"ece": 0.0, "total_predictions": 0, "bins": []}
    service.discard_gliner_candidate_model = lambda *_: None
    service.record_gliner_benchmark = lambda *args, **kwargs: None
    service.write_run_audit = lambda payload: str(tmp_path / "training" / "runs" / f"{payload['run_id']}.json")

    result = asyncio.run(service._run_finetune_pipeline_inner())

    assert result["status"] == "below_threshold"
    assert result.get("diagnostics_path")
    diag_path = Path(result["diagnostics_path"])
    assert diag_path.exists()
    payload = json.loads(diag_path.read_text(encoding="utf-8"))
    assert "deltas" in payload
    assert "class_distribution" in payload


def test_rebalance_disabled_is_backward_compatible(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MOLLYGRAPH_TRAIN_REBALANCE_ENABLED", False)
    monkeypatch.setattr(config, "GRAPH_MEMORY_DIR", tmp_path)

    service = GLiNERTrainingService(state_file=tmp_path / "state.json")
    rows = [_make_row(i, "WORKS_AT") for i in range(10)]

    rebalanced, summary = service.rebalance_training_rows(rows, run_id="disabled")
    assert summary["enabled"] is False
    assert service.relation_distribution(rebalanced) == service.relation_distribution(rows)
    assert summary["artifact_path"] is None
