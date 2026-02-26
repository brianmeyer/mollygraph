from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import audit.signals as audit_signals
from extraction.pipeline import ExtractionPipeline
import metrics.model_health as model_health_module
from metrics.model_health import ModelHealthMonitor
import runtime_state


def _cancel_timer(timer: Any) -> None:
    if timer is None:
        return
    timer.cancel()
    if hasattr(timer, "join"):
        timer.join(timeout=0.2)


def test_audit_signal_counts_persist_with_debounce(monkeypatch, tmp_path: Path):
    state_path = tmp_path / "audit_signal_counts.json"
    monkeypatch.setattr(audit_signals, "_AUDIT_SIGNAL_COUNTS_PATH", state_path, raising=False)
    monkeypatch.setattr(audit_signals, "_PERSIST_DEBOUNCE_SECONDS", 60.0, raising=False)

    with audit_signals._counts_lock:
        old_timer = audit_signals._persist_timer
        audit_signals._persist_timer = None
        audit_signals._signal_counts = {}
        audit_signals._counts_loaded = True
    _cancel_timer(old_timer)

    audit_signals._metrics_listener("relationship_verified", {})
    audit_signals._metrics_listener("relationship_verified", {})

    assert not state_path.exists(), "debounce should avoid immediate disk writes"

    audit_signals._flush_signal_counts_to_disk()
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["relationship_verified"] == 2
    assert not list(tmp_path.glob("*.tmp"))

    # Simulate restart: clear in-memory state and reload from disk.
    with audit_signals._counts_lock:
        audit_signals._signal_counts = {}
        audit_signals._counts_loaded = False
    reloaded = audit_signals.get_signal_counts()
    assert reloaded["relationship_verified"] == 2

    with audit_signals._counts_lock:
        trailing_timer = audit_signals._persist_timer
        audit_signals._persist_timer = None
    _cancel_timer(trailing_timer)


def test_nightly_results_persist_and_reload(monkeypatch, tmp_path: Path):
    state_path = tmp_path / "nightly_results.json"
    monkeypatch.setattr(runtime_state, "_NIGHTLY_RESULTS_PATH", state_path, raising=False)

    with runtime_state._LOCK:
        runtime_state._NIGHTLY_RESULTS = []
        runtime_state._NIGHTLY_RESULTS_LOADED = True

    runtime_state.record_nightly_result(
        status="success",
        audit_status="success",
        lora_status="skipped",
        relationships_reviewed=8,
        relationships_approved=6,
        relationships_flagged=1,
        relationships_reclassified=1,
        infra_health={"decision": "healthy"},
    )

    assert state_path.exists()
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert payload[-1]["status"] == "success"
    assert not list(tmp_path.glob("*.tmp"))

    with runtime_state._LOCK:
        runtime_state._NIGHTLY_RESULTS = []
        runtime_state._NIGHTLY_RESULTS_LOADED = False

    reloaded = runtime_state.get_nightly_results()
    assert len(reloaded) == 1
    assert reloaded[0]["relationships_reviewed"] == 8



def test_ingestion_counters_persist_with_debounce_and_rollover(monkeypatch, tmp_path: Path):
    state_path = tmp_path / "ingestion_counters.json"
    monkeypatch.setattr(ExtractionPipeline, "_INGESTION_COUNTERS_PATH", state_path, raising=False)
    monkeypatch.setattr(ExtractionPipeline, "_PERSIST_DEBOUNCE_SECONDS", 60.0, raising=False)

    with ExtractionPipeline._counter_lock:
        old_timer = ExtractionPipeline._persist_timer
        ExtractionPipeline._persist_timer = None
        ExtractionPipeline._counter_date = ""
        ExtractionPipeline._ingestion_counters = ExtractionPipeline._empty_ingestion_counters()
        ExtractionPipeline._counters_loaded = True
    _cancel_timer(old_timer)

    ExtractionPipeline._update_ingestion_counters(
        entities_extracted=4,
        relationships_extracted=3,
        new_entities=2,
        existing_entities=2,
        fallback_count=1,
    )

    assert not state_path.exists(), "debounce should avoid immediate disk writes"

    ExtractionPipeline._flush_ingestion_counters_to_disk()
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["counters"]["jobs_processed"] == 1
    assert payload["counters"]["total_entities_extracted"] == 4
    assert not list(tmp_path.glob("*.tmp"))

    # Simulate restart and ensure counters reload.
    with ExtractionPipeline._counter_lock:
        ExtractionPipeline._counter_date = ""
        ExtractionPipeline._ingestion_counters = ExtractionPipeline._empty_ingestion_counters()
        ExtractionPipeline._counters_loaded = False
    reloaded = ExtractionPipeline._get_ingestion_counters()
    assert reloaded["jobs_processed"] == 1
    assert reloaded["total_entities_extracted"] == 4

    # Simulate day rollover by persisting yesterday's counters.
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    stale_payload = {
        "date": yesterday,
        "counters": {
            "jobs_processed": 7,
            "total_entities_extracted": 14,
            "total_relationships_extracted": 21,
            "total_new_entities": 4,
            "total_existing_entities": 10,
            "total_fallback_relationships": 3,
        },
    }
    state_path.write_text(json.dumps(stale_payload), encoding="utf-8")

    with ExtractionPipeline._counter_lock:
        trailing_timer = ExtractionPipeline._persist_timer
        ExtractionPipeline._persist_timer = None
        ExtractionPipeline._counter_date = ""
        ExtractionPipeline._ingestion_counters = ExtractionPipeline._empty_ingestion_counters()
        ExtractionPipeline._counters_loaded = False
    _cancel_timer(trailing_timer)

    rolled = ExtractionPipeline._get_ingestion_counters()
    assert rolled["date"] == date.today().isoformat()
    assert rolled["jobs_processed"] == 0
    assert rolled["total_entities_extracted"] == 0

    with ExtractionPipeline._counter_lock:
        final_timer = ExtractionPipeline._persist_timer
        ExtractionPipeline._persist_timer = None
    _cancel_timer(final_timer)



def test_model_health_persists_windows_and_supports_legacy_state(monkeypatch, tmp_path: Path):
    state_path = tmp_path / "model_health_state.json"
    monkeypatch.setattr(model_health_module, "MODEL_HEALTH_STATE_PATH", state_path, raising=False)
    monkeypatch.setattr(
        model_health_module._config,
        "MODEL_DEGRADATION_WINDOW_SIZE",
        5,
        raising=False,
    )

    monitor = ModelHealthMonitor()
    monitor.set_baseline(0.10, "model-a")
    monitor.record_extraction(total_relations=10, fallback_count=1)
    monitor.record_extraction(total_relations=8, fallback_count=0)

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["extraction_counter"] == 2
    assert len(payload["rolling_extractions"]) == 2
    assert len(payload["degradation_window"]) == 2

    reloaded = ModelHealthMonitor()
    assert reloaded._extraction_counter == 2
    assert len(reloaded.rolling_extractions) == 2
    assert len(reloaded.degradation_window) == 2

    legacy_payload = {
        "baseline_fallback_rate": 0.15,
        "model_ref": "legacy-model",
        "deployed_at": "2026-02-20T00:00:00+00:00",
        "degradation_detected": True,
    }
    state_path.write_text(json.dumps(legacy_payload), encoding="utf-8")

    legacy = ModelHealthMonitor()
    assert legacy.baseline_fallback_rate == 0.15
    assert legacy.model_ref == "legacy-model"
    assert legacy.degradation_detected is True
    assert legacy._extraction_counter == 0
    assert len(legacy.rolling_extractions) == 0
    assert len(legacy.degradation_window) == 0
