from __future__ import annotations

import pytest


def _sample_rels() -> list[dict]:
    return [
        {
            "head": "Brian",
            "head_type": "Person",
            "tail": "Databricks",
            "tail_type": "Organization",
            "rel_type": "WORKS_AT",
            "context_snippets": ["Brian works at Databricks as an engineer."],
        },
        {
            "head": "Brian",
            "head_type": "Person",
            "tail": "Python",
            "tail_type": "Technology",
            "rel_type": "WORKS_AT",
            "context_snippets": ["Brian uses Python daily."],
        },
        {
            "head": "Brian",
            "head_type": "Person",
            "tail": "Mars",
            "tail_type": "Place",
            "rel_type": "WORKS_AT",
            "context_snippets": ["Brian works at Mars."],
        },
    ]


def _sample_verdicts() -> list[dict]:
    return [
        {"index": 1, "verdict": "verify"},
        {"index": 2, "verdict": "reclassify", "suggested_type": "USES"},
        {"index": 3, "verdict": "delete"},
    ]


def test_audit_feedback_records_and_loads(tmp_path, monkeypatch) -> None:
    from evolution import audit_feedback

    monkeypatch.setattr(audit_feedback.config, "TRAINING_DIR", tmp_path / "training")

    summary = audit_feedback.record_audit_feedback_batch(
        _sample_rels(),
        _sample_verdicts(),
        schedule="nightly",
        provider="ollama",
        model="llama3.1:8b",
        dry_run=False,
    )

    assert summary["written"] == 3
    assert summary["positive_labels"] == 2  # verify + reclassify(corrected)
    assert summary["negative_labels"] == 2  # reclassify(original) + delete
    assert summary["file"]

    rows = audit_feedback.load_audit_feedback_entries(limit=20)
    assert len(rows) == 3
    assert all(str(r.get("feedback_id") or "").strip() for r in rows)
    assert any(r.get("decision") == "delete" for r in rows)
    assert any(r.get("decision") == "reclassify" for r in rows)


def test_gliner_training_builds_feedback_examples(tmp_path, monkeypatch) -> None:
    from evolution import audit_feedback
    from evolution.gliner_training import GLiNERTrainingService

    monkeypatch.setattr(audit_feedback.config, "TRAINING_DIR", tmp_path / "training")
    monkeypatch.setattr("evolution.gliner_training.config.TRAINING_DIR", tmp_path / "training")

    audit_feedback.record_audit_feedback_batch(
        _sample_rels(),
        _sample_verdicts(),
        schedule="nightly",
        provider="ollama",
        model="llama3.1:8b",
        dry_run=False,
    )

    service = GLiNERTrainingService(state_file=tmp_path / "state.json")
    examples, stats = service.build_training_examples_from_feedback(set(), limit=20)

    assert stats["feedback_rows_scanned"] == 3
    assert stats["feedback_examples_added"] == 3
    assert stats["feedback_positive_examples"] >= 1
    assert stats["feedback_negative_examples"] >= 1
    assert len(examples) == 3
    assert all(str(e.get("episode_id") or "").startswith("audit-feedback:") for e in examples)
    assert any(e.get("hard_negative_relations") for e in examples)


def test_strict_ai_blocks_hash_embedding(monkeypatch) -> None:
    from extraction.pipeline import ExtractionPipeline

    monkeypatch.setattr("extraction.pipeline.service_config.STRICT_AI", True)
    monkeypatch.setattr("extraction.pipeline.service_config.EMBEDDING_BACKEND", "hash")

    with pytest.raises(RuntimeError):
        ExtractionPipeline._text_embedding("hello world")


def test_embedding_status_reports_strict_ai_blocking_errors(monkeypatch) -> None:
    import embedding_registry

    monkeypatch.setattr("embedding_registry.config.STRICT_AI", True)
    monkeypatch.setattr("embedding_registry.config.RUNTIME_PROFILE", "strict_ai")
    monkeypatch.setattr(
        embedding_registry,
        "_REGISTRY_CACHE",
        {
            "active_provider": "hash",
            "active_model": "",
            "models": {"huggingface": [], "ollama": []},
            "supported_providers": ["hash", "huggingface", "ollama"],
        },
    )

    status = embedding_registry.get_embedding_status()
    assert status["strict_ai"] is True
    assert status["active_provider"] == "hash"
    assert status["blocking_errors"]
