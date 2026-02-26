"""Standalone GLiNER training pipeline for graph-memory.

This is adapted from Molly's ``evolution/gliner_training.py`` with Molly-specific
engine/infra dependencies removed in favor of a simple state file.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import shutil
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import config
from evolution.audit_feedback import load_audit_feedback_entries

log = logging.getLogger(__name__)

GLINER_BASE_MODEL = config.GLINER_BASE_MODEL
GLINER_BENCHMARK_SEED = config.GLINER_BENCHMARK_SEED
GLINER_BENCHMARK_EVAL_RATIO = config.GLINER_BENCHMARK_EVAL_RATIO
GLINER_BENCHMARK_THRESHOLD = config.GLINER_BENCHMARK_THRESHOLD
GLINER_FINETUNE_COOLDOWN_DAYS = config.GLINER_FINETUNE_COOLDOWN_DAYS
GLINER_LORA_COOLDOWN_DAYS = config.GLINER_LORA_COOLDOWN_DAYS
GLINER_TRAINING_SCAN_LIMIT = config.GLINER_TRAINING_SCAN_LIMIT
GLINER_MIN_NEW_EXAMPLES = config.GLINER_MIN_NEW_EXAMPLES

_ALLOWED_ENTITY_TYPES = {
    "Person",
    "Organization",
    "Technology",
    "Place",
    "Project",
    "Concept",
    "Event",
}


def _get_allowed_rel_types() -> frozenset[str]:
    """Derive allowed relation type labels from the active extractor schema.

    Returns labels as normalized lowercase strings (matching the training record format).
    Falls back to VALID_REL_TYPES from graph constants when the extractor schema
    cannot be loaded, ensuring training examples are always schema-filtered.
    Returns empty frozenset (= allow all) only as a last resort.
    """
    try:
        from extractor_schema_registry import get_effective_extractor_schema

        schema = get_effective_extractor_schema()
        rels = schema.get("relations") or {}
        if rels:
            return frozenset(
                str(k).strip().lower().replace("_", " ") for k in rels if k
            )
    except Exception:
        pass
    # Fall back to the hardcoded VALID_REL_TYPES from graph constants so that
    # training examples are always schema-filtered even when the extractor
    # schema registry is unavailable.
    try:
        from memory.graph.constants import VALID_REL_TYPES

        return frozenset(k.strip().lower().replace("_", " ") for k in VALID_REL_TYPES)
    except Exception:
        pass
    return frozenset()


# Lazily computed allowed relation types (populated on first use)
_ALLOWED_REL_TYPES: frozenset[str] = frozenset()


class GLiNERTrainingService:

    def __init__(self, state_file: Path | None = None):
        self._state_file = state_file or config.STATE_FILE
        self._state_lock = threading.Lock()
        self.state: dict[str, Any] = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        if not self._state_file.exists():
            return {}
        try:
            payload = json.loads(self._state_file.read_text(encoding="utf-8"))
        except Exception:
            log.warning("Failed to parse state file %s", self._state_file, exc_info=True)
            return {}
        return payload if isinstance(payload, dict) else {}

    def save_state(self) -> None:
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        with self._state_lock:
            fd, tmp_name = tempfile.mkstemp(
                dir=self._state_file.parent,
                prefix=".state_tmp_",
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(json.dumps(self.state, indent=2, ensure_ascii=True) + "\n")
                Path(tmp_name).replace(self._state_file)
            except Exception:
                try:
                    Path(tmp_name).unlink(missing_ok=True)
                except Exception:
                    pass
                raise

    def refresh_state(self) -> None:
        self.state = self._load_state()

    def gliner_training_dir(self) -> Path:
        return config.TRAINING_DIR

    def gliner_models_dir(self) -> Path:
        return config.MODELS_DIR

    def gliner_candidate_model_dir(self) -> Path:
        return self.gliner_models_dir() / "gliner_candidate"

    def gliner_active_model_dir(self) -> Path:
        return self.gliner_models_dir() / "gliner_active"

    def gliner_training_config_path(self) -> Path:
        return config.GRAPH_MEMORY_DIR / "gliner_finetune_config.json"

    async def run_gliner_nightly_cycle(self, force: bool = False) -> dict[str, Any]:
        self.refresh_state()

        accumulation = await asyncio.to_thread(
            self.accumulate_gliner_training_data,
            GLINER_TRAINING_SCAN_LIMIT,
        )
        total_examples = int(accumulation.get("total_examples", 0))
        # Cursor-based new-example gate: count examples accumulated since the last
        # successful training run (tracked via gliner_examples_at_last_train in state.json).
        examples_at_last_train = int(self.state.get("gliner_examples_at_last_train", 0))
        new_examples = max(0, total_examples - examples_at_last_train)
        required_new = int(config.GLINER_MIN_NEW_EXAMPLES)

        progress_line = (
            f"GLiNER training data: {total_examples} total, {new_examples}/{required_new} new examples since last train"
        )
        self.state["gliner_training_examples"] = total_examples
        self.state["gliner_last_result"] = progress_line
        self.state["gliner_last_cycle_status"] = "accumulated"
        self.save_state()

        if new_examples < required_new:
            return {
                "status": "insufficient_new_examples",
                "count": total_examples,
                "new_examples": new_examples,
                "required_new": required_new,
                "examples_at_last_train": examples_at_last_train,
                "accumulation": accumulation,
                "message": progress_line,
            }

        now_utc = datetime.now(timezone.utc)
        last_run_iso = str(self.state.get("gliner_last_finetune_at", "")).strip()
        last_run = _parse_datetime(last_run_iso)

        # Strategy-aware cooldown: LoRA = 2 days, full fine-tune = 7 days
        last_strategy = str(
            self.state.get("gliner_last_training_strategy")
            or self.state.get("gliner_last_strategy")
            or "lora"
        ).strip().lower()
        if last_strategy not in {"lora", "full"}:
            last_strategy = "lora"
        cooldown_hours = GLINER_LORA_COOLDOWN_DAYS * 24 if last_strategy == "lora" else GLINER_FINETUNE_COOLDOWN_DAYS * 24

        if not force and last_run and (now_utc - last_run) < timedelta(hours=cooldown_hours):
            elapsed = now_utc - last_run
            remaining = timedelta(hours=cooldown_hours) - elapsed
            hours_remaining = max(0, int(remaining.total_seconds() // 3600))
            cooldown_line = (
                f"GLiNER fine-tune skipped: last {last_strategy} run {last_run.date().isoformat()} "
                f"({hours_remaining}h cooldown, {cooldown_hours}h for {last_strategy})."
            )
            self.state["gliner_last_result"] = cooldown_line
            self.state["gliner_last_cycle_status"] = "cooldown_active"
            self.save_state()
            return {
                "status": "cooldown_active",
                "count": total_examples,
                "new_examples": new_examples,
                "required_new": required_new,
                "last_run": last_run.isoformat(),
                "accumulation": accumulation,
                "message": cooldown_line,
                "force": bool(force),
            }

        pipeline = await self.run_gliner_finetune_pipeline()
        return {
            "status": "finetune_triggered",
            "accumulation": accumulation,
            "pipeline": pipeline,
            "force": bool(force),
        }

    _finetune_running = False  # class-level concurrency guard

    async def run_gliner_finetune_pipeline(self) -> dict[str, Any]:
        if GLiNERTrainingService._finetune_running:
            log.warning("GLiNER fine-tune already running, skipping concurrent trigger")
            return {"status": "already_running", "message": "Fine-tune pipeline already in progress"}
        GLiNERTrainingService._finetune_running = True
        try:
            return await self._run_finetune_pipeline_inner()
        finally:
            GLiNERTrainingService._finetune_running = False

    async def _run_finetune_pipeline_inner(self) -> dict[str, Any]:
        rows = await asyncio.to_thread(self.load_accumulated_gliner_examples)
        total_rows = len(rows)
        required = int(config.GLINER_FINETUNE_MIN_EXAMPLES)

        if total_rows < required:
            msg = f"GLiNER training data: {total_rows}/{required} examples accumulated"
            self.state["gliner_training_examples"] = total_rows
            self.state["gliner_last_result"] = msg
            self.state["gliner_last_cycle_status"] = "insufficient_examples"
            self.save_state()
            return {
                "status": "insufficient_examples",
                "count": total_rows,
                "required": required,
                "message": msg,
            }

        # ── Pre-split data stats logging ──────────────────────────────────────
        _pre_pos = sum(len(r.get("relations") or []) for r in rows)
        _pre_neg = sum(len(r.get("negative_relations") or []) for r in rows)
        log.info("=" * 60)
        log.info("GLiNER fine-tune pipeline starting")
        log.info("  Total examples:    %d (required: %d)", total_rows, required)
        log.info("  Positive rels:     %d", _pre_pos)
        log.info("  Negative rels:     %d", _pre_neg)
        log.info("  Eval ratio:        %.2f  seed=%d", GLINER_BENCHMARK_EVAL_RATIO, GLINER_BENCHMARK_SEED)
        log.info("=" * 60)

        train_rows, eval_rows = self.split_holdout_rows(
            rows,
            eval_ratio=GLINER_BENCHMARK_EVAL_RATIO,
            seed=GLINER_BENCHMARK_SEED,
        )
        if not train_rows or not eval_rows:
            self.state["gliner_last_cycle_status"] = "split_failed"
            self.state["gliner_last_result"] = "GLiNER fine-tune skipped: invalid train/eval split."
            self.save_state()
            return {
                "status": "split_failed",
                "count": total_rows,
                "train_count": len(train_rows),
                "eval_count": len(eval_rows),
            }

        # ── Post-split logging ────────────────────────────────────────────────
        _train_ent_dist: dict[str, int] = {}
        _train_rel_dist: dict[str, int] = {}
        for _r in train_rows:
            for _e in (_r.get("entities") or []):
                _lbl = str(_e.get("label") or "Concept") if isinstance(_e, dict) else "Concept"
                _train_ent_dist[_lbl] = _train_ent_dist.get(_lbl, 0) + 1
            for _rel in (_r.get("relations") or []):
                _lbl = str(_rel.get("label") or "").strip() if isinstance(_rel, dict) else ""
                if _lbl:
                    _train_rel_dist[_lbl] = _train_rel_dist.get(_lbl, 0) + 1

        log.info("After split: train=%d  eval=%d", len(train_rows), len(eval_rows))
        log.info("  Train entity types:   %s", dict(sorted(_train_ent_dist.items())))
        log.info("  Train relation types: %s", dict(sorted(_train_rel_dist.items())))

        strategy = await asyncio.to_thread(self.select_gliner_training_strategy, total_rows)
        mode = str(strategy.get("mode") or "lora").strip().lower()
        if mode not in {"lora", "full"}:
            mode = "lora"

        # Pre-training audit with Kimi (gates the training)
        pretrain_audit = await self._run_pretraining_audit(mode)
        if not pretrain_audit.get("passed", True):
            self.state["gliner_last_cycle_status"] = "pretrain_audit_failed"
            self.state["gliner_last_result"] = f"GLiNER {mode} blocked: Kimi pre-training audit failed - {pretrain_audit.get('reason', 'quality checks not met')}"
            self.save_state()
            return {
                "status": "pretrain_audit_failed",
                "count": total_rows,
                "pretrain_audit": pretrain_audit,
                "message": self.state["gliner_last_result"],
            }

        # NOTE: gliner_last_finetune_at is intentionally NOT set here.
        # It is only stamped after training succeeds so that a failed attempt
        # does NOT consume the cooldown window — allowing an immediate retry.
        self.state["gliner_training_examples"] = total_rows
        self.state["gliner_last_cycle_status"] = "finetune_started"
        self.state["gliner_last_training_strategy"] = mode
        self.save_state()

        fine_tune = await asyncio.to_thread(self.fine_tune_gliner_candidate, train_rows, mode)
        if not fine_tune.get("ok", False):
            error_msg = fine_tune.get("error", "unknown error")
            self.state["gliner_last_cycle_status"] = "finetune_failed"
            self.state["gliner_last_result"] = f"GLiNER {mode} fine-tune failed: {error_msg}"
            self.save_state()
            return {
                "status": "finetune_failed",
                "error": error_msg,
                "count": total_rows,
                "train_count": len(train_rows),
                "eval_count": len(eval_rows),
                "training_strategy": strategy,
                "fine_tune": fine_tune,
            }

        # Training succeeded — stamp the cooldown timer and update status.
        self.state["gliner_last_finetune_at"] = datetime.now(timezone.utc).isoformat()
        # Cursor-based new-example gate: snapshot total on-disk example count so the next
        # nightly cycle can compute how many NEW examples have arrived since this run.
        self.state["gliner_examples_at_last_train"] = self.count_accumulated_gliner_examples()
        self.state["gliner_last_cycle_status"] = "finetune_completed"
        self.state["gliner_last_result"] = (
            f"GLiNER {mode} training completed. {len(train_rows)} train / {len(eval_rows)} eval. "
            f"Running benchmark..."
        )
        self.save_state()

        candidate_model_ref = str(fine_tune.get("candidate_model") or "").strip()
        benchmark = await asyncio.to_thread(
            self.benchmark_finetune_candidate,
            eval_rows,
            candidate_model_ref,
            len(train_rows),
        )

        benchmark_ok = bool(benchmark.get("ok", False))
        # Use combined improvement (0.5 * entity + 0.5 * relation) for deploy decision
        combined_improvement = float(benchmark.get("combined_improvement", 0.0) or 0.0)
        entity_improvement = float(benchmark.get("entity_improvement", 0.0) or 0.0)
        relation_improvement = float(benchmark.get("relation_improvement", 0.0) or 0.0)
        # Legacy: improvement = entity improvement for backward compat logs
        improvement = float(benchmark.get("improvement", 0.0) or 0.0)
        threshold = float(config.GLINER_FINETUNE_BENCHMARK_THRESHOLD)
        should_deploy = benchmark_ok and combined_improvement >= threshold

        log.info(
            "Deploy decision: combined=%.4f (entity=%.4f, relation=%.4f) vs threshold=%.4f → %s",
            combined_improvement, entity_improvement, relation_improvement,
            threshold, "DEPLOY" if should_deploy else "REJECT",
        )

        # ── Confidence calibration pass ───────────────────────────────────────
        calibration: dict[str, Any] = {}
        try:
            log.info("Running confidence calibration on %d eval rows...", len(eval_rows))
            calibration = await asyncio.to_thread(
                self.calibrate_confidence,
                candidate_model_ref,
                eval_rows,
            )
            ece = float(calibration.get("ece", 0.0) or 0.0)
            log.info(
                "Calibration complete: ECE=%.4f  total_predictions=%d",
                ece,
                calibration.get("total_predictions", 0),
            )
            if calibration.get("warning"):
                log.warning("Calibration warning: %s", calibration["warning"])
        except Exception:
            log.warning("Calibration pass failed", exc_info=True)
            calibration = {"ece": 0.0, "total_predictions": 0, "bins": [], "warning": None}

        payload = {
            "count": total_rows,
            "train_count": len(train_rows),
            "eval_count": len(eval_rows),
            "training_strategy": strategy,
            "fine_tune": fine_tune,
            "benchmark": benchmark,
            "calibration": calibration,
            "threshold": threshold,
            "should_deploy": should_deploy,
        }

        status = "below_threshold"
        decision_reason = f"Combined improvement {combined_improvement:.4f} < threshold {threshold:.4f}"

        if should_deploy:
            decision_reason = f"Combined improvement {combined_improvement:.4f} >= threshold {threshold:.4f}"
            # ── Shadow evaluation before deploy ────────────────────────────
            shadow: dict[str, Any] = {}
            if config.GLINER_SHADOW_ENABLED:
                base_model_ref = self.active_gliner_model_ref()
                shadow = await asyncio.to_thread(
                    self.run_shadow_evaluation,
                    base_model_ref,
                    candidate_model_ref,
                    config.GLINER_SHADOW_EPISODES,
                )
                payload["shadow"] = shadow
                log.info("Shadow evaluation result: %s", shadow)

            shadow_passed = (
                not config.GLINER_SHADOW_ENABLED
                or shadow.get("skipped", False)
                or shadow.get("passed", True)
            )

            if not shadow_passed:
                await asyncio.to_thread(
                    self.discard_gliner_candidate_model,
                    Path(candidate_model_ref) if candidate_model_ref else None,
                )
                status = "shadow_failed"
                reason = shadow.get("failure_reason", "candidate_fallback_rate_regression")
                decision_reason = f"Benchmark passed but shadow FAILED: {reason}"
                self.state["gliner_last_result"] = (
                    f"GLiNER {mode} benchmark passed (+{combined_improvement:.4f}) but shadow FAILED: {reason}"
                )
                self.state["gliner_last_cycle_status"] = "shadow_failed"
                self.save_state()
            else:
                deploy = await asyncio.to_thread(
                    self.deploy_gliner_candidate_model,
                    Path(candidate_model_ref),
                    benchmark,
                    fine_tune,
                )
                payload["deploy"] = deploy
                if deploy.get("ok"):
                    status = "deployed"
                    decision_reason = (
                        f"Combined improvement {combined_improvement:.4f} >= threshold {threshold:.4f}; deployed"
                    )
                    self.state["gliner_last_result"] = (
                        f"GLiNER {mode} +{combined_improvement:.4f} combined F1 on {len(eval_rows)} eval rows; deployed."
                    )
                    self.state["gliner_last_cycle_status"] = "deployed"
                    self.save_state()
                    # Archive training example files older than GLINER_ARCHIVE_DAYS post-deploy.
                    try:
                        archive_result = await asyncio.to_thread(self.archive_old_training_examples)
                        log.info("Post-deploy archival: %s", archive_result)
                    except Exception:
                        log.warning("Post-deploy archival failed (non-fatal)", exc_info=True)
                else:
                    status = "deploy_failed"
                    decision_reason = f"Combined improvement {combined_improvement:.4f} >= threshold but deploy failed"
                    self.state["gliner_last_result"] = (
                        f"GLiNER {mode} benchmark passed (+{combined_improvement:.4f}) but deploy failed."
                    )
                    self.state["gliner_last_cycle_status"] = "deploy_failed"
                    self.save_state()
        else:
            await asyncio.to_thread(
                self.discard_gliner_candidate_model,
                Path(candidate_model_ref) if candidate_model_ref else None,
            )
            self.state["gliner_last_cycle_status"] = "below_threshold"
            self.state["gliner_last_result"] = (
                f"GLiNER {mode} candidate rejected (+{combined_improvement:.4f} combined < {threshold:.4f})."
            )
            self.save_state()

        log.info("Deploy decision finalized: status=%s reason=%s", status, decision_reason)

        await asyncio.to_thread(
            self.record_gliner_benchmark,
            mode,
            benchmark,
            status,
            total_rows,
        )

        # ── LoRA run audit trail ──────────────────────────────────────────────
        run_audit_path: str | None = None
        try:
            positive_counts = sum(len(r.get("relations", [])) for r in rows)
            negative_counts = sum(len(r.get("negative_relations", [])) for r in rows)
            type_dist: dict[str, int] = {}
            for r in rows:
                for ent in r.get("entities", []):
                    if isinstance(ent, dict):
                        label = str(ent.get("label") or "Concept")
                    else:
                        label = "Concept"
                    type_dist[label] = type_dist.get(label, 0) + 1

            rollback_baseline: dict = {}
            try:
                from metrics.model_health import model_health_monitor
                if model_health_monitor.baseline_fallback_rate is not None:
                    rollback_baseline = {
                        "fallback_rate": model_health_monitor.baseline_fallback_rate,
                        "model_ref": model_health_monitor.model_ref,
                        "set_at": model_health_monitor.deployed_at.isoformat() if model_health_monitor.deployed_at else None,
                    }
            except Exception:
                pass

            unique_entity_types = sorted({
                str(ent.get("label") or "Concept")
                for r in rows
                for ent in (r.get("entities") or [])
                if isinstance(ent, dict)
            })
            unique_relation_types = sorted({
                str(rel.get("label") or "").strip()
                for r in rows
                for rel in (r.get("relations") or [])
                if isinstance(rel, dict) and rel.get("label")
            })
            avg_entities = (
                sum(len(r.get("entities") or []) for r in rows) / len(rows) if rows else 0.0
            )
            avg_relations = (
                sum(len(r.get("relations") or []) for r in rows) / len(rows) if rows else 0.0
            )

            _ft_meta = fine_tune.get("metadata") or {}
            if isinstance(_ft_meta, dict):
                _ft_epochs = _ft_meta.get("num_epochs")
                _ft_lr = _ft_meta.get("learning_rate")
                _ft_batch = _ft_meta.get("batch_size")
            else:
                _ft_epochs = None
                _ft_lr = None
                _ft_batch = None

            _ft_result = fine_tune.get("result") or {}
            loss_curve: list[float] = []
            final_loss: float | None = None
            if isinstance(_ft_result, dict):
                loss_curve = [
                    float(x) for x in (_ft_result.get("loss_curve") or [])
                    if x is not None
                ]
                final_loss = float(_ft_result.get("final_loss")) if _ft_result.get("final_loss") is not None else None
            if not final_loss and loss_curve:
                final_loss = loss_curve[-1]

            _cal_ece = float(calibration.get("ece", 0.0) or 0.0)
            _cal_bins = calibration.get("bins") or []

            audit_payload: dict[str, Any] = {
                "run_id": datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "mode": mode,
                "data": {
                    "total_examples": total_rows,
                    "train_count": len(train_rows),
                    "eval_count": len(eval_rows),
                    "positive_relations": positive_counts,
                    "negative_relations": negative_counts,
                    "unique_entity_types": unique_entity_types,
                    "unique_relation_types": unique_relation_types,
                    "avg_entities_per_example": round(avg_entities, 2),
                    "avg_relations_per_example": round(avg_relations, 2),
                    "entity_type_distribution": type_dist,
                },
                "training": {
                    "epochs": _ft_epochs,
                    "learning_rate": _ft_lr,
                    "batch_size": _ft_batch,
                    "final_loss": final_loss,
                    "loss_curve": loss_curve,
                    "base_model": self.active_gliner_model_ref(),
                    "seed": GLINER_BENCHMARK_SEED,
                },
                "benchmark": {
                    "entity_f1": {
                        "base": float(benchmark.get("base_entity_f1", benchmark.get("base_score", 0.0)) or 0.0),
                        "candidate": float(benchmark.get("candidate_entity_f1", benchmark.get("candidate_score", 0.0)) or 0.0),
                        "improvement": float(benchmark.get("entity_improvement", benchmark.get("improvement", 0.0)) or 0.0),
                    },
                    "relation_f1": {
                        "base": float(benchmark.get("base_relation_f1", 0.0) or 0.0),
                        "candidate": float(benchmark.get("candidate_relation_f1", 0.0) or 0.0),
                        "improvement": float(benchmark.get("relation_improvement", 0.0) or 0.0),
                    },
                    "per_type_f1": benchmark.get("per_type_relation_f1") or {},
                    "combined_improvement": float(benchmark.get("combined_improvement", 0.0) or 0.0),
                },
                "calibration": {
                    "ece": _cal_ece,
                    "bins": _cal_bins,
                    "total_predictions": int(calibration.get("total_predictions", 0) or 0),
                    "warning": calibration.get("warning"),
                },
                "decision": status,
                "decision_reason": decision_reason,
                "status": status,
                "deploy_decision": {
                    "should_deploy": status == "deployed",
                    "combined_improvement": float(benchmark.get("combined_improvement", 0.0) or 0.0),
                    "entity_improvement": float(benchmark.get("entity_improvement", 0.0) or 0.0),
                    "relation_improvement": float(benchmark.get("relation_improvement", 0.0) or 0.0),
                    "threshold": float(config.GLINER_FINETUNE_BENCHMARK_THRESHOLD),
                    "reason": decision_reason,
                },
                "rollback_baseline": rollback_baseline,
                "strategy": strategy,
                "shadow": payload.get("shadow"),
                "fine_tune": fine_tune,
            }

            run_audit_path = await asyncio.to_thread(self.write_run_audit, audit_payload)
        except Exception:
            log.warning("Failed to write run audit trail", exc_info=True)

        result: dict[str, Any] = {
            "status": status,
            "count": total_rows,
            "train_count": len(train_rows),
            "eval_count": len(eval_rows),
            "training_strategy": strategy,
            "benchmark": benchmark,
            "fine_tune": fine_tune,
            **({"shadow": payload.get("shadow")} if "shadow" in payload else {}),
            **({"deploy": payload.get("deploy")} if "deploy" in payload else {}),
        }
        if run_audit_path:
            result["run_audit_path"] = run_audit_path
        return result

    def accumulate_gliner_training_data(self, limit: int = GLINER_TRAINING_SCAN_LIMIT) -> dict[str, Any]:
        from runtime_graph import require_graph_instance

        training_dir = self.gliner_training_dir()
        training_dir.mkdir(parents=True, exist_ok=True)

        seen_episode_ids = self.load_existing_gliner_episode_ids(training_dir)
        opus_analysis_text = self.latest_maintenance_analysis_text()
        cursor = str(self.state.get("gliner_training_cursor", "")).strip()
        # Load the episode IDs seen at the previous cursor boundary so we can
        # exclude them from the >= query and avoid re-processing them.
        # Guard against corrupt state (non-list or non-string elements).
        _raw_seen = self.state.get("gliner_training_cursor_seen_ids", [])
        cursor_seen_ids: list[str] = [
            str(x) for x in (_raw_seen if isinstance(_raw_seen, list) else [])
            if x and str(x).strip()
        ]

        # Only accumulate episodes older than 24 hours, giving the nightly audit
        # time to review and quarantine bad data before it enters training.
        cutoff_time = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        log.info(
            "GLiNER accumulation: 24-hour audit delay active, cutoff=%s", cutoff_time
        )

        # Issue 6: Use >= (not >) so episodes that share the exact cursor
        # timestamp are not permanently skipped.  We deduplicate via NOT IN on
        # the IDs that were already processed in the previous batch.
        where_cursor = (
            "AND datetime(coalesce(ep.created_at, toString(ep.ingested_at), toString(ep.occurred_at))) >= datetime($cursor)"
            " AND NOT ep.id IN $seen_ids"
            if cursor
            else ""
        )
        params: dict[str, Any] = {"limit": int(limit), "cutoff_time": cutoff_time}
        if cursor:
            params["cursor"] = cursor
            params["seen_ids"] = cursor_seen_ids

        driver = require_graph_instance().driver
        with driver.session() as session:
            rows = [
                dict(record)
                for record in session.run(
                    f"""
                    MATCH (ep:Episode)
                    WHERE ep.content_preview IS NOT NULL
                      AND trim(ep.content_preview) <> ''
                      AND ep.entities_extracted IS NOT NULL
                      AND size(ep.entities_extracted) > 0
                      AND datetime(coalesce(ep.created_at, toString(ep.ingested_at), toString(ep.occurred_at))) < datetime($cutoff_time)
                      {where_cursor}
                    RETURN ep.id AS episode_id,
                           coalesce(ep.created_at, toString(ep.ingested_at), toString(ep.occurred_at)) AS created_at,
                           ep.content_preview AS source_text,
                           ep.entities_extracted AS entity_names
                    ORDER BY coalesce(ep.created_at, toString(ep.ingested_at), toString(ep.occurred_at)) ASC
                    LIMIT $limit
                    """,
                    **params,
                )
            ]

            new_examples: list[dict[str, Any]] = []
            latest_seen_created_at = cursor

            for row in rows:
                episode_id = str(row.get("episode_id") or "").strip()
                if not episode_id:
                    continue

                if episode_id in seen_episode_ids:
                    if row.get("created_at"):
                        latest_seen_created_at = str(row["created_at"])
                    continue

                example = self.build_training_example_from_episode(
                    session=session,
                    episode=row,
                    opus_analysis_text=opus_analysis_text,
                )
                if example:
                    new_examples.append(example)
                    seen_episode_ids.add(episode_id)

                if row.get("created_at"):
                    latest_seen_created_at = str(row["created_at"])

        feedback_examples, feedback_stats = self.build_training_examples_from_feedback(
            seen_episode_ids=seen_episode_ids,
            limit=max(25, int(limit)),
        )
        if feedback_examples:
            new_examples.extend(feedback_examples)

        batch_path: str | None = None
        if new_examples:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S%f")
            path = training_dir / f"examples-{ts}.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for example in new_examples:
                    f.write(json.dumps(example, ensure_ascii=True) + "\n")
            batch_path = str(path)
            log.info("GLiNER accumulation: wrote %d examples to %s", len(new_examples), path)
        else:
            log.info("GLiNER accumulation: no new examples this run")

        if latest_seen_created_at:
            self.state["gliner_training_cursor"] = latest_seen_created_at
            # Issue 6: Persist the episode IDs that sit exactly at the new
            # cursor timestamp.  The next run queries >= cursor, so we must
            # exclude these already-processed IDs to avoid re-processing them.
            boundary_ids = [
                str(row.get("episode_id") or "")
                for row in rows
                if str(row.get("created_at") or "") == latest_seen_created_at
                and str(row.get("episode_id") or "").strip()
            ]
            self.state["gliner_training_cursor_seen_ids"] = boundary_ids
            log.debug(
                "GLiNER cursor advanced to %s with %d boundary IDs",
                latest_seen_created_at,
                len(boundary_ids),
            )
        else:
            # No episodes processed this run; clear stale boundary IDs.
            self.state.pop("gliner_training_cursor_seen_ids", None)

        total_examples = self.count_accumulated_gliner_examples(training_dir)
        self.state["gliner_training_examples"] = total_examples
        self.save_state()

        return {
            "new_examples": len(new_examples),
            "total_examples": total_examples,
            "batch_path": batch_path,
            "cursor": str(self.state.get("gliner_training_cursor", "")),
            **feedback_stats,
        }

    def load_existing_gliner_episode_ids(self, training_dir: Path) -> set[str]:
        seen: set[str] = set()
        for path in sorted(training_dir.glob("*.jsonl")):
            try:
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(payload, dict):
                            continue
                        episode_id = str(payload.get("episode_id") or "").strip()
                        if episode_id:
                            seen.add(episode_id)
            except Exception:
                log.debug("Failed scanning training file %s", path, exc_info=True)
        return seen

    def cleanup_stale_training_examples(self) -> dict[str, Any]:
        """Remove or strip training examples that contain quarantined relations.

        For each .jsonl file in the training directory:
        - Loads every example and checks its ``extracted_relations`` against
          Neo4j for any that now have ``audit_status IN ['quarantined', 'deleted']``.
        - Relations that have been quarantined are stripped from the example.
        - Examples that have *no* positive relations remaining (after stripping)
          are dropped entirely.
        - The file is atomically rewritten only when changes were made.

        Returns a summary dict with counts of files touched, examples removed,
        and relations stripped.
        """
        training_dir = self.gliner_training_dir()
        if not training_dir.exists():
            log.info("cleanup_stale_training_examples: training dir does not exist, nothing to do")
            return {
                "files_scanned": 0,
                "files_modified": 0,
                "examples_removed": 0,
                "relations_stripped": 0,
            }

        try:
            from runtime_graph import require_graph_instance
            driver = require_graph_instance().driver
        except Exception as exc:
            log.warning(
                "cleanup_stale_training_examples: cannot connect to graph, skipping: %s", exc
            )
            return {
                "files_scanned": 0,
                "files_modified": 0,
                "examples_removed": 0,
                "relations_stripped": 0,
                "skipped": True,
                "reason": str(exc),
            }

        files_scanned = 0
        files_modified = 0
        total_examples_removed = 0
        total_relations_stripped = 0

        jsonl_files = sorted(training_dir.glob("*.jsonl"))
        log.info(
            "cleanup_stale_training_examples: scanning %d .jsonl files in %s",
            len(jsonl_files),
            training_dir,
        )

        for path in jsonl_files:
            files_scanned += 1
            try:
                with path.open("r", encoding="utf-8") as f:
                    raw_lines = f.readlines()
            except Exception:
                log.warning("cleanup_stale_training_examples: failed to read %s", path, exc_info=True)
                continue

            examples: list[dict[str, Any]] = []
            for line in raw_lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    examples.append(payload)

            if not examples:
                continue

            # Collect all (head, tail, label) relation keys present in this file
            # so we can check Neo4j in a single batched query.
            relation_keys: list[tuple[str, str, str]] = []
            for ex in examples:
                for rel in (ex.get("extracted_relations") or []):
                    if not isinstance(rel, dict):
                        continue
                    head = str(rel.get("head") or "").strip()
                    tail = str(rel.get("tail") or "").strip()
                    label = str(rel.get("label") or "").strip()
                    if head and tail and label:
                        relation_keys.append((head, label, tail))

            if not relation_keys:
                continue

            # Query Neo4j for any of these relations that are now quarantined/deleted.
            quarantined_keys: set[tuple[str, str, str]] = set()
            try:
                with driver.session() as session:
                    # Build params as head_names + tail_names to limit the MATCH scope,
                    # then filter by relation type and audit_status in Python.
                    heads = list({h for h, _, _ in relation_keys})
                    tails = list({t for _, _, t in relation_keys})
                    records = session.run(
                        """
                        MATCH (h:Entity)-[r]->(t:Entity)
                        WHERE h.name IN $heads
                          AND t.name IN $tails
                          AND r.audit_status IN ['quarantined', 'deleted']
                        RETURN h.name AS head, t.name AS tail, type(r) AS label
                        """,
                        heads=heads,
                        tails=tails,
                    )
                    for record in records:
                        h = str(record.get("head") or "").strip()
                        t = str(record.get("tail") or "").strip()
                        lb = str(record.get("label") or "").strip()
                        if h and t and lb:
                            quarantined_keys.add((h, lb, t))
            except Exception:
                log.warning(
                    "cleanup_stale_training_examples: Neo4j query failed for %s, skipping file",
                    path,
                    exc_info=True,
                )
                continue

            if not quarantined_keys:
                continue

            # Now filter each example.
            file_examples_removed = 0
            file_relations_stripped = 0
            cleaned_examples: list[dict[str, Any]] = []

            for ex in examples:
                original_rels = ex.get("extracted_relations") or []
                kept_rels: list[dict[str, Any]] = []
                stripped = 0

                for rel in original_rels:
                    if not isinstance(rel, dict):
                        kept_rels.append(rel)
                        continue
                    head = str(rel.get("head") or "").strip()
                    tail = str(rel.get("tail") or "").strip()
                    label = str(rel.get("label") or "").strip()
                    if (head, label, tail) in quarantined_keys:
                        stripped += 1
                        log.debug(
                            "cleanup_stale_training_examples: stripping quarantined relation "
                            "(%s)-[%s]->(%s) from episode %s",
                            head, label, tail, ex.get("episode_id"),
                        )
                    else:
                        kept_rels.append(rel)

                if stripped > 0:
                    file_relations_stripped += stripped
                    if kept_rels:
                        # Keep the example but with stale relations removed.
                        ex = {**ex, "extracted_relations": kept_rels}
                        cleaned_examples.append(ex)
                    else:
                        # No positive relations remain; drop the whole example.
                        file_examples_removed += 1
                        log.info(
                            "cleanup_stale_training_examples: dropping example episode_id=%s "
                            "(all %d relations quarantined)",
                            ex.get("episode_id"), stripped,
                        )
                else:
                    cleaned_examples.append(ex)

            if file_examples_removed == 0 and file_relations_stripped == 0:
                continue

            # Atomically rewrite the file.
            total_examples_removed += file_examples_removed
            total_relations_stripped += file_relations_stripped
            files_modified += 1

            tmp_path = path.with_suffix(".tmp")
            try:
                with tmp_path.open("w", encoding="utf-8") as f:
                    for ex in cleaned_examples:
                        f.write(json.dumps(ex, ensure_ascii=True) + "\n")
                tmp_path.replace(path)
                log.info(
                    "cleanup_stale_training_examples: rewrote %s "
                    "(removed %d examples, stripped %d relations)",
                    path.name, file_examples_removed, file_relations_stripped,
                )
            except Exception:
                log.warning(
                    "cleanup_stale_training_examples: failed to rewrite %s", path, exc_info=True
                )
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

        log.info(
            "cleanup_stale_training_examples: done — files_scanned=%d files_modified=%d "
            "examples_removed=%d relations_stripped=%d",
            files_scanned, files_modified, total_examples_removed, total_relations_stripped,
        )
        return {
            "files_scanned": files_scanned,
            "files_modified": files_modified,
            "examples_removed": total_examples_removed,
            "relations_stripped": total_relations_stripped,
        }

    def latest_maintenance_analysis_text(self) -> str:
        maintenance_dir = config.MAINTENANCE_DIR
        if not maintenance_dir.exists():
            return ""

        candidates = sorted(maintenance_dir.glob("*.md"))
        if not candidates:
            return ""

        latest_path = candidates[-1]
        try:
            text = latest_path.read_text(encoding="utf-8")
        except Exception:
            return ""

        marker = "\n## Analysis\n"
        if marker in text:
            text = text.split(marker, 1)[1]

        return self.normalize_entity_text(text)

    def build_training_example_from_episode(
        self,
        session: Any,
        episode: dict[str, Any],
        opus_analysis_text: str,
    ) -> dict[str, Any] | None:
        source_text = str(episode.get("source_text") or "").strip()
        if not source_text:
            return None

        entity_names_raw = episode.get("entity_names") or []
        if not isinstance(entity_names_raw, list):
            return None

        entity_names = sorted({str(name).strip() for name in entity_names_raw if str(name).strip()})
        if not entity_names:
            return None

        entity_rows = [
            dict(record)
            for record in session.run(
                """
                MATCH (e:Entity)
                WHERE e.name IN $names
                RETURN e.name AS name,
                       e.entity_type AS entity_type,
                       coalesce(e.mention_count, 0) AS mention_count,
                       EXISTS { MATCH (e)--(:Entity) } AS has_relationship
                """,
                names=entity_names,
            )
        ]
        if not entity_rows:
            return None

        selected_entities: list[dict[str, Any]] = []
        signal_counts = {
            "opus_confirmed": 0,
            "multi_mentions": 0,
            "relationship_backed": 0,
        }

        for entity in entity_rows:
            name = str(entity.get("name") or "").strip()
            # Always normalize the entity type against the allowed set before
            # adding it to any training example.  This prevents wrong labels
            # (e.g. "Databricks" classified as "Person") from poisoning training.
            label = self._normalize_entity_type(entity.get("entity_type"))
            mention_count = int(entity.get("mention_count") or 0)
            has_relationship = bool(entity.get("has_relationship"))
            normalized_name = self.normalize_entity_text(name)

            if not normalized_name:
                continue

            opus_confirmed = bool(opus_analysis_text) and normalized_name in opus_analysis_text
            multi_mentions = mention_count >= 2
            relationship_backed = has_relationship

            if not (opus_confirmed or multi_mentions or relationship_backed):
                continue

            if opus_confirmed:
                signal_counts["opus_confirmed"] += 1
            if multi_mentions:
                signal_counts["multi_mentions"] += 1
            if relationship_backed:
                signal_counts["relationship_backed"] += 1

            selected_entities.append({"text": name, "label": label})

        if not selected_entities:
            return None

        selected_names = sorted({str(ent["text"]).strip() for ent in selected_entities})
        episode_id = str(episode.get("episode_id") or "").strip()
        # Filter strictly by episode_id to prevent label bleed from other
        # episodes whose entity names happen to overlap with this episode.
        relation_rows = [
            dict(record)
            for record in session.run(
                """
                MATCH (h:Entity)-[r]->(t:Entity)
                WHERE h.name IN $names AND t.name IN $names
                  AND r.episode_id = $episode_id
                  AND (r.audit_status IS NULL OR NOT r.audit_status IN ['quarantined', 'deleted'])
                RETURN h.name AS head,
                       t.name AS tail,
                       type(r) AS label,
                       coalesce(r.mention_count, 0) AS mention_count
                """,
                names=selected_names,
                episode_id=episode_id,
            )
        ]

        relations: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str, str]] = set()
        for rel in relation_rows:
            head = str(rel.get("head") or "").strip()
            tail = str(rel.get("tail") or "").strip()
            label = str(rel.get("label") or "").strip()
            if not head or not tail or not label:
                continue
            key = (head, label, tail)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            relations.append({"head": head, "tail": tail, "label": label})

        # ── Hard negative relations (quarantined / deleted audit_status) ───
        # Filter strictly by episode_id to prevent negatives from bleeding in
        # from other episodes that share entity names.
        negative_relation_rows = [
            dict(record)
            for record in session.run(
                """
                MATCH (h:Entity)-[r]->(t:Entity)
                WHERE h.name IN $names AND t.name IN $names
                  AND r.episode_id = $episode_id
                  AND r.audit_status IN ['quarantined', 'deleted']
                RETURN h.name AS head,
                       t.name AS tail,
                       type(r) AS label,
                       r.audit_status AS audit_status
                """,
                names=selected_names,
                episode_id=episode_id,
            )
        ]

        negative_relations: list[dict[str, Any]] = []
        seen_neg_keys: set[tuple[str, str, str]] = set()
        for rel in negative_relation_rows:
            head = str(rel.get("head") or "").strip()
            tail = str(rel.get("tail") or "").strip()
            label = str(rel.get("label") or "").strip()
            audit_status = str(rel.get("audit_status") or "").strip()
            if not head or not tail or not label:
                continue
            key = (head, label, tail)
            if key in seen_neg_keys:
                continue
            seen_neg_keys.add(key)
            negative_relations.append({
                "head": head,
                "tail": tail,
                "label": label,
                "reason": audit_status,
            })

        # ── Also pull negatives from audit_signals.jsonl ───────────────────
        try:
            from pathlib import Path as _Path
            signals_path = _Path.home() / ".graph-memory" / "training" / "audit_signals.jsonl"
            if signals_path.exists():
                with signals_path.open("r", encoding="utf-8") as _f:
                    for _line in _f:
                        _line = _line.strip()
                        if not _line:
                            continue
                        try:
                            _sig = json.loads(_line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(_sig, dict):
                            continue
                        if _sig.get("signal") != "negative":
                            continue
                        _head = str(_sig.get("head") or "").strip()
                        _tail = str(_sig.get("tail") or "").strip()
                        _label = str(_sig.get("relation") or "").strip()
                        if _head not in selected_names or _tail not in selected_names:
                            continue
                        _key = (_head, _label, _tail)
                        if _key in seen_neg_keys:
                            continue
                        seen_neg_keys.add(_key)
                        neg_entry: dict[str, Any] = {
                            "head": _head,
                            "tail": _tail,
                            "label": _label,
                            "reason": str(_sig.get("source") or "audit"),
                        }
                        if _sig.get("corrected_to"):
                            neg_entry["corrected_to"] = str(_sig["corrected_to"])
                        negative_relations.append(neg_entry)
        except Exception:
            log.debug("Failed loading audit signals for negatives", exc_info=True)

        return {
            "episode_id": str(episode.get("episode_id") or ""),
            "created_at": str(episode.get("created_at") or ""),
            "source_text": source_text,
            "extracted_entities": selected_entities,
            "extracted_relations": relations,
            "negative_relations": negative_relations,
            "quality_signals": signal_counts,
        }

    @staticmethod
    def _normalize_entity_type(label: Any) -> str:
        value = str(label or "Concept").strip()
        if value in _ALLOWED_ENTITY_TYPES:
            return value
        return "Concept"

    def _build_feedback_training_example(self, row: dict[str, Any]) -> dict[str, Any] | None:
        feedback_id = str(row.get("feedback_id") or "").strip()
        if not feedback_id:
            return None

        episode_id = f"audit-feedback:{feedback_id}"
        decision = str(row.get("decision") or "").strip().lower()
        if decision not in {"verify", "reclassify", "quarantine", "delete"}:
            return None

        head = str(row.get("head") or "").strip()
        tail = str(row.get("tail") or "").strip()
        if not head or not tail:
            return None

        source_text = " ".join(str(row.get("source_text") or "").split()).strip()
        rel_type = str(row.get("rel_type") or "").strip()
        corrected_rel_type = str(row.get("corrected_rel_type") or rel_type).strip()
        if not source_text:
            relation_phrase = corrected_rel_type or rel_type or "related to"
            source_text = f"{head} {relation_phrase.replace('_', ' ').lower()} {tail}"

        # Ensure entity names are present in text so GLiNER trainers can align spans.
        text_norm = self.normalize_entity_text(source_text)
        missing_entities: list[str] = []
        if self.normalize_entity_text(head) not in text_norm:
            missing_entities.append(head)
        if self.normalize_entity_text(tail) not in text_norm:
            missing_entities.append(tail)
        if missing_entities:
            source_text = f"{source_text} {' '.join(missing_entities)}".strip()

        head_type = self._normalize_entity_type(row.get("head_type"))
        tail_type = self._normalize_entity_type(row.get("tail_type"))
        entities = [
            {"text": head, "label": head_type},
            {"text": tail, "label": tail_type},
        ]

        relations: list[dict[str, Any]] = []
        hard_negative_relations: list[dict[str, Any]] = []

        if decision == "verify":
            if rel_type:
                relations.append({"head": head, "tail": tail, "label": rel_type})
        elif decision == "reclassify":
            if rel_type:
                hard_negative_relations.append({"head": head, "tail": tail, "label": rel_type})
            if corrected_rel_type and corrected_rel_type != rel_type:
                relations.append({"head": head, "tail": tail, "label": corrected_rel_type})
        elif decision in {"quarantine", "delete"}:
            if rel_type:
                hard_negative_relations.append({"head": head, "tail": tail, "label": rel_type})

        quality_signals = {
            "audit_feedback": 1,
            "audit_positive": 1 if relations else 0,
            "audit_negative": 1 if hard_negative_relations else 0,
        }

        return {
            "episode_id": episode_id,
            "created_at": str(row.get("created_at") or ""),
            "source_text": source_text,
            "extracted_entities": entities,
            "extracted_relations": relations,
            "hard_negative_relations": hard_negative_relations,
            "quality_signals": quality_signals,
        }

    def build_training_examples_from_feedback(
        self,
        seen_episode_ids: set[str],
        limit: int = GLINER_TRAINING_SCAN_LIMIT,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        safe_limit = max(1, int(limit))
        # Load a larger tail to account for dedupe against seen episode_ids.
        feedback_rows = load_audit_feedback_entries(limit=safe_limit * 4)

        out: list[dict[str, Any]] = []
        stats = {
            "feedback_rows_scanned": len(feedback_rows),
            "feedback_examples_added": 0,
            "feedback_positive_examples": 0,
            "feedback_negative_examples": 0,
        }

        for row in feedback_rows:
            example = self._build_feedback_training_example(row)
            if not example:
                continue

            episode_id = str(example.get("episode_id") or "").strip()
            if not episode_id or episode_id in seen_episode_ids:
                continue

            out.append(example)
            seen_episode_ids.add(episode_id)
            stats["feedback_examples_added"] += 1
            if (example.get("extracted_relations") or []):
                stats["feedback_positive_examples"] += 1
            if (example.get("hard_negative_relations") or []):
                stats["feedback_negative_examples"] += 1

            if len(out) >= safe_limit:
                break

        return out, stats

    def count_accumulated_gliner_examples(self, training_dir: Path | None = None) -> int:
        target_dir = training_dir or self.gliner_training_dir()
        if not target_dir.exists():
            return 0

        total = 0
        for path in target_dir.glob("*.jsonl"):
            try:
                with path.open("r", encoding="utf-8") as f:
                    total += sum(1 for line in f if line.strip())
            except Exception:
                log.debug("Failed counting examples in %s", path, exc_info=True)

        return total

    def archive_old_training_examples(self) -> dict[str, Any]:
        """Compress training example files older than GLINER_ARCHIVE_DAYS days to
        ~/.graph-memory/training/archive/ as .jsonl.gz files.

        Originals are left on disk (the sliding-window loader skips them in practice
        once the archive passes 10 K examples).  Archival is idempotent: files that
        have already been archived are skipped.
        """
        import gzip

        training_dir = self.gliner_training_dir()
        archive_dir = config.GRAPH_MEMORY_DIR / "training" / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)

        cutoff = datetime.now(timezone.utc) - timedelta(days=int(config.GLINER_ARCHIVE_DAYS))
        archived = 0
        skipped = 0
        errors = 0

        for path in sorted(training_dir.glob("*.jsonl")):
            try:
                mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            except Exception:
                continue
            if mtime >= cutoff:
                continue  # file is recent enough, leave it alone

            archive_path = archive_dir / (path.name + ".gz")
            if archive_path.exists():
                skipped += 1
                continue  # already archived

            try:
                with path.open("rb") as f_in, gzip.open(archive_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                archived += 1
                log.info(
                    "archive_old_training_examples: archived %s → %s",
                    path.name,
                    archive_path,
                )
            except Exception:
                log.warning(
                    "archive_old_training_examples: failed to archive %s", path, exc_info=True
                )
                errors += 1

        log.info(
            "archive_old_training_examples: archived=%d skipped=%d errors=%d",
            archived,
            skipped,
            errors,
        )
        return {"archived": archived, "skipped": skipped, "errors": errors}

    def load_accumulated_gliner_examples(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        training_dir = self.gliner_training_dir()
        if not training_dir.exists():
            return rows

        # Files are sorted oldest-first by filename timestamp; examples are appended
        # in the same order so rows[-max:] yields the most recent window.
        for path in sorted(training_dir.glob("*.jsonl")):
            try:
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(payload, dict):
                            continue
                        row = self.to_benchmark_row(payload)
                        if row:
                            rows.append(row)
            except Exception:
                log.debug("Failed loading examples from %s", path, exc_info=True)

        # Sliding window: cap at GLINER_MAX_TRAINING_EXAMPLES most recent examples.
        # Older examples remain on disk but are not loaded for training.
        max_examples = int(config.GLINER_MAX_TRAINING_EXAMPLES)
        if len(rows) > max_examples:
            log.info(
                "Sliding window: %d total examples on disk; using most recent %d for training",
                len(rows),
                max_examples,
            )
            rows = rows[-max_examples:]

        return rows

    @staticmethod
    def to_benchmark_row(payload: dict[str, Any]) -> dict[str, Any] | None:
        text = str(payload.get("source_text") or payload.get("text") or "").strip()
        entities = payload.get("extracted_entities", payload.get("entities", []))
        relations = payload.get("extracted_relations", payload.get("relations", []))
        negative_relations = payload.get("negative_relations", payload.get("hard_negative_relations", []))

        if not text or not isinstance(entities, list) or not entities:
            return None

        return {
            "episode_id": str(payload.get("episode_id") or ""),
            "created_at": str(payload.get("created_at") or ""),
            "text": text,
            "entities": entities,
            "relations": relations if isinstance(relations, list) else [],
            "negative_relations": negative_relations if isinstance(negative_relations, list) else [],
        }

    async def _run_pretraining_audit(self, mode: str) -> dict[str, Any]:
        """Run Kimi pre-training audit to gate GLiNER training.
        
        Kimi reviews training data quality before allowing LoRA or full training.
        """
        try:
            from audit.llm_audit import call_audit_model, build_audit_prompt
            from runtime_graph import require_graph_instance
            
            # Sample recent relationships for quality check.
            # Keep sample modest to avoid JSON truncation/parse failures.
            rels = require_graph_instance().get_relationships_for_audit(limit=25)
            if not rels:
                return {"passed": True, "reason": "no_relationships_to_audit"}

            # Call pretrain audit model with retry + fallback on parse failure.
            audit_model = getattr(config, "AUDIT_MODEL_PRETRAIN", "kimi-k2.5")
            fallback_model = getattr(config, "AUDIT_MODEL_PRETRAIN_FALLBACK", "gemini-2.5-flash")

            from audit.llm_audit import parse_verdicts
            verdicts: list[dict[str, Any]] = []
            provider = "unknown"
            fallback = ""

            async def _invoke_and_parse(model_name: str, sample: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str, str]:
                prompt = build_audit_prompt(sample)
                try:
                    llm_result = await call_audit_model(prompt, schedule="nightly", model_override=model_name)
                    content = str(llm_result.get("content") or "")
                    prov = str(llm_result.get("provider") or "unknown")
                    fb = str(llm_result.get("fallback") or "")
                    return parse_verdicts(content, len(sample)), prov, fb
                except Exception as _exc:
                    log.warning("Pre-training audit model invocation failed for %s: %s", model_name, _exc)
                    return [], str(model_name or "unknown"), ""

            # Attempt 1: primary model on full sample.
            verdicts, provider, fallback = await _invoke_and_parse(audit_model, rels)

            # Attempt 2: primary model on smaller sample (reduce truncation risk).
            if not verdicts and len(rels) > 15:
                smaller = rels[:15]
                verdicts, provider, fallback = await _invoke_and_parse(audit_model, smaller)
                rels = smaller if verdicts else rels

            # Attempt 3: fallback model if primary still unparsable.
            if not verdicts and fallback_model and fallback_model != audit_model:
                verdicts, provider, fallback = await _invoke_and_parse(fallback_model, rels)

            if not verdicts:
                # Deterministic fallback: avoid blocking LoRA solely on parser/model formatting failures.
                rel_types = [str(r.get("rel_type") or r.get("type") or "").upper() for r in rels]
                related_to = sum(1 for t in rel_types if t == "RELATED_TO")
                related_to_ratio = (related_to / len(rel_types)) if rel_types else 0.0
                passed = related_to_ratio <= 0.20
                return {
                    "passed": passed,
                    "reason": (
                        f"audit_parse_failed_fallback_related_to_ratio={related_to_ratio:.2f}"
                        if passed
                        else f"audit_parse_failed_fallback_blocked_related_to_ratio={related_to_ratio:.2f}"
                    ),
                    "provider": provider,
                    "fallback": fallback,
                    "fallback_mode": "deterministic",
                    "total_audited": len(rel_types),
                    "related_to_ratio": related_to_ratio,
                }
            
            # Calculate quality metrics
            total = len(verdicts)
            verify_count = sum(1 for v in verdicts if v.get("verdict") == "verify")
            delete_count = sum(1 for v in verdicts if v.get("verdict") == "delete")
            quarantine_count = sum(1 for v in verdicts if v.get("verdict") == "quarantine")
            
            verify_ratio = verify_count / total if total > 0 else 0
            delete_ratio = delete_count / total if total > 0 else 0
            
            # Quality gates
            passed = verify_ratio >= 0.6 and delete_ratio <= 0.15
            
            return {
                "passed": passed,
                "reason": f"verify_ratio={verify_ratio:.2f}, delete_ratio={delete_ratio:.2f}" if not passed else "quality_acceptable",
                "provider": provider,
                "fallback": fallback,
                "total_audited": total,
                "verify_count": verify_count,
                "delete_count": delete_count,
                "quarantine_count": quarantine_count,
            }
        except Exception as exc:
            log.error("Pre-training audit failed — BLOCKING training to protect data quality: %s", exc, exc_info=True)
            return {"passed": False, "reason": f"audit_error_blocking_train: {exc}"}

    def to_gliner_training_record(self, row: dict[str, Any]) -> dict[str, Any] | None:
        text = str(row.get("text") or "").strip()
        if not text:
            return None

        text_norm = self.normalize_entity_text(text)

        entities_map: dict[str, list[str]] = {}
        entity_names: set[str] = set()

        for entity in row.get("entities") or []:
            if isinstance(entity, str):
                name = entity.strip()
                label = "Concept"
            elif isinstance(entity, dict):
                name = str(entity.get("text") or entity.get("name") or "").strip()
                label = str(entity.get("label") or entity.get("type") or "Concept").strip() or "Concept"
            else:
                continue

            if not name:
                continue
            if self.normalize_entity_text(name) not in text_norm:
                continue

            entities_map.setdefault(label, [])
            if name not in entities_map[label]:
                entities_map[label].append(name)
                entity_names.add(name)

        # ── Derive allowed relation types from schema (cached lazily) ────────
        global _ALLOWED_REL_TYPES
        if not _ALLOWED_REL_TYPES:
            _ALLOWED_REL_TYPES = _get_allowed_rel_types()
        # Empty frozenset means "allow all"
        _check_rel = bool(_ALLOWED_REL_TYPES)

        relations_out: list[dict[str, Any]] = []
        seen_rel_keys: set[tuple[str, str, str]] = set()
        for rel in row.get("relations") or []:
            if not isinstance(rel, dict):
                continue
            head = str(rel.get("head") or "").strip()
            tail = str(rel.get("tail") or "").strip()
            label = str(rel.get("label") or "").strip().lower().replace("_", " ")

            if not head or not tail or not label:
                continue
            if head not in entity_names or tail not in entity_names:
                continue
            if self.normalize_entity_text(head) not in text_norm:
                continue
            if self.normalize_entity_text(tail) not in text_norm:
                continue
            # Filter by allowed relation types
            if _check_rel and label not in _ALLOWED_REL_TYPES:
                continue

            key = (head, label, tail)
            if key in seen_rel_keys:
                continue
            seen_rel_keys.add(key)
            relations_out.append({label: {"head": head, "tail": tail}})

        # ── Hard negatives from quarantine/delete/reclassify audit verdicts ─
        # reclassify → the original label is hard-negative + corrected label becomes positive
        # quarantine / delete → the label is an omission signal (hard negative only)
        negative_relations_out: list[dict[str, Any]] = []
        seen_neg_keys: set[tuple[str, str, str]] = set()
        for rel in row.get("negative_relations") or row.get("hard_negative_relations") or []:
            if not isinstance(rel, dict):
                continue
            head = str(rel.get("head") or "").strip()
            tail = str(rel.get("tail") or "").strip()
            label = str(rel.get("label") or "").strip().lower().replace("_", " ")

            if not head or not tail or not label:
                continue
            # Negatives don't need to appear in text - they are hard negatives from audit

            neg_key = (head, label, tail)
            if neg_key not in seen_neg_keys:
                seen_neg_keys.add(neg_key)
                entry: dict[str, Any] = {label: {"head": head, "tail": tail}}
                corrected_to = str(rel.get("corrected_to") or "").strip().lower().replace("_", " ")
                if corrected_to:
                    entry[label]["corrected_to"] = corrected_to
                negative_relations_out.append(entry)

            # reclassify signal: also add corrected label as a positive relation
            corrected_to = str(rel.get("corrected_to") or "").strip().lower().replace("_", " ")
            if corrected_to and corrected_to != label:
                if not _check_rel or corrected_to in _ALLOWED_REL_TYPES:
                    pos_key = (head, corrected_to, tail)
                    if pos_key not in seen_rel_keys:
                        seen_rel_keys.add(pos_key)
                        relations_out.append({corrected_to: {"head": head, "tail": tail}})

        output: dict[str, Any] = {}
        if entities_map:
            output["entities"] = entities_map
        if relations_out:
            output["relations"] = relations_out
        if negative_relations_out:
            output["negative_relations"] = negative_relations_out
        if not output:
            return None

        return {"input": text, "output": output}

    def active_gliner_model_ref(self) -> str:
        active_dir = self.gliner_active_model_dir()
        if active_dir.exists():
            return str(active_dir)

        active_state_ref = str(self.state.get("gliner_active_model_ref", "")).strip()
        if active_state_ref and Path(active_state_ref).exists():
            return active_state_ref

        return GLINER_BASE_MODEL

    def select_gliner_training_strategy(self, total_examples: int) -> dict[str, Any]:
        full_min_examples = max(1, int(config.GLINER_FULL_FINETUNE_MIN_EXAMPLES))
        plateau_window = max(1, int(config.GLINER_LORA_PLATEAU_WINDOW))
        plateau_epsilon = max(0.0, float(config.GLINER_LORA_PLATEAU_EPSILON))

        if total_examples < full_min_examples:
            return {
                "mode": "lora",
                "reason": "insufficient_examples_for_full_finetune",
                "full_min_examples": full_min_examples,
                "total_examples": total_examples,
            }

        history = self.state.get("gliner_benchmark_history")
        if not isinstance(history, list):
            history = []

        recent_lora = [
            row
            for row in history
            if isinstance(row, dict)
            and str(row.get("strategy", "")).lower() == "lora"
            and bool(row.get("benchmark_ok"))
        ][-plateau_window:]

        if len(recent_lora) < plateau_window:
            return {
                "mode": "lora",
                "reason": "not_enough_lora_history",
                "required_runs": plateau_window,
                "available_runs": len(recent_lora),
                "total_examples": total_examples,
            }

        improvements = [float(row.get("improvement", 0.0) or 0.0) for row in recent_lora]
        max_gain = max(improvements) if improvements else 0.0
        plateaued = max_gain <= plateau_epsilon

        if plateaued:
            return {
                "mode": "full",
                "reason": "lora_plateau_detected",
                "plateau_window": plateau_window,
                "plateau_epsilon": plateau_epsilon,
                "recent_improvements": improvements,
                "total_examples": total_examples,
            }

        return {
            "mode": "lora",
            "reason": "lora_still_improving",
            "plateau_window": plateau_window,
            "plateau_epsilon": plateau_epsilon,
            "recent_improvements": improvements,
            "total_examples": total_examples,
        }

    def record_gliner_benchmark(
        self,
        strategy: str,
        benchmark: dict[str, Any],
        status: str,
        total_examples: int,
    ) -> None:
        history = self.state.get("gliner_benchmark_history")
        if not isinstance(history, list):
            history = []

        history.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "strategy": strategy,
                "status": status,
                "total_examples": int(total_examples),
                "benchmark_ok": bool(benchmark.get("ok", False)),
                # Legacy key
                "improvement": float(benchmark.get("improvement", 0.0) or 0.0),
                "base_score": float(benchmark.get("base_score", 0.0) or 0.0),
                "candidate_score": float(benchmark.get("candidate_score", 0.0) or 0.0),
                "eval_count": int(benchmark.get("split", {}).get("eval_count", 0) or 0),
                # Extended scoring keys
                "entity_improvement": float(benchmark.get("entity_improvement", 0.0) or 0.0),
                "relation_improvement": float(benchmark.get("relation_improvement", 0.0) or 0.0),
                "combined_improvement": float(benchmark.get("combined_improvement", 0.0) or 0.0),
                "base_entity_f1": float(benchmark.get("base_entity_f1", benchmark.get("base_score", 0.0)) or 0.0),
                "candidate_entity_f1": float(benchmark.get("candidate_entity_f1", benchmark.get("candidate_score", 0.0)) or 0.0),
                "base_relation_f1": float(benchmark.get("base_relation_f1", 0.0) or 0.0),
                "candidate_relation_f1": float(benchmark.get("candidate_relation_f1", 0.0) or 0.0),
            }
        )

        self.state["gliner_benchmark_history"] = history[-20:]
        self.save_state()

    def fine_tune_gliner_candidate(self, train_rows: list[dict[str, Any]], mode: str = "lora") -> dict[str, Any]:
        train_records = [record for record in (self.to_gliner_training_record(r) for r in train_rows) if record]
        if not train_records:
            return {"ok": False, "error": "no_valid_train_records"}

        # Log pre-training data stats
        _ent_types: dict[str, int] = {}
        _rel_types: dict[str, int] = {}
        _neg_types: dict[str, int] = {}
        for _rec in train_records:
            _out = _rec.get("output") or {}
            for _lbl, _ents in (_out.get("entities") or {}).items():
                _ent_types[_lbl] = _ent_types.get(_lbl, 0) + len(_ents)
            for _rel in (_out.get("relations") or []):
                for _lbl in _rel:
                    _rel_types[_lbl] = _rel_types.get(_lbl, 0) + 1
            for _neg in (_out.get("negative_relations") or []):
                for _lbl in _neg:
                    _neg_types[_lbl] = _neg_types.get(_lbl, 0) + 1
        log.info(
            "fine_tune_gliner_candidate: %d records  entity_types=%s  rel_types=%s  neg_types=%s",
            len(train_records),
            dict(sorted(_ent_types.items())),
            dict(sorted(_rel_types.items())),
            dict(sorted(_neg_types.items())),
        )

        try:
            from gliner2.training.trainer import train_gliner2
        except Exception as exc:
            return {"ok": False, "error": f"training_import_failed: {exc}"}

        models_dir = self.gliner_models_dir()
        candidate_dir = self.gliner_candidate_model_dir()
        runs_dir = models_dir / "gliner_runs"
        splits_dir = self.gliner_training_dir() / "splits"

        models_dir.mkdir(parents=True, exist_ok=True)
        runs_dir.mkdir(parents=True, exist_ok=True)
        splits_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        train_split_path = splits_dir / f"train-{ts}.jsonl"
        with train_split_path.open("w", encoding="utf-8") as f:
            for record in train_records:
                f.write(json.dumps(record, ensure_ascii=True) + "\n")

        # Transient training output dir — deleted after candidate is populated.
        # gliner_runs/<ts>/ receives ONLY KB-sized metadata (no model weights).
        tmp_train_dir = models_dir / f".training-{ts}"
        tmp_train_dir.mkdir(parents=True, exist_ok=True)

        batch_size = min(4, max(1, len(train_records)))
        normalized_mode = "full" if str(mode).strip().lower() == "full" else "lora"
        use_lora = normalized_mode == "lora"
        num_epochs = 1 if use_lora else 2

        _train_kwargs = dict(
            model_path=self.active_gliner_model_ref(),
            train_data=train_records,
            output_dir=str(tmp_train_dir),
            num_epochs=num_epochs,
            batch_size=batch_size,
            eval_strategy="no",
            fp16=False,
            bf16=False,
            num_workers=0,
            logging_steps=max(1, min(25, len(train_records))),
            use_lora=use_lora,
            save_adapter_only=False,
            seed=GLINER_BENCHMARK_SEED,
        )
        try:
            result = train_gliner2(**_train_kwargs)
        except TypeError as type_exc:
            # Older train_gliner2 versions may not accept all kwargs; retry
            # with minimal required args only.
            log.warning(
                "train_gliner2 TypeError (likely unsupported kwargs), retrying with minimal signature: %s",
                type_exc,
            )
            _minimal_kwargs = {
                k: v for k, v in _train_kwargs.items()
                if k in {"model_path", "train_data", "output_dir", "num_epochs", "batch_size", "use_lora", "seed"}
            }
            try:
                result = train_gliner2(**_minimal_kwargs)
            except Exception as exc2:
                log.error("GLiNER fine-tune failed (minimal signature)", exc_info=True)
                shutil.rmtree(tmp_train_dir, ignore_errors=True)
                return {"ok": False, "error": str(exc2)}
        except Exception as exc:
            log.error("GLiNER fine-tune failed", exc_info=True)
            shutil.rmtree(tmp_train_dir, ignore_errors=True)
            return {"ok": False, "error": str(exc)}

        # Locate trained model checkpoint inside tmp dir.
        final_dir = tmp_train_dir / "final"
        if not final_dir.exists():
            best_dir = tmp_train_dir / "best"
            if best_dir.exists():
                final_dir = best_dir
            else:
                shutil.rmtree(tmp_train_dir, ignore_errors=True)
                return {
                    "ok": False,
                    "error": "trained_model_not_found",
                    "output_dir": str(tmp_train_dir),
                }

        # Populate gliner_candidate from the trained checkpoint.
        self.discard_gliner_candidate_model(candidate_dir)
        try:
            final_dir.rename(candidate_dir)  # atomic on same filesystem
        except OSError:
            shutil.copytree(final_dir, candidate_dir, dirs_exist_ok=False)

        # Delete the transient training dir now that candidate is populated.
        shutil.rmtree(tmp_train_dir, ignore_errors=True)

        # Write KB-sized metadata to gliner_runs/<ts>/ — no model weights here.
        run_meta_dir = runs_dir / ts
        run_meta_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "base_model": self.active_gliner_model_ref(),
            "train_examples": len(train_records),
            "batch_size": batch_size,
            "mode": normalized_mode,
            "num_epochs": num_epochs,
            "candidate_dir": str(candidate_dir),
            "result": result,
        }
        metadata_path = run_meta_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

        # Also write a copy into candidate for introspection.
        try:
            (candidate_dir / "fine_tune_metadata.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
            )
        except Exception:
            pass  # non-fatal

        self.prune_gliner_dirs()

        return {
            "ok": True,
            "candidate_model": str(candidate_dir),
            "train_split_path": str(train_split_path),
            "output_dir": str(run_meta_dir),
            "metadata_path": str(metadata_path),
            "mode": normalized_mode,
            "result": result,
        }

    def prune_gliner_dirs(self) -> dict[str, Any]:
        """Delete any model directory in models_dir that isn't gliner_active or gliner_candidate.

        gliner_runs/ is kept because it stores KB-sized metadata; its subdirectory contents
        are NOT pruned here.  Dotfiles/temp dirs (e.g. .training-*, .deploy-*) are skipped
        so concurrent operations are not disrupted.
        """
        pruned: list[str] = []
        models_dir = self.gliner_models_dir()
        if not models_dir.exists():
            return {"pruned": pruned}

        # Only these dirs are permitted to exist in models_dir.
        keep = {"gliner_active", "gliner_candidate", "gliner_runs"}

        for item in models_dir.iterdir():
            if item.name.startswith("."):
                continue  # skip hidden/temp dirs
            if not item.is_dir():
                continue
            if item.name in keep:
                continue
            try:
                shutil.rmtree(item, ignore_errors=True)
                pruned.append(item.name)
                log.info("prune_gliner_dirs: deleted stale model dir %s", item)
            except Exception:
                log.warning("prune_gliner_dirs: failed to delete %s", item, exc_info=True)

        return {"pruned": pruned}

    @staticmethod
    def discard_gliner_candidate_model(candidate_path: Path | None) -> None:
        if candidate_path and candidate_path.exists():
            shutil.rmtree(candidate_path, ignore_errors=True)

    def deploy_gliner_candidate_model(
        self,
        candidate_path: Path,
        benchmark: dict[str, Any],
        fine_tune: dict[str, Any],
    ) -> dict[str, Any]:
        """Deploy gliner_candidate → gliner_active with minimal disk footprint.

        Lifecycle (active+candidate only — no backups, no previous):
          1. Delete gliner_active if it exists (frees disk immediately).
          2. Rename gliner_candidate → gliner_active (atomic on same filesystem).

        gliner_candidate ceases to exist after this call succeeds.
        """
        if not candidate_path.exists():
            return {"ok": False, "error": "candidate_path_missing"}

        active_dir = self.gliner_active_model_dir()
        active_dir.parent.mkdir(parents=True, exist_ok=True)
        previous_active_ref = self.active_gliner_model_ref()

        # Step 1: Delete existing active model to free disk before rename.
        if active_dir.exists():
            shutil.rmtree(active_dir, ignore_errors=True)

        # Step 2: Rename candidate → active (atomic on same filesystem, zero extra disk).
        try:
            candidate_path.rename(active_dir)
        except OSError as rename_exc:
            # Cross-device rename: fall back to copy + delete.
            log.warning(
                "deploy: rename failed (%s), falling back to copy+delete", rename_exc
            )
            shutil.copytree(candidate_path, active_dir, dirs_exist_ok=False)
            shutil.rmtree(candidate_path, ignore_errors=True)

        deployed_at = datetime.now(timezone.utc).isoformat()
        self.state["gliner_active_model_ref"] = str(active_dir)
        self.state["gliner_last_deployed_at"] = deployed_at
        self.save_state()

        config_payload = {
            "updated_at": deployed_at,
            "active_model_ref": str(active_dir),
            "previous_model_ref": previous_active_ref,
            "benchmark": benchmark,
            "fine_tune": {
                "mode": fine_tune.get("mode", "lora"),
                "train_split_path": fine_tune.get("train_split_path"),
                "output_dir": fine_tune.get("output_dir"),
                "metadata_path": fine_tune.get("metadata_path"),
            },
        }

        config_path = self.gliner_training_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config_payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

        # ── Notify model health monitor of the new baseline ───────────────────
        try:
            from metrics.model_health import model_health_monitor

            base_eval = benchmark.get("base") or {}
            # Compute pre-deploy fallback rate from base benchmark: (FP+FN) / total
            tp = int(base_eval.get("counts", {}).get("tp", 0) or 0)
            fp = int(base_eval.get("counts", {}).get("fp", 0) or 0)
            fn = int(base_eval.get("counts", {}).get("fn", 0) or 0)
            total_ops = tp + fp + fn
            baseline_fallback_rate = (fp + fn) / max(total_ops, 1)
            model_health_monitor.set_baseline(baseline_fallback_rate, str(active_dir))
            log.info(
                "Model health monitor notified: baseline_fallback_rate=%.4f model=%s",
                baseline_fallback_rate, str(active_dir),
            )
        except Exception:
            log.warning("Failed to notify model health monitor after deploy", exc_info=True)

        pruned = self.prune_gliner_dirs()

        return {
            "ok": True,
            "active_model": str(active_dir),
            "training_config": str(config_path),
            "pruned": pruned,
        }

    @staticmethod
    def split_holdout_rows(
        rows: list[dict[str, Any]],
        eval_ratio: float = GLINER_BENCHMARK_EVAL_RATIO,
        seed: int = GLINER_BENCHMARK_SEED,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not rows:
            return [], []

        indices = list(range(len(rows)))
        random.Random(seed).shuffle(indices)

        if len(rows) == 1:
            eval_count = 1
        else:
            eval_count = max(1, int(round(len(rows) * eval_ratio)))
            eval_count = min(eval_count, len(rows) - 1)

        eval_idx = set(indices[:eval_count])
        train_rows = [rows[i] for i in range(len(rows)) if i not in eval_idx]
        eval_rows = [rows[i] for i in range(len(rows)) if i in eval_idx]

        return train_rows, eval_rows

    @staticmethod
    def normalize_entity_text(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip().lower()
        return re.sub(r"\s+", " ", text)

    def extract_expected_entity_set(self, row: dict[str, Any]) -> set[str]:
        entities = row.get("entities") or []
        if not isinstance(entities, list):
            return set()

        normalized: set[str] = set()
        for entity in entities:
            if isinstance(entity, str):
                name = self.normalize_entity_text(entity)
            elif isinstance(entity, dict):
                name = self.normalize_entity_text(
                    entity.get("text") or entity.get("name") or entity.get("entity")
                )
            else:
                name = ""

            if name:
                normalized.add(name)

        return normalized

    @staticmethod
    def extract_predicted_entity_set(result: Any) -> set[str]:
        if not isinstance(result, dict):
            return set()

        entity_dict = result.get("entities", result)
        if not isinstance(entity_dict, dict):
            return set()

        predicted: set[str] = set()
        for items in entity_dict.values():
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, str):
                    name = GLiNERTrainingService.normalize_entity_text(item)
                elif isinstance(item, dict):
                    name = GLiNERTrainingService.normalize_entity_text(item.get("text"))
                else:
                    name = ""
                if name:
                    predicted.add(name)

        return predicted

    @staticmethod
    def compute_prf_metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
        precision = (tp / (tp + fp)) if (tp + fp) else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    def load_gliner_entity_model(self, model_ref: str) -> tuple[Any, Any]:
        from gliner2 import GLiNER2
        from extractor_schema_registry import get_effective_extractor_schema

        model = GLiNER2.from_pretrained(model_ref)
        entity_schema = get_effective_extractor_schema().get("entities", {})
        schema = model.create_schema().entities(entity_schema)
        return model, schema

    def load_gliner_full_model(self, model_ref: str) -> tuple[Any, Any]:
        """Load GLiNER2 model with both entity and relation schemas."""
        from gliner2 import GLiNER2
        from extractor_schema_registry import get_effective_extractor_schema

        model = GLiNER2.from_pretrained(model_ref)
        schema_cfg = get_effective_extractor_schema()
        entity_schema = schema_cfg.get("entities", {})
        relation_schema = schema_cfg.get("relations", {})
        try:
            schema = model.create_schema().entities(entity_schema).relations(relation_schema)
        except Exception:
            log.debug("Relation schema not supported by model; using entity-only schema")
            schema = model.create_schema().entities(entity_schema)
        return model, schema

    @staticmethod
    def _normalize_relation_label(label: Any) -> str:
        """Normalize a relation label to a canonical lowercase string."""
        return str(label or "").strip().lower().replace("_", " ")

    @staticmethod
    def _extract_gold_relations(row: dict[str, Any]) -> set[tuple[str, str, str]]:
        """Extract (head, label, tail) tuples from a benchmark row."""
        gold: set[tuple[str, str, str]] = set()
        for rel in row.get("relations") or []:
            if not isinstance(rel, dict):
                continue
            head = str(rel.get("head") or "").strip().lower()
            tail = str(rel.get("tail") or "").strip().lower()
            label = GLiNERTrainingService._normalize_relation_label(rel.get("label"))
            if head and tail and label:
                gold.add((head, label, tail))
        return gold

    @staticmethod
    def _extract_predicted_relations(result: Any) -> set[tuple[str, str, str]]:
        """Extract (head, label, tail) tuples from a GLiNER2 extraction result."""
        if not isinstance(result, dict):
            return set()
        predicted: set[tuple[str, str, str]] = set()
        rel_dict = result.get("relation_extraction") or result.get("relations") or {}

        if isinstance(rel_dict, dict):
            for rtype, items in rel_dict.items():
                if not isinstance(items, list):
                    continue
                label = GLiNERTrainingService._normalize_relation_label(rtype)
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    head_raw = item.get("head") or {}
                    tail_raw = item.get("tail") or {}
                    head = str(head_raw.get("text") if isinstance(head_raw, dict) else head_raw).strip().lower()
                    tail = str(tail_raw.get("text") if isinstance(tail_raw, dict) else tail_raw).strip().lower()
                    if head and tail and label:
                        predicted.add((head, label, tail))
        elif isinstance(rel_dict, list):
            for item in rel_dict:
                if not isinstance(item, dict):
                    continue
                label = GLiNERTrainingService._normalize_relation_label(item.get("label"))
                head = str(item.get("head") or "").strip().lower()
                tail = str(item.get("tail") or "").strip().lower()
                if head and tail and label:
                    predicted.add((head, label, tail))

        return predicted

    @staticmethod
    def _extract_predicted_entities_with_confidence(
        result: Any,
    ) -> list[tuple[str, str, float]]:
        """Extract (text, label, confidence) tuples from a GLiNER2 result."""
        if not isinstance(result, dict):
            return []
        entity_dict = result.get("entities") or result
        if not isinstance(entity_dict, dict):
            return []
        out: list[tuple[str, str, float]] = []
        for etype, items in entity_dict.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict):
                    text = str(item.get("text") or "").strip()
                    conf = float(item.get("confidence", 0.5) or 0.5)
                    if text:
                        out.append((text, str(etype), conf))
        return out

    def evaluate_model_on_rows(
        self,
        model_ref: str,
        rows: list[dict[str, Any]],
        threshold: float = GLINER_BENCHMARK_THRESHOLD,
    ) -> dict[str, Any]:
        """Evaluate a GLiNER2 model on eval rows, scoring both entities and relations.

        Returns a dict containing:
        - ``entity_metrics``  — precision/recall/F1 for entities
        - ``relation_metrics`` — precision/recall/F1 for relations (all types)
        - ``per_type_relation_f1`` — per-relation-type F1 dict
        - ``metrics`` — alias for entity_metrics (backward-compat)
        - ``ok`` / ``error`` / ``counts`` / ``rows_evaluated`` / ``latency_ms_avg``
        """
        if not rows:
            _empty = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            return {
                "ok": False,
                "error": "empty_eval_set",
                "metrics": _empty,
                "entity_metrics": _empty,
                "relation_metrics": _empty,
                "per_type_relation_f1": {},
                "counts": {"tp": 0, "fp": 0, "fn": 0},
                "relation_counts": {"tp": 0, "fp": 0, "fn": 0},
                "rows_total": 0,
                "rows_evaluated": 0,
                "rows_failed": 0,
                "latency_ms_avg": 0.0,
            }

        try:
            model, schema = self.load_gliner_full_model(model_ref)
        except Exception as exc:
            _empty = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            return {
                "ok": False,
                "error": f"model_load_failed: {exc}",
                "metrics": _empty,
                "entity_metrics": _empty,
                "relation_metrics": _empty,
                "per_type_relation_f1": {},
                "counts": {"tp": 0, "fp": 0, "fn": 0},
                "relation_counts": {"tp": 0, "fp": 0, "fn": 0},
                "rows_total": len(rows),
                "rows_evaluated": 0,
                "rows_failed": len(rows),
                "latency_ms_avg": 0.0,
            }

        e_tp = e_fp = e_fn = 0
        r_tp: dict[str, int] = {}
        r_fp: dict[str, int] = {}
        r_fn: dict[str, int] = {}
        rows_evaluated = 0
        rows_failed = 0
        latency_sum_ms = 0.0
        failure_samples: list[dict[str, Any]] = []

        for idx, row in enumerate(rows):
            text = str(row.get("text") or "").strip()
            if not text:
                rows_failed += 1
                if len(failure_samples) < 5:
                    failure_samples.append({"row_index": idx, "error": "missing_text"})
                continue

            expected_ents = self.extract_expected_entity_set(row)
            expected_rels = self._extract_gold_relations(row)

            try:
                t0 = time.monotonic()
                result = model.extract(text, schema, threshold=threshold, include_confidence=True)
                latency_sum_ms += (time.monotonic() - t0) * 1000.0
            except Exception as exc:
                rows_failed += 1
                if len(failure_samples) < 5:
                    failure_samples.append({"row_index": idx, "error": str(exc)[:300]})
                continue

            # Entity scoring
            predicted_ents = self.extract_predicted_entity_set(result)
            e_tp += len(predicted_ents & expected_ents)
            e_fp += len(predicted_ents - expected_ents)
            e_fn += len(expected_ents - predicted_ents)

            # Relation scoring (per-type)
            predicted_rels = self._extract_predicted_relations(result)
            all_rel_labels = {rtype for _, rtype, _ in expected_rels | predicted_rels}
            for rtype in all_rel_labels:
                exp_set = {(h, t) for h, rt, t in expected_rels if rt == rtype}
                pred_set = {(h, t) for h, rt, t in predicted_rels if rt == rtype}
                r_tp[rtype] = r_tp.get(rtype, 0) + len(pred_set & exp_set)
                r_fp[rtype] = r_fp.get(rtype, 0) + len(pred_set - exp_set)
                r_fn[rtype] = r_fn.get(rtype, 0) + len(exp_set - pred_set)

            rows_evaluated += 1

        _empty = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        if rows_evaluated == 0:
            return {
                "ok": False,
                "error": "all_inference_failed",
                "metrics": _empty,
                "entity_metrics": _empty,
                "relation_metrics": _empty,
                "per_type_relation_f1": {},
                "counts": {"tp": e_tp, "fp": e_fp, "fn": e_fn},
                "relation_counts": {"tp": sum(r_tp.values()), "fp": sum(r_fp.values()), "fn": sum(r_fn.values())},
                "rows_total": len(rows),
                "rows_evaluated": 0,
                "rows_failed": rows_failed,
                "latency_ms_avg": 0.0,
                "failure_samples": failure_samples,
            }

        entity_metrics = self.compute_prf_metrics(tp=e_tp, fp=e_fp, fn=e_fn)

        # Aggregate relation metrics (micro-average)
        total_r_tp = sum(r_tp.values())
        total_r_fp = sum(r_fp.values())
        total_r_fn = sum(r_fn.values())
        relation_metrics = self.compute_prf_metrics(tp=total_r_tp, fp=total_r_fp, fn=total_r_fn)

        # Per-type relation F1
        per_type_relation_f1: dict[str, float] = {}
        for rtype in sorted(set(list(r_tp.keys()) + list(r_fp.keys()) + list(r_fn.keys()))):
            pt_f1 = self.compute_prf_metrics(
                tp=r_tp.get(rtype, 0),
                fp=r_fp.get(rtype, 0),
                fn=r_fn.get(rtype, 0),
            )
            per_type_relation_f1[rtype] = round(pt_f1["f1"], 4)

        is_partial = rows_failed > 0
        return {
            "ok": not is_partial,
            "error": "" if not is_partial else "partial_inference_failures",
            # Entity
            "entity_metrics": entity_metrics,
            "metrics": entity_metrics,  # backward-compat alias
            "counts": {"tp": e_tp, "fp": e_fp, "fn": e_fn},
            # Relation
            "relation_metrics": relation_metrics,
            "relation_counts": {"tp": total_r_tp, "fp": total_r_fp, "fn": total_r_fn},
            "per_type_relation_f1": per_type_relation_f1,
            # Common
            "rows_total": len(rows),
            "rows_evaluated": rows_evaluated,
            "rows_failed": rows_failed,
            "latency_ms_avg": round(latency_sum_ms / rows_evaluated, 2),
            "failure_samples": failure_samples,
        }

    # ── Confidence calibration ────────────────────────────────────────────────

    def calibrate_confidence(
        self,
        model_ref: str,
        rows: list[dict[str, Any]],
        n_bins: int = 10,
        threshold: float = GLINER_BENCHMARK_THRESHOLD,
    ) -> dict[str, Any]:
        """Compute Expected Calibration Error (ECE) for a candidate model.

        Buckets predictions by confidence score and compares average confidence
        to empirical accuracy within each bucket.  Returns ECE and per-bin stats.
        """
        if not rows:
            return {"ece": 0.0, "total_predictions": 0, "bins": [], "warning": "empty_eval_set"}

        try:
            model, schema = self.load_gliner_full_model(model_ref)
        except Exception as exc:
            return {"ece": 0.0, "total_predictions": 0, "bins": [], "warning": f"model_load_failed: {exc}"}

        bin_confidence: list[list[float]] = [[] for _ in range(n_bins)]
        bin_accuracy: list[list[float]] = [[] for _ in range(n_bins)]
        total_predictions = 0

        for row in rows:
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            expected_ents = self.extract_expected_entity_set(row)
            try:
                result = model.extract(text, schema, threshold=threshold, include_confidence=True)
            except Exception:
                continue

            for text_pred, _label, conf in self._extract_predicted_entities_with_confidence(result):
                bin_idx = min(int(conf * n_bins), n_bins - 1)
                is_correct = 1.0 if self.normalize_entity_text(text_pred) in expected_ents else 0.0
                bin_confidence[bin_idx].append(conf)
                bin_accuracy[bin_idx].append(is_correct)
                total_predictions += 1

        if total_predictions == 0:
            return {"ece": 0.0, "total_predictions": 0, "bins": [], "warning": "no_predictions"}

        bins_out: list[dict[str, Any]] = []
        ece = 0.0
        for i in range(n_bins):
            if not bin_confidence[i]:
                continue
            avg_conf = sum(bin_confidence[i]) / len(bin_confidence[i])
            avg_acc = sum(bin_accuracy[i]) / len(bin_accuracy[i])
            frac = len(bin_confidence[i]) / total_predictions
            ece += frac * abs(avg_conf - avg_acc)
            bins_out.append({
                "bin": i,
                "avg_confidence": round(avg_conf, 4),
                "avg_accuracy": round(avg_acc, 4),
                "count": len(bin_confidence[i]),
                "fraction": round(frac, 4),
                "calibration_gap": round(abs(avg_conf - avg_acc), 4),
            })

        warning: str | None = None
        if ece > 0.25:
            warning = f"High ECE={ece:.4f}: model may be overconfident or underconfident"
        elif ece > 0.15:
            warning = f"Moderate ECE={ece:.4f}: consider temperature scaling"

        return {
            "ece": round(ece, 4),
            "total_predictions": total_predictions,
            "bins": bins_out,
            "warning": warning,
        }

    # ── Run audit trail ───────────────────────────────────────────────────────

    def write_run_audit(self, audit_payload: dict[str, Any]) -> str:
        """Write a JSON audit trail for this training run.

        Saved to ~/.graph-memory/training/runs/YYYY-MM-DD-HHMMSS.json.
        Returns the path as a string.
        """
        runs_dir = config.GRAPH_MEMORY_DIR / "training" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        run_id = str(audit_payload.get("run_id") or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
        out_path = runs_dir / f"{run_id}.json"
        out_path.write_text(
            json.dumps(audit_payload, indent=2, default=str, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        log.info("Run audit written to %s", out_path)
        return str(out_path)

    def list_training_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recent training run audits."""
        runs_dir = config.GRAPH_MEMORY_DIR / "training" / "runs"
        if not runs_dir.exists():
            return []

        paths = sorted(runs_dir.glob("*.json"), reverse=True)[:limit]
        results: list[dict[str, Any]] = []
        for path in paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("run payload is not an object")

                deploy_decision = payload.get("deploy_decision")
                deploy_decision = deploy_decision if isinstance(deploy_decision, dict) else {}

                benchmark = payload.get("benchmark")
                benchmark = benchmark if isinstance(benchmark, dict) else {}
                entity_f1 = benchmark.get("entity_f1")
                entity_f1 = entity_f1 if isinstance(entity_f1, dict) else {}
                relation_f1 = benchmark.get("relation_f1")
                relation_f1 = relation_f1 if isinstance(relation_f1, dict) else {}

                mode = payload.get("mode")
                if mode in (None, ""):
                    strategy = payload.get("strategy")
                    if isinstance(strategy, dict):
                        mode = strategy.get("mode")
                if mode in (None, ""):
                    mode = payload.get("run_type")

                status = payload.get("status")
                if status in (None, ""):
                    status = payload.get("decision")
                if status in (None, ""):
                    status = payload.get("verdict")

                started_at = payload.get("started_at")
                if started_at in (None, ""):
                    started_at = payload.get("timestamp")

                entity_improvement = deploy_decision.get("entity_improvement")
                if entity_improvement in (None, ""):
                    entity_improvement = benchmark.get("entity_improvement")
                if entity_improvement in (None, ""):
                    entity_improvement = entity_f1.get("improvement")

                relation_improvement = deploy_decision.get("relation_improvement")
                if relation_improvement in (None, ""):
                    relation_improvement = benchmark.get("relation_improvement")
                if relation_improvement in (None, ""):
                    relation_improvement = relation_f1.get("improvement")

                combined_improvement = deploy_decision.get("combined_improvement")
                if combined_improvement in (None, ""):
                    combined_improvement = benchmark.get("combined_improvement")
                if combined_improvement in (None, "") and entity_improvement not in (None, "") and relation_improvement in (None, ""):
                    # Legacy runs sometimes only tracked a single improvement score.
                    combined_improvement = entity_improvement

                results.append({
                    "run_id": payload.get("run_id", path.stem),
                    "mode": mode,
                    "status": status,
                    "started_at": started_at,
                    "combined_improvement": combined_improvement,
                    "entity_improvement": entity_improvement,
                    "relation_improvement": relation_improvement,
                    "path": str(path),
                })
            except Exception:
                results.append({"path": str(path), "error": "parse_failed"})
        return results

    # ── Shadow evaluation ─────────────────────────────────────────────────────

    def run_shadow_evaluation(
        self,
        base_model_ref: str,
        candidate_model_ref: str,
        n_episodes: int = 20,
    ) -> dict[str, Any]:
        """Run both models on recent live episodes; compare fallback rates.

        The candidate passes if its fallback rate is not significantly worse
        than the base model (within a 20% tolerance).
        """
        if not candidate_model_ref:
            return {"skipped": True, "reason": "no_candidate_model_ref"}

        episodes = self._query_recent_episode_texts(limit=n_episodes)
        if not episodes:
            return {"skipped": True, "reason": "no_episodes_available"}

        base_results: list[dict[str, Any]] = []
        cand_results: list[dict[str, Any]] = []

        try:
            base_model, base_schema = self.load_gliner_full_model(base_model_ref)
        except Exception as exc:
            return {"skipped": True, "reason": f"base_model_load_failed: {exc}"}

        try:
            cand_model, cand_schema = self.load_gliner_full_model(candidate_model_ref)
        except Exception as exc:
            return {"skipped": True, "reason": f"candidate_model_load_failed: {exc}"}

        for ep in episodes:
            text = str(ep.get("text") or "").strip()
            if not text:
                continue
            base_res = self._shadow_extract(base_model, base_schema, text)
            cand_res = self._shadow_extract(cand_model, cand_schema, text)
            base_results.append(base_res)
            cand_results.append(cand_res)

        if not base_results:
            return {"skipped": True, "reason": "all_extractions_failed"}

        def _fallback_rate(results: list[dict[str, Any]]) -> float:
            total = sum(r.get("n_entities", 0) + r.get("n_relations", 0) for r in results)
            fallbacks = sum(r.get("fallback", 0) for r in results)
            return fallbacks / max(total, 1)

        base_rate = _fallback_rate(base_results)
        cand_rate = _fallback_rate(cand_results)

        # Candidate fails if fallback rate is >LORA_BENCHMARK_TOLERANCE worse than base
        # (sourced from config; default 1.20 = 20% worse allowed)
        TOLERANCE = config.LORA_BENCHMARK_TOLERANCE
        passed = cand_rate <= base_rate * TOLERANCE

        failure_reason: str | None = None
        if not passed:
            failure_reason = (
                f"candidate_fallback_rate={cand_rate:.4f} > "
                f"base_fallback_rate={base_rate:.4f} * {TOLERANCE:.2f}"
            )

        return {
            "skipped": False,
            "passed": passed,
            "failure_reason": failure_reason,
            "episodes_evaluated": len(base_results),
            "base_fallback_rate": round(base_rate, 4),
            "candidate_fallback_rate": round(cand_rate, 4),
            "base_avg_entities": round(
                sum(r.get("n_entities", 0) for r in base_results) / max(len(base_results), 1), 2
            ),
            "candidate_avg_entities": round(
                sum(r.get("n_entities", 0) for r in cand_results) / max(len(cand_results), 1), 2
            ),
            "base_avg_relations": round(
                sum(r.get("n_relations", 0) for r in base_results) / max(len(base_results), 1), 2
            ),
            "candidate_avg_relations": round(
                sum(r.get("n_relations", 0) for r in cand_results) / max(len(cand_results), 1), 2
            ),
        }

    def _query_recent_episode_texts(self, limit: int = 20) -> list[dict[str, Any]]:
        """Query Neo4j for recent episode texts for shadow evaluation."""
        try:
            from runtime_graph import require_graph_instance

            driver = require_graph_instance().driver
            with driver.session() as session:
                records = session.run(
                    """
                    MATCH (ep:Episode)
                    WHERE ep.content_preview IS NOT NULL
                      AND trim(ep.content_preview) <> ''
                    RETURN ep.content_preview AS text,
                           ep.id AS episode_id
                    ORDER BY coalesce(ep.created_at, ep.ingested_at, ep.occurred_at) DESC
                    LIMIT $limit
                    """,
                    limit=int(limit),
                )
                return [{"text": r["text"], "episode_id": r["episode_id"]} for r in records]
        except Exception:
            log.warning("Failed to query episode texts for shadow eval", exc_info=True)
            return []

    @staticmethod
    def _shadow_extract(model: Any, schema: Any, text: str) -> dict[str, Any]:
        """Run a single extraction and return stats for shadow evaluation."""
        try:
            result = model.extract(text, schema, threshold=0.35, include_confidence=True)
            entity_dict = result.get("entities") or {}
            n_entities = sum(len(v) for v in entity_dict.values() if isinstance(v, list)) if isinstance(entity_dict, dict) else 0
            rel_dict = result.get("relation_extraction") or result.get("relations") or {}
            n_relations = sum(len(v) for v in rel_dict.values() if isinstance(v, list)) if isinstance(rel_dict, dict) else 0
            return {
                "n_entities": n_entities,
                "n_relations": n_relations,
                "fallback": 0,
                "ok": True,
            }
        except Exception:
            return {"n_entities": 0, "n_relations": 0, "fallback": 1, "ok": False}

    def benchmark_finetune_candidate(
        self,
        rows: list[dict[str, Any]],
        candidate_model_ref: str | None = None,
        train_count: int | None = None,
    ) -> dict[str, Any]:
        eval_rows = rows
        resolved_train_count = int(train_count or 0)

        if train_count is None:
            split_train, split_eval = self.split_holdout_rows(rows)
            eval_rows = split_eval
            resolved_train_count = len(split_train)

        if not eval_rows:
            return {
                "ok": False,
                "base_score": 0.0,
                "candidate_score": 0.0,
                "improvement": 0.0,
                "split": {
                    "seed": GLINER_BENCHMARK_SEED,
                    "train_count": resolved_train_count,
                    "eval_count": 0,
                },
                "failure": {"reason": "no_eval_rows"},
            }

        base_model_ref = self.active_gliner_model_ref()
        candidate_model_ref = str(candidate_model_ref or "").strip()
        if not candidate_model_ref:
            candidate_path = self.gliner_candidate_model_dir()
            if candidate_path.exists():
                candidate_model_ref = str(candidate_path)

        if not candidate_model_ref:
            return {
                "ok": False,
                "base_score": 0.0,
                "candidate_score": 0.0,
                "improvement": 0.0,
                "split": {
                    "seed": GLINER_BENCHMARK_SEED,
                    "train_count": resolved_train_count,
                    "eval_count": len(eval_rows),
                },
                "failure": {"reason": "candidate_model_missing"},
            }

        base_eval = self.evaluate_model_on_rows(base_model_ref, eval_rows)
        candidate_eval = self.evaluate_model_on_rows(candidate_model_ref, eval_rows)

        # ── Entity F1 ──────────────────────────────────────────────────────
        base_entity_f1 = float(base_eval.get("entity_metrics", base_eval.get("metrics", {})).get("f1", 0.0) or 0.0)
        cand_entity_f1 = float(candidate_eval.get("entity_metrics", candidate_eval.get("metrics", {})).get("f1", 0.0) or 0.0)

        # ── Relation F1 ───────────────────────────────────────────────────
        base_relation_f1 = float((base_eval.get("relation_metrics") or {}).get("f1", 0.0) or 0.0)
        cand_relation_f1 = float((candidate_eval.get("relation_metrics") or {}).get("f1", 0.0) or 0.0)

        # ── Combined F1 (0.5 entity + 0.5 relation) ───────────────────────
        base_combined = 0.5 * base_entity_f1 + 0.5 * base_relation_f1
        cand_combined = 0.5 * cand_entity_f1 + 0.5 * cand_relation_f1

        benchmark_ok = bool(base_eval.get("ok")) and bool(candidate_eval.get("ok"))
        entity_improvement = (cand_entity_f1 - base_entity_f1) if benchmark_ok else 0.0
        relation_improvement = (cand_relation_f1 - base_relation_f1) if benchmark_ok else 0.0
        combined_improvement = (cand_combined - base_combined) if benchmark_ok else 0.0
        # Legacy key for backward compatibility
        improvement = entity_improvement

        # Per-type relation F1
        base_per_type = base_eval.get("per_type_relation_f1") or {}
        cand_per_type = candidate_eval.get("per_type_relation_f1") or {}
        all_types = sorted(set(list(base_per_type.keys()) + list(cand_per_type.keys())))
        per_type_improvement = {
            rtype: round(
                float(cand_per_type.get(rtype, 0.0) or 0.0)
                - float(base_per_type.get(rtype, 0.0) or 0.0),
                4,
            )
            for rtype in all_types
        }

        failure_details = []
        if not base_eval.get("ok"):
            failure_details.append({"model": "base", "error": base_eval.get("error", "unknown")})
        if not candidate_eval.get("ok"):
            failure_details.append(
                {"model": "candidate", "error": candidate_eval.get("error", "unknown")}
            )

        return {
            "ok": benchmark_ok,
            "split": {
                "seed": GLINER_BENCHMARK_SEED,
                "train_count": resolved_train_count,
                "eval_count": len(eval_rows),
            },
            "base_model": base_model_ref,
            "candidate_model": candidate_model_ref or None,
            "base": base_eval,
            "candidate": candidate_eval,
            # Legacy keys
            "base_score": round(base_entity_f1, 4),
            "candidate_score": round(cand_entity_f1, 4),
            "improvement": round(improvement, 4),
            # Extended keys
            "base_entity_f1": round(base_entity_f1, 4),
            "candidate_entity_f1": round(cand_entity_f1, 4),
            "entity_improvement": round(entity_improvement, 4),
            "base_relation_f1": round(base_relation_f1, 4),
            "candidate_relation_f1": round(cand_relation_f1, 4),
            "relation_improvement": round(relation_improvement, 4),
            "combined_improvement": round(combined_improvement, 4),
            "per_type_relation_f1": {
                "base": base_per_type,
                "candidate": cand_per_type,
                "improvement": per_type_improvement,
            },
            "failure": None if benchmark_ok else {"reason": "model_evaluation_failed", "details": failure_details},
        }

    def status(self) -> dict[str, Any]:
        self.refresh_state()

        total_examples = int(self.state.get("gliner_training_examples", 0) or 0)
        required = int(config.GLINER_FINETUNE_MIN_EXAMPLES)

        last_finetune_iso = str(self.state.get("gliner_last_finetune_at", "") or "")
        last_finetune = _parse_datetime(last_finetune_iso)

        cooldown_remaining_hours = 0
        if last_finetune:
            now = datetime.now(timezone.utc)
            elapsed = now - last_finetune
            last_strategy = str(self.state.get("gliner_last_training_strategy", "lora") or "lora")
            cooldown_days = GLINER_LORA_COOLDOWN_DAYS if last_strategy == "lora" else GLINER_FINETUNE_COOLDOWN_DAYS
            cooldown = timedelta(days=cooldown_days)
            if elapsed < cooldown:
                cooldown_remaining_hours = int((cooldown - elapsed).total_seconds() // 3600)

        return {
            "examples_accumulated": total_examples,
            "required": required,
            "last_finetune_at": last_finetune_iso,
            "last_strategy": str(self.state.get("gliner_last_training_strategy", "lora") or "lora"),
            "last_result": str(self.state.get("gliner_last_result", "") or ""),
            "last_cycle_status": str(self.state.get("gliner_last_cycle_status", "") or ""),
            "cooldown_remaining_hours": max(0, cooldown_remaining_hours),
            "active_model": self.active_gliner_model_ref(),
            "base_model": GLINER_BASE_MODEL,
        }


def _parse_datetime(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


_SERVICE: GLiNERTrainingService | None = None


def _get_service() -> GLiNERTrainingService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = GLiNERTrainingService()
    return _SERVICE


async def run_gliner_finetune_pipeline(force: bool = False) -> dict[str, Any]:
    """Run nightly GLiNER cycle: accumulate + (optional) fine-tune pipeline."""
    service = _get_service()
    return await service.run_gliner_nightly_cycle(force=force)


async def run_gliner_accumulation(limit: int = GLINER_TRAINING_SCAN_LIMIT) -> dict[str, Any]:
    """Accumulate training data without triggering fine-tune."""
    service = _get_service()
    return await asyncio.to_thread(service.accumulate_gliner_training_data, limit)


def get_gliner_stats() -> dict[str, Any]:
    """Return GLiNER status snapshot from state.json and model dirs."""
    service = _get_service()
    return service.status()


async def cleanup_stale_gliner_training_examples() -> dict[str, Any]:
    """Run stale training example cleanup as a coroutine (thread-dispatched)."""
    service = _get_service()
    return await asyncio.to_thread(service.cleanup_stale_training_examples)


__all__ = [
    "GLINER_BASE_MODEL",
    "GLINER_BENCHMARK_EVAL_RATIO",
    "GLINER_BENCHMARK_SEED",
    "GLINER_BENCHMARK_THRESHOLD",
    "GLINER_FINETUNE_COOLDOWN_DAYS",
    "GLINER_TRAINING_SCAN_LIMIT",
    "GLiNERTrainingService",
    "cleanup_stale_gliner_training_examples",
    "get_gliner_stats",
    "run_gliner_accumulation",
    "run_gliner_finetune_pipeline",
]
