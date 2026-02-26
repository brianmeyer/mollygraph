"""Model health monitoring for MollyGraph.

Tracks extraction quality post-deploy and triggers rollback if quality degrades.
Also provides continuous degradation detection between training runs via a
separate rolling window (last MODEL_DEGRADATION_WINDOW_SIZE extractions).
"""
from __future__ import annotations

import json
import logging
import shutil
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config as _config

log = logging.getLogger(__name__)

ROLLBACKS_DIR = Path.home() / ".graph-memory" / "training" / "rollbacks"
TRAINING_CONFIG_PATH = Path.home() / ".graph-memory" / "gliner_finetune_config.json"
MODEL_HEALTH_STATE_PATH = Path.home() / ".graph-memory" / "metrics" / "model_health_state.json"


class ModelHealthMonitor:
    """Tracks extraction quality post-deploy and triggers rollback if quality degrades.

    Two monitoring windows run in parallel:

    * ``rolling_extractions`` (``WINDOW_SIZE=50``) — short-window spike detector
      that can trigger an automatic rollback when the fallback rate is 2× baseline.

    * ``degradation_window`` (``MODEL_DEGRADATION_WINDOW_SIZE`` from config,
      default 100) — longer window for *continuous* monitoring between training
      runs.  When the window is full (≥ 100 samples) and the rolling fallback rate
      exceeds ``baseline + MODEL_DEGRADATION_THRESHOLD`` (default +0.15), a WARNING
      is logged and ``degradation_detected`` is set to ``True``.  No automatic
      action is taken; the flag is surfaced via ``GET /model-health/status``.
    """

    WINDOW_SIZE = 50        # rolling window of last N extractions (rollback guard)
    SPIKE_THRESHOLD = 2.0   # rollback if fallback rate > 2x baseline

    def __init__(self):
        self.baseline_fallback_rate: float | None = None  # set at deploy time
        self.rolling_extractions: deque[dict[str, Any]] = deque(maxlen=self.WINDOW_SIZE)
        # ── Continuous degradation detection ─────────────────────────────────
        self.degradation_window: deque[dict[str, Any]] = deque(
            maxlen=_config.MODEL_DEGRADATION_WINDOW_SIZE
        )
        self.degradation_detected: bool = False
        # ─────────────────────────────────────────────────────────────────────
        self.deployed_at: datetime | None = None
        self.model_ref: str | None = None
        self.rollback_triggered: bool = False
        self._extraction_counter: int = 0
        self._load_state()

    def _load_state(self) -> None:
        """Load persisted baseline state so restart does not reset monitoring."""
        try:
            if not MODEL_HEALTH_STATE_PATH.exists():
                return
            payload = json.loads(MODEL_HEALTH_STATE_PATH.read_text(encoding="utf-8"))
            self.baseline_fallback_rate = (
                float(payload["baseline_fallback_rate"])
                if payload.get("baseline_fallback_rate") is not None
                else None
            )
            self.model_ref = str(payload.get("model_ref") or "") or None
            deployed_at = payload.get("deployed_at")
            self.deployed_at = datetime.fromisoformat(deployed_at) if deployed_at else None
            self.degradation_detected = bool(payload.get("degradation_detected", False))
        except Exception:
            log.warning("Failed to load model health state", exc_info=True)

    def _save_state(self) -> None:
        """Persist baseline state for restart-safe monitoring."""
        try:
            MODEL_HEALTH_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "baseline_fallback_rate": self.baseline_fallback_rate,
                "model_ref": self.model_ref,
                "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
                "degradation_detected": self.degradation_detected,
            }
            tmp_path = MODEL_HEALTH_STATE_PATH.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp_path.replace(MODEL_HEALTH_STATE_PATH)
        except Exception:
            log.warning("Failed to save model health state", exc_info=True)

    def set_baseline(self, fallback_rate: float, model_ref: str) -> None:
        """Called after each training deploy with the benchmark fallback rate.

        Resets both rolling windows and clears the degradation flag so that
        fresh monitoring begins from a clean state after every training run.
        """
        self.baseline_fallback_rate = fallback_rate
        self.deployed_at = datetime.now(timezone.utc)
        self.model_ref = model_ref
        self.rolling_extractions.clear()
        self.degradation_window.clear()
        self.degradation_detected = False
        self.rollback_triggered = False
        self._extraction_counter = 0
        self._save_state()
        log.info(
            "Model health baseline set: fallback_rate=%.3f model=%s",
            fallback_rate,
            model_ref,
        )

    def record_extraction(self, total_relations: int, fallback_count: int) -> None:
        """Called after each extraction to track quality.

        Updates both the short rollback-guard window and the longer degradation
        detection window.  Rollback check runs every 10 extractions; degradation
        check runs every 10 extractions once the degradation window is full.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_relations": total_relations,
            "fallback_count": fallback_count,
        }
        self.rolling_extractions.append(entry)
        self.degradation_window.append(entry)
        self._extraction_counter += 1

        # Check every 10 extractions
        if self._extraction_counter % 10 == 0:
            health = self.check_health()
            if health.get("should_rollback") and not self.rollback_triggered:
                self._trigger_rollback(health)
            # Continuous degradation detection (separate from rollback guard)
            self._check_degradation()

    def check_health(self) -> dict[str, Any]:
        """Check if current model is performing acceptably."""
        if not self.baseline_fallback_rate or len(self.rolling_extractions) < 10:
            return {"status": "monitoring", "samples": len(self.rolling_extractions)}

        total_rels = sum(e["total_relations"] for e in self.rolling_extractions)
        total_fallbacks = sum(e["fallback_count"] for e in self.rolling_extractions)
        current_rate = total_fallbacks / max(total_rels, 1)

        spike_ratio = current_rate / max(self.baseline_fallback_rate, 0.01)

        if spike_ratio > self.SPIKE_THRESHOLD:
            return {
                "status": "degraded",
                "current_fallback_rate": round(current_rate, 4),
                "baseline_fallback_rate": round(self.baseline_fallback_rate, 4),
                "spike_ratio": round(spike_ratio, 2),
                "should_rollback": True,
                "samples": len(self.rolling_extractions),
            }

        return {
            "status": "healthy",
            "current_fallback_rate": round(current_rate, 4),
            "baseline_fallback_rate": round(self.baseline_fallback_rate, 4),
            "spike_ratio": round(spike_ratio, 2),
            "should_rollback": False,
            "samples": len(self.rolling_extractions),
        }

    def _check_degradation(self) -> None:
        """Check the long degradation window for sustained quality decline.

        If the window has accumulated at least MODEL_DEGRADATION_WINDOW_SIZE
        samples and the rolling RELATED_TO fallback rate exceeds
        ``baseline + MODEL_DEGRADATION_THRESHOLD``, a WARNING is logged and
        ``degradation_detected`` is set to ``True``.  No automatic action is
        taken — the flag is surfaced via ``GET /model-health/status``.
        """
        required = _config.MODEL_DEGRADATION_WINDOW_SIZE
        if len(self.degradation_window) < required:
            return  # not enough data yet
        if self.baseline_fallback_rate is None:
            return  # no baseline established

        total_rels = sum(e["total_relations"] for e in self.degradation_window)
        total_fallbacks = sum(e["fallback_count"] for e in self.degradation_window)
        current_rate = total_fallbacks / max(total_rels, 1)

        threshold = self.baseline_fallback_rate + _config.MODEL_DEGRADATION_THRESHOLD
        if current_rate > threshold:
            if not self.degradation_detected:
                log.warning(
                    "MODEL DEGRADATION DETECTED: rolling_fallback_rate=%.4f "
                    "baseline=%.4f threshold=%.4f samples=%d",
                    current_rate,
                    self.baseline_fallback_rate,
                    threshold,
                    len(self.degradation_window),
                )
            if not self.degradation_detected:
                self.degradation_detected = True
                self._save_state()
        else:
            # Rate has recovered — clear the flag
            if self.degradation_detected:
                log.info(
                    "Model degradation flag cleared: rolling_fallback_rate=%.4f "
                    "is back within threshold=%.4f",
                    current_rate,
                    threshold,
                )
                self.degradation_detected = False
                self._save_state()

    def get_status(self) -> dict[str, Any]:
        """Return rolling stats from both windows plus the degradation flag.

        Used by ``GET /model-health/status`` to surface continuous monitoring
        state without triggering a rollback check.
        """
        health = self.check_health()

        # Degradation window stats
        dw_samples = len(self.degradation_window)
        dw_total_rels = sum(e["total_relations"] for e in self.degradation_window)
        dw_total_fallbacks = sum(e["fallback_count"] for e in self.degradation_window)
        dw_rate = dw_total_fallbacks / max(dw_total_rels, 1) if dw_samples else 0.0

        return {
            **health,
            "degradation_detected": self.degradation_detected,
            "degradation_window": {
                "samples": dw_samples,
                "window_size": _config.MODEL_DEGRADATION_WINDOW_SIZE,
                "fallback_rate": round(dw_rate, 4),
                "baseline_fallback_rate": (
                    round(self.baseline_fallback_rate, 4)
                    if self.baseline_fallback_rate is not None
                    else None
                ),
                "threshold": (
                    round(
                        self.baseline_fallback_rate + _config.MODEL_DEGRADATION_THRESHOLD, 4
                    )
                    if self.baseline_fallback_rate is not None
                    else None
                ),
                "full": dw_samples >= _config.MODEL_DEGRADATION_WINDOW_SIZE,
            },
            "model_ref": self.model_ref,
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
            "rollback_triggered": self.rollback_triggered,
        }

    def _trigger_rollback(self, health: dict[str, Any]) -> None:
        """Execute rollback: copy backup model to active, log event."""
        if self.rollback_triggered:
            return  # don't double-rollback

        self.rollback_triggered = True
        self._save_state()
        log.critical(
            "MODEL HEALTH DEGRADED – triggering rollback! "
            "current_fallback_rate=%.4f baseline=%.4f spike_ratio=%.2f samples=%d",
            health.get("current_fallback_rate", 0),
            health.get("baseline_fallback_rate", 0),
            health.get("spike_ratio", 0),
            health.get("samples", 0),
        )

        try:
            # Read backup model path from training config
            if not TRAINING_CONFIG_PATH.exists():
                log.error("Rollback aborted: training config not found at %s", TRAINING_CONFIG_PATH)
                return

            config_data = json.loads(TRAINING_CONFIG_PATH.read_text(encoding="utf-8"))
            backup_model_ref = str(config_data.get("backup_model_ref") or "").strip()
            active_model_ref = str(config_data.get("active_model_ref") or "").strip()

            if not backup_model_ref:
                log.error("Rollback aborted: no backup_model_ref in training config")
                return

            backup_path = Path(backup_model_ref)
            if not backup_path.exists():
                log.error("Rollback aborted: backup model path does not exist: %s", backup_path)
                return

            active_path = Path(active_model_ref) if active_model_ref else None

            # Atomic swap: copy backup → temp, rename active → .old, rename temp → active.
            # Avoids the race where rmtree races with a concurrent model read.
            if active_path:
                temp_dir = active_path.parent / f".rollback-{int(time.time())}"
                shutil.copytree(backup_path, temp_dir)

                old_dir = active_path.parent / f".old-{int(time.time())}"
                try:
                    active_path.rename(old_dir)  # atomic on same filesystem
                except FileNotFoundError:
                    pass  # active dir already gone — no problem

                temp_dir.rename(active_path)  # atomic on same filesystem

                # Best-effort cleanup of the displaced old directory
                try:
                    shutil.rmtree(old_dir, ignore_errors=True)
                except Exception:
                    pass

                log.critical("ROLLBACK COMPLETE: restored %s from backup %s", active_path, backup_path)
            else:
                log.error("Rollback: active_model_ref not set in config, cannot copy")

            # Log rollback event to disk
            ROLLBACKS_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")
            rollback_event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_ref": self.model_ref,
                "backup_model_ref": backup_model_ref,
                "active_model_ref": active_model_ref,
                "health_snapshot": health,
                "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
            }
            rollback_path = ROLLBACKS_DIR / f"{ts}.json"
            rollback_path.write_text(
                json.dumps(rollback_event, indent=2, default=str) + "\n", encoding="utf-8"
            )
            log.info("Rollback event logged to %s", rollback_path)

        except Exception:
            log.error("Rollback procedure failed", exc_info=True)


# ── Module-level singleton ────────────────────────────────────────────────────
model_health_monitor = ModelHealthMonitor()
