"""Model health monitoring for MollyGraph.

Tracks extraction quality post-deploy and triggers rollback if quality degrades.
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

log = logging.getLogger(__name__)

ROLLBACKS_DIR = Path.home() / ".graph-memory" / "training" / "rollbacks"
TRAINING_CONFIG_PATH = Path.home() / ".graph-memory" / "gliner_finetune_config.json"


class ModelHealthMonitor:
    """Tracks extraction quality post-deploy and triggers rollback if quality degrades."""

    WINDOW_SIZE = 50        # rolling window of last N extractions
    SPIKE_THRESHOLD = 2.0   # rollback if fallback rate > 2x baseline

    def __init__(self):
        self.baseline_fallback_rate: float | None = None  # set at deploy time
        self.rolling_extractions: deque[dict[str, Any]] = deque(maxlen=self.WINDOW_SIZE)
        self.deployed_at: datetime | None = None
        self.model_ref: str | None = None
        self.rollback_triggered: bool = False
        self._extraction_counter: int = 0

    def set_baseline(self, fallback_rate: float, model_ref: str) -> None:
        """Called after each deploy with the pre-deploy fallback rate."""
        self.baseline_fallback_rate = fallback_rate
        self.deployed_at = datetime.now(timezone.utc)
        self.model_ref = model_ref
        self.rolling_extractions.clear()
        self.rollback_triggered = False
        self._extraction_counter = 0
        log.info(
            "Model health baseline set: fallback_rate=%.3f model=%s",
            fallback_rate,
            model_ref,
        )

    def record_extraction(self, total_relations: int, fallback_count: int) -> None:
        """Called after each extraction to track quality."""
        self.rolling_extractions.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_relations": total_relations,
                "fallback_count": fallback_count,
            }
        )
        self._extraction_counter += 1

        # Check every 10 extractions
        if self._extraction_counter % 10 == 0:
            health = self.check_health()
            if health.get("should_rollback") and not self.rollback_triggered:
                self._trigger_rollback(health)

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

    def _trigger_rollback(self, health: dict[str, Any]) -> None:
        """Execute rollback: copy backup model to active, log event."""
        if self.rollback_triggered:
            return  # don't double-rollback

        self.rollback_triggered = True
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
