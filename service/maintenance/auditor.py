"""Nightly maintenance orchestration for MollyGraph."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from audit.llm_audit import run_llm_audit
from evolution.gliner_training import run_gliner_finetune_pipeline
from memory.graph import (
    delete_orphan_entities_sync,
    delete_self_referencing_rels,
    run_strength_decay_sync,
)
from memory.graph_suggestions import build_suggestion_digest, run_auto_adoption

log = logging.getLogger(__name__)


async def run_maintenance_cycle() -> dict[str, Any]:
    """Run deterministic cleanup + audit + suggestions + GLiNER cycle."""
    started_at = datetime.now(timezone.utc).isoformat()

    cleanup = {
        "strength_decay_updates": 0,
        "orphans_deleted": 0,
        "self_refs_deleted": 0,
    }

    try:
        cleanup["strength_decay_updates"] = run_strength_decay_sync()
    except Exception:
        log.warning("Strength decay failed", exc_info=True)

    try:
        cleanup["orphans_deleted"] = delete_orphan_entities_sync()
    except Exception:
        log.warning("Orphan cleanup failed", exc_info=True)

    try:
        cleanup["self_refs_deleted"] = delete_self_referencing_rels()
    except Exception:
        log.warning("Self-reference cleanup failed", exc_info=True)

    audit_result: dict[str, Any]
    try:
        audit_result = await run_llm_audit(limit=500, dry_run=False, schedule="nightly")
    except Exception as exc:
        log.error("LLM audit failed", exc_info=True)
        audit_result = {"status": "error", "error": str(exc)}

    try:
        adoption_summary = run_auto_adoption()
    except Exception:
        log.warning("Suggestion auto-adoption failed", exc_info=True)
        adoption_summary = "auto_adoption_failed"

    try:
        suggestion_digest = build_suggestion_digest()
    except Exception:
        log.warning("Suggestion digest build failed", exc_info=True)
        suggestion_digest = ""

    try:
        gliner_result = await run_gliner_finetune_pipeline(force=False)
    except Exception as exc:
        log.warning("GLiNER nightly cycle failed", exc_info=True)
        gliner_result = {"status": "error", "error": str(exc)}

    completed_at = datetime.now(timezone.utc).isoformat()
    return {
        "started_at": started_at,
        "completed_at": completed_at,
        "cleanup": cleanup,
        "audit": audit_result,
        "suggestions": {
            "digest": suggestion_digest,
            "auto_adoption": adoption_summary,
        },
        "gliner": gliner_result,
    }
