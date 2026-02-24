"""Nightly maintenance orchestration for MollyGraph."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from audit.llm_audit import run_llm_audit
from evolution.gliner_training import (
    cleanup_stale_gliner_training_examples,
    run_gliner_finetune_pipeline,
)
from maintenance.lock import maintenance_lock
from memory.graph_suggestions import build_suggestion_digest, run_auto_adoption
from runtime_graph import require_graph_instance
from runtime_vector_store import get_vector_store_instance

log = logging.getLogger(__name__)


async def run_maintenance_cycle() -> dict[str, Any]:
    """Run deterministic cleanup + audit + suggestions + GLiNER cycle.
    
    Acquires maintenance lock so the extraction queue pauses during
    bulk graph mutations (merge, delete, reclassify).
    """
    with maintenance_lock():
        return await _run_maintenance_cycle_inner()


async def _run_maintenance_cycle_inner() -> dict[str, Any]:
    """Inner maintenance logic — runs under the maintenance lock."""
    started_at = datetime.now(timezone.utc).isoformat()
    graph = require_graph_instance()

    cleanup = {
        "strength_decay_updates": 0,
        "orphans_deleted": 0,
        "self_refs_deleted": 0,
    }

    try:
        cleanup["strength_decay_updates"] = graph.run_strength_decay_sync()
    except Exception:
        log.warning("Strength decay failed", exc_info=True)

    try:
        vs = get_vector_store_instance()
        cleanup["orphans_deleted"] = graph.delete_orphan_entities_sync(vector_store=vs)
    except Exception:
        log.warning("Orphan cleanup failed", exc_info=True)

    try:
        cleanup["self_refs_deleted"] = graph.delete_self_referencing_rels()
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

    # Stale example cleanup runs BEFORE accumulation so that quarantined
    # relations from the LLM audit above are removed before new examples
    # are built on top of them.
    stale_cleanup_result: dict
    try:
        stale_cleanup_result = await cleanup_stale_gliner_training_examples()
        log.info(
            "Stale training example cleanup: files_modified=%d examples_removed=%d relations_stripped=%d",
            stale_cleanup_result.get("files_modified", 0),
            stale_cleanup_result.get("examples_removed", 0),
            stale_cleanup_result.get("relations_stripped", 0),
        )
    except Exception as exc:
        log.warning("Stale training example cleanup failed", exc_info=True)
        stale_cleanup_result = {"status": "error", "error": str(exc)}

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
        "stale_training_cleanup": stale_cleanup_result,
        "gliner": gliner_result,
    }
