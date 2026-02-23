"""Audit feedback persistence for GLiNER training.

This module captures relationship-level audit verdicts so the training
pipeline can consume both positive and hard-negative supervision signals.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config

log = logging.getLogger(__name__)

_VALID_REL_TYPES = {
    "WORKS_ON",
    "WORKS_AT",
    "KNOWS",
    "USES",
    "LOCATED_IN",
    "DISCUSSED_WITH",
    "INTERESTED_IN",
    "CREATED",
    "MANAGES",
    "DEPENDS_ON",
    "RELATED_TO",
    "MENTIONS",
}


def audit_feedback_dir() -> Path:
    return config.TRAINING_DIR / "audit_feedback"


def _normalize_rel_type(value: Any) -> str:
    return str(value or "").strip().upper().replace(" ", "_")


def _normalize_snippets(value: Any, limit: int = 3) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = " ".join(str(item or "").split()).strip()
        if not text:
            continue
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _source_text_from_rel(rel: dict[str, Any]) -> str:
    snippets = _normalize_snippets(rel.get("context_snippets"))
    if snippets:
        return snippets[0]
    head = str(rel.get("head") or "").strip()
    rel_type = _normalize_rel_type(rel.get("rel_type"))
    tail = str(rel.get("tail") or "").strip()
    if head and tail and rel_type:
        return f"{head} {rel_type.replace('_', ' ').lower()} {tail}"
    return ""


def _build_feedback_id(base: str) -> str:
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()
    return digest[:24]


def _feedback_path_for_date(now: datetime) -> Path:
    filename = f"feedback-{now.strftime('%Y%m%d')}.jsonl"
    return audit_feedback_dir() / filename


def record_audit_feedback_batch(
    rels: list[dict[str, Any]],
    verdicts: list[dict[str, Any]],
    *,
    schedule: str,
    provider: str,
    model: str,
    dry_run: bool,
) -> dict[str, Any]:
    """Persist structured audit verdict feedback for downstream training."""
    if dry_run:
        return {
            "written": 0,
            "positive_labels": 0,
            "negative_labels": 0,
            "file": "",
        }

    if not rels or not verdicts:
        return {
            "written": 0,
            "positive_labels": 0,
            "negative_labels": 0,
            "file": "",
        }

    now = datetime.now(timezone.utc)
    created_at = now.isoformat()
    out_path = _feedback_path_for_date(now)

    entries: list[dict[str, Any]] = []
    positive_labels = 0
    negative_labels = 0

    for verdict in verdicts:
        index_raw = verdict.get("index")
        if not isinstance(index_raw, int):
            continue
        idx = index_raw - 1
        if not (0 <= idx < len(rels)):
            continue

        rel = rels[idx]
        decision = str(verdict.get("verdict") or "").strip().lower()
        if decision not in {"verify", "reclassify", "quarantine", "delete"}:
            continue

        rel_type = _normalize_rel_type(rel.get("rel_type"))
        suggested_type = _normalize_rel_type(verdict.get("suggested_type"))
        has_valid_suggestion = bool(suggested_type and suggested_type in _VALID_REL_TYPES)
        corrected_rel_type = suggested_type if (decision == "reclassify" and has_valid_suggestion) else rel_type

        training_targets: list[dict[str, str]] = []
        if decision == "verify":
            if rel_type:
                training_targets.append({"kind": "positive", "label": rel_type})
                positive_labels += 1
        elif decision == "reclassify":
            if rel_type:
                training_targets.append({"kind": "negative", "label": rel_type})
                negative_labels += 1
            if has_valid_suggestion and corrected_rel_type:
                training_targets.append({"kind": "positive", "label": corrected_rel_type})
                positive_labels += 1
        elif decision in {"quarantine", "delete"}:
            if rel_type:
                training_targets.append({"kind": "negative", "label": rel_type})
                negative_labels += 1

        head = str(rel.get("head") or "").strip()
        tail = str(rel.get("tail") or "").strip()
        head_type = str(rel.get("head_type") or "Concept").strip() or "Concept"
        tail_type = str(rel.get("tail_type") or "Concept").strip() or "Concept"
        snippets = _normalize_snippets(rel.get("context_snippets"))
        source_text = _source_text_from_rel(rel)
        note = str(verdict.get("note") or "").strip()

        seed = "|".join(
            [
                created_at,
                str(idx),
                decision,
                head,
                rel_type,
                tail,
                corrected_rel_type,
                note,
            ]
        )
        feedback_id = _build_feedback_id(seed)

        entry = {
            "feedback_id": feedback_id,
            "created_at": created_at,
            "schedule": str(schedule or "nightly").strip().lower(),
            "provider": str(provider or "").strip(),
            "model": str(model or "").strip(),
            "decision": decision,
            "head": head,
            "head_type": head_type,
            "tail": tail,
            "tail_type": tail_type,
            "rel_type": rel_type,
            "suggested_type": suggested_type,
            "corrected_rel_type": corrected_rel_type,
            "note": note,
            "context_snippets": snippets,
            "source_text": source_text,
            "training_targets": training_targets,
        }
        entries.append(entry)

    if not entries:
        return {
            "written": 0,
            "positive_labels": 0,
            "negative_labels": 0,
            "file": "",
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=True) + "\n")

    return {
        "written": len(entries),
        "positive_labels": positive_labels,
        "negative_labels": negative_labels,
        "file": str(out_path),
    }


def load_audit_feedback_entries(limit: int = 5000) -> list[dict[str, Any]]:
    """Load recent audit feedback records for training consumption."""
    target_dir = audit_feedback_dir()
    if not target_dir.exists():
        return []

    safe_limit = max(1, int(limit))
    rows: list[dict[str, Any]] = []

    for path in sorted(target_dir.glob("feedback-*.jsonl")):
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(payload, dict):
                        continue
                    rows.append(payload)
        except Exception:
            log.debug("Failed loading audit feedback file %s", path, exc_info=True)

    rows.sort(key=lambda r: str(r.get("created_at") or ""))
    if len(rows) > safe_limit:
        rows = rows[-safe_limit:]
    return rows


__all__ = [
    "audit_feedback_dir",
    "load_audit_feedback_entries",
    "record_audit_feedback_batch",
]
