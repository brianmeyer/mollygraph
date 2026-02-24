"""GLiREL second-pass relation enrichment for GLiNER2 extraction output."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import config

log = logging.getLogger(__name__)


class GLiRELEnrichment:
    """Second-pass relation extraction + silver-label generation."""

    _model: Any | None = None
    _model_ref: str = ""
    _model_lock = threading.Lock()

    @classmethod
    def _get_model(cls) -> Any:
        model_ref = str(getattr(config, "GLIREL_MODEL", "jackboyla/glirel-large-v0")).strip()
        if cls._model is not None and cls._model_ref == model_ref:
            return cls._model

        with cls._model_lock:
            if cls._model is not None and cls._model_ref == model_ref:
                return cls._model

            from glirel import GLiREL  # lazy import (optional dependency at runtime)

            log.info("Loading GLiREL model (%s)...", model_ref)
            cls._model = GLiREL.from_pretrained(model_ref, trust_remote_code=True)
            cls._model_ref = model_ref
            log.info("GLiREL model loaded.")
            return cls._model

    def enrich_relations(self, text: str, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run GLiREL relation extraction over known entities from GLiNER2."""
        if not text or not entities:
            return []
        if not getattr(config, "GLIREL_ENABLED", False):
            return []

        spans = self._build_entity_spans(text, entities)
        if len(spans) < 2:
            return []

        labels = self._relation_labels()
        if not labels:
            return []

        model = self._get_model()
        raw = self._run_inference(model, text=text, spans=spans, labels=labels)
        return self._normalize_relations(raw, spans)

    def generate_training_examples(
        self,
        text: str,
        glirel_relations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Build GLiNER-compatible silver-label records from high-confidence GLiREL output."""
        if not text:
            return []

        min_conf = float(getattr(config, "GLIREL_TRAINING_THRESHOLD", 0.8))
        selected = [
            rel for rel in glirel_relations
            if self._safe_float(rel.get("confidence"), 0.0) >= min_conf
        ]
        if not selected:
            return []

        entities: dict[str, dict[str, str]] = {}
        rels: list[dict[str, str]] = []
        seen_rels: set[tuple[str, str, str]] = set()

        for rel in selected:
            head = str(rel.get("head") or "").strip()
            tail = str(rel.get("tail") or "").strip()
            label = str(rel.get("rel_type") or "").strip().upper().replace(" ", "_")
            if not head or not tail or not label:
                continue
            key = (head, label, tail)
            if key in seen_rels:
                continue
            seen_rels.add(key)

            entities.setdefault(head.lower(), {"text": head, "label": "Concept"})
            entities.setdefault(tail.lower(), {"text": tail, "label": "Concept"})
            rels.append({"head": head, "tail": tail, "label": label})

        if not rels:
            return []

        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return [
            {
                "episode_id": f"glirel-silver:{digest}",
                "created_at": datetime.now(UTC).isoformat(),
                "source_text": text,
                "source": "glirel_silver",
                "extracted_entities": list(entities.values()),
                "extracted_relations": rels,
                "negative_relations": [],
                "quality_signals": {"glirel_silver_relations": len(rels)},
            }
        ]

    def persist_training_examples(self, examples: list[dict[str, Any]]) -> str | None:
        """Append silver-label examples to the shared GLiNER training directory."""
        if not examples:
            return None

        training_dir = Path(getattr(config, "TRAINING_DIR"))
        training_dir.mkdir(parents=True, exist_ok=True)
        path = training_dir / f"glirel-silver-{datetime.now(UTC).strftime('%Y%m%d')}.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            for row in examples:
                fh.write(json.dumps(row, ensure_ascii=True) + "\n")
        return str(path)

    def _run_inference(
        self,
        model: Any,
        text: str,
        spans: list[dict[str, Any]],
        labels: list[str],
    ) -> Any:
        threshold = float(getattr(config, "GLIREL_CONFIDENCE_THRESHOLD", 0.5))

        # Different GLiREL versions expose different call signatures.
        if hasattr(model, "predict_relations"):
            fn = getattr(model, "predict_relations")
            call_specs = (
                lambda: fn(text=text, labels=labels, ner=spans, threshold=threshold),
                lambda: fn(text=text, labels=labels, entities=spans, threshold=threshold),
                lambda: fn(text, labels, spans, threshold=threshold),
                lambda: fn(text, labels=labels, threshold=threshold),
                lambda: fn(text=text, labels=labels, threshold=threshold),
            )
        elif hasattr(model, "predict"):
            fn = getattr(model, "predict")
            call_specs = (
                lambda: fn(text=text, labels=labels, ner=spans, threshold=threshold),
                lambda: fn(text=text, labels=labels, entities=spans, threshold=threshold),
                lambda: fn(text, labels),
                lambda: fn(text=text, labels=labels),
            )
        else:
            if not callable(model):
                raise RuntimeError("GLiREL model has no supported inference method")
            call_specs = (
                lambda: model(text=text, labels=labels, ner=spans, threshold=threshold),
                lambda: model(text=text, labels=labels, entities=spans, threshold=threshold),
                lambda: model(text, labels),
            )

        last_error: Exception | None = None
        for call in call_specs:
            try:
                return call()
            except TypeError as exc:
                last_error = exc
                continue
        raise RuntimeError(f"GLiREL inference failed: {last_error}") from last_error

    def _normalize_relations(
        self,
        payload: Any,
        spans: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        rows = payload if isinstance(payload, list) else payload.get("relations", []) if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            return []

        canonical = {
            self._normalize(str(ent.get("text") or "")): str(ent.get("text") or "")
            for ent in spans
            if str(ent.get("text") or "").strip()
        }

        out: dict[tuple[str, str, str], dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue

            head = self._coerce_entity_text(
                row.get("head"),
                row.get("head_text"),
                row.get("source"),
                row.get("subject"),
                row.get("arg1"),
            )
            tail = self._coerce_entity_text(
                row.get("tail"),
                row.get("tail_text"),
                row.get("target"),
                row.get("object"),
                row.get("arg2"),
            )
            rel_type = str(
                row.get("rel_type")
                or row.get("relation")
                or row.get("label")
                or row.get("predicate")
                or row.get("type")
                or ""
            ).strip().upper().replace(" ", "_")

            if not head or not tail or not rel_type:
                continue
            if self._normalize(head) == self._normalize(tail):
                continue

            head = canonical.get(self._normalize(head), head)
            tail = canonical.get(self._normalize(tail), tail)
            confidence = max(
                0.0,
                min(
                    1.0,
                    self._safe_float(
                        row.get("confidence", row.get("score", row.get("probability", row.get("prob", 0.0)))),
                        0.0,
                    ),
                ),
            )
            key = (self._normalize(head), rel_type, self._normalize(tail))
            prev = out.get(key)
            if prev is None or confidence > float(prev.get("confidence") or 0.0):
                out[key] = {
                    "head": head,
                    "tail": tail,
                    "rel_type": rel_type,
                    "confidence": confidence,
                }

        return sorted(out.values(), key=lambda item: float(item.get("confidence", 0.0)), reverse=True)

    @staticmethod
    def _coerce_entity_text(*candidates: Any) -> str:
        for candidate in candidates:
            if isinstance(candidate, dict):
                text = str(candidate.get("text") or candidate.get("name") or "").strip()
                if text:
                    return text
            elif isinstance(candidate, str):
                text = candidate.strip()
                if text:
                    return text
        return ""

    def _build_entity_spans(self, text: str, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        spans: list[dict[str, Any]] = []
        seen: set[tuple[int, int, str]] = set()

        for ent in entities:
            if not isinstance(ent, dict):
                continue
            raw_text = str(ent.get("text") or "").strip()
            if not raw_text:
                continue
            label = str(ent.get("label") or "Concept").strip() or "Concept"

            # Use first mention only to keep inference cost predictable.
            match = re.search(re.escape(raw_text), text, flags=re.IGNORECASE)
            if not match:
                continue

            start, end = int(match.start()), int(match.end())
            key = (start, end, label)
            if key in seen:
                continue
            seen.add(key)
            spans.append(
                {
                    "text": text[start:end],
                    "start": start,
                    "end": end,
                    "label": label,
                }
            )
        return spans

    @staticmethod
    def _relation_labels() -> list[str]:
        try:
            from extractor_schema_registry import get_effective_extractor_schema

            schema = get_effective_extractor_schema()
            relation_map = schema.get("relations", {}) if isinstance(schema, dict) else {}
            labels = [
                str(name).strip().lower().replace("_", " ")
                for name in relation_map.keys()
                if str(name).strip()
            ]
            # GLiREL labels should be unique, stable-ordered.
            return sorted(set(labels))
        except Exception:
            log.debug("Unable to load extractor relation labels for GLiREL", exc_info=True)
            return []

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().lower())
