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

# Token regex matching GLiREL's internal tokenizer
_GLIREL_TOKEN_RE = re.compile(r'\w+(?:[-_]\w+)*|\S')


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

            cls._apply_glirel_compat_patch(GLiREL)

            log.info("Loading GLiREL model (%s)...", model_ref)
            cls._model = GLiREL.from_pretrained(model_ref, trust_remote_code=True)
            cls._model_ref = model_ref
            log.info("GLiREL model loaded.")
            return cls._model

    @staticmethod
    def _apply_glirel_compat_patch(GLiREL: Any) -> None:
        """Patch GLiREL._from_pretrained for huggingface_hub >= 1.0 compatibility.

        huggingface_hub dropped 'proxies' and 'resume_download' from the
        ModelHubMixin.from_pretrained() call chain in v1.0. GLiREL 1.2.1 still
        declares them as required keyword args in _from_pretrained, causing a
        TypeError. This patch makes them optional with safe defaults and removes
        them from hf_hub_download calls that no longer accept them.
        """
        import inspect
        from pathlib import Path as _Path
        from typing import Dict as _Dict, Optional as _Optional, Union as _Union

        try:
            orig_sig = inspect.signature(GLiREL._from_pretrained)
        except Exception:
            return  # can't introspect; skip patch

        proxies_param = orig_sig.parameters.get("proxies")
        resume_param = orig_sig.parameters.get("resume_download")

        # Only patch if the params exist AND lack defaults (i.e. are still required)
        needs_patch = (
            proxies_param is not None
            and proxies_param.default is inspect.Parameter.empty
        ) or (
            resume_param is not None
            and resume_param.default is inspect.Parameter.empty
        )
        if not needs_patch:
            return

        log.debug("Applying GLiREL hf_hub compat patch (_from_pretrained)")

        import torch as _torch
        from huggingface_hub import hf_hub_download as _hf_hub_download

        try:
            from glirel.model import load_config_as_namespace as _load_cfg
        except ImportError:
            return  # can't patch safely

        @classmethod  # type: ignore[misc]
        def _patched_from_pretrained(
            klass,
            *,
            model_id: str,
            revision: _Optional[str] = None,
            cache_dir: _Optional[_Union[str, _Path]] = None,
            force_download: bool = False,
            proxies: _Optional[_Dict] = None,  # kept for API compat, not forwarded
            resume_download: bool = False,      # kept for API compat, not forwarded
            local_files_only: bool = False,
            token: _Union[str, bool, None] = None,
            map_location: str = "cpu",
            strict: bool = False,
            **model_kwargs: Any,
        ) -> Any:
            model_file = _Path(model_id) / "pytorch_model.bin"
            if not model_file.exists():
                model_file = _hf_hub_download(
                    repo_id=model_id,
                    filename="pytorch_model.bin",
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            config_file = _Path(model_id) / "glirel_config.json"
            if not config_file.exists():
                config_file = _hf_hub_download(
                    repo_id=model_id,
                    filename="glirel_config.json",
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            cfg = _load_cfg(config_file)
            model = klass(cfg)
            state_dict = _torch.load(model_file, map_location=_torch.device(map_location))
            model.load_state_dict(state_dict, strict=strict, assign=True)
            model.to(map_location)
            return model

        GLiREL._from_pretrained = _patched_from_pretrained  # type: ignore[method-assign]

    def enrich_relations(self, text: str, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run GLiREL relation extraction over known entities from GLiNER2."""
        if not text or not entities:
            return []
        if not getattr(config, "GLIREL_ENABLED", False):
            return []

        tokens, ner_spans = self._build_entity_spans(text, entities)
        if len(ner_spans) < 2:
            log.debug("GLiREL: fewer than 2 entity spans found, skipping. spans=%s", ner_spans)
            return []

        labels = self._relation_labels()
        if not labels:
            log.debug("GLiREL: no relation labels configured, skipping.")
            return []

        model = self._get_model()
        threshold = float(getattr(config, "GLIREL_CONFIDENCE_THRESHOLD", 0.5))
        raw = model.predict_relations(
            tokens,
            labels,
            flat_ner=True,
            threshold=threshold,
            ner=ner_spans,
            top_k=-1,
        )
        return self._normalize_relations(raw)

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

    def _build_entity_spans(
        self, text: str, entities: list[dict[str, Any]]
    ) -> tuple[list[str], list[list]]:
        """Tokenize text and build GLiREL NER spans as token-index lists.

        Returns:
            tokens: list of token strings (GLiREL's tokenized input)
            ner_spans: list of [start_token_idx, end_token_idx, entity_type, entity_text]
        """
        # Tokenize using GLiREL's internal tokenizer pattern
        token_matches = list(_GLIREL_TOKEN_RE.finditer(text))
        tokens = [m.group() for m in token_matches]
        tok_starts = [m.start() for m in token_matches]
        tok_ends = [m.end() for m in token_matches]

        ner_spans: list[list] = []
        seen: set[tuple[int, int]] = set()

        for ent in entities:
            if not isinstance(ent, dict):
                continue
            # Support both 'text' (GLiNER2 output) and 'name' (MollyGraph entity format)
            raw_text = str(ent.get("text") or ent.get("name") or "").strip()
            if not raw_text:
                continue
            # Support both 'label' and 'entity_type'
            label = str(ent.get("label") or ent.get("entity_type") or "Concept").strip() or "Concept"

            # Find first mention of entity name in the text (case-insensitive)
            match = re.search(re.escape(raw_text), text, flags=re.IGNORECASE)
            if not match:
                log.debug("GLiREL: entity %r not found in text", raw_text)
                continue

            char_start, char_end = match.start(), match.end()

            # Map character offsets → token indices
            # start_tok: first token whose start >= char_start
            # end_tok: last token whose end <= char_end
            start_tok: int | None = None
            end_tok: int | None = None
            for i, (ts, te) in enumerate(zip(tok_starts, tok_ends)):
                if start_tok is None and ts >= char_start:
                    start_tok = i
                if te <= char_end:
                    end_tok = i

            if start_tok is None or end_tok is None or end_tok < start_tok:
                log.debug("GLiREL: could not map entity %r to token indices", raw_text)
                continue

            key = (start_tok, end_tok)
            if key in seen:
                continue
            seen.add(key)

            matched_text = " ".join(tokens[start_tok : end_tok + 1])
            ner_spans.append([start_tok, end_tok, label, matched_text])

        log.debug("GLiREL: tokens=%d, ner_spans=%d", len(tokens), len(ner_spans))
        return tokens, ner_spans

    def _normalize_relations(
        self,
        payload: Any,
    ) -> list[dict[str, Any]]:
        """Normalize GLiREL output to our standard relation format.

        GLiREL returns list of dicts:
          {'head_text': [tokens...], 'tail_text': [tokens...], 'label': str, 'score': float, ...}
        """
        rows = payload if isinstance(payload, list) else []

        out: dict[tuple[str, str, str], dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue

            # head_text and tail_text are lists of token strings from GLiREL
            head_raw = row.get("head_text", "")
            tail_raw = row.get("tail_text", "")
            head = " ".join(head_raw) if isinstance(head_raw, list) else str(head_raw or "").strip()
            tail = " ".join(tail_raw) if isinstance(tail_raw, list) else str(tail_raw or "").strip()

            rel_type = str(
                row.get("label") or row.get("rel_type") or row.get("relation") or ""
            ).strip().upper().replace(" ", "_")

            if not head or not tail or not rel_type:
                continue
            if self._normalize(head) == self._normalize(tail):
                continue

            confidence = max(
                0.0,
                min(
                    1.0,
                    self._safe_float(
                        row.get("score", row.get("confidence", row.get("probability", 0.0))),
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
