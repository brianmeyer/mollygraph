"""Pluggable entity/relationship extraction runtime.

Backends:
- gliner2: entities + relations
- hf_token_classification: entities + relations (with separate relation model)

Environment variables (all via config.py):
- MOLLYGRAPH_EXTRACTOR_BACKEND     : "gliner2" (default) or "hf_token_classification"
- MOLLYGRAPH_EXTRACTOR_MODEL       : override entity model path/name
- MOLLYGRAPH_EXTRACTOR_RELATION_MODEL : override relation model for HF backend
- MOLLYGRAPH_EXTRACTOR_RELATION_BACKEND : "hf" or "rebel" to force HF relation
  extraction even when the entity backend is gliner2 (mixed-mode)
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

import config
from extractor_schema_registry import get_effective_extractor_schema

log = logging.getLogger(__name__)

_model: Any | None = None
# For the HF relation backend we store (model, tokenizer) instead of a pipeline.
# The pipeline("text-generation") approach loads REBEL as BartForCausalLM with
# mismatched weights on transformers 5.x.  Using AutoModelForSeq2SeqLM directly
# is the only reliable approach for seq2seq relation extraction models.
_relation_model: tuple[Any, Any] | None = None  # (AutoModelForSeq2SeqLM, AutoTokenizer)
_model_lock = threading.Lock()
_model_backend: str = ""
_model_ref: str = ""
_model_mtime: float | None = None   # mtime of active model dir for hot-reload detection
_relation_model_backend: str = ""
_relation_model_ref: str = ""

_HF_LABEL_MAP = {
    "PER": "Person",
    "PERSON": "Person",
    "ORG": "Organization",
    "ORGANIZATION": "Organization",
    "NORP": "Organization",
    "GPE": "Place",
    "LOC": "Place",
    "LOCATION": "Place",
    "FAC": "Place",
    "EVENT": "Concept",
    "PRODUCT": "Technology",
    "WORK_OF_ART": "Concept",
    "MISC": "Concept",
}


def _normalize_backend(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"gliner", "gliner2"}:
        return "gliner2"
    if raw in {"hf", "ner", "uie", "token-classification", "token_classification", "hf_token_classification"}:
        return "hf_token_classification"
    return "gliner2"


def _active_backend() -> str:
    return _normalize_backend(getattr(config, "EXTRACTOR_BACKEND", "gliner2"))


def _active_model_ref() -> str:
    return str(getattr(config, "EXTRACTOR_MODEL", "") or "").strip()


def _active_relation_model_ref() -> str:
    return str(getattr(config, "EXTRACTOR_RELATION_MODEL", "") or "").strip()


def _active_relation_backend() -> str:
    """Return the relation extraction backend to use.

    Normally this is the same as the entity backend.  When
    MOLLYGRAPH_EXTRACTOR_RELATION_BACKEND is set to "hf" or "rebel" the
    caller should use the HF seq2seq path for relations even if entities come
    from gliner2 (mixed-mode).
    """
    override = str(getattr(config, "EXTRACTOR_RELATION_BACKEND", "") or "").strip().lower()
    if override in {"hf", "rebel", "hf_token_classification"}:
        return "hf_token_classification"
    return _active_backend()


def invalidate_model_cache() -> None:
    """Clear cached extractor model so runtime config changes take effect."""
    global _model, _relation_model, _model_backend, _model_ref, _model_mtime, _relation_model_backend, _relation_model_ref
    with _model_lock:
        _model = None
        _relation_model = None
        _model_backend = ""
        _model_ref = ""
        _model_mtime = None
        _relation_model_backend = ""
        _relation_model_ref = ""


def _resolve_gliner_model_ref(selected_model: str) -> str:
    """Resolve which GLiNER model to load.

    Priority:
    1. selected_model (from MOLLYGRAPH_EXTRACTOR_MODEL env var or API)
    2. ~/.graph-memory/models/gliner_active (local fine-tuned, if present)
    3. config.GLINER_BASE_MODEL (HuggingFace fallback)
    """
    if selected_model:
        return selected_model
    active_dir = config.MODELS_DIR / "gliner_active"
    if active_dir.exists() and (active_dir / "model.safetensors").exists():
        return str(active_dir)
    return str(config.GLINER_BASE_MODEL)


def _load_gliner_model(model_ref: str) -> Any:
    """Load a GLiNER2 model from a local path or a HuggingFace model ID.

    GLiNER2.from_pretrained() handles both cases natively — this is a thin
    wrapper that gives us a consistent single entry point.
    """
    from gliner2 import GLiNER2

    log.info("Loading GLiNER2 model (%s)...", model_ref)
    m = GLiNER2.from_pretrained(model_ref)
    log.info("GLiNER2 model loaded.")
    return m


# Kept for backwards compat with callers that used these names.
def _load_base_gliner_model() -> Any:
    return _load_gliner_model(str(config.GLINER_BASE_MODEL))


def _load_gliner_model_from_dir(model_dir: str) -> Any:
    return _load_gliner_model(model_dir)


def _get_model() -> Any:
    """Load the active extractor model according to runtime config.

    For gliner2 backend, also performs mtime-based hot-reload: if the
    active model directory content changes (e.g. after a LoRA deploy) the
    model is reloaded automatically on the next extraction call.
    """
    global _model, _model_backend, _model_ref, _model_mtime

    backend = _active_backend()
    requested_model = _active_model_ref()
    resolved_ref = _resolve_gliner_model_ref(requested_model) if backend == "gliner2" else requested_model

    if backend == "hf_token_classification" and not resolved_ref:
        raise ValueError(
            "No extractor model configured for hf_token_classification. "
            "Set one via POST /extractors/config or POST /extractors/models."
        )

    # ── GLiNER2 mtime hot-reload check ────────────────────────────────────
    # When the resolved path is a local directory (deployed model), check its
    # mtime.  A rename-swap deploy changes the mtime while the path string
    # stays the same, so we need this extra check to trigger a reload.
    if backend == "gliner2":
        import os as _os
        current_mtime: float | None = None
        if resolved_ref and _os.path.isdir(resolved_ref):
            try:
                current_mtime = _os.path.getmtime(resolved_ref)
            except OSError:
                current_mtime = None

        # Fast path: same backend, same ref, same mtime → return cached model
        if (
            _model is not None
            and _model_backend == backend
            and _model_ref == resolved_ref
            and current_mtime is not None
            and _model_mtime == current_mtime
        ):
            return _model

        # Also fast-path non-directory refs (HuggingFace model IDs) when unchanged
        if (
            _model is not None
            and _model_backend == backend
            and _model_ref == resolved_ref
            and current_mtime is None
            and _model_mtime is None
        ):
            return _model

        with _model_lock:
            # Re-read inside lock
            if resolved_ref and _os.path.isdir(resolved_ref):
                try:
                    current_mtime = _os.path.getmtime(resolved_ref)
                except OSError:
                    current_mtime = None

            if (
                _model is not None
                and _model_backend == backend
                and _model_ref == resolved_ref
                and _model_mtime == current_mtime
            ):
                return _model

            if current_mtime is not None:
                log.info(
                    "GLiNER2 hot-reload triggered: path=%s mtime=%.3f",
                    resolved_ref, current_mtime,
                )
                try:
                    model = _load_gliner_model(resolved_ref)
                except Exception as exc:
                    log.error(
                        "Failed to hot-reload GLiNER2 model from %s: %s", resolved_ref, exc
                    )
                    if _model is not None:
                        return _model  # keep old model as safe fallback
                    model = _load_base_gliner_model()
                    resolved_ref = str(config.GLINER_BASE_MODEL)
                    current_mtime = None
            else:
                # resolved_ref may be a local directory without a detectable mtime
                # change, or a HuggingFace model ID.  Both cases are handled by
                # _load_gliner_model() which calls GLiNER2.from_pretrained().
                model = _load_gliner_model(resolved_ref)

            _model = model
            _model_backend = backend
            _model_ref = resolved_ref
            _model_mtime = current_mtime
            return _model

    # ── Non-gliner2 backends (no mtime tracking) ──────────────────────────
    if _model is not None and _model_backend == backend and _model_ref == resolved_ref:
        return _model

    with _model_lock:
        if _model is not None and _model_backend == backend and _model_ref == resolved_ref:
            return _model

        from transformers import pipeline

        log.info("Loading HF token-classification model (%s)...", resolved_ref)
        model = pipeline(
            "token-classification",
            model=resolved_ref,
            tokenizer=resolved_ref,
            aggregation_strategy="simple",
        )
        log.info("HF token-classification model loaded.")

        _model = model
        _model_backend = backend
        _model_ref = resolved_ref
        _model_mtime = None
        return _model


def _should_use_hf_relation() -> bool:
    """Return True when the relation extraction path should use HF seq2seq.

    True when:
    - entity backend is hf_token_classification (always uses HF relation model)
    - OR MOLLYGRAPH_EXTRACTOR_RELATION_BACKEND is set to "hf"/"rebel" (mixed mode)
    """
    return _active_relation_backend() == "hf_token_classification"


def _get_relation_model_hf() -> tuple[Any, Any] | None:
    """Load relation extraction model (model, tokenizer) for HF/seq2seq path.

    Returns a (AutoModelForSeq2SeqLM, AutoTokenizer) tuple, or None if no
    relation model is configured.

    Uses AutoModelForSeq2SeqLM directly instead of the pipeline() API because
    transformers 5.x removed text2text-generation from the pipeline registry
    and the text-generation pipeline loads BART/T5 models with the wrong
    architecture class (BartForCausalLM instead of BartForConditionalGeneration),
    producing garbage output and weight mismatch warnings.
    """
    global _relation_model, _relation_model_backend, _relation_model_ref

    if not _should_use_hf_relation():
        return None

    relation_ref = _active_relation_model_ref()
    if not relation_ref:
        return None

    # Cache key uses the canonical relation backend string.
    rel_backend_key = "hf_relation"

    if (
        _relation_model is not None
        and _relation_model_backend == rel_backend_key
        and _relation_model_ref == relation_ref
    ):
        return _relation_model  # type: ignore[return-value]

    with _model_lock:
        if (
            _relation_model is not None
            and _relation_model_backend == rel_backend_key
            and _relation_model_ref == relation_ref
        ):
            return _relation_model  # type: ignore[return-value]

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        log.info("Loading HF seq2seq relation model (%s) via AutoModelForSeq2SeqLM...", relation_ref)
        tokenizer = AutoTokenizer.from_pretrained(relation_ref)
        model = AutoModelForSeq2SeqLM.from_pretrained(relation_ref)
        log.info("HF seq2seq relation model loaded.")

        pair: tuple[Any, Any] = (model, tokenizer)
        _relation_model = pair  # type: ignore[assignment]
        _relation_model_backend = rel_backend_key
        _relation_model_ref = relation_ref
        return pair


def _build_schema():
    model = _get_model()
    schema = get_effective_extractor_schema()
    return model.create_schema().entities(schema["entities"]).relations(schema["relations"])


def _build_entity_schema():
    model = _get_model()
    schema = get_effective_extractor_schema()
    return model.create_schema().entities(schema["entities"])


def _normalize_hf_label(raw_label: Any) -> str:
    label = str(raw_label or "").strip().upper()
    for prefix in ("B-", "I-", "L-", "U-", "S-", "E-"):
        if label.startswith(prefix):
            label = label[len(prefix):]
            break
    if label.startswith("LABEL_"):
        label = label[6:]
    return _HF_LABEL_MAP.get(label, "Concept")


def _normalize_hf_token_text(raw_text: Any) -> str:
    text = str(raw_text or "")
    text = text.replace("##", "")
    text = text.replace("▁", " ")
    return " ".join(text.split()).strip()


def _extract_entities_hf(text: str, threshold: float = 0.4) -> list[dict[str, Any]]:
    model = _get_model()
    raw = model(text)
    if not isinstance(raw, list):
        return []

    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        token_text = _normalize_hf_token_text(item.get("word"))
        if not token_text:
            continue
        label = _normalize_hf_label(item.get("entity_group") or item.get("entity"))
        score = float(item.get("score") or 0.5)
        if score < threshold:
            continue
        key = (token_text.lower(), label)
        if key in seen:
            continue
        seen.add(key)
        out.append({"text": token_text, "label": label, "score": score})
    return out


def _parse_rebel_triplets(raw_text: str) -> list[dict[str, str]]:
    """Parse REBEL-style generated text into triplets."""
    cleaned = (
        str(raw_text or "")
        .replace("<s>", " ")
        .replace("</s>", " ")
        .replace("<pad>", " ")
        .strip()
    )
    if not cleaned:
        return []

    tokens = cleaned.split()
    triplets: list[dict[str, str]] = []
    subject = ""
    obj = ""
    relation = ""
    mode = ""

    def flush() -> None:
        nonlocal subject, obj, relation
        subj_text = " ".join(subject.split()).strip()
        obj_text = " ".join(obj.split()).strip()
        rel_text = " ".join(relation.split()).strip()
        if subj_text and obj_text and rel_text:
            triplets.append({"head": subj_text, "tail": obj_text, "label": rel_text})
        subject = ""
        obj = ""
        relation = ""

    for token in tokens:
        if token == "<triplet>":
            flush()
            mode = "subject"
            continue
        if token == "<subj>":
            mode = "object"
            continue
        if token == "<obj>":
            mode = "relation"
            continue

        if mode == "subject":
            subject = f"{subject} {token}".strip()
        elif mode == "object":
            obj = f"{obj} {token}".strip()
        elif mode == "relation":
            relation = f"{relation} {token}".strip()

    flush()
    return triplets


def _resolve_entity_name(candidate: str, entities: list[dict[str, Any]]) -> str:
    text = " ".join(str(candidate or "").split()).strip()
    if not text:
        return ""
    if not entities:
        return text

    lowered = text.lower()
    for ent in entities:
        name = str(ent.get("text") or "").strip()
        if name and name.lower() == lowered:
            return name

    for ent in entities:
        name = str(ent.get("text") or "").strip()
        if not name:
            continue
        name_low = name.lower()
        if lowered in name_low or name_low in lowered:
            return name

    return text


def _extract_relations_hf(
    text: str,
    entities: list[dict[str, Any]],
    threshold: float = 0.4,
) -> list[dict[str, Any]]:
    """Extract relations using a HF seq2seq model (e.g. REBEL).

    Uses AutoModelForSeq2SeqLM + tokenizer directly — NOT the pipeline API —
    because transformers 5.x removed text2text-generation from the pipeline
    registry and the fallback text-generation pipeline loads seq2seq models
    with the wrong architecture class.
    """
    result = _get_relation_model_hf()
    if result is None:
        return []

    rel_model, rel_tokenizer = result

    try:
        import torch
        inputs = rel_tokenizer(
            text[:5000],
            return_tensors="pt",
            max_length=512,
            truncation=True,
        )
        with torch.no_grad():
            generated_ids = rel_model.generate(
                **inputs,
                max_new_tokens=256,
            )
        raw_text = rel_tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    except Exception as exc:
        if getattr(config, "STRICT_AI", False):
            raise RuntimeError("HF relation extraction failed in strict_ai mode.") from exc
        log.error("HF relation extraction failed", exc_info=True)
        return []

    base_score = 0.6
    if base_score < threshold:
        return []

    relations: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for triplet in _parse_rebel_triplets(raw_text):
        head = _resolve_entity_name(str(triplet.get("head") or ""), entities)
        tail = _resolve_entity_name(str(triplet.get("tail") or ""), entities)
        label = " ".join(str(triplet.get("label") or "").split()).strip().lower()

        if not head or not tail or not label:
            continue
        if head.lower() == tail.lower():
            continue

        key = (head.lower(), label, tail.lower())
        if key in seen:
            continue
        seen.add(key)

        relations.append(
            {
                "head": head,
                "tail": tail,
                "label": label,
                "score": base_score,
            }
        )
    return relations


def extract_entities(text: str, threshold: float = 0.4) -> list[dict[str, Any]]:
    """Extract entities only.

    Returns list of {"text": str, "label": str, "score": float}.
    """
    if getattr(config, "TEST_MODE", False):
        return []

    backend = _active_backend()
    try:
        if backend == "hf_token_classification":
            return _extract_entities_hf(text, threshold=threshold)

        model = _get_model()
        schema = _build_entity_schema()
        result = model.extract(text, schema, threshold=threshold, include_confidence=True)
    except Exception as exc:
        if getattr(config, "STRICT_AI", False):
            raise RuntimeError("Entity extraction failed in strict_ai mode.") from exc
        log.error("Entity extraction failed", exc_info=True)
        return []

    entity_dict = result.get("entities", result)
    entities: list[dict[str, Any]] = []
    for etype, items in entity_dict.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                score = float(item.get("confidence", 0.5) or 0.5)
                if score < threshold:
                    continue
                entities.append({"text": item.get("text", ""), "label": etype, "score": score})
    return entities


def extract(text: str, threshold: float = 0.4) -> dict[str, Any]:
    """Full extraction entry point used by the pipeline."""
    if getattr(config, "TEST_MODE", False):
        return {"entities": [], "relations": [], "latency_ms": 0}

    backend = _active_backend()
    relation_backend = _active_relation_backend()
    log.info(
        "extract() called with entity_backend=%s relation_backend=%s "
        "(config.EXTRACTOR_BACKEND=%s EXTRACTOR_RELATION_BACKEND=%s)",
        backend,
        relation_backend,
        getattr(config, "EXTRACTOR_BACKEND", "?"),
        getattr(config, "EXTRACTOR_RELATION_BACKEND", ""),
    )
    t0 = time.monotonic()

    if backend == "hf_token_classification":
        # Pure HF path: token-classification entities + seq2seq relations
        entities = extract_entities(text, threshold=threshold)
        relations = _extract_relations_hf(text, entities=entities, threshold=threshold)
        latency_ms = int((time.monotonic() - t0) * 1000)
        return {"entities": entities, "relations": relations, "latency_ms": latency_ms}

    # ── GLiNER2 entity backend ─────────────────────────────────────────────
    if relation_backend == "hf_token_classification":
        # Mixed mode: gliner2 entities + HF seq2seq relations
        log.info(
            "Mixed extraction mode: gliner2 entities + HF seq2seq relations "
            "(EXTRACTOR_RELATION_BACKEND=%s)",
            getattr(config, "EXTRACTOR_RELATION_BACKEND", ""),
        )
        entities = extract_entities(text, threshold=threshold)
        relations = _extract_relations_hf(text, entities=entities, threshold=threshold)
        latency_ms = int((time.monotonic() - t0) * 1000)
        log.debug(
            "Mixed-mode extracted %d entities, %d relations in %dms",
            len(entities), len(relations), latency_ms,
        )
        return {"entities": entities, "relations": relations, "latency_ms": latency_ms}

    # ── Default: GLiNER2 handles both entities and relations ───────────────
    try:
        model = _get_model()
        schema = _build_schema()
        result = model.extract(text, schema, threshold=threshold, include_confidence=True)
    except Exception as exc:
        if getattr(config, "STRICT_AI", False):
            raise RuntimeError("Extraction failed in strict_ai mode.") from exc
        log.error("Extraction failed", exc_info=True)
        return {"entities": [], "relations": [], "latency_ms": 0}

    latency_ms = int((time.monotonic() - t0) * 1000)

    entity_dict = result.get("entities", {})
    entities_out: list[dict[str, Any]] = []
    for etype, items in entity_dict.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                entities_out.append(
                    {
                        "text": item.get("text", ""),
                        "label": etype,
                        "score": item.get("confidence", 0.5),
                    }
                )

    rel_dict = result.get("relation_extraction", {})
    relations_out: list[dict[str, Any]] = []
    for rtype, items in rel_dict.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            head = item.get("head", {})
            tail = item.get("tail", {})
            head_text = head.get("text", "") if isinstance(head, dict) else str(head)
            tail_text = tail.get("text", "") if isinstance(tail, dict) else str(tail)
            head_conf = head.get("confidence", 0.5) if isinstance(head, dict) else 0.5
            tail_conf = tail.get("confidence", 0.5) if isinstance(tail, dict) else 0.5

            if head_text and tail_text:
                relations_out.append(
                    {
                        "head": head_text,
                        "tail": tail_text,
                        "label": rtype,
                        "score": min(head_conf, tail_conf),
                    }
                )

    log.debug(
        "Extracted %d entities, %d relations in %dms",
        len(entities_out), len(relations_out), latency_ms,
    )

    return {"entities": entities_out, "relations": relations_out, "latency_ms": latency_ms}


def prefetch_model(
    backend: str | None = None,
    model: str | None = None,
    relation_model: str | None = None,
) -> dict[str, Any]:
    """Download/cache a model now without waiting for first extraction call."""
    if getattr(config, "TEST_MODE", False):
        selected_backend = _normalize_backend(backend or _active_backend())
        selected_model = str(model or "").strip()
        selected_relation_model = str(relation_model or "").strip()
        return {
            "backend": selected_backend,
            "model": selected_model,
            "relation_model": selected_relation_model,
            "status": "skipped_test_mode",
        }

    selected_backend = _normalize_backend(backend or _active_backend())
    selected_model = str(model or "").strip()
    selected_relation_model = str(relation_model or "").strip()

    if selected_backend == "gliner2":
        model_ref = _resolve_gliner_model_ref(selected_model)
        _load_gliner_model(model_ref)
        return {"backend": selected_backend, "model": model_ref, "relation_model": "", "status": "ready"}

    model_ref = selected_model or _active_model_ref()
    if not model_ref:
        raise ValueError(
            "No model provided for hf_token_classification prefetch. "
            "Set one via /extractors/config or pass a model explicitly."
        )
    relation_ref = selected_relation_model or _active_relation_model_ref()
    if not relation_ref:
        raise ValueError(
            "No relation model provided for hf_token_classification prefetch. "
            "Set one via /extractors/config or pass relation_model explicitly."
        )
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

    pipeline("token-classification", model=model_ref, tokenizer=model_ref, aggregation_strategy="simple")
    # Use AutoModelForSeq2SeqLM for relation model — not the pipeline API
    # (transformers 5.x removed text2text-generation from the pipeline registry)
    AutoTokenizer.from_pretrained(relation_ref)
    AutoModelForSeq2SeqLM.from_pretrained(relation_ref)
    return {
        "backend": selected_backend,
        "model": model_ref,
        "relation_model": relation_ref,
        "status": "ready",
    }
