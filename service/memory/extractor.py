"""Pluggable entity/relationship extraction runtime.

Backend: GLiNER2 (entities + relations). This is the only supported backend.
If GLiNER2 fails to load, an error is raised — there is no silent fallback.

Environment variables (all via config.py):
- MOLLYGRAPH_EXTRACTOR_BACKEND : must be "gliner2" (any other value is an error)
- MOLLYGRAPH_EXTRACTOR_MODEL   : override entity model path/name
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
_model_lock = threading.Lock()
_model_backend: str = ""
_model_ref: str = ""
_model_mtime: float | None = None   # mtime of active model dir for hot-reload detection


def _active_backend() -> str:
    backend = str(getattr(config, "EXTRACTOR_BACKEND", "gliner2") or "gliner2").strip().lower()
    if backend not in {"gliner2", "gliner"}:
        log.error(
            "Unsupported EXTRACTOR_BACKEND=%r — GLiNER2 is the only supported backend. "
            "Overriding to 'gliner2'.",
            backend,
        )
    return "gliner2"


def _active_model_ref() -> str:
    return str(getattr(config, "EXTRACTOR_MODEL", "") or "").strip()


def invalidate_model_cache() -> None:
    """Clear cached extractor model so runtime config changes take effect."""
    global _model, _model_backend, _model_ref, _model_mtime
    with _model_lock:
        _model = None
        _model_backend = ""
        _model_ref = ""
        _model_mtime = None


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

    Raises RuntimeError if the model fails to load (no silent fallback).
    """
    from gliner2 import GLiNER2

    log.info("Loading GLiNER2 model (%s)...", model_ref)
    try:
        m = GLiNER2.from_pretrained(model_ref)
    except Exception as exc:
        raise RuntimeError(
            f"GLiNER2 model failed to load from {model_ref!r}. "
            "Fix the model configuration — there is no fallback backend."
        ) from exc
    log.info("GLiNER2 model loaded.")
    return m


# Kept for backwards compat with callers that used these names.
def _load_base_gliner_model() -> Any:
    return _load_gliner_model(str(config.GLINER_BASE_MODEL))


def _load_gliner_model_from_dir(model_dir: str) -> Any:
    return _load_gliner_model(model_dir)


def _get_model() -> Any:
    """Load the active GLiNER2 model according to runtime config.

    Performs mtime-based hot-reload: if the active model directory content
    changes (e.g. after a LoRA deploy) the model is reloaded automatically
    on the next extraction call.

    Raises RuntimeError if GLiNER2 fails to load — no silent fallback.
    """
    global _model, _model_backend, _model_ref, _model_mtime

    requested_model = _active_model_ref()
    resolved_ref = _resolve_gliner_model_ref(requested_model)

    # ── GLiNER2 mtime hot-reload check ────────────────────────────────────
    # When the resolved path is a local directory (deployed model), check its
    # mtime.  A rename-swap deploy changes the mtime while the path string
    # stays the same, so we need this extra check to trigger a reload.
    import os as _os
    current_mtime: float | None = None
    if resolved_ref and _os.path.isdir(resolved_ref):
        try:
            current_mtime = _os.path.getmtime(resolved_ref)
        except OSError:
            current_mtime = None

    # Fast path: same ref, same mtime → return cached model
    if (
        _model is not None
        and _model_ref == resolved_ref
        and current_mtime is not None
        and _model_mtime == current_mtime
    ):
        return _model

    # Also fast-path non-directory refs (HuggingFace model IDs) when unchanged
    if (
        _model is not None
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
            and _model_ref == resolved_ref
            and _model_mtime == current_mtime
        ):
            return _model

        if current_mtime is not None:
            log.info(
                "GLiNER2 hot-reload triggered: path=%s mtime=%.3f",
                resolved_ref, current_mtime,
            )

        # Load the model — raises RuntimeError on failure (no fallback).
        model = _load_gliner_model(resolved_ref)

        _model = model
        _model_backend = "gliner2"
        _model_ref = resolved_ref
        _model_mtime = current_mtime
        return _model


def _build_schema():
    model = _get_model()
    schema = get_effective_extractor_schema()
    return model.create_schema().entities(schema["entities"]).relations(schema["relations"])


def _build_entity_schema():
    model = _get_model()
    schema = get_effective_extractor_schema()
    return model.create_schema().entities(schema["entities"])


def extract(text: str, threshold: float = 0.4) -> dict[str, Any]:
    """Full extraction entry point used by the pipeline.

    Uses GLiNER2 for both entity and relation extraction.
    Raises RuntimeError in strict_ai mode on failure; logs + returns empty otherwise.
    """
    if getattr(config, "TEST_MODE", False):
        return {"entities": [], "relations": [], "latency_ms": 0}

    log.info(
        "extract() called (backend=gliner2, config.EXTRACTOR_BACKEND=%s)",
        getattr(config, "EXTRACTOR_BACKEND", "?"),
    )
    t0 = time.monotonic()

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
    """Download/cache the GLiNER2 model now without waiting for first extraction call."""
    if getattr(config, "TEST_MODE", False):
        selected_model = str(model or "").strip()
        return {
            "backend": "gliner2",
            "model": selected_model,
            "status": "skipped_test_mode",
        }

    if backend is not None:
        requested = str(backend or "").strip().lower()
        if requested not in {"gliner2", "gliner", ""}:
            raise ValueError(
                f"Unsupported backend {backend!r} for prefetch. "
                "GLiNER2 is the only supported backend."
            )

    selected_model = str(model or "").strip()
    model_ref = _resolve_gliner_model_ref(selected_model)
    _load_gliner_model(model_ref)
    return {"backend": "gliner2", "model": model_ref, "status": "ready"}
