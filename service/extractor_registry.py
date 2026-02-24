"""Runtime registry for extraction backend/models.

Backend: GLiNER2 only (entities + relations).
"""
from __future__ import annotations

import copy
import importlib.util
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

import config

log = logging.getLogger(__name__)

SUPPORTED_BACKENDS = ("gliner2",)

_CURATED_MODELS: dict[str, list[dict[str, Any]]] = {
    "entity": [
        {
            "model": "fastino/gliner2-large-v1",
            "family": "GLiNER2",
            "note": "General-purpose entities+relations for this pipeline.",
        },
        {
            "model": "knowledgator/gliner-x-small-v1.0",
            "family": "GLiNER",
            "note": "Smaller/faster GLiNER variant with lower quality ceiling.",
        },
    ],
}

_REGISTRY_LOCK = threading.Lock()
_REGISTRY_CACHE: dict[str, Any] | None = None


def normalize_backend(backend: str | None) -> str:
    raw = str(backend or "").strip().lower()
    if raw in {"gliner2", "gliner"}:
        return "gliner2"
    raise ValueError(
        f"Unsupported extractor backend: {backend!r}. "
        "Only 'gliner2' is supported."
    )


def normalize_role(role: str | None) -> str:
    raw = str(role or "entity").strip().lower()
    if raw in {"entity", "entities"}:
        return "entity"
    raise ValueError(
        "Extractor model role must be 'entity'. "
        "GLiNER2 handles relation extraction natively — no separate relation model needed."
    )


def _registry_path() -> Path:
    configured = str(
        os.environ.get(
            "MOLLYGRAPH_EXTRACTOR_CONFIG_FILE",
            str(config.GRAPH_MEMORY_DIR / "extractor_config.json"),
        )
    ).strip()
    return Path(configured).expanduser()


def _fallback_registry_path() -> Path:
    return Path("/tmp/mollygraph-extractor-config.json")


def _dedupe_models(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _default_registry() -> dict[str, Any]:
    gliner_model = str(getattr(config, "GLINER_BASE_MODEL", "") or "fastino/gliner2-large-v1").strip()
    selected_model = str(getattr(config, "EXTRACTOR_MODEL", "") or "").strip()

    return {
        "active_backend": "gliner2",
        "active_model": selected_model,
        "models": {
            "gliner2": _dedupe_models(
                [gliner_model] + [entry["model"] for entry in _CURATED_MODELS["entity"]]
            ),
        },
        "supported_backends": list(SUPPORTED_BACKENDS),
    }


def _merge_registry(base: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    models_payload = payload.get("models")
    if isinstance(models_payload, dict):
        extras_raw = models_payload.get("gliner2", [])
        extras = extras_raw if isinstance(extras_raw, list) else []
        merged = list(base["models"].get("gliner2", [])) + [str(item) for item in extras]
        base["models"]["gliner2"] = _dedupe_models(merged)

    active_backend_raw = payload.get("active_backend")
    try:
        normalize_backend(str(active_backend_raw))
    except Exception:
        pass  # keep default

    active_model = str(payload.get("active_model") or "").strip()
    if active_model and active_model not in base["models"]["gliner2"]:
        base["models"]["gliner2"].insert(0, active_model)

    base["active_backend"] = "gliner2"
    base["active_model"] = active_model
    base["supported_backends"] = list(SUPPORTED_BACKENDS)
    return base


def _apply_runtime_config(registry: dict[str, Any]) -> None:
    entity_model = str(registry.get("active_model") or "").strip()

    # ── Env-var priority: NEVER let the registry file override explicit env vars ──
    # If MOLLYGRAPH_EXTRACTOR_MODEL is set explicitly, honour it; otherwise apply
    # whatever the registry resolved.
    if not os.environ.get("MOLLYGRAPH_EXTRACTOR_MODEL"):
        config.EXTRACTOR_MODEL = entity_model

    registry["active_backend"] = "gliner2"


def _load_registry_locked() -> dict[str, Any]:
    registry = _default_registry()
    path = _registry_path()
    path_candidates = [path]
    fallback = _fallback_registry_path()
    if fallback not in path_candidates:
        path_candidates.append(fallback)

    selected_path: Path | None = None
    for candidate in path_candidates:
        if candidate.exists():
            selected_path = candidate
            break

    if selected_path is not None:
        try:
            payload = json.loads(selected_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                registry = _merge_registry(registry, payload)
        except Exception:
            log.warning("Failed to parse extractor registry at %s", selected_path, exc_info=True)

    _apply_runtime_config(registry)
    return registry


def _save_registry_locked(registry: dict[str, Any]) -> None:
    payload = {
        "active_backend": "gliner2",
        "active_model": registry.get("active_model", ""),
        "models": registry.get("models", {}),
    }

    for path in (_registry_path(), _fallback_registry_path()):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
            tmp.replace(path)
            return
        except OSError:
            log.warning("Extractor registry path not writable: %s", path, exc_info=True)

    log.warning("Failed to persist extractor registry; keeping in-memory config only")


def initialize_extractor_registry() -> dict[str, Any]:
    global _REGISTRY_CACHE
    with _REGISTRY_LOCK:
        if _REGISTRY_CACHE is None:
            _REGISTRY_CACHE = _load_registry_locked()
        else:
            _apply_runtime_config(_REGISTRY_CACHE)
        return copy.deepcopy(_REGISTRY_CACHE)


def get_extractor_registry() -> dict[str, Any]:
    return initialize_extractor_registry()


def set_active_extractor_backend(
    backend: str,
    model: str | None = None,
    relation_model: str | None = None,
) -> dict[str, Any]:
    """Set the active extractor configuration. Only 'gliner2' is supported."""
    global _REGISTRY_CACHE
    normalize_backend(backend)  # raises ValueError for anything other than gliner2
    selected_model = str(model or "").strip()

    if relation_model:
        log.warning(
            "set_active_extractor_backend: relation_model=%r ignored — "
            "GLiNER2 handles relation extraction natively.",
            relation_model,
        )

    with _REGISTRY_LOCK:
        if _REGISTRY_CACHE is None:
            _REGISTRY_CACHE = _load_registry_locked()

        registry = _REGISTRY_CACHE

        backend_models = registry["models"].setdefault("gliner2", [])
        backend_models = _dedupe_models([str(item) for item in backend_models])
        registry["models"]["gliner2"] = backend_models

        if selected_model and selected_model not in backend_models:
            backend_models.insert(0, selected_model)

        registry["active_backend"] = "gliner2"
        registry["active_model"] = selected_model

        _apply_runtime_config(registry)
        _save_registry_locked(registry)
        return copy.deepcopy(registry)


def add_extractor_model(
    backend: str,
    model: str,
    activate: bool = False,
    role: str = "entity",
) -> dict[str, Any]:
    """Register a GLiNER2 model. Only 'entity' role is valid."""
    global _REGISTRY_CACHE
    normalize_backend(backend)   # raises ValueError for non-gliner2
    normalize_role(role)          # raises ValueError for 'relation'

    model_name = str(model or "").strip()
    if not model_name:
        raise ValueError("Model name cannot be empty.")

    with _REGISTRY_LOCK:
        if _REGISTRY_CACHE is None:
            _REGISTRY_CACHE = _load_registry_locked()

        registry = _REGISTRY_CACHE
        current = [str(item) for item in registry["models"].setdefault("gliner2", [])]
        if model_name not in current:
            current.append(model_name)
        registry["models"]["gliner2"] = _dedupe_models(current)

        if activate:
            registry["active_backend"] = "gliner2"
            registry["active_model"] = model_name

        _apply_runtime_config(registry)
        _save_registry_locked(registry)
        return copy.deepcopy(registry)


def get_extractor_status() -> dict[str, Any]:
    registry = get_extractor_registry()
    active_model = str(registry.get("active_model") or "")
    models = registry.get("models", {})

    gliner_available = importlib.util.find_spec("gliner2") is not None
    active_ready = gliner_available

    strict_ai = bool(getattr(config, "STRICT_AI", False))
    blocking_errors: list[str] = []
    if strict_ai and not active_ready:
        blocking_errors.append(
            "GLiNER2 extractor backend is not ready (gliner2 package not found)."
        )

    return {
        "active_backend": "gliner2",
        "active_model": active_model,
        "active_ready": active_ready,
        "supports_relations": True,
        "strict_ai": strict_ai,
        "runtime_profile": str(getattr(config, "RUNTIME_PROFILE", "hybrid") or "hybrid"),
        "blocking_errors": blocking_errors,
        "supported_backends": list(SUPPORTED_BACKENDS),
        "backends": {
            "gliner2": {
                "available": gliner_available,
                "supports_relations": True,
                "entity_models": [str(m) for m in models.get("gliner2", [])],
                "note": (
                    "GLiNER2 is the only extraction backend. "
                    "It handles both entities and relations natively and participates "
                    "in the self-improving LoRA training loop."
                ),
            },
        },
        "curated_models": copy.deepcopy(_CURATED_MODELS),
        "notes": [
            "GLiNER2 supports both entity and relationship extraction natively.",
            "Fine-tuned LoRA checkpoints are hot-swapped via ~/.graph-memory/models/gliner_active.",
        ],
    }
