"""Runtime registry for extraction backends/models.

Backends:
- gliner2 (entities + relations)
- hf_token_classification + relation model (entities + relations)
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

SUPPORTED_BACKENDS = ("gliner2", "hf_token_classification")
_BACKEND_ALIASES = {
    "gliner": "gliner2",
    "hf": "hf_token_classification",
    "ner": "hf_token_classification",
    "token-classification": "hf_token_classification",
    "token_classification": "hf_token_classification",
    "uie": "hf_token_classification",
}

_DEFAULT_HF_NER_MODEL = "dslim/bert-base-NER"
_DEFAULT_HF_REL_MODEL = "Babelscape/rebel-large"

_CURATED_MODELS: dict[str, dict[str, list[dict[str, Any]]]] = {
    "gliner2": {
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
        "relation": [],
    },
    "hf_token_classification": {
        "entity": [
            {
                "model": "dslim/bert-base-NER",
                "family": "Transformers token-classification",
                "note": "English NER baseline.",
            },
            {
                "model": "dbmdz/bert-large-cased-finetuned-conll03-english",
                "family": "Transformers token-classification",
                "note": "Stronger English NER model.",
            },
            {
                "model": "Babelscape/wikineural-multilingual-ner",
                "family": "Transformers token-classification",
                "note": "Multilingual NER model.",
            },
        ],
        "relation": [
            {
                "model": "Babelscape/rebel-large",
                "family": "Transformers seq2seq relation extraction",
                "note": "Open relation extraction model used to decode subject/predicate/object triplets.",
            }
        ],
    },
}

_REGISTRY_LOCK = threading.Lock()
_REGISTRY_CACHE: dict[str, Any] | None = None


def normalize_backend(backend: str | None) -> str:
    raw = str(backend or "").strip().lower()
    normalized = _BACKEND_ALIASES.get(raw, raw)
    if normalized in SUPPORTED_BACKENDS:
        return normalized
    raise ValueError(
        f"Unsupported extractor backend: {backend!r}. "
        f"Expected one of {', '.join(SUPPORTED_BACKENDS)}."
    )


def normalize_role(role: str | None) -> str:
    raw = str(role or "entity").strip().lower()
    if raw in {"entity", "entities"}:
        return "entity"
    if raw in {"relation", "relations", "re", "rel"}:
        return "relation"
    raise ValueError("Extractor model role must be 'entity' or 'relation'.")


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
    hf_entity_model = str(
        getattr(config, "HF_NER_MODEL", "") or os.environ.get("MOLLYGRAPH_HF_NER_MODEL", _DEFAULT_HF_NER_MODEL)
    ).strip()
    hf_relation_model = str(
        getattr(config, "EXTRACTOR_RELATION_MODEL", "")
        or os.environ.get("MOLLYGRAPH_EXTRACTOR_RELATION_MODEL", _DEFAULT_HF_REL_MODEL)
    ).strip()

    selected_model = str(getattr(config, "EXTRACTOR_MODEL", "") or "").strip()
    selected_relation_model = str(getattr(config, "EXTRACTOR_RELATION_MODEL", "") or "").strip()

    backend_raw = str(getattr(config, "EXTRACTOR_BACKEND", "gliner2") or "gliner2").strip().lower()
    backend = _BACKEND_ALIASES.get(backend_raw, backend_raw)
    if backend not in SUPPORTED_BACKENDS:
        backend = "gliner2"

    return {
        "active_backend": backend,
        "active_model": selected_model,
        "active_relation_model": selected_relation_model if backend == "hf_token_classification" else "",
        "models": {
            "gliner2": _dedupe_models(
                [gliner_model] + [entry["model"] for entry in _CURATED_MODELS["gliner2"]["entity"]]
            ),
            "hf_token_classification": _dedupe_models(
                [hf_entity_model] + [entry["model"] for entry in _CURATED_MODELS["hf_token_classification"]["entity"]]
            ),
        },
        "relation_models": {
            "gliner2": [],
            "hf_token_classification": _dedupe_models(
                [hf_relation_model] + [entry["model"] for entry in _CURATED_MODELS["hf_token_classification"]["relation"]]
            ),
        },
        "supported_backends": list(SUPPORTED_BACKENDS),
    }


def _merge_registry(base: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    models_payload = payload.get("models")
    if isinstance(models_payload, dict):
        for backend in SUPPORTED_BACKENDS:
            extras_raw = models_payload.get(backend, [])
            extras = extras_raw if isinstance(extras_raw, list) else []
            merged = list(base["models"].get(backend, [])) + [str(item) for item in extras]
            base["models"][backend] = _dedupe_models(merged)

    relation_models_payload = payload.get("relation_models")
    if isinstance(relation_models_payload, dict):
        for backend in SUPPORTED_BACKENDS:
            extras_raw = relation_models_payload.get(backend, [])
            extras = extras_raw if isinstance(extras_raw, list) else []
            merged = list(base["relation_models"].get(backend, [])) + [str(item) for item in extras]
            base["relation_models"][backend] = _dedupe_models(merged)

    active_backend_raw = payload.get("active_backend")
    try:
        active_backend = normalize_backend(str(active_backend_raw))
    except Exception:
        active_backend = str(base.get("active_backend", "gliner2"))

    active_model = str(payload.get("active_model") or "").strip()
    if active_model and active_model not in base["models"][active_backend]:
        base["models"][active_backend].insert(0, active_model)

    active_relation_model = str(payload.get("active_relation_model") or "").strip()
    if active_backend == "hf_token_classification":
        if active_relation_model and active_relation_model not in base["relation_models"][active_backend]:
            base["relation_models"][active_backend].insert(0, active_relation_model)
    else:
        active_relation_model = ""

    base["active_backend"] = active_backend
    base["active_model"] = active_model
    base["active_relation_model"] = active_relation_model
    base["supported_backends"] = list(SUPPORTED_BACKENDS)
    return base


def _apply_runtime_config(registry: dict[str, Any]) -> None:
    backend = str(registry.get("active_backend") or "gliner2").strip().lower()
    if backend not in SUPPORTED_BACKENDS:
        backend = "gliner2"

    entity_model = str(registry.get("active_model") or "").strip()
    relation_model = str(registry.get("active_relation_model") or "").strip()

    if backend == "hf_token_classification" and not relation_model:
        relation_candidates = registry.get("relation_models", {}).get("hf_token_classification", [])
        relation_model = str(relation_candidates[0] if relation_candidates else "").strip()
        registry["active_relation_model"] = relation_model

    if backend != "hf_token_classification":
        relation_model = ""
        registry["active_relation_model"] = ""

    # ── Env-var priority: NEVER let the registry file override explicit env vars ──
    #
    # config.EXTRACTOR_BACKEND is already canonical (never overridden from registry).
    # We apply the same protection to EXTRACTOR_MODEL and EXTRACTOR_RELATION_MODEL:
    # if the user set either env var explicitly, we must not clobber it with a stale
    # registry value, especially when the registry's backend differs from the active
    # backend (e.g. registry saved hf_token_classification model but env says gliner2).
    canonical_backend = str(getattr(config, "EXTRACTOR_BACKEND", "gliner2") or "gliner2").strip().lower()
    canonical_backend = _BACKEND_ALIASES.get(canonical_backend, canonical_backend)
    if canonical_backend not in SUPPORTED_BACKENDS:
        canonical_backend = "gliner2"

    log.debug(
        "Registry backend=%s (not applied — config.EXTRACTOR_BACKEND=%s is canonical)",
        backend,
        canonical_backend,
    )

    # Only apply entity_model from registry when:
    # 1. MOLLYGRAPH_EXTRACTOR_MODEL env var is not explicitly set, AND
    # 2. The registry's backend matches the active canonical backend
    #    (prevents a stale hf model from leaking into a gliner2 session)
    if not os.environ.get("MOLLYGRAPH_EXTRACTOR_MODEL"):
        if backend == canonical_backend:
            config.EXTRACTOR_MODEL = entity_model
        else:
            # Backend mismatch: registry has a model for a different backend.
            # Leave config.EXTRACTOR_MODEL at its default (empty) so
            # _resolve_gliner_model_ref() picks the correct local model.
            log.debug(
                "Registry entity_model=%r skipped (registry backend=%s != canonical=%s)",
                entity_model, backend, canonical_backend,
            )

    # Only apply relation_model from registry when env var is not explicitly set
    if not os.environ.get("MOLLYGRAPH_EXTRACTOR_RELATION_MODEL"):
        config.EXTRACTOR_RELATION_MODEL = relation_model


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
        "active_backend": registry.get("active_backend", "gliner2"),
        "active_model": registry.get("active_model", ""),
        "active_relation_model": registry.get("active_relation_model", ""),
        "models": registry.get("models", {}),
        "relation_models": registry.get("relation_models", {}),
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
    global _REGISTRY_CACHE
    normalized_backend = normalize_backend(backend)
    selected_model = str(model or "").strip()
    selected_relation_model = str(relation_model or "").strip()

    with _REGISTRY_LOCK:
        if _REGISTRY_CACHE is None:
            _REGISTRY_CACHE = _load_registry_locked()

        registry = _REGISTRY_CACHE

        backend_models = registry["models"].setdefault(normalized_backend, [])
        backend_models = _dedupe_models([str(item) for item in backend_models])
        registry["models"][normalized_backend] = backend_models

        if selected_model:
            if selected_model not in backend_models:
                backend_models.insert(0, selected_model)
        elif normalized_backend == "hf_token_classification":
            if backend_models:
                selected_model = backend_models[0]
            else:
                raise ValueError(
                    "No entity model registered for backend 'hf_token_classification'. "
                    "Add one first via POST /extractors/models."
                )

        relation_models = registry["relation_models"].setdefault(normalized_backend, [])
        relation_models = _dedupe_models([str(item) for item in relation_models])
        registry["relation_models"][normalized_backend] = relation_models

        if normalized_backend == "hf_token_classification":
            if selected_relation_model:
                if selected_relation_model not in relation_models:
                    relation_models.insert(0, selected_relation_model)
            elif relation_models:
                selected_relation_model = relation_models[0]
            else:
                raise ValueError(
                    "No relation model registered for backend 'hf_token_classification'. "
                    "Add one first via POST /extractors/models with role='relation'."
                )
        else:
            selected_relation_model = ""

        registry["active_backend"] = normalized_backend
        registry["active_model"] = selected_model
        registry["active_relation_model"] = selected_relation_model

        _apply_runtime_config(registry)
        _save_registry_locked(registry)
        return copy.deepcopy(registry)


def add_extractor_model(
    backend: str,
    model: str,
    activate: bool = False,
    role: str = "entity",
) -> dict[str, Any]:
    global _REGISTRY_CACHE
    normalized_backend = normalize_backend(backend)
    normalized_role = normalize_role(role)

    model_name = str(model or "").strip()
    if not model_name:
        raise ValueError("Model name cannot be empty.")

    if normalized_backend == "gliner2" and normalized_role == "relation":
        raise ValueError("Backend 'gliner2' does not use a separate relation model.")

    with _REGISTRY_LOCK:
        if _REGISTRY_CACHE is None:
            _REGISTRY_CACHE = _load_registry_locked()

        registry = _REGISTRY_CACHE
        target_key = "models" if normalized_role == "entity" else "relation_models"
        current = [str(item) for item in registry[target_key].setdefault(normalized_backend, [])]
        if model_name not in current:
            current.append(model_name)
        registry[target_key][normalized_backend] = _dedupe_models(current)

        if activate:
            if normalized_role == "entity":
                registry["active_backend"] = normalized_backend
                registry["active_model"] = model_name
                if normalized_backend == "hf_token_classification":
                    relation_candidates = registry["relation_models"].get(normalized_backend, [])
                    registry["active_relation_model"] = (
                        registry.get("active_relation_model")
                        or (relation_candidates[0] if relation_candidates else "")
                    )
                else:
                    registry["active_relation_model"] = ""
            else:
                registry["active_backend"] = normalized_backend
                registry["active_relation_model"] = model_name
                if normalized_backend == "hf_token_classification":
                    entity_candidates = registry["models"].get(normalized_backend, [])
                    if not registry.get("active_model") and entity_candidates:
                        registry["active_model"] = entity_candidates[0]

        _apply_runtime_config(registry)
        _save_registry_locked(registry)
        return copy.deepcopy(registry)


def get_extractor_status() -> dict[str, Any]:
    registry = get_extractor_registry()
    active_backend = str(registry.get("active_backend") or "gliner2")
    active_model = str(registry.get("active_model") or "")
    active_relation_model = str(registry.get("active_relation_model") or "")

    models = registry.get("models", {})
    relation_models = registry.get("relation_models", {})

    gliner_available = importlib.util.find_spec("gliner2") is not None
    transformers_available = importlib.util.find_spec("transformers") is not None

    if active_backend == "gliner2":
        active_ready = bool(gliner_available)
        supports_relations = True
    else:
        active_ready = bool(transformers_available and active_model and active_relation_model)
        supports_relations = bool(active_relation_model)

    strict_ai = bool(getattr(config, "STRICT_AI", False))
    blocking_errors: list[str] = []
    if strict_ai and not active_ready:
        blocking_errors.append(
            f"Active extractor backend '{active_backend}' is not ready."
        )

    return {
        "active_backend": active_backend,
        "active_model": active_model,
        "active_relation_model": active_relation_model,
        "active_ready": active_ready,
        "supports_relations": supports_relations,
        "strict_ai": strict_ai,
        "runtime_profile": str(getattr(config, "RUNTIME_PROFILE", "hybrid") or "hybrid"),
        "blocking_errors": blocking_errors,
        "supported_backends": list(SUPPORTED_BACKENDS),
        "backends": {
            "gliner2": {
                "available": gliner_available,
                "supports_relations": True,
                "entity_models": [str(m) for m in models.get("gliner2", [])],
                "relation_models": [],
            },
            "hf_token_classification": {
                "available": transformers_available,
                "supports_relations": True,
                "entity_models": [str(m) for m in models.get("hf_token_classification", [])],
                "relation_models": [str(m) for m in relation_models.get("hf_token_classification", [])],
                "note": (
                    "Uses token-classification for entities and seq2seq relation extraction "
                    "for subject/predicate/object triplets."
                ),
            },
        },
        "curated_models": copy.deepcopy(_CURATED_MODELS),
        "notes": [
            "Only extraction configurations that support both entities and relationships are marked ready.",
            "UIE-style models are supported when they expose a HuggingFace token-classification interface.",
        ],
    }
