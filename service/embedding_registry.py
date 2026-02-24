"""Runtime registry for embedding providers/models.

Supports switching active embeddings between:
- hash (deterministic local baseline)
- huggingface (sentence-transformers)
- ollama
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

SUPPORTED_PROVIDERS = ("hash", "huggingface", "ollama")
_PROVIDER_ALIASES = {
    "sentence-transformers": "huggingface",
    "sentence_transformers": "huggingface",
    "st": "huggingface",
    "hf": "huggingface",
    "local": "ollama",
}

_DEFAULT_HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_OLLAMA_MODEL = "nomic-embed-text"

_REGISTRY_LOCK = threading.Lock()
_REGISTRY_CACHE: dict[str, Any] | None = None


def normalize_provider(provider: str | None) -> str:
    raw = str(provider or "").strip().lower()
    normalized = _PROVIDER_ALIASES.get(raw, raw)
    if normalized in SUPPORTED_PROVIDERS:
        return normalized
    raise ValueError(
        f"Unsupported embedding provider: {provider!r}. "
        f"Expected one of {', '.join(SUPPORTED_PROVIDERS)}."
    )


def _registry_path() -> Path:
    configured = str(
        os.environ.get(
            "MOLLYGRAPH_EMBEDDING_CONFIG_FILE",
            str(config.GRAPH_MEMORY_DIR / "embedding_config.json"),
        )
    ).strip()
    return Path(configured).expanduser()


def _fallback_registry_path() -> Path:
    return Path("/tmp/mollygraph-embedding-config.json")


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
    hf_model = str(getattr(config, "EMBEDDING_MODEL", "") or _DEFAULT_HF_MODEL).strip() or _DEFAULT_HF_MODEL
    ollama_model = str(getattr(config, "OLLAMA_EMBED_MODEL", "") or _DEFAULT_OLLAMA_MODEL).strip() or _DEFAULT_OLLAMA_MODEL

    backend_raw = str(getattr(config, "EMBEDDING_BACKEND", "hash") or "hash").strip().lower()
    backend = _PROVIDER_ALIASES.get(backend_raw, backend_raw)
    if backend not in SUPPORTED_PROVIDERS:
        backend = "hash"

    active_model = ""
    if backend == "huggingface":
        active_model = hf_model
    elif backend == "ollama":
        active_model = ollama_model

    return {
        "active_provider": backend,
        "active_model": active_model,
        "models": {
            "huggingface": [hf_model],
            "ollama": [ollama_model],
        },
        "supported_providers": list(SUPPORTED_PROVIDERS),
    }


def _merge_registry(base: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    models_payload = payload.get("models")
    if isinstance(models_payload, dict):
        for provider in ("huggingface", "ollama"):
            extras_raw = models_payload.get(provider, [])
            extras = extras_raw if isinstance(extras_raw, list) else []
            merged = list(base["models"].get(provider, [])) + [str(item) for item in extras]
            base["models"][provider] = _dedupe_models(merged)

    active_provider_raw = payload.get("active_provider")
    try:
        active_provider = normalize_provider(str(active_provider_raw))
    except Exception:
        active_provider = str(base.get("active_provider", "hash"))

    active_model = str(payload.get("active_model") or "").strip()
    if active_provider == "hash":
        active_model = ""
    else:
        if active_model and active_model not in base["models"][active_provider]:
            base["models"][active_provider].insert(0, active_model)
        if not active_model:
            candidates = base["models"].get(active_provider, [])
            active_model = candidates[0] if candidates else ""

    base["active_provider"] = active_provider
    base["active_model"] = active_model
    base["supported_providers"] = list(SUPPORTED_PROVIDERS)
    return base


def _apply_runtime_config(registry: dict[str, Any]) -> None:
    """Sync per-provider model names from registry into config.

    IMPORTANT: This must NEVER override ``config.EMBEDDING_BACKEND``.
    ``config.py`` (+ env vars) is the sole authority for the active backend.
    The registry only maintains model lists so the API can report/switch them.
    Overriding EMBEDDING_BACKEND here was the root cause of the tier chain
    being bypassed (same class of bug as the extractor registry override).
    """
    active_model = str(registry.get("active_model") or "").strip()

    hf_models = _dedupe_models([str(m) for m in registry.get("models", {}).get("huggingface", [])])
    ollama_models = _dedupe_models([str(m) for m in registry.get("models", {}).get("ollama", [])])

    # Set per-provider model names (these are harmless — just model paths)
    if hf_models:
        config.EMBEDDING_MODEL = hf_models[0]
    if ollama_models:
        config.OLLAMA_EMBED_MODEL = ollama_models[0]

    # If the registry's active provider matches, update the model for that provider
    provider = str(registry.get("active_provider") or "hash").strip().lower()
    if provider == "huggingface" and active_model:
        config.EMBEDDING_MODEL = active_model
    elif provider == "ollama" and active_model:
        config.OLLAMA_EMBED_MODEL = active_model


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
            log.warning("Failed to parse embedding registry at %s", selected_path, exc_info=True)
    _apply_runtime_config(registry)
    return registry


def _save_registry_locked(registry: dict[str, Any]) -> None:
    payload = {
        "active_provider": registry.get("active_provider", "hash"),
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
            log.warning("Embedding registry path not writable: %s", path, exc_info=True)

    log.warning("Failed to persist embedding registry; keeping in-memory config only")


def initialize_embedding_registry() -> dict[str, Any]:
    global _REGISTRY_CACHE
    with _REGISTRY_LOCK:
        if _REGISTRY_CACHE is None:
            _REGISTRY_CACHE = _load_registry_locked()
        else:
            _apply_runtime_config(_REGISTRY_CACHE)
        return copy.deepcopy(_REGISTRY_CACHE)


def get_embedding_registry() -> dict[str, Any]:
    return initialize_embedding_registry()


def set_active_embedding_provider(provider: str, model: str | None = None) -> dict[str, Any]:
    global _REGISTRY_CACHE
    normalized_provider = normalize_provider(provider)
    selected_model = str(model or "").strip()

    with _REGISTRY_LOCK:
        if _REGISTRY_CACHE is None:
            _REGISTRY_CACHE = _load_registry_locked()

        registry = _REGISTRY_CACHE
        if normalized_provider == "hash":
            registry["active_provider"] = "hash"
            registry["active_model"] = ""
            _apply_runtime_config(registry)
            _save_registry_locked(registry)
            return copy.deepcopy(registry)

        provider_models = registry["models"].setdefault(normalized_provider, [])
        provider_models = _dedupe_models([str(item) for item in provider_models])
        registry["models"][normalized_provider] = provider_models

        if not selected_model:
            if registry.get("active_provider") == normalized_provider and str(registry.get("active_model") or "").strip():
                selected_model = str(registry["active_model"]).strip()
            elif provider_models:
                selected_model = provider_models[0]
            else:
                raise ValueError(
                    f"No model registered for provider '{normalized_provider}'. "
                    "Add one first via POST /embeddings/models."
                )

        if selected_model not in provider_models:
            provider_models.insert(0, selected_model)

        registry["active_provider"] = normalized_provider
        registry["active_model"] = selected_model
        _apply_runtime_config(registry)
        _save_registry_locked(registry)
        return copy.deepcopy(registry)


def add_embedding_model(provider: str, model: str, activate: bool = False) -> dict[str, Any]:
    global _REGISTRY_CACHE
    normalized_provider = normalize_provider(provider)
    if normalized_provider == "hash":
        raise ValueError("Provider 'hash' does not take model registrations.")

    model_name = str(model or "").strip()
    if not model_name:
        raise ValueError("Model name cannot be empty.")

    with _REGISTRY_LOCK:
        if _REGISTRY_CACHE is None:
            _REGISTRY_CACHE = _load_registry_locked()

        registry = _REGISTRY_CACHE
        current = [str(item) for item in registry["models"].setdefault(normalized_provider, [])]
        if model_name not in current:
            current.append(model_name)
        registry["models"][normalized_provider] = _dedupe_models(current)

        if activate:
            registry["active_provider"] = normalized_provider
            registry["active_model"] = model_name

        _apply_runtime_config(registry)
        _save_registry_locked(registry)
        return copy.deepcopy(registry)


def _probe_ollama_models(timeout_seconds: float = 2.0) -> dict[str, Any]:
    base_url = str(getattr(config, "OLLAMA_BASE_URL", "http://127.0.0.1:11434") or "http://127.0.0.1:11434").strip()
    endpoint = f"{base_url.rstrip('/')}/api/tags"
    try:
        import httpx

        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.get(endpoint)
            response.raise_for_status()
            payload = response.json()

        items = payload.get("models", []) if isinstance(payload, dict) else []
        models: list[str] = []
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    name = str(item.get("name") or "").strip()
                else:
                    name = str(item or "").strip()
                if name:
                    models.append(name)
        return {"reachable": True, "available_models": _dedupe_models(models), "error": ""}
    except Exception as exc:
        return {"reachable": False, "available_models": [], "error": str(exc)}


def get_embedding_status() -> dict[str, Any]:
    registry = get_embedding_registry()
    active_provider = str(registry.get("active_provider") or "hash")
    active_model = str(registry.get("active_model") or "")
    models = registry.get("models", {})

    hf_models = [str(m) for m in models.get("huggingface", [])]
    ollama_models = [str(m) for m in models.get("ollama", [])]

    hf_available = importlib.util.find_spec("sentence_transformers") is not None
    ollama_probe = _probe_ollama_models()
    ollama_available_models = [str(m) for m in ollama_probe.get("available_models", [])]
    ollama_reachable = bool(ollama_probe.get("reachable"))

    if active_provider == "hash":
        active_ready = True
    elif active_provider == "huggingface":
        active_ready = hf_available and bool(active_model)
    elif active_provider == "ollama":
        active_ready = ollama_reachable and (
            not active_model or active_model in ollama_available_models or not ollama_available_models
        )
    else:
        active_ready = False

    strict_ai = bool(getattr(config, "STRICT_AI", False))
    blocking_errors: list[str] = []
    if strict_ai and active_provider == "hash":
        blocking_errors.append(
            "strict_ai mode does not allow hash embeddings. "
            "Use huggingface or ollama via /embeddings/config."
        )
    if strict_ai and not active_ready:
        blocking_errors.append(
            f"Active embedding provider '{active_provider}' is not ready."
        )

    return {
        "active_provider": active_provider,
        "active_model": active_model,
        "active_ready": active_ready,
        "strict_ai": strict_ai,
        "runtime_profile": str(getattr(config, "RUNTIME_PROFILE", "hybrid") or "hybrid"),
        "blocking_errors": blocking_errors,
        "providers": {
            "hash": {
                "ready": True,
                "registered_models": [],
            },
            "huggingface": {
                "ready": hf_available,
                "dependency_installed": hf_available,
                "registered_models": _dedupe_models(hf_models),
                "default_model": _DEFAULT_HF_MODEL,
            },
            "ollama": {
                "ready": ollama_reachable,
                "base_url": str(getattr(config, "OLLAMA_BASE_URL", "http://127.0.0.1:11434")),
                "registered_models": _dedupe_models(ollama_models),
                "available_models": _dedupe_models(ollama_available_models),
                "error": str(ollama_probe.get("error") or ""),
            },
        },
    }
