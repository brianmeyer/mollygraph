"""Runtime registry for extractor schemas (default/preset/custom)."""
from __future__ import annotations

import copy
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

import yaml

import config

log = logging.getLogger(__name__)

SUPPORTED_SCHEMA_MODES = ("default", "preset", "custom")

DEFAULT_ENTITY_SCHEMA: dict[str, dict[str, Any]] = {
    "Person": {
        "description": "Full name or nickname of a person mentioned in conversation",
        "threshold": 0.4,
    },
    "Technology": {
        "description": "Programming language, framework, tool, platform, or technical system",
        "threshold": 0.4,
    },
    "Organization": {
        "description": "Company, institution, team, or named group",
        "threshold": 0.5,
    },
    "Project": {
        "description": "Named project, product, app, or initiative being worked on",
        "threshold": 0.45,
    },
    "Place": {
        "description": "City, country, region, office, or named location",
        "threshold": 0.5,
    },
    "Concept": {
        "description": "Abstract idea, field of study, methodology, or domain topic discussed",
        "threshold": 0.5,
    },
}

DEFAULT_RELATION_SCHEMA: dict[str, dict[str, Any]] = {
    "works on": {
        "description": "Person actively works on a project or task",
        "threshold": 0.45,
    },
    "works at": {
        "description": "Person is employed at or affiliated with an organization",
        "threshold": 0.5,
    },
    "knows": {
        "description": "Person knows or has a personal connection with another person",
        "threshold": 0.5,
    },
    "uses": {
        "description": "Person or project uses a technology, tool, or platform",
        "threshold": 0.45,
    },
    "located in": {
        "description": "Person, organization, or project is located in or based at a place",
        "threshold": 0.5,
    },
    "discussed with": {
        "description": "Person discussed a topic or entity with another person",
        "threshold": 0.5,
    },
    "interested in": {
        "description": "Person expressed interest in a topic, technology, or project",
        "threshold": 0.45,
    },
    "created": {
        "description": "Person or organization created or built a project or technology",
        "threshold": 0.5,
    },
    "manages": {
        "description": "Person manages or leads a project, team, or organization",
        "threshold": 0.5,
    },
    "depends on": {
        "description": "Project or technology depends on or requires another technology",
        "threshold": 0.5,
    },
    "related to": {
        "description": "General association between two entities discussed together",
        "threshold": 0.4,
    },
    "classmate of": {
        "description": "Person attended the same program, cohort, or school as another person",
        "threshold": 0.45,
    },
    "studied at": {
        "description": "Person attended or was enrolled at an educational institution",
        "threshold": 0.45,
    },
    "alumni of": {
        "description": "Person graduated from an educational institution or program",
        "threshold": 0.45,
    },
    "mentors": {
        "description": "Person mentors or advises another person",
        "threshold": 0.5,
    },
    "mentored by": {
        "description": "Person is mentored or advised by another person",
        "threshold": 0.5,
    },
    "reports to": {
        "description": "Person directly reports to another person in a management hierarchy",
        "threshold": 0.5,
    },
    "collaborates with": {
        "description": "Person works together with another person but not at the same organization",
        "threshold": 0.45,
    },
    "customer of": {
        "description": "Person is a customer, subscriber, or account holder at a company or service",
        "threshold": 0.5,
    },
    "attends": {
        "description": "Person attends or is enrolled at a school, program, or recurring event",
        "threshold": 0.45,
    },
    "parent of": {
        "description": "Person is the parent or guardian of another person (typically a child)",
        "threshold": 0.5,
    },
    "child of": {
        "description": "Person is the child of another person",
        "threshold": 0.5,
    },
    "received from": {
        "description": "Person received a delivery, package, email, or communication from an organization or person",
        "threshold": 0.5,
    },
    "contact of": {
        "description": "Person is a known contact of another person or organization",
        "threshold": 0.5,
    },
}

PRESET_SCHEMAS: dict[str, dict[str, Any]] = {
    "personal_network": {
        "name": "Personal Network",
        "description": "People, teams, and life context for personal memory use cases.",
        "entities": copy.deepcopy(DEFAULT_ENTITY_SCHEMA),
        "relations": copy.deepcopy(DEFAULT_RELATION_SCHEMA),
    },
    "project_delivery": {
        "name": "Project Delivery",
        "description": "Engineering and delivery planning across projects, tools, and teams.",
        "entities": {
            "Person": {"description": "Individual contributor, manager, or stakeholder", "threshold": 0.4},
            "Team": {"description": "Named engineering or cross-functional team", "threshold": 0.45},
            "Project": {"description": "Project, epic, product stream, or workstream", "threshold": 0.45},
            "Technology": {"description": "Language, framework, infra system, or service", "threshold": 0.4},
            "Artifact": {"description": "PR, ticket, doc, RFC, or release artifact", "threshold": 0.45},
            "Milestone": {"description": "Deadline, launch date, sprint goal, or checkpoint", "threshold": 0.5},
        },
        "relations": {
            "owns": {"description": "Person or team owns a project or artifact", "threshold": 0.45},
            "assigned to": {"description": "Artifact or task is assigned to a person or team", "threshold": 0.45},
            "blocked by": {"description": "Task, project, or milestone is blocked by another item", "threshold": 0.5},
            "depends on": {"description": "Project or artifact depends on another component", "threshold": 0.5},
            "uses": {"description": "Project or team uses a technology", "threshold": 0.45},
            "delivers": {"description": "Project delivers an artifact or milestone", "threshold": 0.45},
            "reports to": {"description": "Person reports to another person", "threshold": 0.5},
            "collaborates with": {"description": "Team or person collaborates with another", "threshold": 0.45},
        },
    },
    "customer_support": {
        "name": "Customer Support",
        "description": "Support operations around customers, accounts, incidents, and products.",
        "entities": {
            "Person": {"description": "Support agent, customer contact, or stakeholder", "threshold": 0.4},
            "Organization": {"description": "Customer company or vendor", "threshold": 0.45},
            "Product": {"description": "Product, service plan, or feature area", "threshold": 0.45},
            "Ticket": {"description": "Case, support ticket, or issue identifier", "threshold": 0.5},
            "Incident": {"description": "Outage, degraded service, or production event", "threshold": 0.5},
            "Channel": {"description": "Support channel like email, chat, or phone", "threshold": 0.45},
        },
        "relations": {
            "reports": {"description": "Customer or contact reports a ticket or incident", "threshold": 0.45},
            "owns": {"description": "Support agent or team owns a ticket", "threshold": 0.45},
            "affects": {"description": "Incident affects product or organization", "threshold": 0.5},
            "requested by": {"description": "Ticket or change requested by a contact", "threshold": 0.45},
            "resolved by": {"description": "Incident or ticket resolved by person or team", "threshold": 0.5},
            "related to": {"description": "General association across support entities", "threshold": 0.4},
            "uses": {"description": "Organization uses a product or plan", "threshold": 0.45},
            "contact of": {"description": "Person is contact for an organization", "threshold": 0.5},
        },
    },
}

_REGISTRY_LOCK = threading.Lock()
_REGISTRY_CACHE: dict[str, Any] | None = None


def _registry_path() -> Path:
    configured = str(
        os.environ.get(
            "MOLLYGRAPH_EXTRACTOR_SCHEMA_FILE",
            str(config.GRAPH_MEMORY_DIR / "extractor_schema.json"),
        )
    ).strip()
    return Path(configured).expanduser()


def _fallback_registry_path() -> Path:
    return Path("/tmp/mollygraph-extractor-schema.json")


def _normalize_threshold(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(0.0, min(1.0, numeric))


def _normalize_schema_items(
    raw: Any,
    *,
    item_type: str,
    default_description_prefix: str,
    default_threshold: float,
) -> dict[str, dict[str, Any]]:
    if raw is None:
        return {}

    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        items = [(str(item), {}) for item in raw]
    else:
        raise ValueError(f"{item_type} schema must be an object or list.")

    normalized: dict[str, dict[str, Any]] = {}
    for raw_name, raw_cfg in items:
        name = " ".join(str(raw_name or "").split()).strip()
        if not name:
            continue
        if item_type == "relation":
            name = name.lower()

        cfg = raw_cfg if isinstance(raw_cfg, dict) else {}
        description = str(cfg.get("description") or f"{default_description_prefix}: {name}").strip()
        threshold = _normalize_threshold(cfg.get("threshold"), default_threshold)
        normalized[name] = {"description": description, "threshold": threshold}
    return normalized


def _load_user_relation_schema() -> dict[str, dict[str, Any]]:
    """Load optional relation schema extensions from RELATION_SCHEMA_FILE."""
    path = getattr(config, "RELATION_SCHEMA_FILE", None)
    if path is None:
        return {}
    schema_path = path.expanduser()
    if not schema_path.exists():
        return {}

    try:
        payload = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    except Exception:
        log.warning("Failed to parse relation schema file: %s", schema_path, exc_info=True)
        return {}

    try:
        return _normalize_schema_items(
            payload,
            item_type="relation",
            default_description_prefix="User-defined relation",
            default_threshold=0.45,
        )
    except ValueError:
        log.warning("Unsupported relation schema format at %s", schema_path)
        return {}


def _default_registry_state() -> dict[str, Any]:
    return {
        "mode": "default",
        "active_preset": "",
        "custom_schema": {"entities": {}, "relations": {}},
    }


def _normalize_mode(value: str | None) -> str:
    mode = str(value or "default").strip().lower()
    if mode not in SUPPORTED_SCHEMA_MODES:
        raise ValueError(
            f"Unsupported schema mode: {value!r}. Expected one of {', '.join(SUPPORTED_SCHEMA_MODES)}."
        )
    return mode


def _parse_custom_schema(payload: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    entities = _normalize_schema_items(
        payload.get("entities"),
        item_type="entity",
        default_description_prefix="Custom entity",
        default_threshold=0.45,
    )
    relations = _normalize_schema_items(
        payload.get("relations"),
        item_type="relation",
        default_description_prefix="Custom relation",
        default_threshold=0.45,
    )
    if not entities:
        raise ValueError("Custom schema must include at least one entity type.")
    if not relations:
        raise ValueError("Custom schema must include at least one relation type.")
    return {"entities": entities, "relations": relations}


def _effective_schema_for_registry(registry: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    mode = str(registry.get("mode") or "default")
    active_preset = str(registry.get("active_preset") or "")

    if mode == "preset":
        preset = PRESET_SCHEMAS.get(active_preset)
        if preset is None:
            mode = "default"
        else:
            return {
                "entities": copy.deepcopy(preset["entities"]),
                "relations": copy.deepcopy(preset["relations"]),
            }

    if mode == "custom":
        custom = registry.get("custom_schema")
        if isinstance(custom, dict):
            try:
                return _parse_custom_schema(custom)
            except ValueError:
                pass
        mode = "default"

    if mode == "default":
        relations = copy.deepcopy(DEFAULT_RELATION_SCHEMA)
        relations.update(_load_user_relation_schema())
        return {
            "entities": copy.deepcopy(DEFAULT_ENTITY_SCHEMA),
            "relations": relations,
        }

    # Defensive fallback.
    return {
        "entities": copy.deepcopy(DEFAULT_ENTITY_SCHEMA),
        "relations": copy.deepcopy(DEFAULT_RELATION_SCHEMA),
    }


def _load_registry_locked() -> dict[str, Any]:
    registry = _default_registry_state()
    path_candidates = [_registry_path(), _fallback_registry_path()]

    selected_path: Path | None = None
    for candidate in path_candidates:
        if candidate.exists():
            selected_path = candidate
            break

    if selected_path is not None:
        try:
            payload = json.loads(selected_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                mode = payload.get("mode")
                active_preset = payload.get("active_preset")
                custom_schema = payload.get("custom_schema")

                try:
                    registry["mode"] = _normalize_mode(mode)
                except ValueError:
                    log.warning("Invalid schema mode in %s; using default", selected_path)

                if isinstance(active_preset, str) and active_preset in PRESET_SCHEMAS:
                    registry["active_preset"] = active_preset

                if isinstance(custom_schema, dict):
                    entities = custom_schema.get("entities")
                    relations = custom_schema.get("relations")
                    registry["custom_schema"] = {
                        "entities": entities if isinstance(entities, (dict, list)) else {},
                        "relations": relations if isinstance(relations, (dict, list)) else {},
                    }
        except Exception:
            log.warning("Failed to parse extractor schema registry at %s", selected_path, exc_info=True)

    # Ensure mode is usable with current payload.
    if registry["mode"] == "preset" and registry["active_preset"] not in PRESET_SCHEMAS:
        registry["mode"] = "default"
        registry["active_preset"] = ""
    if registry["mode"] == "custom":
        try:
            _parse_custom_schema(registry["custom_schema"])
        except ValueError:
            registry["mode"] = "default"

    return registry


def _save_registry_locked(registry: dict[str, Any]) -> None:
    payload = {
        "mode": registry.get("mode", "default"),
        "active_preset": registry.get("active_preset", ""),
        "custom_schema": registry.get("custom_schema", {"entities": {}, "relations": {}}),
    }
    for path in (_registry_path(), _fallback_registry_path()):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
            tmp.replace(path)
            return
        except OSError:
            log.warning("Extractor schema registry path not writable: %s", path, exc_info=True)
    log.warning("Failed to persist extractor schema registry; keeping in-memory config only")


def _serialize_preset_entry(key: str, payload: dict[str, Any], include_schema: bool) -> dict[str, Any]:
    entities = payload.get("entities", {})
    relations = payload.get("relations", {})
    out: dict[str, Any] = {
        "id": key,
        "name": str(payload.get("name") or key),
        "description": str(payload.get("description") or ""),
        "entity_count": len(entities) if isinstance(entities, dict) else 0,
        "relation_count": len(relations) if isinstance(relations, dict) else 0,
    }
    if include_schema:
        out["entities"] = copy.deepcopy(entities)
        out["relations"] = copy.deepcopy(relations)
    return out


def initialize_extractor_schema_registry() -> dict[str, Any]:
    global _REGISTRY_CACHE
    with _REGISTRY_LOCK:
        if _REGISTRY_CACHE is None:
            _REGISTRY_CACHE = _load_registry_locked()
        return copy.deepcopy(_REGISTRY_CACHE)


def get_extractor_schema_presets(include_schema: bool = False) -> dict[str, Any]:
    presets = [
        _serialize_preset_entry(key, value, include_schema=include_schema)
        for key, value in PRESET_SCHEMAS.items()
    ]
    return {"presets": presets, "preset_ids": [p["id"] for p in presets]}


def get_extractor_schema_status(*, include_schema: bool = True) -> dict[str, Any]:
    registry = initialize_extractor_schema_registry()
    mode = str(registry.get("mode") or "default")
    active_preset = str(registry.get("active_preset") or "")
    effective = _effective_schema_for_registry(registry)
    custom_schema = registry.get("custom_schema", {"entities": {}, "relations": {}})
    custom_ready = False
    if isinstance(custom_schema, dict):
        try:
            _parse_custom_schema(custom_schema)
            custom_ready = True
        except ValueError:
            custom_ready = False

    payload: dict[str, Any] = {
        "mode": mode,
        "active_preset": active_preset,
        "supported_modes": list(SUPPORTED_SCHEMA_MODES),
        "custom_schema_ready": custom_ready,
        "preset_catalog": get_extractor_schema_presets(include_schema=False)["presets"],
    }
    if include_schema:
        payload["schema"] = effective
    return payload


def get_effective_extractor_schema() -> dict[str, dict[str, dict[str, Any]]]:
    registry = initialize_extractor_schema_registry()
    return _effective_schema_for_registry(registry)


def set_active_extractor_schema(mode: str, preset: str | None = None) -> dict[str, Any]:
    global _REGISTRY_CACHE
    normalized_mode = _normalize_mode(mode)
    selected_preset = str(preset or "").strip()

    with _REGISTRY_LOCK:
        if _REGISTRY_CACHE is None:
            _REGISTRY_CACHE = _load_registry_locked()

        registry = _REGISTRY_CACHE
        if normalized_mode == "preset":
            if selected_preset not in PRESET_SCHEMAS:
                raise ValueError(
                    f"Unknown preset: {selected_preset!r}. "
                    f"Available presets: {', '.join(sorted(PRESET_SCHEMAS.keys()))}."
                )
            registry["active_preset"] = selected_preset
        elif normalized_mode == "custom":
            _parse_custom_schema(registry.get("custom_schema", {}))
            registry["active_preset"] = ""
        else:
            registry["active_preset"] = ""

        registry["mode"] = normalized_mode
        _save_registry_locked(registry)

    return get_extractor_schema_status(include_schema=True)


def upload_custom_extractor_schema(schema: dict[str, Any], activate: bool = False) -> dict[str, Any]:
    global _REGISTRY_CACHE
    normalized = _parse_custom_schema(schema)

    with _REGISTRY_LOCK:
        if _REGISTRY_CACHE is None:
            _REGISTRY_CACHE = _load_registry_locked()
        registry = _REGISTRY_CACHE
        registry["custom_schema"] = normalized
        if activate:
            registry["mode"] = "custom"
            registry["active_preset"] = ""
        _save_registry_locked(registry)

    status = get_extractor_schema_status(include_schema=True)
    status["custom_schema_updated"] = True
    status["activated"] = bool(activate)
    return status
