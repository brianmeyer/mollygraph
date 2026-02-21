"""Context retrieval router."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from memory.graph import query_entities_for_context, query_entity

router = APIRouter(tags=["context"])


class ContextRequest(BaseModel):
    entities: list[str] = Field(default_factory=list)
    max_relationships: int = Field(default=10, ge=1, le=100)


def _normalize_entities(raw: list[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for item in raw:
        value = str(item or "").strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(value)
    return normalized


def _missing_entities(entities: list[str]) -> list[str]:
    missing: list[str] = []
    for name in entities:
        try:
            if query_entity(name) is None:
                missing.append(name)
        except Exception:
            missing.append(name)
    return missing


@router.get("/context")
async def get_context_endpoint(entities: list[str] = Query(default_factory=list)) -> dict:
    entity_names = _normalize_entities(entities)
    try:
        context = query_entities_for_context(entity_names)
        missing = _missing_entities(entity_names)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Context lookup failed: {exc}") from exc

    return {
        "context": context,
        "entities_found": max(0, len(entity_names) - len(missing)),
        "entities_missing": missing,
    }


@router.post("/context")
async def post_context_endpoint(req: ContextRequest) -> dict:
    entity_names = _normalize_entities(req.entities)
    try:
        context = query_entities_for_context(entity_names)
        missing = _missing_entities(entity_names)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Context lookup failed: {exc}") from exc

    return {
        "context": context,
        "entities_found": max(0, len(entity_names) - len(missing)),
        "entities_missing": missing,
    }
