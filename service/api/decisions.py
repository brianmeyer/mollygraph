"""Decision trace API routes."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from runtime_graph import get_graph_instance

from api.deps import require_runtime_ready, verify_api_key

router = APIRouter()


class CreateDecisionRequest(BaseModel):
    decision: str
    reasoning: str
    alternatives: list[str] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    outcome: str
    decided_by: str
    related_entities: list[str] = Field(default_factory=list)
    preceded_by_decision_id: str | None = None
    source_episode_id: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    timestamp: datetime | None = None

    @field_validator("alternatives", "inputs", "related_entities")
    @classmethod
    def _normalize_list_values(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for raw in values:
            item = str(raw).strip()
            if not item:
                continue
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(item)
        return cleaned


@router.post("/decisions", operation_id="post_decisions")
async def create_decision(
    body: CreateDecisionRequest,
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    require_runtime_ready()
    graph = get_graph_instance()
    try:
        return graph.create_decision(
            decision=body.decision,
            reasoning=body.reasoning,
            alternatives=body.alternatives,
            inputs=body.inputs,
            outcome=body.outcome,
            decided_by=body.decided_by,
            related_entities=body.related_entities,
            preceded_by_decision_id=body.preceded_by_decision_id,
            source_episode_id=body.source_episode_id,
            confidence=body.confidence,
            timestamp=body.timestamp or datetime.now(UTC),
        )
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "not found" in message else 422
        raise HTTPException(status_code=status_code, detail=message) from exc


@router.get("/decisions", operation_id="get_decisions")
async def list_decisions(
    q: str | None = None,
    decided_by: str | None = None,
    limit: int = Query(default=20, ge=1, le=200),
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    require_runtime_ready()
    graph = get_graph_instance()
    items = graph.list_decisions(q=q, decided_by=decided_by, limit=limit)
    return {
        "items": items,
        "count": len(items),
        "limit": limit,
    }


@router.get("/decisions/{decision_id}", operation_id="get_decision")
async def get_decision(
    decision_id: str,
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    require_runtime_ready()
    graph = get_graph_instance()
    row = graph.get_decision(decision_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Decision '{decision_id}' not found")
    return row
