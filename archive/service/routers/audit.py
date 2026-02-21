"""Audit router."""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from audit.llm_audit import run_llm_audit

router = APIRouter(tags=["audit"])


class AuditRequest(BaseModel):
    limit: int = Field(default=500, ge=1, le=5000)
    dry_run: bool = False
    schedule: Literal["nightly", "weekly"] = "nightly"
    model: str | None = None


@router.post("/audit")
async def audit_endpoint(req: AuditRequest) -> dict:
    try:
        return await run_llm_audit(
            limit=req.limit,
            dry_run=req.dry_run,
            schedule=req.schedule,
            model_override=req.model,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Audit failed: {exc}") from exc
