"""Training router."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from evolution.gliner_training import get_gliner_stats, run_gliner_finetune_pipeline

router = APIRouter(tags=["train"])


class TrainRequest(BaseModel):
    force: bool = False


@router.post("/train/gliner")
async def train_gliner_endpoint(req: TrainRequest) -> dict:
    try:
        return await run_gliner_finetune_pipeline(force=req.force)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"GLiNER training failed: {exc}") from exc


@router.get("/train/status")
async def train_status_endpoint() -> dict:
    try:
        return {"gliner": get_gliner_stats()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read training status: {exc}") from exc
