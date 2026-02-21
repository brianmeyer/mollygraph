"""Dedup router — async with job polling (like extract)."""
from __future__ import annotations

import asyncio
import time
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from memory.dedup import run_dedup

router = APIRouter(tags=["dedup"])

_DEDUP_JOBS: dict[str, dict] = {}


class DedupRequest(BaseModel):
    dry_run: bool = False
    batch_size: int = Field(default=20, ge=1, le=200)
    background: bool = True


async def _run_dedup_job(job_id: str, dry_run: bool, batch_size: int):
    started = time.monotonic()
    try:
        result = await run_dedup(dry_run=dry_run, batch_size=batch_size)
        result["duration_seconds"] = round(time.monotonic() - started, 3)
        _DEDUP_JOBS[job_id] = {"status": "completed", "result": result}
    except Exception as exc:
        _DEDUP_JOBS[job_id] = {"status": "failed", "error": str(exc)}


@router.post("/dedup")
async def dedup_endpoint(req: DedupRequest) -> dict:
    if req.background:
        job_id = str(uuid.uuid4())
        _DEDUP_JOBS[job_id] = {"status": "running"}
        asyncio.create_task(_run_dedup_job(job_id, req.dry_run, req.batch_size))
        return {"status": "accepted", "job_id": job_id}

    # Sync mode
    started = time.monotonic()
    try:
        result = await run_dedup(dry_run=req.dry_run, batch_size=req.batch_size)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Dedup failed: {exc}") from exc
    result["duration_seconds"] = round(time.monotonic() - started, 3)
    return result


@router.get("/dedup/{job_id}")
async def dedup_status(job_id: str) -> dict:
    job = _DEDUP_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
