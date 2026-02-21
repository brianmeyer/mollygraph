"""Extraction router."""
from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from memory.processor import process_text

router = APIRouter(tags=["extract"])

_EXTRACT_JOBS: dict[str, dict[str, Any]] = {}
_JOB_TTL_SECONDS = 3600


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source: str = "agent"
    threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    background: bool = True


class ExtractJobStatus(BaseModel):
    job_id: str
    status: str
    created_at: str
    updated_at: str
    result: dict[str, Any] | None = None
    error: str | None = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cleanup_jobs() -> None:
    now = time.time()
    stale_ids = [
        job_id
        for job_id, entry in _EXTRACT_JOBS.items()
        if now - float(entry.get("created_ts", now)) > _JOB_TTL_SECONDS
    ]
    for job_id in stale_ids:
        _EXTRACT_JOBS.pop(job_id, None)


async def _run_extract_job(job_id: str, req: ExtractRequest) -> None:
    entry = _EXTRACT_JOBS.get(job_id)
    if not entry:
        return

    entry["status"] = "running"
    entry["updated_at"] = _utc_now_iso()

    try:
        result = await process_text(req.text, source=req.source, threshold=req.threshold)
        entry["status"] = "completed"
        entry["result"] = result
        entry["updated_at"] = _utc_now_iso()
    except Exception as exc:
        entry["status"] = "failed"
        entry["error"] = str(exc)
        entry["updated_at"] = _utc_now_iso()


@router.post("/extract")
async def extract_endpoint(req: ExtractRequest) -> dict[str, Any]:
    """Extract entities/relationships from text and store to Neo4j.

    - background=true (default): queue async processing and return immediately.
    - background=false: process inline and return extraction summary.
    """
    _cleanup_jobs()

    if req.background:
        job_id = str(uuid.uuid4())
        now_iso = _utc_now_iso()
        _EXTRACT_JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": now_iso,
            "updated_at": now_iso,
            "created_ts": time.time(),
            "result": None,
            "error": None,
        }
        asyncio.create_task(_run_extract_job(job_id, req))
        return {
            "status": "accepted",
            "job_id": job_id,
            "message": "Extraction queued",
        }

    try:
        result = await process_text(req.text, source=req.source, threshold=req.threshold)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}") from exc

    return {
        "status": "ok",
        **result,
    }


@router.get("/extract/{job_id}", response_model=ExtractJobStatus)
async def extract_job_status(job_id: str) -> ExtractJobStatus:
    _cleanup_jobs()
    entry = _EXTRACT_JOBS.get(job_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Extract job not found")

    return ExtractJobStatus(
        job_id=job_id,
        status=str(entry.get("status") or "unknown"),
        created_at=str(entry.get("created_at") or ""),
        updated_at=str(entry.get("updated_at") or ""),
        result=entry.get("result"),
        error=entry.get("error"),
    )
