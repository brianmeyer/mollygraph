"""Service/graph statistics router."""
from __future__ import annotations

import time

from fastapi import APIRouter, Request

from evolution.gliner_training import get_gliner_stats
from memory.graph import get_graph_summary, get_relationship_type_distribution

router = APIRouter(tags=["stats"])


@router.get("/stats")
async def stats_endpoint(request: Request) -> dict:
    started_at = float(getattr(request.app.state, "started_at", time.time()))
    uptime = max(0, int(time.time() - started_at))

    graph_summary = {
        "entity_count": 0,
        "relationship_count": 0,
        "episode_count": 0,
        "top_connected": [],
        "recent": [],
    }
    relationship_distribution: dict[str, int] = {}
    neo4j_status = "offline"

    try:
        graph_summary = get_graph_summary()
        relationship_distribution = get_relationship_type_distribution()
        neo4j_status = "online"
    except Exception:
        neo4j_status = "offline"

    gliner = get_gliner_stats()

    return {
        **graph_summary,
        "relationship_type_distribution": relationship_distribution,
        "gliner_training": {
            "examples_accumulated": int(gliner.get("examples_accumulated", 0)),
            "last_finetune": str(gliner.get("last_finetune_at", "")),
            "last_strategy": str(gliner.get("last_strategy", "")),
            "last_result": str(gliner.get("last_result", "")),
            "last_cycle_status": str(gliner.get("last_cycle_status", "")),
            "active_model": str(gliner.get("active_model", "")),
            "base_model": str(gliner.get("base_model", "")),
        },
        "neo4j_status": neo4j_status,
        "service_uptime_seconds": uptime,
    }
