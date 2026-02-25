"""Ingest and mutation API routes."""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, UTC
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel

import config
from memory.graph import VALID_REL_TYPES
from memory.models import ExtractionJob
from runtime_graph import get_graph_instance
from runtime_queue import get_queue_instance
from runtime_vector_store import get_vector_store_instance

from api.deps import (
    DeleteRelationshipRequest,
    PruneRequest,
    _delete_entity_and_vector,
    require_no_maintenance,
    require_runtime_ready,
    verify_api_key,
)

# Cypher-safe relationship type: uppercase letters, digits, underscores only
_REL_TYPE_SAFE_RE = re.compile(r'^[A-Z][A-Z0-9_]{0,63}$')

log = logging.getLogger("mollygraph")
router = APIRouter()


class _IngestBody(BaseModel):
    content: str
    source: str = "manual"
    priority: int = 1


@router.post("/ingest", operation_id="post_ingest")
@router.post("/extract", operation_id="post_extract_legacy")
async def ingest(
    content: str | None = None,
    source: str = "manual",
    priority: int = 1,
    _api_key: str = Depends(verify_api_key),
    body: _IngestBody | None = Body(None),
) -> dict[str, Any]:
    if body is not None and content is None:
        content = body.content
        source = body.source
        priority = body.priority
    if not content:
        raise HTTPException(status_code=422, detail="content is required (query param or JSON body)")
    require_runtime_ready()
    queue = get_queue_instance()

    job = ExtractionJob(
        content=content,
        source=source,
        priority=priority,
        reference_time=datetime.now(UTC),
    )
    job_id = await asyncio.to_thread(queue.submit, job)
    queue_depth = await asyncio.to_thread(queue.get_pending_count)

    return {
        "job_id": job_id,
        "status": "queued",
        "queue_depth": queue_depth,
    }


@router.delete("/entity/{name}", operation_id="delete_entity")
async def delete_entity_endpoint(
    name: str,
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """Delete an entity by name from Neo4j (DETACH DELETE) and from the vector store."""
    require_runtime_ready()
    require_no_maintenance()
    graph = get_graph_instance()
    vector_store = get_vector_store_instance()

    with graph.driver.session() as _session:
        rec = _session.run(
            """
            MATCH (e:Entity)
            WHERE toLower(e.name) = $name
               OR ANY(a IN coalesce(e.aliases, []) WHERE toLower(a) = $name)
            OPTIONAL MATCH (e)-[r]-()
            RETURN coalesce(e.id, toLower(e.name)) AS entity_id,
                   count(r) AS rels
            """,
            name=name.lower(),
        ).single()

    if not rec or rec["entity_id"] is None:
        raise HTTPException(status_code=404, detail=f"Entity '{name}' not found")

    entity_id: str = str(rec["entity_id"])
    rels_removed: int = int(rec["rels"]) if rec["rels"] is not None else 0

    deleted = await asyncio.to_thread(graph.delete_entity, name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Entity '{name}' not found")

    vector_removed = False
    if vector_store is not None:
        try:
            vector_removed = bool(await asyncio.to_thread(vector_store.remove_entity, entity_id))
        except Exception:
            log.debug("delete_entity: vector remove failed for %s", entity_id, exc_info=True)

    log.info(
        "delete_entity: name=%s entity_id=%s rels_removed=%d vector_removed=%s",
        name, entity_id, rels_removed, vector_removed,
    )
    return {
        "deleted": True,
        "entity": name,
        "relationships_removed": rels_removed,
        "vector_removed": vector_removed,
    }


@router.delete("/relationship", operation_id="delete_relationship")
async def delete_relationship_endpoint(
    req: DeleteRelationshipRequest,
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """Delete a specific relationship (or all relationships) between two entities."""
    require_runtime_ready()
    require_no_maintenance()
    graph = get_graph_instance()

    source = req.source.strip()
    target = req.target.strip()
    rel_type = req.rel_type.strip() if req.rel_type else None
    rel_type_validated: str | None = None  # set after validation

    if not source or not target:
        raise HTTPException(status_code=422, detail="source and target are required")

    if rel_type:
        # Cypher injection guard: validate rel_type against the known-safe allow-list
        # and the safe-character pattern before interpolating into the query.
        _rel_upper = rel_type.upper().replace(" ", "_")
        if _rel_upper not in VALID_REL_TYPES or not _REL_TYPE_SAFE_RE.match(_rel_upper):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Invalid rel_type '{rel_type}'. Must be one of the valid relationship types: "
                    + ", ".join(sorted(VALID_REL_TYPES))
                ),
            )
        rel_type_validated = _rel_upper
        with graph.driver.session() as _session:
            rec = _session.run(
                f"""
                MATCH (h:Entity {{name: $source}})-[r:`{rel_type_validated}`]-(t:Entity {{name: $target}})
                DELETE r
                RETURN count(r) AS deleted
                """,
                source=source,
                target=target,
            ).single()
        deleted_count = int(rec["deleted"]) if rec else 0
    else:
        with graph.driver.session() as _session:
            rec = _session.run(
                """
                MATCH (h:Entity {name: $source})-[r]-(t:Entity {name: $target})
                DELETE r
                RETURN count(r) AS deleted
                """,
                source=source,
                target=target,
            ).single()
        deleted_count = int(rec["deleted"]) if rec else 0

    log.info(
        "delete_relationship: source=%s target=%s rel_type=%s deleted=%d",
        source, target, rel_type_validated, deleted_count,
    )
    return {
        "deleted": deleted_count,
        "source": source,
        "target": target,
        "rel_type": rel_type_validated,
    }


@router.post("/entities/prune", operation_id="post_entities_prune")
async def prune_entities_endpoint(
    req: PruneRequest,
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """Bulk delete entities. Pass names list or set orphans=true."""
    require_runtime_ready()
    require_no_maintenance()
    graph = get_graph_instance()

    names_to_prune: list[str] = []
    if req.names:
        names_to_prune = [n.strip() for n in req.names if n.strip()]
    elif req.orphans:
        with graph.driver.session() as _session:
            rows = _session.run(
                """
                MATCH (e:Entity)
                WHERE NOT (e)--()
                RETURN e.name AS name
                """
            ).data()
        names_to_prune = [r["name"] for r in rows if r.get("name")]
    else:
        raise HTTPException(
            status_code=422,
            detail="Provide 'names' (list of entity names) or 'orphans': true",
        )

    pruned: list[str] = []
    vectors_removed = 0
    for name in names_to_prune:
        try:
            neo4j_ok, vec_ok = await _delete_entity_and_vector(name)
            if neo4j_ok:
                pruned.append(name)
            if vec_ok:
                vectors_removed += 1
        except Exception:
            log.warning("prune_entities: failed to delete %s", name, exc_info=True)

    log.info(
        "prune_entities: requested=%d pruned=%d vectors_removed=%d",
        len(names_to_prune), len(pruned), vectors_removed,
    )
    return {
        "pruned": len(pruned),
        "entities": pruned,
        "vectors_removed": vectors_removed,
    }


@router.get("/entities", operation_id="get_entities")
async def list_entities(
    limit: int = 50,
    offset: int = 0,
    type: str | None = None,
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """List entities with pagination. Filter by ?type=Person."""
    require_runtime_ready()
    graph = get_graph_instance()

    if type:
        cypher = (
            "MATCH (e:Entity) WHERE e.entity_type = $type "
            "RETURN e.name AS name, e.entity_type AS entity_type, "
            "e.confidence AS confidence, e.id AS id "
            "ORDER BY e.name ASC SKIP $offset LIMIT $limit"
        )
        count_cypher = "MATCH (e:Entity) WHERE e.entity_type = $type RETURN count(e) AS total"
        params: dict[str, Any] = {"type": type, "limit": limit, "offset": offset}
        count_params: dict[str, Any] = {"type": type}
    else:
        cypher = (
            "MATCH (e:Entity) "
            "RETURN e.name AS name, e.entity_type AS entity_type, "
            "e.confidence AS confidence, e.id AS id "
            "ORDER BY e.name ASC SKIP $offset LIMIT $limit"
        )
        count_cypher = "MATCH (e:Entity) RETURN count(e) AS total"
        params = {"limit": limit, "offset": offset}
        count_params = {}

    try:
        with graph.driver.session() as _session:
            rows = _session.run(cypher, **params).data()
            total_rec = _session.run(count_cypher, **count_params).single()
            total = int(total_rec["total"]) if total_rec else 0
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"entities_list_error: {exc}") from exc

    entities = [
        {
            "id": r.get("id"),
            "name": r.get("name"),
            "entity_type": r.get("entity_type"),
            "confidence": r.get("confidence"),
        }
        for r in rows
    ]

    return {
        "entities": entities,
        "total": total,
        "limit": limit,
        "offset": offset,
        "type_filter": type,
        "timestamp": datetime.now(UTC).isoformat(),
    }
