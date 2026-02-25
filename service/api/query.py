"""Query API routes: GET /query, POST /query, GET /entity/{name}."""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, UTC
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import config
from extraction.pipeline import ExtractionPipeline
from metrics.stats_logger import log_retrieval
from runtime_graph import get_graph_instance
from runtime_vector_store import get_vector_store_instance

from api.deps import (
    EntityResponse,
    QueryResponse,
    _extract_query_entities,
    require_runtime_ready,
    verify_api_key,
)

log = logging.getLogger("mollygraph")
router = APIRouter()


class _PostQueryBody(BaseModel):
    q: str
    limit: int = Field(default=5, ge=1, le=50)


async def _run_query(q: str, result_limit: int = 5) -> QueryResponse:
    """Core query logic shared by GET and POST /query."""
    graph = get_graph_instance()
    vector_store = get_vector_store_instance()

    query_start = time.perf_counter()
    entity_extraction_ms = 0.0
    graph_exact_lookup_ms = 0.0
    graph_fuzzy_lookup_ms = 0.0
    embedding_ms = 0.0
    vector_search_ms = 0.0
    reranker_ms = 0.0

    _entity_extract_start = time.perf_counter()
    entities = _extract_query_entities(q)
    entity_extraction_ms = (time.perf_counter() - _entity_extract_start) * 1000

    # ── Parallel branch: graph (exact → fuzzy) ────────────────────────────────
    async def _graph_branch() -> tuple[list[dict[str, Any]], float, float, str]:
        _exact_start = time.perf_counter()
        _graph_results: list[dict[str, Any]] = []
        for entity_name in entities[:5]:
            facts = graph.get_current_facts(entity_name)
            if facts:
                _graph_results.append({
                    "entity": entity_name,
                    "facts": facts[:10],
                    "retrieval_source": "graph_exact",
                })
        _exact_ms = (time.perf_counter() - _exact_start) * 1000

        _fuzzy_ms = 0.0
        _source = "graph_exact" if _graph_results else "none"

        if not _graph_results:
            _fuzzy_start = time.perf_counter()
            try:
                with graph.driver.session() as _session:
                    for entity_name in entities[:5]:
                        _contains_q = entity_name.lower()
                        _rows = _session.run(
                            "MATCH (e:Entity) WHERE toLower(e.name) CONTAINS $q "
                            "RETURN e.name AS name LIMIT 5",
                            q=_contains_q,
                        )
                        for _row in _rows:
                            _name = str(_row.get("name") or "").strip()
                            if _name:
                                _facts = graph.get_current_facts(_name)
                                if _facts:
                                    _graph_results.append({
                                        "entity": _name,
                                        "facts": _facts[:10],
                                        "match": "fuzzy_contains",
                                        "retrieval_source": "graph_fuzzy",
                                    })
                        if _graph_results:
                            break
            except Exception:
                log.debug("Fuzzy CONTAINS fallback failed", exc_info=True)
            _fuzzy_ms = (time.perf_counter() - _fuzzy_start) * 1000
            if _graph_results:
                _source = "graph_fuzzy"

        return _graph_results, _exact_ms, _fuzzy_ms, _source

    # ── Parallel branch: vector similarity search ─────────────────────────────
    async def _vector_branch() -> tuple[list[dict[str, Any]], float, float]:
        if vector_store is None:
            return [], 0.0, 0.0

        _emb_start = time.perf_counter()
        try:
            embedding = await asyncio.to_thread(ExtractionPipeline._text_embedding, q)
        except Exception as exc:
            log.debug("Vector embedding failed: %s", exc)
            return [], (time.perf_counter() - _emb_start) * 1000, 0.0
        _emb_ms = (time.perf_counter() - _emb_start) * 1000

        _vec_start = time.perf_counter()
        try:
            vector_hits = await asyncio.to_thread(vector_store.similarity_search, embedding, 5)
        except Exception as exc:
            log.debug("Vector similarity_search failed: %s", exc)
            return [], _emb_ms, (time.perf_counter() - _vec_start) * 1000
        _vec_ms = (time.perf_counter() - _vec_start) * 1000

        _vec_results: list[dict[str, Any]] = []
        for item in vector_hits:
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            facts = graph.get_current_facts(name)
            if facts:
                _vec_results.append({
                    "entity": name,
                    "facts": facts[:5],
                    "score": item.get("score", 0.0),
                    "retrieval_source": "vector",
                })
        return _vec_results, _emb_ms, _vec_ms

    # ── Run both branches in parallel ─────────────────────────────────────────
    (graph_results, graph_exact_lookup_ms, graph_fuzzy_lookup_ms, graph_source), \
    (vector_results, embedding_ms, vector_search_ms) = await asyncio.gather(
        _graph_branch(),
        _vector_branch(),
    )

    # ── Merge: graph results have priority; vector fills gaps ─────────────────
    seen_entities: set[str] = set()
    results: list[dict[str, Any]] = []

    for r in graph_results:
        key = r["entity"].lower()
        if key not in seen_entities:
            seen_entities.add(key)
            results.append(r)

    for r in vector_results:
        key = r["entity"].lower()
        if key not in seen_entities:
            seen_entities.add(key)
            results.append(r)

    results = results[:result_limit]

    has_graph = bool(graph_results)
    has_vector = bool(vector_results)
    if has_graph and has_vector:
        retrieval_source = "combined"
    elif has_graph:
        retrieval_source = graph_source
    elif has_vector:
        retrieval_source = "vector"
    else:
        retrieval_source = "none"

    # ── Optional reranker ─────────────────────────────────────────────────────
    reranked = False
    if getattr(config, "RERANKER_ENABLED", False) and len(results) > 1:
        _rerank_start = time.perf_counter()
        try:
            reranker = await asyncio.to_thread(ExtractionPipeline._get_reranker_model)
            if reranker is not None:
                pairs = [(q, " ".join(str(f) for f in r.get("facts", []))) for r in results]
                scores = await asyncio.to_thread(reranker.predict, pairs)
                scored = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
                results = [r for _, r in scored]
                reranked = True
        except Exception as exc:
            log.debug("Reranker step failed: %s", exc)
        reranker_ms = (time.perf_counter() - _rerank_start) * 1000

    total_latency_ms = (time.perf_counter() - query_start) * 1000
    try:
        log_retrieval(
            query=q,
            retrieval_source=retrieval_source,
            result_count=len(results),
            latency_ms=total_latency_ms,
            vector_search_ms=vector_search_ms,
            embedding_ms=embedding_ms,
            entity_extraction_ms=entity_extraction_ms,
            graph_exact_lookup_ms=graph_exact_lookup_ms,
            graph_fuzzy_lookup_ms=graph_fuzzy_lookup_ms,
            entities_queried=entities,
            reranker_ms=reranker_ms,
            graph_result_count=len(graph_results),
            vector_result_count=len(vector_results),
            graph_entity_names=[r["entity"] for r in graph_results],
            vector_entity_names=[r["entity"] for r in vector_results],
        )
    except Exception:
        log.debug("metrics log_retrieval failed", exc_info=True)

    return QueryResponse(
        query=q,
        entities_found=entities,
        results=results,
        result_count=len(results),
        timestamp=datetime.now(UTC).isoformat(),
        reranked=reranked,
    )


@router.get("/query", response_model=QueryResponse)
async def query_get(
    q: str,
    limit: int = 5,
    _api_key: str = Depends(verify_api_key),
) -> QueryResponse:
    require_runtime_ready()
    return await _run_query(q, result_limit=limit)


@router.post("/query", response_model=QueryResponse, operation_id="post_query")
async def query_post(
    body: _PostQueryBody,
    _api_key: str = Depends(verify_api_key),
) -> QueryResponse:
    """POST alternative to GET /query; accepts {q, limit} in the request body."""
    require_runtime_ready()
    return await _run_query(body.q, result_limit=body.limit)


@router.get("/entity/{name}", response_model=EntityResponse)
async def get_entity(
    name: str,
    _api_key: str = Depends(verify_api_key),
) -> EntityResponse:
    require_runtime_ready()
    graph = get_graph_instance()
    facts = graph.get_current_facts(name)
    if not facts:
        raise HTTPException(status_code=404, detail=f"Entity '{name}' not found")
    context = graph.get_entity_context(name, hops=2)
    return EntityResponse(
        entity=name,
        facts=facts,
        context=context,
        timestamp=datetime.now(UTC).isoformat(),
    )
