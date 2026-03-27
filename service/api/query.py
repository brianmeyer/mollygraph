"""Query API routes: GET /query, POST /query, GET /entity/{name}."""
from __future__ import annotations

import asyncio
import math
import logging
import time
from datetime import datetime, UTC
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import config
from extraction.pipeline import ExtractionPipeline
from metrics.retrieval_quality import compute_retrieval_quality
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

_REL_INTENT_KEYWORDS: dict[str, list[str]] = {
    "WORKS_ON": ["work", "working", "develop", "build", "building"],
    "WORKS_AT": ["work", "employed", "job", "company", "at"],
    "KNOWS": ["know", "friend", "colleague", "contact"],
    "USES": ["use", "using", "uses", "tool", "with"],
    "LOCATED_IN": ["located", "lives", "in", "city", "country"],
    "DISCUSSED_WITH": ["discuss", "talked", "meeting", "with"],
    "INTERESTED_IN": ["interest", "interested", "likes", "hobby"],
    "CREATED": ["create", "created", "made", "built", "founded"],
    "MANAGES": ["manage", "manages", "lead", "leads"],
    "DEPENDS_ON": ["depend", "depends", "require", "requires"],
    "RELATED_TO": ["related", "about"],
    "MENTIONS": ["mention", "about"],
    "ATTENDS": ["attend", "attends", "going to"],
    "COLLABORATES_WITH": ["collaborate", "partner", "together"],
    "MENTORED_BY": ["mentor", "mentored", "teaches"],
    "REPORTS_TO": ["report", "reports to", "under"],
    "ALUMNI_OF": ["alumni", "graduated", "went to"],
    "STUDIED_AT": ["study", "studied", "school", "university"],
}

_NAME_MATCH_BOOST = 1.5
_REL_RELEVANCE_BOOST = 0.05
_REL_RELEVANCE_MAX = 0.10


class _PostQueryBody(BaseModel):
    q: str
    limit: int = Field(default=5, ge=1, le=50)


def _extract_query_verbs(query: str) -> list[str]:
    words = [word for word in "".join(ch if ch.isalpha() else " " for ch in query.lower()).split() if word]
    bigrams = [f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)]
    return words + bigrams


def _rel_relevance_boost(facts: list[dict[str, Any]], query_verbs: list[str]) -> float:
    boost = 0.0
    seen_rel_types: set[str] = set()
    for fact in facts:
        rel_type = str(fact.get("relation_type") or fact.get("type") or "").upper().replace(" ", "_")
        if not rel_type or rel_type in seen_rel_types:
            continue
        seen_rel_types.add(rel_type)
        for keyword in _REL_INTENT_KEYWORDS.get(rel_type, []):
            if keyword in query_verbs:
                boost += _REL_RELEVANCE_BOOST
                break
    return min(boost, _REL_RELEVANCE_MAX)


def _graph_rerank_results(
    graph: Any,
    results: list[dict[str, Any]],
    query: str,
    query_entities: list[str],
) -> list[dict[str, Any]]:
    if not results:
        return results

    neighbor_weight = config.MOLLYGRAPH_RERANK_NEIGHBOR_WEIGHT
    strength_weight = config.MOLLYGRAPH_RERANK_STRENGTH_WEIGHT
    path_bonus = config.MOLLYGRAPH_RERANK_PATH_BONUS
    query_verbs = _extract_query_verbs(query)
    query_lower = query.lower()

    seen: dict[str, dict[str, Any]] = {}
    for result in results:
        key = result["entity"].lower()
        if key not in seen:
            seen[key] = dict(result)
            seen[key].setdefault("score", 0.5)
        else:
            existing = seen[key]
            if existing.get("retrieval_source", "") != result.get("retrieval_source", ""):
                existing["retrieval_source"] = "combined"
            existing["score"] = max(
                float(existing.get("score") or 0.5),
                float(result.get("score") or 0.5),
            )
            merged_facts: dict[str, Any] = {}
            for fact in (existing.get("facts") or []) + (result.get("facts") or []):
                fkey = f"{fact.get('relation_type', '')}/{fact.get('target', '')}/{fact.get('source', '')}"
                if fkey not in merged_facts:
                    merged_facts[fkey] = fact
            existing["facts"] = list(merged_facts.values())

    deduped = list(seen.values())
    for item in deduped:
        entity_name = str(item.get("entity") or "").strip()
        if not entity_name:
            continue

        base_score = float(item.get("score") or 0.5)
        facts = item.get("facts") or []

        name_lower = entity_name.lower()
        name_match = name_lower in query_lower or any(
            part in query_lower for part in name_lower.split() if len(part) > 2
        )
        neighbor_count, avg_strength = graph.get_neighborhood_stats(entity_name)
        graph_score = base_score * (
            1.0
            + math.log(1 + neighbor_count) * neighbor_weight
            + avg_strength * strength_weight
        )

        max_path_boost = 0.0
        for q_entity in query_entities[:3]:
            if q_entity.lower() == name_lower:
                continue
            dist = graph.get_path_distance(q_entity, entity_name)
            if dist is not None and dist <= 2:
                max_path_boost = max(max_path_boost, path_bonus * (3 - dist))
        graph_score += max_path_boost
        graph_score += _rel_relevance_boost(facts, query_verbs)
        if name_match:
            graph_score += _NAME_MATCH_BOOST

        item["graph_score"] = round(graph_score, 6)
        item["graph_neighbor_count"] = neighbor_count
        item["graph_avg_strength"] = round(avg_strength, 4)
        item["graph_name_match"] = name_match

    return sorted(deduped, key=lambda x: x.get("graph_score", 0.0), reverse=True)


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
                for entity_name in entities[:5]:
                    for _name in graph.find_entities_containing(entity_name, limit=5):
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

    # ── Optional graph-aware reranker ────────────────────────────────────────
    graph_reranked = False
    graph_rerank_ms = 0.0
    if getattr(config, "GRAPH_RERANK_ENABLED", False) and len(results) > 1:
        _gr_start = time.perf_counter()
        try:
            reranked_results = await asyncio.to_thread(
                _graph_rerank_results,
                graph,
                results,
                q,
                entities,
            )
            if reranked_results:
                results = reranked_results[:result_limit]
                graph_reranked = True
        except Exception as _gr_exc:
            log.debug("graph_rerank failed: %s", _gr_exc)
        graph_rerank_ms = (time.perf_counter() - _gr_start) * 1000
        log.debug("graph_rerank: %.1fms graph_reranked=%s", graph_rerank_ms, graph_reranked)

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

    # ── Retrieval quality metrics ──────────────────────────────────────────────
    quality_metrics: dict[str, Any] = {}
    try:
        quality_metrics = compute_retrieval_quality(results)
    except Exception as _qm_exc:
        log.debug("compute_retrieval_quality failed: %s", _qm_exc)

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
        graph_reranked=graph_reranked,
        quality_metrics=quality_metrics,
        retrieval_metadata={
            "total_latency_ms": round(total_latency_ms, 2),
            "entity_extraction_ms": round(entity_extraction_ms, 2),
            "embedding_ms": round(embedding_ms, 2),
            "vector_search_ms": round(vector_search_ms, 2),
            "graph_exact_lookup_ms": round(graph_exact_lookup_ms, 2),
            "graph_fuzzy_lookup_ms": round(graph_fuzzy_lookup_ms, 2),
            "reranker_ms": round(reranker_ms, 2),
            "graph_rerank_ms": round(graph_rerank_ms, 2),
            "retrieval_source": retrieval_source,
        },
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
