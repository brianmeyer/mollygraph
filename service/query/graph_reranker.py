"""Graph-aware reranking for MollyGraph query results.

Applies three scoring signals on top of raw vector similarity:

1. Neighborhood score — 1-hop neighbor count + avg relationship strength.
   Score = base * (1 + log(1 + neighbor_count) * NEIGHBOR_WEIGHT
                     + avg_strength * STRENGTH_WEIGHT)

2. Path distance bonus — if the query mentions entity A and a result is
   entity B, check A→B shortest path (≤ 2 hops).  Boost B by PATH_BONUS
   per hop closer (i.e., 2-hop: +PATH_BONUS, 1-hop: +2*PATH_BONUS).

3. Relationship type relevance — verbs in the query are matched against
   canonical relationship-type labels; results with matching rel types
   in their fact set get a small relevance lift.

4. Dedup by entity — merges graph+vector results for the same entity;
   merged items carry retrieval_source="combined".

5. A/B metrics — raw vector ranking and graph-reranked ranking are both
   logged so rank_improvement_avg and rerank_lift_pct can be tracked.
"""
from __future__ import annotations

import logging
import math
import re
import time
from typing import Any

import config

log = logging.getLogger(__name__)

# Relationship type → intent keywords (used for relevance matching)
_REL_INTENT_KEYWORDS: dict[str, list[str]] = {
    "WORKS_ON":        ["work", "working", "develop", "build", "building"],
    "WORKS_AT":        ["work", "employed", "job", "company", "at"],
    "KNOWS":           ["know", "friend", "colleague", "contact"],
    "USES":            ["use", "using", "uses", "tool", "with"],
    "LOCATED_IN":      ["located", "lives", "in", "city", "country"],
    "DISCUSSED_WITH":  ["discuss", "talked", "meeting", "with"],
    "INTERESTED_IN":   ["interest", "interested", "likes", "hobby"],
    "CREATED":         ["create", "created", "made", "built", "founded"],
    "MANAGES":         ["manage", "manages", "lead", "leads"],
    "DEPENDS_ON":      ["depend", "depends", "require", "requires"],
    "RELATED_TO":      ["related", "about"],
    "MENTIONS":        ["mention", "about"],
    "ATTENDS":         ["attend", "attends", "going to"],
    "COLLABORATES_WITH": ["collaborate", "partner", "together"],
    "MENTORED_BY":     ["mentor", "mentored", "teaches"],
    "REPORTS_TO":      ["report", "reports to", "under"],
    "ALUMNI_OF":       ["alumni", "graduated", "went to"],
    "STUDIED_AT":      ["study", "studied", "school", "university"],
}

# Small relevance boost per matching rel-type keyword hit (caps at 0.1)
_REL_RELEVANCE_BOOST = 0.05
_REL_RELEVANCE_MAX = 0.10


def _extract_query_verbs(query: str) -> list[str]:
    """Return lowercased words/bigrams from query for intent matching."""
    q_lower = query.lower()
    words = re.findall(r"[a-z]+", q_lower)
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    return words + bigrams


def _get_neighborhood_stats(
    driver: Any,  # neo4j Driver
    entity_name: str,
) -> tuple[int, float]:
    """Return (neighbor_count, avg_strength) for entity's 1-hop neighborhood."""
    try:
        with driver.session() as session:
            rec = session.run(
                """
                MATCH (e:Entity)
                WHERE toLower(e.name) = $name
                   OR ANY(a IN coalesce(e.aliases, []) WHERE toLower(a) = $name)
                OPTIONAL MATCH (e)-[r]-(neighbor:Entity)
                RETURN count(DISTINCT neighbor) AS neighbor_count,
                       avg(coalesce(r.strength, r.confidence, 0.5)) AS avg_strength
                """,
                name=entity_name.lower(),
            ).single()
        if rec is None:
            return 0, 0.0
        n = int(rec["neighbor_count"] or 0)
        s = float(rec["avg_strength"] or 0.0)
        return n, s
    except Exception as exc:
        log.debug("neighborhood_stats failed for %r: %s", entity_name, exc)
        return 0, 0.0


def _get_path_distance(
    driver: Any,
    from_name: str,
    to_name: str,
    max_hops: int = 2,
) -> int | None:
    """Return shortest path length (1 or 2 hops) or None if unreachable."""
    try:
        with driver.session() as session:
            rec = session.run(
                f"""
                MATCH p = shortestPath(
                    (a:Entity)-[*1..{max_hops}]-(b:Entity)
                )
                WHERE (toLower(a.name) = $from_name
                    OR ANY(al IN coalesce(a.aliases, []) WHERE toLower(al) = $from_name))
                  AND (toLower(b.name) = $to_name
                    OR ANY(bl IN coalesce(b.aliases, []) WHERE toLower(bl) = $to_name))
                RETURN length(p) AS dist
                LIMIT 1
                """,
                from_name=from_name.lower(),
                to_name=to_name.lower(),
            ).single()
        if rec is None:
            return None
        return int(rec["dist"])
    except Exception as exc:
        log.debug("path_distance failed %r→%r: %s", from_name, to_name, exc)
        return None


def _rel_relevance_boost(facts: list[dict[str, Any]], query_verbs: list[str]) -> float:
    """Compute a small relevance boost based on relationship-type/fact keyword overlap."""
    boost = 0.0
    seen_rel_types: set[str] = set()
    for fact in facts:
        rel_type = str(fact.get("relation_type") or fact.get("type") or "").upper().replace(" ", "_")
        if not rel_type or rel_type in seen_rel_types:
            continue
        seen_rel_types.add(rel_type)
        keywords = _REL_INTENT_KEYWORDS.get(rel_type, [])
        for kw in keywords:
            if kw in query_verbs:
                boost += _REL_RELEVANCE_BOOST
                break  # one match per rel type
    return min(boost, _REL_RELEVANCE_MAX)


# ── A/B metrics (in-memory, session-level) ────────────────────────────────────
import threading as _threading

_ab_lock = _threading.Lock()
_ab_stats: dict[str, Any] = {
    "total_reranked_queries": 0,
    "rank_improvements": [],   # list of int (positive = improved)
    "rerank_lift_count": 0,    # queries where top result changed
}


def _record_ab_metrics(
    original_order: list[str],
    reranked_order: list[str],
) -> None:
    """Record rank improvement between original and reranked orderings."""
    with _ab_lock:
        _ab_stats["total_reranked_queries"] += 1
        improvements: list[int] = []
        for new_rank, entity in enumerate(reranked_order):
            try:
                old_rank = original_order.index(entity)
                improvements.append(old_rank - new_rank)  # positive = moved up
            except ValueError:
                pass
        if improvements:
            avg_improvement = sum(improvements) / len(improvements)
            _ab_stats["rank_improvements"].append(avg_improvement)
            # Keep last 1000
            if len(_ab_stats["rank_improvements"]) > 1000:
                del _ab_stats["rank_improvements"][:500]

        if reranked_order and original_order:
            if reranked_order[0] != original_order[0]:
                _ab_stats["rerank_lift_count"] += 1


def get_ab_metrics() -> dict[str, Any]:
    """Return current A/B reranking metrics."""
    with _ab_lock:
        improvements = list(_ab_stats["rank_improvements"])
        total = _ab_stats["total_reranked_queries"]
        lift_count = _ab_stats["rerank_lift_count"]

    avg_improvement = round(sum(improvements) / len(improvements), 3) if improvements else 0.0
    rerank_lift_pct = round(lift_count / total * 100, 1) if total > 0 else 0.0

    return {
        "total_reranked_queries": total,
        "rank_improvement_avg": avg_improvement,
        "rerank_lift_pct": rerank_lift_pct,
        "lift_count": lift_count,
    }


# ── Main reranking entry point ────────────────────────────────────────────────

def graph_rerank(
    results: list[dict[str, Any]],
    query: str,
    query_entities: list[str],
    driver: Any,  # neo4j Driver
) -> list[dict[str, Any]]:
    """Merge, deduplicate, and rerank query results using graph signals.

    Args:
        results: merged list of result dicts from graph + vector branches.
            Each has keys: entity, facts, score (optional), retrieval_source.
        query: raw query string (for intent/verb matching).
        query_entities: entities extracted from the query (for path bonus).
        driver: live neo4j Driver instance.

    Returns:
        Reranked list of result dicts, each with 'graph_score' and
        'retrieval_source' set appropriately.
    """
    if not results:
        return results

    neighbor_weight = config.MOLLYGRAPH_RERANK_NEIGHBOR_WEIGHT
    strength_weight = config.MOLLYGRAPH_RERANK_STRENGTH_WEIGHT
    path_bonus = config.MOLLYGRAPH_RERANK_PATH_BONUS

    query_verbs = _extract_query_verbs(query)
    original_order = [r["entity"] for r in results]

    # ── Step 1: Deduplicate by entity, merge provenance ───────────────────────
    seen: dict[str, dict[str, Any]] = {}
    for r in results:
        key = r["entity"].lower()
        if key not in seen:
            seen[key] = dict(r)
            seen[key].setdefault("score", 0.5)
        else:
            # Merge: keep highest score, update retrieval_source → combined
            existing = seen[key]
            existing_src = existing.get("retrieval_source", "")
            new_src = r.get("retrieval_source", "")
            if existing_src != new_src:
                existing["retrieval_source"] = "combined"
            # Merge score (take max)
            existing["score"] = max(
                float(existing.get("score") or 0.5),
                float(r.get("score") or 0.5),
            )
            # Extend facts (deduplicated)
            existing_facts = existing.get("facts") or []
            new_facts = r.get("facts") or []
            merged_facts_map: dict[str, Any] = {}
            for f in existing_facts + new_facts:
                fkey = f"{f.get('relation_type','')}/{f.get('target','')}/{f.get('source','')}"
                if fkey not in merged_facts_map:
                    merged_facts_map[fkey] = f
            existing["facts"] = list(merged_facts_map.values())

    deduped = list(seen.values())

    # ── Step 2: Compute graph-aware scores ───────────────────────────────────
    _t0 = time.perf_counter()
    for item in deduped:
        base_score = float(item.get("score") or 0.5)
        entity_name = item["entity"]
        facts = item.get("facts") or []

        # 2a. Neighborhood score
        neighbor_count, avg_strength = _get_neighborhood_stats(driver, entity_name)
        neighborhood_boost = (
            math.log(1 + neighbor_count) * neighbor_weight
            + avg_strength * strength_weight
        )
        graph_score = base_score * (1.0 + neighborhood_boost)

        # 2b. Path distance bonus
        max_path_boost = 0.0
        for q_entity in query_entities[:3]:  # limit to top 3 query entities
            if q_entity.lower() == entity_name.lower():
                continue
            dist = _get_path_distance(driver, q_entity, entity_name)
            if dist is not None and dist <= 2:
                # 1-hop: +2*PATH_BONUS, 2-hop: +1*PATH_BONUS
                hops_bonus = path_bonus * (3 - dist)  # dist=1→2x, dist=2→1x
                max_path_boost = max(max_path_boost, hops_bonus)
        graph_score += max_path_boost

        # 2c. Relationship type relevance
        relevance = _rel_relevance_boost(facts, query_verbs)
        graph_score += relevance

        item["graph_score"] = round(graph_score, 6)
        item["graph_neighbor_count"] = neighbor_count
        item["graph_avg_strength"] = round(avg_strength, 4)

    graph_score_ms = (time.perf_counter() - _t0) * 1000
    log.debug("graph_rerank: scored %d items in %.1fms", len(deduped), graph_score_ms)

    # ── Step 3: Sort by graph_score descending ────────────────────────────────
    reranked = sorted(deduped, key=lambda x: x.get("graph_score", 0.0), reverse=True)
    reranked_order = [r["entity"] for r in reranked]

    # ── Step 4: A/B metrics ───────────────────────────────────────────────────
    _record_ab_metrics(original_order, reranked_order)

    return reranked
