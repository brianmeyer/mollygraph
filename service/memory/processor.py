"""Graph processing pipeline — adapted from Molly memory/processor.py.

Changes from original:
- Removed embed_and_store / batch_embed_and_store (no vectorstore)
- Removed process_conversation / _append_daily_log (WhatsApp specific)
- Extended _ENTITY_BLOCKLIST with agent-runtime noise terms
- Added public process_text() function as the service's main entry point
- extract_to_graph now takes (content, source) without chat_jid
"""
from __future__ import annotations

import asyncio
import logging
import re

log = logging.getLogger(__name__)

# Extended blocklist: system artifacts, pronouns, agent-runtime noise
_ENTITY_BLOCKLIST = {
    # System artifacts
    "molly", "agent", "user", "heartbeat", "brian's approval", "approval",
    "context", "system", "assistant", "claude", "opus", "haiku", "sonnet",
    "grok", "gemini",
    # Pronouns / generic words that GLiNER2 over-extracts
    "him", "her", "his", "them", "it", "its", "they", "we", "you", "i",
    "me", "my", "mine", "your", "yours", "he", "she",
    # Common noise
    "someone", "something", "nothing", "everything", "anyone",
}

_MIN_ENTITY_LEN = 3  # Raised from 2 — single/double char entities are always noise


def _filter_entities(entities: list[dict]) -> list[dict]:
    """Remove noise entities: blocklisted, too short, or containing newlines."""
    filtered = []
    for ent in entities:
        name = ent.get("text", "").strip()
        if not name or len(name) < _MIN_ENTITY_LEN:
            continue
        if name.lower() in _ENTITY_BLOCKLIST:
            continue
        if "\n" in name:
            continue
        filtered.append(ent)
    return filtered


def _filter_relations(relations: list[dict]) -> list[dict]:
    """Remove relations involving blocked entities or self-references."""
    filtered = []
    for rel in relations:
        head = rel.get("head", "").strip()
        tail = rel.get("tail", "").strip()
        if not head or not tail:
            continue
        if head.lower() in _ENTITY_BLOCKLIST or tail.lower() in _ENTITY_BLOCKLIST:
            continue
        if head.lower() == tail.lower():
            continue
        filtered.append(rel)
    return filtered


_URL_ONLY_RE = re.compile(r"^\s*https?://\S+\s*$")
_HEARTBEAT_PREFIX = "HEARTBEAT CHECK"


def _is_junk_chunk(content: str) -> bool:
    """Return True if content is noise that shouldn't be processed."""
    text = content.strip()
    if len(text) < 20:
        return True
    if _URL_ONLY_RE.match(text):
        return True
    if text.startswith(_HEARTBEAT_PREFIX) or text.startswith(f"User: {_HEARTBEAT_PREFIX}"):
        return True
    return False


async def extract_to_graph(content: str, source: str = "agent", threshold: float = 0.4) -> dict:
    """Extract entities/relations from text and upsert to Neo4j.

    Returns summary dict with counts and episode_id.
    Raises ValueError for junk content.
    """
    if _is_junk_chunk(content):
        raise ValueError(f"Junk content rejected ({len(content.strip())} chars)")

    try:
        from memory.extractor import extract
        from memory import graph

        loop = asyncio.get_running_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, extract, content, threshold),
            timeout=30.0,
        )
        entities = _filter_entities(result["entities"])
        relations = _filter_relations(result["relations"])

        if not entities:
            return {
                "entities": [],
                "relations": [],
                "episode_id": None,
                "entities_stored": 0,
                "relations_stored": 0,
                "latency_ms": result.get("latency_ms", 0),
            }

        # Build raw→canonical name mapping so relationships use the right names
        raw_to_canonical: dict[str, str] = {}
        entity_results = []
        entity_names = []

        for ent in entities:
            canonical = await graph.upsert_entity(
                name=ent["text"],
                entity_type=ent["label"],
                confidence=ent["score"],
            )
            raw_to_canonical[ent["text"]] = canonical
            entity_names.append(canonical)
            entity_results.append({
                "text": ent["text"],
                "label": ent["label"],
                "score": ent["score"],
                "canonical": canonical,
            })

        relations_stored = 0
        relation_results = []
        for rel in relations:
            head = raw_to_canonical.get(rel["head"], rel["head"])
            tail = raw_to_canonical.get(rel["tail"], rel["tail"])
            await graph.upsert_relationship(
                head_name=head,
                tail_name=tail,
                rel_type=rel["label"],
                confidence=rel["score"],
                context_snippet=content[:200],
            )
            relations_stored += 1
            relation_results.append({
                "head": head,
                "tail": tail,
                "label": rel["label"],
                "score": rel["score"],
            })

        episode_id = await graph.create_episode(
            content_preview=content,
            source=source,
            entity_names=list(set(entity_names)),
        )

        log.debug(
            "Graph updated: %d entities, %d relations (%dms)",
            len(entities), relations_stored, result.get("latency_ms", 0),
        )

        return {
            "entities": entity_results,
            "relations": relation_results,
            "episode_id": episode_id,
            "entities_stored": len(entity_names),
            "relations_stored": relations_stored,
            "latency_ms": result.get("latency_ms", 0),
        }

    except ValueError:
        raise
    except Exception:
        log.error("Graph extraction failed", exc_info=True)
        raise


async def process_text(content: str, source: str = "agent", threshold: float = 0.4) -> dict:
    """Public API: extract entities/rels from text, upsert to Neo4j, return summary.

    This is the main entry point called by POST /extract.
    """
    return await extract_to_graph(content, source, threshold)
