"""Deduplication engine — adapted from Molly memory/dedup.py.

Changes from original:
- Replaced Kimi verification with Gemini Flash (using GOOGLE_API_KEY)
- Import from service config
- All algorithmic scoring logic preserved verbatim
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import combinations
from typing import Iterable, Mapping

log = logging.getLogger(__name__)

AliasMap = Mapping[str, Iterable[str]]

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_MULTI_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class DedupConfig:
    """Configuration for dedup scoring and near-duplicate decisions."""

    near_duplicate_threshold: float = 0.82
    max_length_delta: int = 6
    sequence_weight: float = 0.55
    edit_weight: float = 0.4
    token_weight: float = 0.05
    acronym_floor: float = 0.9
    alias_floor: float = 0.95
    enable_acronym_helper: bool = True
    enable_alias_helper: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.near_duplicate_threshold <= 1:
            raise ValueError("near_duplicate_threshold must be in [0, 1]")
        if self.max_length_delta < 0:
            raise ValueError("max_length_delta must be >= 0")
        if self.sequence_weight < 0 or self.edit_weight < 0 or self.token_weight < 0:
            raise ValueError("weights must be >= 0")
        if (self.sequence_weight + self.edit_weight + self.token_weight) <= 0:
            raise ValueError("at least one weight must be > 0")
        if not 0 <= self.acronym_floor <= 1:
            raise ValueError("acronym_floor must be in [0, 1]")
        if not 0 <= self.alias_floor <= 1:
            raise ValueError("alias_floor must be in [0, 1]")


@dataclass(frozen=True)
class DedupScore:
    """Detailed score for a single name pair."""

    left: str
    right: str
    canonical_left: str
    canonical_right: str
    compact_left: str
    compact_right: str
    length_delta: int
    sequence_ratio: float
    edit_similarity: float
    token_overlap: float
    compact_match: bool
    acronym_match: bool
    alias_match: bool
    score: float


def canonical_normalize(value: str) -> str:
    """Canonical normalization used for all dedup comparisons."""
    if not value:
        return ""
    folded = unicodedata.normalize("NFKD", value)
    without_marks = "".join(ch for ch in folded if not unicodedata.combining(ch))
    lowered = without_marks.lower().replace("&", " and ")
    cleaned = _NON_ALNUM_RE.sub(" ", lowered)
    return _MULTI_SPACE_RE.sub(" ", cleaned).strip()


def acronym_key(value: str) -> str:
    tokens = [tok for tok in canonical_normalize(value).split(" ") if tok]
    if not tokens:
        return ""
    if len(tokens) == 1:
        token = tokens[0]
        return token if len(token) <= 4 else token[0]
    return "".join(tok[0] for tok in tokens if tok and tok[0].isalnum())


def aliases_equivalent(left: str, right: str, aliases: AliasMap | None) -> bool:
    if not aliases:
        return False
    lookup = _prepare_alias_lookup(aliases)
    left_key = _compact_key(left)
    right_key = _compact_key(right)
    if not left_key or not right_key:
        return False
    return right_key in lookup.get(left_key, set())


def score_pair(
    left: str,
    right: str,
    *,
    config: DedupConfig | None = None,
    aliases: AliasMap | None = None,
) -> DedupScore:
    cfg = config or DedupConfig()
    alias_lookup = _prepare_alias_lookup(aliases) if aliases else {}
    return _score_pair_prepared(left, right, config=cfg, alias_lookup=alias_lookup)


def is_near_duplicate(
    left: str,
    right: str,
    *,
    config: DedupConfig | None = None,
    aliases: AliasMap | None = None,
) -> bool:
    cfg = config or DedupConfig()
    scored = score_pair(left, right, config=cfg, aliases=aliases)
    if scored.compact_match or scored.alias_match:
        return True
    if scored.length_delta > cfg.max_length_delta and not scored.acronym_match:
        return False
    return scored.score >= cfg.near_duplicate_threshold


def find_near_duplicates(
    names: Iterable[str],
    *,
    config: DedupConfig | None = None,
    aliases: AliasMap | None = None,
) -> list[DedupScore]:
    cfg = config or DedupConfig()
    cleaned = [name for name in names if isinstance(name, str) and name.strip()]
    alias_lookup = _prepare_alias_lookup(aliases) if aliases else {}

    matches: list[DedupScore] = []
    for left, right in combinations(cleaned, 2):
        ordered_left, ordered_right = sorted((left, right))
        scored = _score_pair_prepared(ordered_left, ordered_right, config=cfg, alias_lookup=alias_lookup)
        if scored.compact_match or scored.alias_match:
            matches.append(scored)
            continue
        if scored.length_delta > cfg.max_length_delta and not scored.acronym_match:
            continue
        if scored.score >= cfg.near_duplicate_threshold:
            matches.append(scored)

    return sorted(
        matches,
        key=lambda item: (
            -item.score,
            item.canonical_left,
            item.canonical_right,
            item.left,
            item.right,
        ),
    )


def _score_pair_prepared(
    left: str,
    right: str,
    *,
    config: DedupConfig,
    alias_lookup: Mapping[str, set[str]],
) -> DedupScore:
    canonical_left = canonical_normalize(left)
    canonical_right = canonical_normalize(right)
    compact_left = _compact_from_canonical(canonical_left)
    compact_right = _compact_from_canonical(canonical_right)
    length_delta = abs(len(compact_left) - len(compact_right))

    sequence_ratio = SequenceMatcher(None, canonical_left, canonical_right).ratio()
    max_len = max(len(compact_left), len(compact_right), 1)
    edit_distance = _levenshtein(compact_left, compact_right)
    edit_similarity = 1.0 - (edit_distance / max_len)
    token_overlap = _token_jaccard(canonical_left, canonical_right)

    compact_match = bool(compact_left) and compact_left == compact_right
    acronym_match = (
        config.enable_acronym_helper
        and _acronym_equivalent(left, right, compact_left, compact_right)
    )
    alias_match = config.enable_alias_helper and _alias_equivalent_from_lookup(
        compact_left, compact_right, alias_lookup
    )

    score = _blended_score(
        sequence_ratio=sequence_ratio,
        edit_similarity=edit_similarity,
        token_overlap=token_overlap,
        config=config,
    )
    if compact_match:
        score = 1.0
    elif acronym_match:
        score = max(score, config.acronym_floor)
    if alias_match:
        score = max(score, config.alias_floor)
    score = max(0.0, min(1.0, score))

    return DedupScore(
        left=left,
        right=right,
        canonical_left=canonical_left,
        canonical_right=canonical_right,
        compact_left=compact_left,
        compact_right=compact_right,
        length_delta=length_delta,
        sequence_ratio=sequence_ratio,
        edit_similarity=edit_similarity,
        token_overlap=token_overlap,
        compact_match=compact_match,
        acronym_match=acronym_match,
        alias_match=alias_match,
        score=score,
    )


def _blended_score(
    *,
    sequence_ratio: float,
    edit_similarity: float,
    token_overlap: float,
    config: DedupConfig,
) -> float:
    total_weight = config.sequence_weight + config.edit_weight + config.token_weight
    return (
        (sequence_ratio * config.sequence_weight)
        + (edit_similarity * config.edit_weight)
        + (token_overlap * config.token_weight)
    ) / total_weight


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = {token for token in left.split(" ") if token}
    right_tokens = {token for token in right.split(" ") if token}
    if not left_tokens or not right_tokens:
        return 0.0
    if len(left_tokens) == 1 and len(right_tokens) == 1:
        return 1.0 if left_tokens == right_tokens else 0.5
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _acronym_equivalent(
    left: str,
    right: str,
    compact_left: str,
    compact_right: str,
) -> bool:
    left_acronym = acronym_key(left)
    right_acronym = acronym_key(right)
    if not left_acronym or not right_acronym:
        return False
    return (
        left_acronym == compact_right
        or right_acronym == compact_left
        or left_acronym == right_acronym
    )


def _prepare_alias_lookup(aliases: AliasMap | None) -> dict[str, set[str]]:
    lookup: dict[str, set[str]] = {}
    if not aliases:
        return lookup
    for canonical, alt_names in aliases.items():
        group = {_compact_key(canonical)}
        group.update(_compact_key(alias) for alias in alt_names)
        group.discard("")
        if len(group) < 2:
            continue
        for key in group:
            lookup.setdefault(key, set()).update(group)
    return lookup


def _alias_equivalent_from_lookup(
    left_compact: str,
    right_compact: str,
    lookup: Mapping[str, set[str]],
) -> bool:
    if not left_compact or not right_compact:
        return False
    return right_compact in lookup.get(left_compact, set())


def _compact_key(value: str) -> str:
    return _compact_from_canonical(canonical_normalize(value))


def _compact_from_canonical(value: str) -> str:
    return value.replace(" ", "")


def _levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for left_idx, left_ch in enumerate(left, start=1):
        current = [left_idx]
        for right_idx, right_ch in enumerate(right, start=1):
            insert_cost = current[right_idx - 1] + 1
            delete_cost = previous[right_idx] + 1
            replace_cost = previous[right_idx - 1] + (left_ch != right_ch)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


# ---------------------------------------------------------------------------
# Gemini Flash verification (replaces Kimi)
# ---------------------------------------------------------------------------

async def _gemini_verify_duplicates(
    candidates: list[DedupScore],
    batch_size: int = 20,
) -> list[DedupScore]:
    """Send duplicate candidates to Gemini Flash for verification.

    Returns only the candidates Gemini confirms as true duplicates.
    Falls back to high-confidence algorithmic matches if Gemini unavailable.
    """
    import config as _cfg

    api_key = getattr(_cfg, "GOOGLE_API_KEY", "")
    if not api_key:
        log.warning("GOOGLE_API_KEY not set — using algorithmic dedup only")
        return [c for c in candidates if c.score >= 0.90 or c.compact_match]

    model = "gemini-2.5-flash"
    base_url = getattr(_cfg, "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")

    verified: list[DedupScore] = []

    for batch_start in range(0, len(candidates), batch_size):
        batch = candidates[batch_start : batch_start + batch_size]
        pairs_desc = "\n".join(
            f'{i + 1}. "{c.left}" vs "{c.right}" (score={c.score:.2f}, '
            f'compact_match={c.compact_match}, acronym_match={c.acronym_match})'
            for i, c in enumerate(batch)
        )

        prompt = (
            "You are a knowledge graph deduplication auditor.\n\n"
            "Below are candidate duplicate entity pairs from a personal knowledge graph "
            "(extracted from conversations). For each pair, decide if they truly "
            "refer to the same entity and should be merged.\n\n"
            "Consider:\n"
            "- Name variations (nicknames, abbreviations, typos)\n"
            "- Different entities that happen to have similar names\n"
            "- Context: this is a personal assistant's graph with people, places, organizations, projects, technologies\n\n"
            f"Candidates:\n{pairs_desc}\n\n"
            "Return a JSON array of objects with:\n"
            '- "index": candidate number (1-based)\n'
            '- "merge": true if they are the same entity, false if different\n'
            '- "reason": brief explanation\n\n'
            "Return ONLY the JSON array."
        )

        try:
            raw = await _invoke_gemini_dedup(prompt, model, base_url, api_key)
            approved_indices = _parse_dedup_response(raw, len(batch))
            for idx in approved_indices:
                if 0 <= idx < len(batch):
                    verified.append(batch[idx])
        except Exception:
            log.warning("Gemini dedup verification failed for batch — using algorithmic fallback", exc_info=True)
            verified.extend(c for c in batch if c.score >= 0.90 or c.compact_match)

    log.info(
        "Dedup verification: %d candidates → %d Gemini-approved",
        len(candidates), len(verified),
    )
    return verified


async def _invoke_gemini_dedup(prompt: str, model: str, base_url: str, api_key: str) -> str:
    """Invoke Gemini Flash for dedup verification."""
    import asyncio
    import httpx

    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a knowledge graph deduplication auditor. Respond with valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 8192,
    }

    async with httpx.AsyncClient(timeout=120) as client:
        for attempt in range(3):
            try:
                resp = await client.post(
                    f"{base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                )
                if resp.status_code >= 400:
                    log.warning("Gemini dedup HTTP %d: %s", resp.status_code, resp.text[:500])
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except Exception as exc:
                if attempt == 2:
                    raise
                await asyncio.sleep(5 * (attempt + 1))
                log.debug("Gemini dedup retry %d: %s", attempt + 1, exc)

    return ""


def _parse_dedup_response(raw: str, batch_len: int) -> list[int]:
    """Parse Gemini's JSON response and return 0-based indices of approved merges."""
    import json as _json

    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

    try:
        verdicts = _json.loads(text)
    except _json.JSONDecodeError:
        log.warning("Failed to parse Gemini dedup response: %s", text[:200])
        return []

    if not isinstance(verdicts, list):
        return []

    approved: list[int] = []
    for v in verdicts:
        if not isinstance(v, dict):
            continue
        idx = v.get("index")
        merge = v.get("merge")
        if isinstance(idx, int) and merge is True and 1 <= idx <= batch_len:
            approved.append(idx - 1)
    return approved


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------

def _merge_entity_pair(driver, name_a: str, name_b: str) -> bool:
    """Merge entity name_b into name_a (keeping the higher-mention entity)."""
    with driver.session() as session:
        result = session.run(
            """MATCH (a:Entity {name: $a}), (b:Entity {name: $b})
               RETURN a.mention_count AS a_mentions, b.mention_count AS b_mentions""",
            a=name_a, b=name_b,
        )
        record = result.single()
        if not record:
            return False

        a_mentions = record["a_mentions"] or 0
        b_mentions = record["b_mentions"] or 0
        if b_mentions > a_mentions:
            keep, discard = name_b, name_a
        else:
            keep, discard = name_a, name_b

        # Transfer outgoing relationships
        session.run(
            """MATCH (discard:Entity {name: $discard})-[r]->(t)
               WHERE t.name <> $keep
               MATCH (keep:Entity {name: $keep})
               MERGE (keep)-[nr:RELATED_TO]->(t)
               ON CREATE SET nr.mention_count = r.mention_count,
                             nr.first_mentioned = r.first_mentioned,
                             nr.last_mentioned = r.last_mentioned
               ON MATCH SET nr.mention_count = nr.mention_count + coalesce(r.mention_count, 1)
               DELETE r""",
            discard=discard, keep=keep,
        )

        # Transfer incoming relationships
        session.run(
            """MATCH (s)-[r]->(discard:Entity {name: $discard})
               WHERE s.name <> $keep
               MATCH (keep:Entity {name: $keep})
               MERGE (s)-[nr:RELATED_TO]->(keep)
               ON CREATE SET nr.mention_count = r.mention_count,
                             nr.first_mentioned = r.first_mentioned,
                             nr.last_mentioned = r.last_mentioned
               ON MATCH SET nr.mention_count = nr.mention_count + coalesce(r.mention_count, 1)
               DELETE r""",
            discard=discard, keep=keep,
        )

        # Sum mention counts and delete the discard entity
        session.run(
            """MATCH (keep:Entity {name: $keep}), (discard:Entity {name: $discard})
               SET keep.mention_count = coalesce(keep.mention_count, 0) + coalesce(discard.mention_count, 0)
               DETACH DELETE discard""",
            keep=keep, discard=discard,
        )

    return True


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_dedup(dry_run: bool = False, batch_size: int = 20) -> dict:
    """Query Neo4j for all entity names, find near-duplicates, verify with Gemini, merge.

    Returns summary dict with counts and examples.
    """
    try:
        from memory.graph import get_driver
    except Exception:
        log.error("Cannot import graph driver for dedup", exc_info=True)
        return {"status": "error", "entities_merged": 0, "error": "neo4j unavailable"}

    driver = get_driver()
    with driver.session() as session:
        result = session.run("MATCH (e:Entity) RETURN e.name AS name")
        names = [r["name"] for r in result if r["name"]]

    # Filter out short names that cause bad merge proposals
    names = [n for n in names if len(n.strip()) >= 3]
    entities_scanned = len(names)
    if len(names) < 2:
        return {
            "status": "ok",
            "entities_scanned": entities_scanned,
            "candidates_found": 0,
            "gemini_approved": 0,
            "audit_model_approved": 0,
            "entities_merged": 0,
            "examples": [],
        }

    candidates = find_near_duplicates(names)
    candidates_found = len(candidates)
    if not candidates:
        return {
            "status": "ok",
            "entities_scanned": entities_scanned,
            "candidates_found": 0,
            "gemini_approved": 0,
            "audit_model_approved": 0,
            "entities_merged": 0,
            "examples": [],
        }

    # Gemini verification
    verified = await _gemini_verify_duplicates(candidates, batch_size=batch_size)
    gemini_approved = len(verified)

    merged = 0
    examples = []
    for dup in verified:
        if dry_run:
            examples.append({"kept": dup.left, "merged": dup.right, "score": round(dup.score, 3)})
            merged += 1
            continue
        try:
            success = _merge_entity_pair(driver, dup.left, dup.right)
            if success:
                merged += 1
                if len(examples) < 10:
                    examples.append({"kept": dup.left, "merged": dup.right, "score": round(dup.score, 3)})
                log.info("Merged duplicate: '%s' -> '%s' (score=%.2f)", dup.right, dup.left, dup.score)
        except Exception:
            log.debug("Failed to merge %s -> %s", dup.right, dup.left, exc_info=True)

    return {
        "status": "ok",
        "entities_scanned": entities_scanned,
        "candidates_found": candidates_found,
        "gemini_approved": gemini_approved,
        "audit_model_approved": gemini_approved,
        "entities_merged": merged,
        "examples": examples,
    }


__all__ = [
    "AliasMap",
    "DedupConfig",
    "DedupScore",
    "acronym_key",
    "aliases_equivalent",
    "canonical_normalize",
    "find_near_duplicates",
    "is_near_duplicate",
    "run_dedup",
    "score_pair",
]
