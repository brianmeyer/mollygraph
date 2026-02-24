"""GLiREL synonym management: defaults, persistent JSON overrides, and auto-generation.

The code-level ``_RELATION_SYNONYM_GROUPS`` dict in ``glirel_enrichment.py``
acts as built-in defaults.  This module:

1. Provides a deterministic ``generate_synonyms_for_label`` that converts
   any relation label (e.g. ``WORKS_AT``, ``collaborates with``) into
   2–4 natural-language phrasings GLiREL works best with.

2. Loads / saves a persistent JSON file at
   ``~/.graph-memory/glirel_synonyms.json`` so that synonyms for
   auto-adopted or manually approved relation types survive restarts.

3. Exposes ``get_merged_synonyms(defaults)`` that the enrichment layer
   calls at inference time to get the full effective synonym map.

4. Exposes ``add_synonym_group(canonical, synonyms=None)`` that the
   auto-adoption pipeline calls whenever a new relation type is approved.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_SYNONYMS_PATH = Path.home() / ".graph-memory" / "glirel_synonyms.json"

# ---------------------------------------------------------------------------
# Deterministic synonym generation
# ---------------------------------------------------------------------------

# Suffix-based expansion rules.
# Each entry: (suffix_with_underscore, [template, ...]) where templates may
# use {base} (full base phrase), {stem} (part before suffix), {verb} (first
# word of stem).
_SUFFIX_RULES: list[tuple[str, list[str]]] = [
    ("_at",   ["{base}", "{verb}s at", "employed at"]),
    ("_for",  ["{base}", "{verb}s for", "working for"]),
    ("_in",   ["{base}", "based in", "situated in"]),
    ("_of",   ["{base}", "part of", "member of"]),
    ("_to",   ["{base}", "connected to", "reporting to"]),
    ("_by",   ["{base}", "done by", "handled by"]),
    ("_with", ["{base}", "{verb}s with", "co-{verb}s with"]),
    ("_on",   ["{base}", "{verb}s on", "contributing to"]),
]

# Prefix-based expansion rules.
_PREFIX_RULES: list[tuple[str, list[str]]] = [
    ("mentored_", ["{base}", "coached by", "guided by", "advised by"]),
    ("managed_",  ["{base}", "run by", "led by", "overseen by"]),
]

# Static overrides for well-known relation labels that need hand-crafted synonyms.
# Keys are lowercase with spaces (normalised label form).
_STATIC_SYNONYMS: dict[str, tuple[str, ...]] = {
    "works at":        ("works at",        "employed by",       "works for",        "affiliated with"),
    "works on":        ("works on",        "is working on",     "contributes to",   "developing"),
    "founded":         ("founded",         "founder of",        "co-founded",       "established"),
    "knows":           ("knows",           "is acquainted with","is friends with",  "connected to"),
    "uses":            ("uses",            "utilizes",          "relies on",        "works with"),
    "located in":      ("located in",      "based in",          "headquartered in", "situated in"),
    "discussed with":  ("discussed with",  "talked about with", "conversed with",   "spoke with about"),
    "interested in":   ("interested in",   "focused on",        "enthusiastic about","passionate about"),
    "created":         ("created",         "built",             "developed",        "authored"),
    "manages":         ("manages",         "leads",             "oversees",         "is responsible for"),
    "depends on":      ("depends on",      "requires",          "is dependent on",  "needs"),
    "related to":      ("related to",      "associated with",   "linked to",        "connected to"),
    "classmate of":    ("classmate of",    "studied with",      "went to school with","cohort of"),
    "studied at":      ("studied at",      "attended",          "enrolled at",      "went to school at"),
    "alumni of":       ("alumni of",       "graduated from",    "alumnus of",       "alum of"),
    "mentors":         ("mentors",         "coaches",           "guides",           "advises"),
    "mentored by":     ("mentored by",     "coached by",        "guided by",        "advised by"),
    "reports to":      ("reports to",      "answers to",        "works under",      "is managed by"),
    "collaborates with":("collaborates with","works together with","partners with",  "co-works with"),
    "customer of":     ("customer of",     "client of",         "subscriber of",    "account at"),
    "attends":         ("attends",         "goes to",           "enrolled in",      "is attending"),
    "parent of":       ("parent of",       "father of",         "mother of",        "guardian of"),
    "child of":        ("child of",        "son of",            "daughter of",      "born to"),
    "received from":   ("received from",   "got from",          "delivered by",     "sent by"),
    "contact of":      ("contact of",      "is in contact with","associated with",  "connected to"),
    "lives in":        ("lives in",        "resides in",        "based in",         "located at"),
    "member of":       ("member of",       "belongs to",        "part of",          "affiliated with"),
    # Preset-schema relations
    "owns":            ("owns",            "is owner of",       "possesses",        "holds"),
    "assigned to":     ("assigned to",     "allocated to",      "given to",         "working on"),
    "blocked by":      ("blocked by",      "waiting on",        "dependent on",     "held up by"),
    "delivers":        ("delivers",        "ships",             "produces",         "outputs"),
    "reports":         ("reports",         "files",             "submits",          "raises"),
    "affects":         ("affects",         "impacts",           "touches",          "influences"),
    "requested by":    ("requested by",    "asked for by",      "initiated by",     "raised by"),
    "resolved by":     ("resolved by",     "fixed by",          "closed by",        "handled by"),
}


def _label_to_normalized(label: str) -> str:
    """Return lowercase-with-spaces canonical form for any label variant."""
    return label.strip().lower().replace("_", " ")


def generate_synonyms_for_label(label: str) -> tuple[str, ...]:
    """Generate 2–4 natural-language synonym phrasings for a relation type label.

    First checks the static override table; if missing, applies suffix/prefix
    pattern rules; falls back to the base phrase plus a simple s-verb variant.

    Args:
        label: Relation type like ``'WORKS_AT'``, ``'works at'``, or ``'KNOWS'``.

    Returns:
        Tuple of 1–4 distinct natural-language phrasings.  The normalised base
        phrase is always first.
    """
    base = _label_to_normalized(label)
    if not base:
        return ()

    # Static table takes priority.
    if base in _STATIC_SYNONYMS:
        return _STATIC_SYNONYMS[base]

    normalized_key = base.replace(" ", "_")
    parts = base.split()
    verb = parts[0] if parts else base

    synonyms: list[str] = [base]

    # Prefix rules
    for prefix, templates in _PREFIX_RULES:
        if normalized_key.startswith(prefix):
            rest = " ".join(normalized_key[len(prefix):].split("_"))
            for tmpl in templates:
                variant = tmpl.format(base=base, stem=rest, verb=rest.split()[0] if rest else rest)
                variant = variant.strip()
                if variant and variant not in synonyms:
                    synonyms.append(variant)
            if len(synonyms) >= 2:
                break

    # Suffix rules (only if prefix didn't produce enough)
    if len(synonyms) < 2:
        for suffix, templates in _SUFFIX_RULES:
            if normalized_key.endswith(suffix):
                stem_raw = normalized_key[: -len(suffix)]
                stem = " ".join(stem_raw.split("_"))
                stem_verb = stem.split()[0] if stem.split() else stem
                for tmpl in templates:
                    variant = tmpl.format(base=base, stem=stem, verb=stem_verb)
                    variant = variant.strip()
                    if variant and variant not in synonyms:
                        synonyms.append(variant)
                break

    # Generic fallback: add a simple present-tense s-verb variant.
    if len(synonyms) < 2 and not verb.endswith(("s", "ed", "ing")):
        s_variant = (verb + "s " + " ".join(parts[1:])).strip() if len(parts) > 1 else verb + "s"
        if s_variant not in synonyms:
            synonyms.append(s_variant)

    return tuple(synonyms[:4])


# ---------------------------------------------------------------------------
# Persistent JSON store
# ---------------------------------------------------------------------------


def load_synonyms() -> dict[str, tuple[str, ...]]:
    """Load GLiREL synonyms from ``~/.graph-memory/glirel_synonyms.json``.

    Returns an empty dict when the file does not exist or cannot be parsed.
    """
    if not _SYNONYMS_PATH.exists():
        return {}

    try:
        payload = json.loads(_SYNONYMS_PATH.read_text(encoding="utf-8"))
    except Exception:
        log.debug("Failed to load glirel_synonyms.json", exc_info=True)
        return {}

    if not isinstance(payload, dict):
        return {}

    raw = payload.get("synonyms", {})
    if not isinstance(raw, dict):
        return {}

    result: dict[str, tuple[str, ...]] = {}
    for canonical, variants in raw.items():
        if not isinstance(canonical, str) or not canonical.strip():
            continue
        key = _label_to_normalized(canonical)
        if isinstance(variants, list):
            result[key] = tuple(str(v) for v in variants if str(v).strip())
        elif isinstance(variants, str) and variants.strip():
            result[key] = (variants.strip(),)

    return result


def save_synonyms(synonyms: dict[str, tuple[str, ...]]) -> None:
    """Atomically write the GLiREL synonym store to disk."""
    _SYNONYMS_PATH.parent.mkdir(parents=True, exist_ok=True)

    serializable: dict[str, list[str]] = {
        canonical: list(variants)
        for canonical, variants in sorted(synonyms.items())
        if variants
    }

    payload: dict = {
        "version": 1,
        "synonyms": serializable,
    }

    try:
        tmp = _SYNONYMS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        tmp.replace(_SYNONYMS_PATH)
    except Exception:
        log.warning("Failed to save glirel_synonyms.json", exc_info=True)


def add_synonym_group(
    canonical: str,
    synonyms: tuple[str, ...] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Add or update synonyms for a relation label, persisting to JSON.

    If ``synonyms`` is ``None``, phrasings are auto-generated from the label.

    Args:
        canonical: Relation label (any case / separator style is normalised).
        synonyms:  Optional explicit tuple of phrasings.  Auto-generated when
                   omitted.

    Returns:
        The updated full synonym dict (defaults NOT included — JSON store only).
    """
    normalized = _label_to_normalized(canonical)
    if not normalized:
        return load_synonyms()

    if synonyms is None:
        synonyms = generate_synonyms_for_label(normalized)

    # Ensure the canonical form itself is present.
    variants: list[str] = list(synonyms)
    if normalized not in variants:
        variants.insert(0, normalized)

    current = load_synonyms()
    current[normalized] = tuple(variants[:4])
    save_synonyms(current)
    log.info("GLiREL synonym group added/updated for '%s': %s", normalized, variants[:4])
    return current


def get_merged_synonyms(
    defaults: dict[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    """Return effective synonym map: hardcoded defaults merged with JSON store.

    JSON store entries override defaults for any key that appears in both,
    allowing user/auto-adoption customisation without editing source code.
    """
    merged = dict(defaults)
    merged.update(load_synonyms())
    return merged
