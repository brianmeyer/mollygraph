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

import asyncio
import concurrent.futures
import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

# Module-level ThreadPoolExecutor singleton shared by all _llm_enrich_synonyms
# calls.  Creating a new executor per call is wasteful (thread creation overhead)
# and can exhaust OS thread limits under concurrent synonym generation.
# max_workers=1 is intentional: LLM calls are I/O-bound and we want at most one
# outstanding synonym request at a time to avoid hammering the audit LLM.
_LLM_SYNONYM_EXECUTOR: concurrent.futures.ThreadPoolExecutor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="glirel-synonyms",
)

_SYNONYMS_PATH = Path.home() / ".graph-memory" / "glirel_synonyms.json"

# ---------------------------------------------------------------------------
# Deterministic synonym generation
# ---------------------------------------------------------------------------

# Suffix-based expansion rules.
# Each entry: (suffix_with_underscore, [template, ...]) where templates may use:
#   {base}  — full natural-language base phrase (e.g. "works at")
#   {stem}  — words before the suffix (e.g. "work")
#   {verb}  — first word of stem, already in infinitive/stem form
#   {verb3} — third-person singular present form of {verb} (e.g. "works")
_SUFFIX_RULES: list[tuple[str, list[str]]] = [
    ("_at",   ["{base}", "{verb3} at", "employed at"]),
    ("_for",  ["{base}", "{verb3} for", "working for"]),
    ("_in",   ["{base}", "based in", "situated in"]),
    ("_of",   ["{base}", "part of", "member of"]),
    ("_to",   ["{base}", "connected to", "reporting to"]),
    ("_by",   ["{base}", "done by", "handled by"]),
    ("_with", ["{base}", "{verb3} with", "co-{verb3} with"]),
    ("_on",   ["{base}", "{verb3} on", "contributing to"]),
]

# Prefix-based expansion rules (same template placeholders as _SUFFIX_RULES).
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


def _to_present_third_person(verb: str) -> str:
    """Return the third-person singular present form of a verb stem.

    E.g. ``'work'`` → ``'works'``, ``'advise'`` → ``'advises'``.
    Already-conjugated forms (ending in 's', 'es') are returned as-is.
    """
    if not verb:
        return verb
    if verb.endswith(("ss", "sh", "ch", "x", "z")):
        return verb + "es"
    if verb.endswith("s"):
        return verb  # already plural/3rd-person
    if verb.endswith("e"):
        return verb + "s"
    return verb + "s"


def _verb_stem(conjugated: str) -> str:
    """Return the approximate infinitive stem of a conjugated verb.

    Strips a trailing plain 's' that is not part of the root, e.g.
    ``'works'`` → ``'work'``, ``'advises'`` → ``'advise'``.
    Does NOT touch 'ss', 'ous', etc.
    """
    if len(conjugated) > 3 and conjugated.endswith("s") and not conjugated.endswith("ss"):
        return conjugated[:-1]
    return conjugated


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

    synonyms: list[str] = [base]

    # Prefix rules
    for prefix, templates in _PREFIX_RULES:
        if normalized_key.startswith(prefix):
            rest_raw = normalized_key[len(prefix):]
            rest = " ".join(rest_raw.split("_"))
            rest_verb_stem = _verb_stem(rest.split()[0]) if rest.split() else rest
            for tmpl in templates:
                variant = tmpl.format(
                    base=base,
                    stem=rest,
                    verb=rest_verb_stem,
                    verb3=_to_present_third_person(rest_verb_stem),
                ).strip()
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
                # Stem may be a conjugated verb (e.g. 'works') — strip the 's' to get stem.
                stem_verb_stem = _verb_stem(stem.split()[0]) if stem.split() else stem
                for tmpl in templates:
                    variant = tmpl.format(
                        base=base,
                        stem=stem,
                        verb=stem_verb_stem,
                        verb3=_to_present_third_person(stem_verb_stem),
                    ).strip()
                    if variant and variant not in synonyms:
                        synonyms.append(variant)
                break

    # Generic fallback: add a simple present-tense s-verb variant.
    if len(synonyms) < 2:
        verb_stem = _verb_stem(parts[0]) if parts else base
        s_variant = (
            (_to_present_third_person(verb_stem) + " " + " ".join(parts[1:])).strip()
            if len(parts) > 1
            else _to_present_third_person(verb_stem)
        )
        if s_variant and s_variant not in synonyms:
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


def _llm_enrich_synonyms(label: str) -> list[str]:
    """Call the audit LLM to generate additional natural-language synonym phrasings.

    This is **optional and non-fatal**: returns an empty list if the feature
    is disabled (``MOLLYGRAPH_GLIREL_LLM_SYNONYMS=false``), the LLM is
    unavailable, or any error occurs.

    Runs the async ``call_audit_model`` in a fresh thread/event-loop so it is
    safe to call from a synchronous context even when an outer async event loop
    is already running (e.g. inside the nightly audit pipeline).

    Args:
        label: Normalised relation label, e.g. ``'consulted on'``.

    Returns:
        Up to 3 additional lowercase short phrasings from the LLM.
    """
    # Feature gate — check env var directly to avoid import-time side-effects.
    enabled = os.environ.get("MOLLYGRAPH_GLIREL_LLM_SYNONYMS", "true").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        log.debug("GLiREL LLM synonym enrichment disabled via env var.")
        return []

    # Lazy import to avoid circular deps at module load time.
    try:
        try:
            from audit.llm_audit import call_audit_model
        except ImportError:
            from service.audit.llm_audit import call_audit_model  # type: ignore[no-redef]
    except ImportError:
        log.debug("audit.llm_audit not importable; skipping LLM synonym enrichment.")
        return []

    prompt = (
        f"Generate 3 short natural language phrasings for the relationship type "
        f"'{label}' between two entities. "
        f"Return only the phrasings, one per line, no explanations, no numbering."
    )

    async def _call() -> dict:
        return await call_audit_model(prompt, schedule="nightly")

    try:
        # Reuse the module-level executor so we don't create a new thread per call.
        # Submitting asyncio.run() to a dedicated thread is safe whether or not an
        # outer async event loop is already running (they run in separate threads).
        result: dict = _LLM_SYNONYM_EXECUTOR.submit(asyncio.run, _call()).result(timeout=30)
    except Exception:
        log.debug("LLM synonym enrichment call failed (non-fatal).", exc_info=True)
        return []

    content: str = result.get("content", "") if isinstance(result, dict) else ""
    if not content or result.get("skipped"):
        return []

    extras: list[str] = []
    for line in content.splitlines():
        # Strip common list prefixes: "1. ", "- ", "• ", etc.
        phrase = line.strip().lower().lstrip("-•*0123456789.) \t")
        if phrase and len(phrase) < 80:
            extras.append(phrase)
        if len(extras) >= 3:
            break

    log.debug("LLM synonym enrichment for '%s' returned: %s", label, extras)
    return extras


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

    auto_generated = synonyms is None
    if synonyms is None:
        synonyms = generate_synonyms_for_label(normalized)

    # Ensure the canonical form itself is present.
    variants: list[str] = list(synonyms)
    if normalized not in variants:
        variants.insert(0, normalized)

    # Optionally enrich with LLM-generated phrasings (non-fatal, cap total at 5).
    # Only attempt enrichment when synonyms were auto-generated (not explicitly provided),
    # and only when there is room below the cap.
    if auto_generated and len(variants) < 5:
        llm_extras = _llm_enrich_synonyms(normalized)
        for phrase in llm_extras:
            if phrase and phrase not in variants:
                variants.append(phrase)
            if len(variants) >= 5:
                break

    current = load_synonyms()
    current[normalized] = tuple(variants[:5])
    save_synonyms(current)
    log.info("GLiREL synonym group added/updated for '%s': %s", normalized, variants[:5])
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
