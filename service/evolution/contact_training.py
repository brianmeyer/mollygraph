"""Direct NER training example generator for structured contact data.

Contacts ingested via source="contacts_json" carry 100% clean labels because
they come from structured fields (employer, title, location, etc.).  This
module converts those fields into entity-span training examples for GLiNER
without needing GLiREL to find relationships first.

Training JSONL is written to:
    ~/.graph-memory/training/gliner/contact-ner-YYYYMMDD.jsonl

Each example looks like:
    {
        "episode_id": "contact-ner:<sha256-prefix>",
        "source_text": "John Doe works at Google as Director of Engineering in Mountain View, CA.",
        "source": "contact_ner",
        "extracted_entities": [
            {"text": "John Doe",                   "label": "Person"},
            {"text": "Google",                     "label": "Organization"},
            {"text": "Director of Engineering",    "label": "Concept"},
            {"text": "Mountain View, CA",          "label": "Place"}
        ],
        "extracted_relations": [],
        "quality_signals": {"contact_ner": true}
    }
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config

log = logging.getLogger(__name__)

# ── Entity label mapping ──────────────────────────────────────────────────────
_PERSON_LABEL = "Person"
_ORG_LABEL = "Organization"
_PLACE_LABEL = "Place"
_CONCEPT_LABEL = "Concept"


# ── Structured field extraction ───────────────────────────────────────────────

def _extract_field(text: str, field: str) -> str | None:
    """Extract a labeled field value from the structured contact string.

    Supports both "FieldName: value." and "FieldName: value\n" patterns.
    Returns stripped value or None if not present.
    """
    # Pattern: "Field: value<stop>" where stop is ". " / "\n" / end-of-string
    pattern = re.compile(
        rf"(?:^|\.\s+){re.escape(field)}:\s*([^.\n]+?)(?=\s*\.|$|\n)",
        re.IGNORECASE | re.MULTILINE,
    )
    m = pattern.search(text)
    if m:
        val = m.group(1).strip()
        return val if val else None

    # Simpler fallback: "Field: value" at any position, stop at period/newline/end
    pattern2 = re.compile(
        rf"{re.escape(field)}:\s*([^.\n]+)",
        re.IGNORECASE,
    )
    m2 = pattern2.search(text)
    if m2:
        val = m2.group(1).strip()
        return val if val else None

    return None


def _extract_contact_name(text: str) -> str | None:
    """Extract contact name from 'Name is a contact of Owner' pattern."""
    m = re.match(r"^(.+?)\s+is\s+a\s+contact\s+of\b", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def _extract_owner_name(text: str) -> str | None:
    """Extract graph-owner name from 'X is a contact of Owner' pattern."""
    m = re.match(r"^.+?\s+is\s+a\s+contact\s+of\s+(.+?)\.", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def _extract_relationship_note(text: str) -> str | None:
    """Extract free-text relationship note (e.g. 'Kellogg EMBA classmates')."""
    # "Relationship: ..." or "Notes: ..."
    for field in ("Relationship", "Notes", "Note"):
        val = _extract_field(text, field)
        if val:
            return val
    return None


# ── Natural language sentence builder ────────────────────────────────────────

def _build_natural_sentences(
    contact_name: str,
    employer: str | None,
    title: str | None,
    location: str | None,
    owner_name: str | None,
    relationship_note: str | None,
) -> list[str]:
    """Build GLiREL-friendly natural language sentences from parsed fields."""
    sentences: list[str] = []

    # Employment sentence
    if employer and title:
        sentences.append(f"{contact_name} works at {employer} as {title}.")
    elif employer:
        sentences.append(f"{contact_name} works at {employer}.")
    elif title:
        sentences.append(f"{contact_name} holds the title of {title}.")

    # Location sentence
    if location:
        sentences.append(f"{contact_name} is based in {location}.")

    # Relationship sentence
    if owner_name and relationship_note:
        sentences.append(f"{contact_name} and {owner_name} are {relationship_note}.")
    elif owner_name:
        sentences.append(f"{contact_name} is a contact of {owner_name}.")

    # Fallback: at minimum identify the person
    if not sentences:
        sentences.append(f"{contact_name} is a person.")

    return sentences


# ── NER span builder ──────────────────────────────────────────────────────────

def _build_entity_spans(
    contact_name: str,
    employer: str | None,
    title: str | None,
    location: str | None,
    owner_name: str | None,
    relationship_note: str | None,
) -> list[dict[str, str]]:
    """Build clean entity-span dicts from structured fields."""
    entities: list[dict[str, str]] = []
    seen: set[str] = set()

    def _add(text: str, label: str) -> None:
        key = text.strip().lower()
        if key and key not in seen:
            seen.add(key)
            entities.append({"text": text.strip(), "label": label})

    _add(contact_name, _PERSON_LABEL)
    if employer:
        _add(employer, _ORG_LABEL)
    if title:
        _add(title, _CONCEPT_LABEL)
    if location:
        _add(location, _PLACE_LABEL)
    if owner_name:
        _add(owner_name, _PERSON_LABEL)

    return entities


# ── Public API ────────────────────────────────────────────────────────────────

def reformat_contact_text(content: str) -> str | None:
    """Reformat a structured contact string into natural language for GLiREL extraction.

    Input (low GLiREL yield):
        "Sarah Smith is a contact of Brian Meyer. Email: sarah@acme.com.
         Employer: Acme Corp. Title: VP Engineering. Location: Chicago, IL."

    Output (GLiREL-friendly):
        "Sarah Smith works at Acme Corp as VP Engineering. Sarah Smith is based in
         Chicago, IL. Sarah Smith is a contact of Brian Meyer."

    Returns the reformatted string, or None if the content cannot be parsed as
    a contact (so the pipeline can fall back to the original text).
    """
    contact_name = _extract_contact_name(content)
    if not contact_name:
        return None

    employer = _extract_field(content, "Employer") or _extract_field(content, "Company")
    title = (
        _extract_field(content, "Title")
        or _extract_field(content, "Role")
        or _extract_field(content, "Job Title")
    )
    location = _extract_field(content, "Location") or _extract_field(content, "City")
    owner_name = _extract_owner_name(content)
    relationship_note = _extract_relationship_note(content)

    sentences = _build_natural_sentences(
        contact_name=contact_name,
        employer=employer,
        title=title,
        location=location,
        owner_name=owner_name,
        relationship_note=relationship_note,
    )

    return " ".join(sentences)


def generate_contact_ner_examples(content: str) -> list[dict[str, Any]]:
    """Generate NER training examples from a structured contact string.

    Returns a list of training example dicts (may be empty if parsing fails).
    """
    contact_name = _extract_contact_name(content)
    if not contact_name:
        log.debug("contact_training: could not extract contact name from content")
        return []

    employer = _extract_field(content, "Employer") or _extract_field(content, "Company")
    title = _extract_field(content, "Title") or _extract_field(content, "Role") or _extract_field(content, "Job Title")
    location = _extract_field(content, "Location") or _extract_field(content, "City")
    owner_name = _extract_owner_name(content)
    relationship_note = _extract_relationship_note(content)

    sentences = _build_natural_sentences(
        contact_name=contact_name,
        employer=employer,
        title=title,
        location=location,
        owner_name=owner_name,
        relationship_note=relationship_note,
    )
    entity_spans = _build_entity_spans(
        contact_name=contact_name,
        employer=employer,
        title=title,
        location=location,
        owner_name=owner_name,
        relationship_note=relationship_note,
    )

    if not entity_spans:
        log.debug("contact_training: no entity spans built for contact '%s'", contact_name)
        return []

    examples: list[dict[str, Any]] = []
    for sentence in sentences:
        # Only include entity spans that actually appear in this sentence
        spans_in_sentence = [
            span for span in entity_spans
            if span["text"].lower() in sentence.lower()
        ]
        if not spans_in_sentence:
            continue

        content_hash = hashlib.sha256(sentence.encode("utf-8")).hexdigest()[:16]
        episode_id = f"contact-ner:{content_hash}"

        examples.append({
            "episode_id": episode_id,
            "source_text": sentence,
            "source": "contact_ner",
            "extracted_entities": spans_in_sentence,
            "extracted_relations": [],
            "quality_signals": {"contact_ner": True},
        })

    log.debug(
        "contact_training: generated %d NER examples for contact '%s'",
        len(examples),
        contact_name,
    )
    return examples


def persist_contact_ner_examples(examples: list[dict[str, Any]]) -> str | None:
    """Write contact NER training examples to the daily JSONL file.

    Returns the file path written to, or None if nothing was written.
    """
    if not examples:
        return None

    training_dir: Path = config.TRAINING_DIR
    training_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_path = training_dir / f"contact-ner-{today}.jsonl"

    # Load existing episode IDs to avoid duplicates
    existing_ids: set[str] = set()
    if out_path.exists():
        try:
            with out_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                        eid = str(payload.get("episode_id") or "").strip()
                        if eid:
                            existing_ids.add(eid)
                    except json.JSONDecodeError:
                        pass
        except Exception:
            log.debug("contact_training: failed reading existing JSONL", exc_info=True)

    new_examples = [
        ex for ex in examples
        if str(ex.get("episode_id") or "").strip() not in existing_ids
    ]
    if not new_examples:
        return None

    try:
        with out_path.open("a", encoding="utf-8") as f:
            for ex in new_examples:
                f.write(json.dumps(ex, ensure_ascii=True) + "\n")
        log.info(
            "contact_training: wrote %d contact NER examples to %s",
            len(new_examples),
            out_path,
        )
        return str(out_path)
    except Exception:
        log.warning("contact_training: failed writing to %s", out_path, exc_info=True)
        return None


__all__ = [
    "generate_contact_ner_examples",
    "persist_contact_ner_examples",
    "reformat_contact_text",
]
