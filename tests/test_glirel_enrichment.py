from __future__ import annotations

from extraction.glirel_enrichment import GLiRELEnrichment


def test_relation_labels_include_synonyms() -> None:
    labels = GLiRELEnrichment._relation_labels(
        configured_relations={"works at", "founded", "located in"}
    )

    assert "works at" in labels
    assert "employed by" in labels
    assert "works for" in labels
    assert "founder of" in labels
    assert "co-founded" in labels
    assert "based in" in labels
    assert "headquartered in" in labels


def test_type_constraints_filter_invalid_directions() -> None:
    enrichment = GLiRELEnrichment()
    payload = [
        {"head_text": ["Alice"], "tail_text": ["Acme"], "label": "employed by", "score": 0.55},
        {"head_text": ["Acme"], "tail_text": ["Alice"], "label": "works for", "score": 0.99},
    ]
    entity_types = {
        "alice": {"person"},
        "acme": {"organization"},
    }

    relations = enrichment._normalize_relations(
        payload,
        entity_types_by_name=entity_types,
        configured_relations={"works at"},
    )

    assert len(relations) == 1
    assert relations[0]["head"] == "Alice"
    assert relations[0]["tail"] == "Acme"
    assert relations[0]["rel_type"] == "works_at"


def test_bidirectional_dedup_keeps_higher_score() -> None:
    enrichment = GLiRELEnrichment()
    payload = [
        {"head_text": ["Alice"], "tail_text": ["Bob"], "label": "parent of", "score": 0.38},
        {"head_text": ["Bob"], "tail_text": ["Alice"], "label": "father of", "score": 0.31},
    ]
    entity_types = {
        "alice": {"person"},
        "bob": {"person"},
    }

    relations = enrichment._normalize_relations(
        payload,
        entity_types_by_name=entity_types,
        configured_relations={"parent of"},
    )

    assert len(relations) == 1
    assert relations[0]["head"] == "Alice"
    assert relations[0]["tail"] == "Bob"
    assert relations[0]["rel_type"] == "parent_of"
    assert relations[0]["confidence"] == 0.38


def test_canonical_mapping_for_synonym_labels() -> None:
    enrichment = GLiRELEnrichment()
    payload = [
        {"head_text": ["Alice"], "tail_text": ["Acme"], "label": "employed by", "score": 0.42},
        {"head_text": ["Alice"], "tail_text": ["Acme"], "label": "founder of", "score": 0.44},
    ]
    entity_types = {
        "alice": {"person"},
        "acme": {"organization"},
    }

    relations = enrichment._normalize_relations(
        payload,
        entity_types_by_name=entity_types,
        configured_relations={"works at", "founded"},
    )

    labels = {row["rel_type"] for row in relations}
    assert "works_at" in labels
    assert "founded" in labels
