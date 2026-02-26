from __future__ import annotations

from evolution.contact_training import generate_contact_ner_examples, reformat_contact_text


def test_kellogg_relationship_note_normalized_to_canonical_school_and_metadata() -> None:
    content = (
        "Jane Doe is a contact of Brian Meyer. "
        "Employer: Databricks. "
        "Relationship: LinkedIn Kellogg EMBA classmates, Miami campus class of 2026."
    )

    reformatted = reformat_contact_text(content)
    assert reformatted is not None
    assert "classmates from Kellogg School of Management (Miami campus, Class of 2026)" in reformatted

    examples = generate_contact_ner_examples(content)
    entity_texts = {e["text"] for row in examples for e in row.get("extracted_entities", [])}
    assert "Kellogg School of Management" in entity_texts


def test_low_confidence_linkedin_note_does_not_emit_classmate_assertion() -> None:
    content = (
        "Jane Doe is a contact of Brian Meyer. "
        "Employer: Databricks. "
        "Relationship: LinkedIn connection from networking event."
    )

    reformatted = reformat_contact_text(content)
    assert reformatted is not None
    assert "classmates" not in reformatted.lower()
    assert "is a contact of Brian Meyer" in reformatted
