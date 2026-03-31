from __future__ import annotations

from pathlib import Path
import sys

_HERE = Path(__file__).parent
_SERVICE_ROOT = _HERE.parent
if str(_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVICE_ROOT))

from extraction.pipeline import ExtractionPipeline


class _DummyGraph:
    pass


class _DummyVector:
    def similarity_search(self, _embedding, top_k: int = 10):
        return [{"entity_id": "dummy", "score": 1.0, "top_k": top_k}]


def _pipeline() -> ExtractionPipeline:
    return ExtractionPipeline(graph=_DummyGraph(), vector_store=_DummyVector())


def test_filter_speaker_relations_normalizes_first_person_head():
    pipeline = _pipeline()

    relations = [
        {"head": "I", "tail": "Acme", "label": "WORKS_AT", "score": 0.94},
    ]

    filtered = pipeline._filter_speaker_relations(
        relations,
        "Alice Example",
        entity_type_map={"acme": "Organization"},
    )

    assert filtered == [
        {"head": "Alice Example", "tail": "Acme", "label": "WORKS_AT", "score": 0.94},
    ]


def test_filter_speaker_relations_keeps_third_party_person_facts():
    pipeline = _pipeline()

    relations = [
        {"head": "Luke", "tail": "Google", "label": "WORKS_AT", "score": 0.88},
    ]

    filtered = pipeline._filter_speaker_relations(
        relations,
        "Alice Example",
        entity_type_map={"luke": "Person", "google": "Organization"},
    )

    assert filtered == relations


def test_filter_speaker_relations_drops_non_person_anchor_leakage():
    pipeline = _pipeline()

    relations = [
        {"head": "Kellogg", "tail": "Google", "label": "WORKS_AT", "score": 0.61},
    ]

    filtered = pipeline._filter_speaker_relations(
        relations,
        "Alice Example",
        entity_type_map={"kellogg": "Organization", "google": "Organization"},
    )

    assert filtered == []
