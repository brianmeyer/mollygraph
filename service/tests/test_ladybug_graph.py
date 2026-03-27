from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path
import sys

_HERE = Path(__file__).parent
_SERVICE_ROOT = _HERE.parent
if str(_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVICE_ROOT))

from memory.graph import LadybugGraph
from memory.models import Entity, Episode, Relationship


def _entity(*, name: str, entity_type: str, aliases: list[str] | None = None, created_from_episode: str = "ep-1") -> Entity:
    now = datetime.now(UTC)
    return Entity(
        name=name,
        entity_type=entity_type,
        aliases=list(aliases or []),
        first_mentioned=now,
        last_mentioned=now,
        confidence=0.9,
        created_from_episode=created_from_episode,
    )


def test_ladybug_graph_entity_and_episode_helpers(tmp_path):
    graph = LadybugGraph(tmp_path / "memory.lbug")

    alice_id, alice_new = graph.upsert_entity(_entity(name="Alice", entity_type="Person", aliases=["Alicia"]))
    acme_id, acme_new = graph.upsert_entity(_entity(name="Acme", entity_type="Organization"))

    assert alice_new is True
    assert acme_new is True
    assert graph.get_entity_id_by_name("Alice") == alice_id
    assert graph.get_entity_id_by_name("Alicia") == alice_id
    assert graph.find_entities_containing("lici", limit=5) == ["Alice"]

    graph.tag_entity_first_seen(alice_id, "manual")

    episode = Episode(
        id="ep-1",
        source="manual",
        content_preview="Alice mentioned Acme",
        content_hash="hash-1",
        occurred_at=datetime.now(UTC),
        entities_extracted=["Alice"],
    )
    episode_id = graph.create_episode(episode)
    assert episode_id == "ep-1"

    assert graph.incomplete_episode_count() == 0
    graph.mark_episode_incomplete("ep-1", "partial write")
    assert graph.incomplete_episode_count() == 1
    graph.finalize_episode("ep-1", ["Alice", "Acme"])
    assert graph.incomplete_episode_count() == 0

    rows, total = graph.list_entities_page(limit=10, offset=0, entity_type="Person")
    assert total == 1
    assert rows[0]["name"] == "Alice"

    summary = graph.get_graph_summary()
    assert summary["entity_count"] == 2
    assert summary["episode_count"] == 1


def test_ladybug_graph_relationship_queries_and_stats(tmp_path):
    graph = LadybugGraph(tmp_path / "relationships.lbug")
    graph.upsert_entity(_entity(name="Alice", entity_type="Person"))
    graph.upsert_entity(_entity(name="Acme", entity_type="Organization"))

    rel = Relationship(
        source_entity="Alice",
        target_entity="Acme",
        relation_type="WORKS_AT",
        confidence=0.95,
        valid_at=datetime.now(UTC),
        observed_at=datetime.now(UTC),
        context_snippets=["Alice works at Acme"],
        episode_ids=["ep-1"],
    )
    rel_id, status = graph.upsert_relationship(rel)

    assert status == "created"
    assert rel_id

    facts = graph.get_current_facts("Alice")
    assert facts[0]["rel_type"] == "WORKS_AT"
    assert facts[0]["target_name"] == "Acme"

    neighbor_count, avg_strength = graph.get_neighborhood_stats("Alice")
    assert neighbor_count == 1
    assert avg_strength > 0

    assert graph.get_path_distance("Alice", "Acme", max_hops=2) == 1

    context = graph.get_entity_context("Alice", hops=2)
    assert context["direct_connections"]
    assert context["direct_connections"][0]["target_name"] == "Acme"

    rel_distribution = graph.get_relationship_type_distribution()
    assert rel_distribution["WORKS_AT"] == 1
    assert graph.relationship_count() == 1


def test_ladybug_graph_delete_helpers_and_embedding_listing(tmp_path):
    graph = LadybugGraph(tmp_path / "cleanup.lbug")
    alice_id, _ = graph.upsert_entity(_entity(name="Alice", entity_type="Person"))
    graph.upsert_entity(_entity(name="Acme", entity_type="Organization"))

    rel = Relationship(
        source_entity="Alice",
        target_entity="Acme",
        relation_type="WORKS_AT",
        confidence=0.95,
        valid_at=datetime.now(UTC),
        observed_at=datetime.now(UTC),
        context_snippets=["Alice works at Acme"],
        episode_ids=["ep-1"],
    )
    graph.upsert_relationship(rel)

    delete_summary = graph.get_entity_delete_summary("Alice")
    assert delete_summary == {"entity_id": alice_id, "relationship_count": 1}

    assert graph.delete_relationships_between("Alice", "Acme", "WORKS_AT") == 1
    assert sorted(graph.list_orphan_entity_names()) == ["Acme", "Alice"]

    embeddings = graph.list_entities_for_embedding(limit=10)
    assert len(embeddings) == 2
    assert {row["name"] for row in embeddings} == {"Alice", "Acme"}
    assert {row["entity_id"] for row in embeddings} == {
        alice_id,
        graph.get_entity_id_by_name("Acme"),
    }

    assert graph.delete_entity("Alice") is True
    assert graph.get_entity_id_by_name("Alice") is None
