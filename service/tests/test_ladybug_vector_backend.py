from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).parent
_SERVICE_ROOT = _HERE.parent
if str(_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVICE_ROOT))

import real_ladybug as lb

from memory.vector_store import LadybugVectorBackend, VectorStore


def _open_vector_database(tmp_path: Path) -> lb.Connection:
    db = lb.Database(str(tmp_path / "ladybug-vector.lbug"))
    conn = lb.Connection(db)
    conn.execute("INSTALL vector;")
    conn.execute("LOAD vector;")
    conn.execute(
        """
        CREATE NODE TABLE MemoryVector(
            entity_id STRING PRIMARY KEY,
            name STRING,
            entity_type STRING,
            content STRING,
            confidence DOUBLE,
            embedding FLOAT[4]
        );
        """
    )
    return conn


def test_ladybug_vector_extension_round_trips_small_embeddings(tmp_path):
    conn = _open_vector_database(tmp_path)

    conn.execute(
        """
        CREATE (v:MemoryVector {
            entity_id: 'e1',
            name: 'Alice',
            entity_type: 'Person',
            content: 'Alice met Bob about launch planning',
            confidence: 0.98,
            embedding: [1.0, 0.0, 0.0, 0.0]
        });
        """
    )
    conn.execute(
        """
        CREATE (v:MemoryVector {
            entity_id: 'e2',
            name: 'Bob',
            entity_type: 'Person',
            content: 'Bob replied with project notes',
            confidence: 0.97,
            embedding: [0.8, 0.2, 0.0, 0.0]
        });
        """
    )

    conn.execute(
        """
        CALL CREATE_VECTOR_INDEX(
            'MemoryVector',
            'memory_vector_idx',
            'embedding',
            metric := 'cosine'
        );
        """
    )

    result = conn.execute(
        """
        CALL QUERY_VECTOR_INDEX(
            'MemoryVector',
            'memory_vector_idx',
            $query_vector,
            $k,
            efs := 64
        )
        RETURN node.entity_id AS entity_id,
               node.name AS name,
               node.entity_type AS entity_type,
               distance
        ORDER BY distance;
        """,
        {"query_vector": [1.0, 0.0, 0.0, 0.0], "k": 2},
    )

    rows = result.rows_as_dict().get_all()

    assert [row["entity_id"] for row in rows] == ["e1", "e2"]
    assert [row["name"] for row in rows] == ["Alice", "Bob"]
    assert rows[0]["distance"] <= rows[1]["distance"]
    assert all(row["entity_type"] == "Person" for row in rows)


def test_ladybug_vector_extension_can_be_reopened_and_reused(tmp_path):
    db_path = tmp_path / "ladybug-vector.lbug"

    first = lb.Connection(lb.Database(str(db_path)))
    first.execute("INSTALL vector;")
    first.execute("LOAD vector;")
    first.execute(
        """
        CREATE NODE TABLE MemoryVector(
            entity_id STRING PRIMARY KEY,
            name STRING,
            entity_type STRING,
            content STRING,
            confidence DOUBLE,
            embedding FLOAT[4]
        );
        """
    )
    first.execute(
        """
        CREATE (v:MemoryVector {
            entity_id: 'e1',
            name: 'Alice',
            entity_type: 'Person',
            content: 'Alice met Bob about launch planning',
            confidence: 0.98,
            embedding: [1.0, 0.0, 0.0, 0.0]
        });
        """
    )
    first.execute(
        """
        CALL CREATE_VECTOR_INDEX(
            'MemoryVector',
            'memory_vector_idx',
            'embedding',
            metric := 'cosine'
        );
        """
    )
    first.close()

    reopened = lb.Connection(lb.Database(str(db_path)))
    reopened.execute("LOAD vector;")
    result = reopened.execute(
        """
        CALL QUERY_VECTOR_INDEX(
            'MemoryVector',
            'memory_vector_idx',
            $query_vector,
            $k
        )
        RETURN node.entity_id AS entity_id, distance
        ORDER BY distance;
        """,
        {"query_vector": [1.0, 0.0, 0.0, 0.0], "k": 1},
    )

    rows = result.rows_as_dict().get_all()

    assert len(rows) == 1
    assert rows[0]["entity_id"] == "e1"
    assert rows[0]["distance"] >= 0


def test_ladybug_backend_supports_add_search_list_remove_and_stats(tmp_path):
    backend = LadybugVectorBackend(tmp_path / "backend.lbug")

    assert backend.degraded is False

    backend.add_entity(
        "e1",
        "Alice",
        "Person",
        [1.0, 0.0, 0.0, 0.0],
        "Alice met Bob about launch planning",
    )
    backend.add_entity(
        "e2",
        "Acme",
        "Organization",
        [0.8, 0.2, 0.0, 0.0],
        "Acme builds rockets",
    )

    stats = backend.get_stats()
    assert stats["backend"] == "ladybug"
    assert stats["entities"] == 2
    assert stats["embedding_dimension"] == 384
    assert stats["vector_index_present"] is True

    ids = backend.list_all_entity_ids()
    assert ids == ["e1", "e2"]

    similar = backend.similarity_search([1.0, 0.0, 0.0, 0.0], top_k=2)
    assert [row["entity_id"] for row in similar] == ["e1", "e2"]
    assert all("score" in row for row in similar)

    filtered = backend.similarity_search(
        [1.0, 0.0, 0.0, 0.0],
        top_k=2,
        entity_type="Organization",
    )
    assert [row["entity_id"] for row in filtered] == ["e2"]

    keyword = backend.keyword_search("rockets", top_k=2)
    assert [row["entity_id"] for row in keyword] == ["e2"]

    assert backend.remove_entity("e2") is True
    assert backend.remove_entity("missing") is False
    assert backend.get_stats()["entities"] == 1


def test_ladybug_backend_upsert_replaces_indexed_embeddings(tmp_path):
    backend = LadybugVectorBackend(tmp_path / "upsert.lbug")

    backend.add_entity(
        "e1",
        "Alice",
        "Person",
        [1.0, 0.0, 0.0, 0.0],
        "Alice before update",
    )
    backend.add_entity(
        "e2",
        "Bob",
        "Person",
        [1.0, 0.0, 0.0, 0.0],
        "Bob stays on the original vector",
    )

    backend.update_entity(
        "e1",
        "Alice",
        "Person",
        [0.0, 1.0, 0.0, 0.0],
        "Alice after update",
    )

    new_direction = backend.similarity_search([0.0, 1.0, 0.0, 0.0], top_k=1)
    old_direction = backend.similarity_search([1.0, 0.0, 0.0, 0.0], top_k=1)

    assert new_direction[0]["entity_id"] == "e1"
    assert old_direction[0]["entity_id"] == "e2"


def test_ladybug_backend_recovers_from_corrupt_wal(tmp_path):
    db_path = tmp_path / "recoverable.lbug"

    backend = LadybugVectorBackend(db_path)
    backend.add_entity(
        "e1",
        "Alice",
        "Person",
        [1.0, 0.0, 0.0, 0.0],
        "Alice before restart",
    )
    backend.conn.close()

    wal_path = Path(f"{db_path}.wal")
    wal_path.write_bytes(b"corrupt-wal")

    recovered = LadybugVectorBackend(db_path)

    assert recovered.degraded is False
    assert not wal_path.exists() or wal_path.read_bytes() != b"corrupt-wal"
    assert list(tmp_path.glob("recoverable.lbug.wal.corrupt-*"))

    recovered.add_entity(
        "e2",
        "Bob",
        "Person",
        [0.0, 1.0, 0.0, 0.0],
        "Bob after recovery",
    )
    ids = recovered.list_all_entity_ids()
    assert ids is not None
    assert "e2" in ids


def test_vector_store_can_select_ladybug_backend(tmp_path):
    store = VectorStore(backend="ladybug", db_path=tmp_path / "wrapper.lbug")

    assert store.get_stats()["backend"] == "ladybug"
    assert store.is_degraded() is False
    assert store.optimize()["status"] == "not_applicable"
