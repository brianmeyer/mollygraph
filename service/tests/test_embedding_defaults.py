from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_HERE = Path(__file__).parent
_SERVICE_ROOT = _HERE.parent
if str(_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVICE_ROOT))

import config
from api import deps
from embedding_registry import _default_registry
from extraction.pipeline import ExtractionPipeline
from memory.vector_store import SqliteVecBackend, ZvecBackend
import memory.vector_store as vector_store


def test_active_embedding_info_reports_snowflake_default(monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_ST_MODEL", "", raising=False)
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "", raising=False)
    monkeypatch.setattr(ExtractionPipeline, "_embedding_active_tier", "sentence-transformers", raising=False)

    provider, model = deps._active_embedding_info()

    assert provider == "sentence-transformers"
    assert model == config.DEFAULT_LOCAL_EMBEDDING_MODEL


def test_embedding_registry_defaults_to_snowflake(monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_ST_MODEL", "", raising=False)
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "", raising=False)
    monkeypatch.setattr(config, "EMBEDDING_BACKEND", "", raising=False)
    monkeypatch.setattr(config, "EMBEDDING_OLLAMA_MODEL", "", raising=False)
    monkeypatch.setattr(config, "OLLAMA_EMBED_MODEL", "", raising=False)

    registry = _default_registry()

    assert registry["active_provider"] == "huggingface"
    assert registry["active_model"] == config.DEFAULT_LOCAL_EMBEDDING_MODEL
    assert registry["models"]["huggingface"][0] == config.DEFAULT_LOCAL_EMBEDDING_MODEL
    assert registry["models"]["ollama"][0] == "nomic-embed-text"


def test_sentence_transformer_tier_loads_without_default_task_model_kwargs(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer),
    )
    monkeypatch.setattr(config, "EMBEDDING_ST_MODEL", "", raising=False)
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "", raising=False)
    monkeypatch.setattr(ExtractionPipeline, "_embedding_model", None, raising=False)
    monkeypatch.setattr(ExtractionPipeline, "_embedding_active_tier", None, raising=False)
    monkeypatch.setattr(ExtractionPipeline, "_embedding_failed_tiers", set(), raising=False)

    assert ExtractionPipeline._try_load_tier("sentence-transformers") is True
    assert captured["args"] == (config.DEFAULT_LOCAL_EMBEDDING_MODEL,)
    assert captured["kwargs"] == {"trust_remote_code": True}


def test_text_embedding_retries_without_prompt_name_if_model_rejects_it(monkeypatch):
    class _FakeVector:
        def __init__(self, values: list[float]):
            self._values = values

        def tolist(self) -> list[float]:
            return list(self._values)

    class _FakeModel:
        def __init__(self):
            self.calls: list[dict[str, object]] = []

        def encode(self, text: str, **kwargs):
            self.calls.append({"text": text, **kwargs})
            if kwargs.get("prompt_name") == "query":
                raise TypeError("unexpected keyword argument 'prompt_name'")
            return _FakeVector([0.25, 0.75])

    fake_model = _FakeModel()
    monkeypatch.setattr(ExtractionPipeline, "_embedding_model", fake_model, raising=False)
    monkeypatch.setattr(ExtractionPipeline, "_embedding_active_tier", "sentence-transformers", raising=False)
    monkeypatch.setattr(ExtractionPipeline, "_embedding_failed_tiers", set(), raising=False)

    vec = ExtractionPipeline._text_embedding("Where does Alice work?", prompt_name="query")

    assert vec == [0.25, 0.75]
    assert fake_model.calls[0]["prompt_name"] == "query"
    assert fake_model.calls[1] == {
        "text": "Where does Alice work?",
        "normalize_embeddings": True,
    }


def test_sqlite_vec_backend_preserves_legacy_dimension_and_normalizes_queries(tmp_path, monkeypatch):
    pytest.importorskip("sqlite_vec")

    db_path = tmp_path / "vectors.db"

    monkeypatch.setattr(config, "EMBEDDING_VECTOR_DIMENSION", 768, raising=False)
    first = SqliteVecBackend(db_path)
    first.add_entity("e1", "Alpha", "Concept", [1.0, 0.5], "alpha content")

    schema_sql = first.db.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'dense_vectors'"
    ).fetchone()[0]
    assert "FLOAT[768]" in schema_sql

    first.db.close()

    monkeypatch.setattr(config, "EMBEDDING_VECTOR_DIMENSION", 384, raising=False)
    second = SqliteVecBackend(db_path)

    assert second.embedding_dimension == 768
    results = second.similarity_search([0.25, 0.75], top_k=1)

    assert results
    assert results[0]["entity_id"] == "e1"

    second.db.close()


def test_zvec_backend_normalizes_embeddings_with_faux_collection(monkeypatch):
    class _DummyDoc:
        def __init__(self, *, id, vectors, fields):
            self.id = id
            self.vectors = vectors
            self.fields = fields
            self.score = 0.9

    class _DummyCollection:
        def __init__(self):
            self.upserts: list[_DummyDoc] = []
            self.queries: list[dict[str, object]] = []

        def upsert(self, doc):
            self.upserts.append(doc)

        def query(self, vectors, topk, filter, output_fields):
            self.queries.append(
                {
                    "vectors": vectors,
                    "topk": topk,
                    "filter": filter,
                    "output_fields": output_fields,
                }
            )
            return [types.SimpleNamespace(id="e1", fields={"name": "Alpha", "entity_type": "Concept"}, score=0.9)]

    dummy_collection = _DummyCollection()
    fake_zvec = types.SimpleNamespace(
        Doc=_DummyDoc,
        VectorQuery=lambda **kwargs: kwargs,
    )

    monkeypatch.setattr(vector_store, "zvec", fake_zvec, raising=False)

    backend = ZvecBackend.__new__(ZvecBackend)
    backend.collection = dummy_collection
    backend.embedding_dimension = 384

    backend.add_entity("e1", "Alpha", "Concept", [1.0, 0.5], "alpha content")
    assert len(dummy_collection.upserts[0].vectors["embedding"]) == 384

    results = backend.similarity_search([0.25, 0.75], top_k=1)
    assert len(dummy_collection.queries[0]["vectors"]["vector"]) == 384
    assert results[0]["entity_id"] == "e1"
