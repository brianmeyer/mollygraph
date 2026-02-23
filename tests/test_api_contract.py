from __future__ import annotations

import pytest


AUTH_HEADERS = {"Authorization": "Bearer dev-key-change-in-production"}

pytestmark = pytest.mark.integration


def test_openapi_includes_canonical_and_legacy_paths(client) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = set(response.json()["paths"].keys())
    expected = {
        "/health",
        "/stats",
        "/ingest",
        "/extract",
        "/entity/{name}",
        "/query",
        "/audit",
        "/audit/run",
        "/maintenance/audit",
        "/suggestions/digest",
        "/suggestions_digest",
        "/train/gliner",
        "/training/gliner",
        "/train/status",
        "/training/status",
        "/embeddings/config",
        "/embeddings/status",
        "/embeddings/models",
        "/embeddings/reindex",
        "/extractors/config",
        "/extractors/status",
        "/extractors/schema",
        "/extractors/schema/presets",
        "/extractors/models",
        "/extractors/schema/upload",
        "/extractors/prefetch",
        "/maintenance/run",
    }
    assert expected.issubset(paths)


def test_protected_routes_require_bearer_auth(client) -> None:
    protected_routes = [
        ("GET", "/stats", {}),
        ("POST", "/ingest", {"params": {"content": "test", "source": "manual", "priority": 1}}),
        ("POST", "/extract", {"params": {"content": "test", "source": "manual", "priority": 1}}),
        ("GET", "/entity/Brian", {}),
        ("GET", "/query", {"params": {"q": "What about Brian?"}}),
        ("POST", "/audit", {"json": {"limit": 25, "dry_run": True, "schedule": "nightly"}}),
        ("POST", "/audit/run", {"json": {"limit": 25, "dry_run": True, "schedule": "nightly"}}),
        ("POST", "/maintenance/audit", {"json": {"limit": 25, "dry_run": True, "schedule": "nightly"}}),
        ("GET", "/suggestions/digest", {}),
        ("GET", "/suggestions_digest", {}),
        ("POST", "/train/gliner", {"json": {"force": True}}),
        ("POST", "/training/gliner", {"json": {"force": True}}),
        ("GET", "/train/status", {}),
        ("GET", "/training/status", {}),
        ("GET", "/embeddings/config", {}),
        ("GET", "/embeddings/status", {}),
        ("POST", "/embeddings/config", {"json": {"provider": "hash"}}),
        ("POST", "/embeddings/models", {"json": {"provider": "ollama", "model": "nomic-embed-text"}}),
        ("POST", "/embeddings/reindex", {"json": {"limit": 2, "dry_run": True}}),
        ("GET", "/extractors/config", {}),
        ("GET", "/extractors/status", {}),
        ("GET", "/extractors/schema", {}),
        ("GET", "/extractors/schema/presets", {}),
        ("POST", "/extractors/schema", {"json": {"mode": "default"}}),
        (
            "POST",
            "/extractors/schema/upload",
            {
                "json": {
                    "entities": {"Person": {"description": "A person", "threshold": 0.4}},
                    "relations": {"works with": {"description": "Collaboration", "threshold": 0.45}},
                    "activate": True,
                }
            },
        ),
        ("POST", "/extractors/config", {"json": {"backend": "gliner2"}}),
        ("POST", "/extractors/models", {"json": {"backend": "gliner2", "model": "fastino/gliner2-large-v1"}}),
        ("POST", "/extractors/prefetch", {"json": {"backend": "gliner2", "model": "fastino/gliner2-large-v1"}}),
        ("POST", "/maintenance/run", {}),
    ]

    for method, route, kwargs in protected_routes:
        resp = client.request(method, route, **kwargs)
        assert resp.status_code == 401, f"{method} {route} should require auth"


def test_ingest_and_extract_aliases_queue_jobs(client) -> None:
    canonical = client.post(
        "/ingest",
        params={"content": "Brian works at Databricks.", "source": "manual", "priority": 1},
        headers=AUTH_HEADERS,
    )
    alias = client.post(
        "/extract",
        params={"content": "Brian uses Python.", "source": "manual", "priority": 1},
        headers=AUTH_HEADERS,
    )

    assert canonical.status_code == 200
    assert alias.status_code == 200
    assert canonical.json()["status"] == alias.json()["status"] == "queued"
    assert canonical.json()["job_id"] != alias.json()["job_id"]


def test_audit_aliases_match_canonical_shape(client) -> None:
    payload = {"limit": 25, "dry_run": True, "schedule": "nightly"}
    canonical = client.post("/audit", json=payload, headers=AUTH_HEADERS)
    action_alias = client.post("/audit/run", json=payload, headers=AUTH_HEADERS)
    legacy_alias = client.post("/maintenance/audit", json=payload, headers=AUTH_HEADERS)

    for resp in (canonical, action_alias, legacy_alias):
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["relationships_scanned"] == 25


def test_suggestions_digest_aliases_match(client) -> None:
    canonical = client.get("/suggestions/digest", headers=AUTH_HEADERS)
    alias = client.get("/suggestions_digest", headers=AUTH_HEADERS)

    assert canonical.status_code == 200
    assert alias.status_code == 200
    assert canonical.json()["digest"] == alias.json()["digest"] == "1 suggestion"
    assert canonical.json()["has_suggestions"] is True
    assert alias.json()["has_suggestions"] is True


def test_training_aliases_match(client) -> None:
    train_canonical = client.post("/train/gliner", json={"force": True}, headers=AUTH_HEADERS)
    train_alias = client.post("/training/gliner", json={"force": True}, headers=AUTH_HEADERS)
    status_canonical = client.get("/train/status", headers=AUTH_HEADERS)
    status_alias = client.get("/training/status", headers=AUTH_HEADERS)

    assert train_canonical.status_code == 200
    assert train_alias.status_code == 200
    assert train_canonical.json()["status"] == "finetune_triggered"
    assert train_alias.json()["status"] == "finetune_triggered"
    assert status_canonical.status_code == 200
    assert status_alias.status_code == 200
    assert status_canonical.json()["gliner"] == status_alias.json()["gliner"]


def test_maintenance_run_contract(client) -> None:
    response = client.post("/maintenance/run", headers=AUTH_HEADERS)
    assert response.status_code == 200
    assert response.json()["status"] == "maintenance_triggered"


def test_embedding_model_registry_supports_huggingface_and_ollama(client) -> None:
    baseline = client.get("/embeddings/config", headers=AUTH_HEADERS)
    assert baseline.status_code == 200
    assert {"hash", "huggingface", "ollama"}.issubset(set(baseline.json()["supported_providers"]))

    add_hf = client.post(
        "/embeddings/models",
        json={"provider": "huggingface", "model": "BAAI/bge-small-en-v1.5"},
        headers=AUTH_HEADERS,
    )
    assert add_hf.status_code == 200
    assert "BAAI/bge-small-en-v1.5" in add_hf.json()["models"]["huggingface"]

    activate_hf = client.post(
        "/embeddings/config",
        json={"provider": "huggingface", "model": "BAAI/bge-small-en-v1.5"},
        headers=AUTH_HEADERS,
    )
    assert activate_hf.status_code == 200
    assert activate_hf.json()["active_provider"] == "huggingface"
    assert activate_hf.json()["active_model"] == "BAAI/bge-small-en-v1.5"

    add_ollama = client.post(
        "/embeddings/models",
        json={"provider": "ollama", "model": "nomic-embed-text-v1.5", "activate": True},
        headers=AUTH_HEADERS,
    )
    assert add_ollama.status_code == 200
    assert add_ollama.json()["active_provider"] == "ollama"
    assert add_ollama.json()["active_model"] == "nomic-embed-text-v1.5"


def test_embedding_status_and_reindex_dry_run(client) -> None:
    status = client.get("/embeddings/status", headers=AUTH_HEADERS)
    assert status.status_code == 200
    payload = status.json()
    assert "active_provider" in payload
    assert "providers" in payload
    assert {"hash", "huggingface", "ollama"}.issubset(set(payload["providers"].keys()))

    dry_run = client.post(
        "/embeddings/reindex",
        json={"limit": 2, "dry_run": True},
        headers=AUTH_HEADERS,
    )
    assert dry_run.status_code == 200
    body = dry_run.json()
    assert body["status"] == "dry_run"
    assert body["entities_found"] >= 1


def test_extractor_registry_supports_gliner_and_hf_token_models(client) -> None:
    baseline = client.get("/extractors/config", headers=AUTH_HEADERS)
    assert baseline.status_code == 200
    assert {"gliner2", "hf_token_classification"}.issubset(set(baseline.json()["supported_backends"]))

    add_hf = client.post(
        "/extractors/models",
        json={"backend": "hf_token_classification", "model": "dslim/bert-base-NER"},
        headers=AUTH_HEADERS,
    )
    assert add_hf.status_code == 200
    assert "dslim/bert-base-NER" in add_hf.json()["models"]["hf_token_classification"]

    activate_hf = client.post(
        "/extractors/config",
        json={
            "backend": "hf_token_classification",
            "model": "dslim/bert-base-NER",
            "relation_model": "Babelscape/rebel-large",
        },
        headers=AUTH_HEADERS,
    )
    assert activate_hf.status_code == 200
    assert activate_hf.json()["active_backend"] == "hf_token_classification"
    assert activate_hf.json()["active_model"] == "dslim/bert-base-NER"
    assert activate_hf.json()["active_relation_model"] == "Babelscape/rebel-large"

    status = client.get("/extractors/status", headers=AUTH_HEADERS)
    assert status.status_code == 200
    payload = status.json()
    assert payload["active_backend"] in {"gliner2", "hf_token_classification"}
    assert payload["supports_relations"] is True
    assert "backends" in payload


def test_extractor_schema_modes_support_default_preset_and_custom_upload(client) -> None:
    baseline = client.get("/extractors/schema", headers=AUTH_HEADERS)
    assert baseline.status_code == 200
    baseline_payload = baseline.json()
    assert baseline_payload["mode"] in {"default", "preset", "custom"}
    assert "schema" in baseline_payload
    assert "entities" in baseline_payload["schema"]
    assert "relations" in baseline_payload["schema"]

    presets = client.get("/extractors/schema/presets", headers=AUTH_HEADERS)
    assert presets.status_code == 200
    presets_payload = presets.json()
    assert "presets" in presets_payload
    assert len(presets_payload["presets"]) >= 1
    preset_id = presets_payload["presets"][0]["id"]

    activate_preset = client.post(
        "/extractors/schema",
        json={"mode": "preset", "preset": preset_id},
        headers=AUTH_HEADERS,
    )
    assert activate_preset.status_code == 200
    preset_payload = activate_preset.json()
    assert preset_payload["mode"] == "preset"
    assert preset_payload["active_preset"] == preset_id

    upload_custom = client.post(
        "/extractors/schema/upload",
        json={
            "entities": {
                "Person": {"description": "A person in conversation", "threshold": 0.4},
                "Topic": {"description": "Discussion theme", "threshold": 0.45},
            },
            "relations": {
                "mentions": {"description": "A person mentions a topic", "threshold": 0.45},
                "knows": {"description": "A person knows another person", "threshold": 0.5},
            },
            "activate": True,
        },
        headers=AUTH_HEADERS,
    )
    assert upload_custom.status_code == 200
    uploaded_payload = upload_custom.json()
    assert uploaded_payload["mode"] == "custom"
    assert uploaded_payload["activated"] is True
    assert uploaded_payload["custom_schema_updated"] is True
    assert "Topic" in uploaded_payload["schema"]["entities"]
    assert "mentions" in uploaded_payload["schema"]["relations"]

    restore_default = client.post(
        "/extractors/schema",
        json={"mode": "default"},
        headers=AUTH_HEADERS,
    )
    assert restore_default.status_code == 200
    assert restore_default.json()["mode"] == "default"
