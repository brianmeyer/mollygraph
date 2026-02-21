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
