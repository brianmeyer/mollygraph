from __future__ import annotations

import pytest


AUTH_HEADERS = {"Authorization": "Bearer dev-key-change-in-production"}

pytestmark = pytest.mark.smoke


def test_health_endpoint(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["test_mode"] is True


def test_core_ingest_entity_query_and_stats(client) -> None:
    ingest_resp = client.post(
        "/ingest",
        params={"content": "Brian works at Databricks.", "source": "manual", "priority": 1},
        headers=AUTH_HEADERS,
    )
    assert ingest_resp.status_code == 200
    assert ingest_resp.json()["status"] == "queued"

    entity_resp = client.get("/entity/Brian", headers=AUTH_HEADERS)
    assert entity_resp.status_code == 200
    assert entity_resp.json()["entity"] == "Brian"

    query_resp = client.get("/query", params={"q": "What about Brian?"}, headers=AUTH_HEADERS)
    assert query_resp.status_code == 200
    assert query_resp.json()["result_count"] >= 1

    stats_resp = client.get("/stats", headers=AUTH_HEADERS)
    assert stats_resp.status_code == 200
    stats_payload = stats_resp.json()
    assert "queue" in stats_payload
    assert "vector_store" in stats_payload
    assert "gliner_training" in stats_payload


def test_parse_verdicts_handles_json_fence() -> None:
    from audit.llm_audit import parse_verdicts

    raw = """```json
    [
      {"index": 1, "verdict": "verify"},
      {"index": 2, "verdict": "reclassify", "suggested_type": "works at"},
      {"index": 3, "verdict": "delete"}
    ]
    ```"""

    parsed = parse_verdicts(raw, batch_len=3)
    assert len(parsed) == 3
    assert parsed[1]["suggested_type"] == "WORKS_AT"
