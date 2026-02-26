# MollyGraph Service

## Infra health evaluator (deterministic-first)

MollyGraph now includes a deterministic infra health policy at `maintenance/infra_health.py` and endpoint:

- `POST /maintenance/infra-health/evaluate`

### Safety model

- Deterministic decision is authoritative.
- Optional LLM advisory is non-blocking and cannot override deterministic guardrails.
- If advisory call fails/parsing fails, service falls back to deterministic-only output.
- `rebuild_vectors` is fallback-only and **not** auto-run in nightly maintenance by default.

### Reconcile vectors fallback

`POST /maintenance/reconcile-vectors` supports:

- `mode=orphan_cleanup` (default): remove vector IDs not found in Neo4j.
- `mode=rebuild`: fallback rebuild path, only allowed when deterministic infra-health returns `rebuild_vectors` (or `force=true`).

Why this exists: zvec does not support `list_all_entity_ids`, so full orphan reconciliation is unavailable on that backend.

## Decision Traces (Phase 1)

New authenticated endpoints:

- `POST /decisions`
- `GET /decisions?q=<text>&decided_by=<name>&limit=<1..200>`
- `GET /decisions/{id}`

`POST /decisions` request body fields:

- `decision` (string)
- `reasoning` (string)
- `alternatives` (string[])
- `inputs` (string[])
- `outcome` (string)
- `decided_by` (string)
- `related_entities` (string[])
- `preceded_by_decision_id` (optional string)
- `source_episode_id` (optional string)
- `confidence` (optional float 0..1)
- `timestamp` (optional ISO datetime)

Minimal example:

```bash
curl -X POST "http://127.0.0.1:7422/decisions" \
  -H "Authorization: Bearer dev-key-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{
    "decision": "Switch embedding model to jina-v5-nano",
    "reasoning": "Better retrieval quality in local benchmarks",
    "alternatives": ["keep current model", "nomic-embed-text"],
    "inputs": ["benchmark run 2026-02-25", "latency targets"],
    "outcome": "Reindexed 903 entities",
    "decided_by": "Brian",
    "related_entities": ["MollyGraph", "Neo4j"],
    "confidence": 0.92
  }'
```
