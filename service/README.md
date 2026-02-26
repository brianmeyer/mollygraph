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
