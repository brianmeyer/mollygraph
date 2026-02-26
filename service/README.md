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

## GLiNER nightly training: auto-rebalance + promotion diagnostics

The nightly LoRA pipeline now runs an automatic, deterministic pre-train rebalance before train/eval split generation:

- Caps dominant relation labels (including `WORKS_AT`) using `MOLLYGRAPH_TRAIN_REBALANCE_WORKSAT_CAP` and `MOLLYGRAPH_TRAIN_REBALANCE_MAX_RATIO`.
- Upsamples target underrepresented labels (default: `CLASSMATE_OF, STUDIED_AT, LOCATED_IN, CONTACT_OF, MEMBER_OF`) via `MOLLYGRAPH_TRAIN_REBALANCE_TARGET_LABELS`, `MOLLYGRAPH_TRAIN_REBALANCE_TARGET_MIN`, and `MOLLYGRAPH_TRAIN_REBALANCE_TARGET_MULTIPLIERS`.
- Writes rebalance provenance to `~/.graph-memory/training/runs/<run_id>-rebalance.json`.

When a candidate is below promotion threshold, MollyGraph now writes:

- `~/.graph-memory/training/runs/<run_id>-diagnostics.json`

This diagnostics artifact includes combined/entity/relation deltas, top per-type relation F1 gains/regressions, class distribution before/after rebalance, and tuning recommendations. The diagnostics path is persisted in training state metadata (`last_diagnostics_path`) and included in pipeline results.

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

## Decision Traces (Phase 2: ingest-time auto-detection)

During ingestion, MollyGraph can now auto-detect decision moments and write
`Decision` nodes automatically.

Behavior:

- Run a cheap deterministic pre-filter for decision-like language.
- Apply source guards to skip obvious low-signal sources (promo/noise markers).
- Only if pre-filter passes, run a dedicated audit-LM chain (`primary` then `fallback`)
  to classify + extract:
  `decision`, `reasoning`, `alternatives[]`, `inputs[]`, `outcome` (optional),
  `decided_by`, `confidence`.
- Write to the existing `graph.create_decision(...)` path only when extraction is
  positive and confidence meets `MOLLYGRAPH_DECISION_TRACES_MIN_CONFIDENCE`.

Cost-control notes:

- Feature is opt-in via `MOLLYGRAPH_DECISION_TRACES_INGEST_ENABLED=false` by default.
- Deterministic pre-filter prevents unnecessary LLM calls on most ingests.
- Provider chain is bounded to `primary -> fallback` only.
- Payload list caps (`MAX_ALTERNATIVES`, `MAX_INPUTS`, `MAX_RELATED_ENTITIES`) limit noise.
