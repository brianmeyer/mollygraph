# MollyGraph Service

This is the service-level guide for the default runtime.

Use [README.md](/Users/brianmeyer/mollygraph/README.md) for the repo-level product story and [docs/DOCS_MAP.md](/Users/brianmeyer/mollygraph/docs/DOCS_MAP.md) for the authoritative docs list.

## Default runtime

The default local-first stack is:

- graph backend: `Ladybug`
- vector backend: `Ladybug`
- extractor: `GLiNER2`
- default embedder: `Snowflake/snowflake-arctic-embed-s`
- surface: `HTTP + MCP + Python SDK`

The canonical service runtime is Python `3.12` in `service/.venv`.

Graph and vector use separate `.lbug` files by default. That keeps the runtime safe.

## Core endpoints

These are the main user-facing endpoints in the default product path:

- `GET /health`
- `GET /stats`
- `POST /ingest`
- `GET /query`
- `POST /query`
- `GET /entity/{name}`
- `DELETE /entity/{name}`
- `POST /entities/prune`

## Operator utilities

These support operating the local runtime, but they are not the main product story:

- `POST /maintenance/infra-health/evaluate`
- `POST /maintenance/reconcile-vectors`
- `POST /maintenance/refresh-embeddings`
- `POST /maintenance/quality-check`

## Concurrent run safety

The local runtime is intentionally conservative for laptop use.

- `scripts/start.sh` warns and stops if another MollyGraph process is already using the same local data directory.
- `GET /health` includes `operator_advisories` when the queue is already busy or when concurrency is set above `1`.
- `MOLLYGRAPH_QUEUE_MAX_CONCURRENT=1` is the recommended default unless you explicitly want more throughput and more local model load at the same time.

## Experimental surfaces

These remain in the codebase but are not part of the default Ladybug product path:

- decision APIs
- LLM audit endpoints
- GLiNER training endpoints
- nightly maintenance compatibility flow
- extractor registry/config endpoints

When the active backend does not support them, they should be hidden from the default API docs or return explicit `501` responses.

## Operations notes

- `service/.env.example` is the runtime configuration reference.
- `service/BACKLOG.md` is the service backlog.
- `service/DECISION_TRACES_PLAN.md` is a later-phase product plan.
- `service/.venv/bin/python scripts/production_smoke.py --json` runs the isolated production-style smoke pass.
