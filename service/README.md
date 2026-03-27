# MollyGraph Service

This is the service-level guide for the current default runtime.

Use [README.md](/Users/brianmeyer/mollygraph/README.md) for the repo-level product story and [docs/DOCS_MAP.md](/Users/brianmeyer/mollygraph/docs/DOCS_MAP.md) for the authoritative docs list.

## Default runtime

The default local-first stack is:

- graph backend: `Ladybug`
- vector backend: `Ladybug`
- extractor: `GLiNER2`
- default embedder: `Snowflake/snowflake-arctic-embed-s`
- surface: `HTTP + MCP + Python SDK`

Graph and vector currently use separate `.lbug` files by default. That keeps the runtime safe while we finish the shared-Ladybug-core refactor.

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

## Experimental surfaces

These remain in the codebase but are not part of the default Ladybug product path:

- decision APIs
- LLM audit endpoints
- GLiNER training endpoints
- legacy nightly maintenance flow
- extractor registry/config endpoints

When the active backend does not support them, they should be hidden from the default API docs or return explicit `501` responses.

## Operations notes

- `service/.env.example` is the current runtime configuration reference.
- `service/BACKLOG.md` is the current service backlog.
- `service/DECISION_TRACES_PLAN.md` is a later-phase product plan, not part of the default runtime.
