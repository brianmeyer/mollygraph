# MollyGraph Architecture

This document summarizes the current default architecture.

For product framing, use [README.md](/Users/brianmeyer/mollygraph/README.md). For the full docs index, use [docs/DOCS_MAP.md](/Users/brianmeyer/mollygraph/docs/DOCS_MAP.md).

## Runtime surfaces

- HTTP service: [service/main.py](/Users/brianmeyer/mollygraph/service/main.py)
- MCP proxy: [service/mcp_server.py](/Users/brianmeyer/mollygraph/service/mcp_server.py)
- Python SDK: [sdk/mollygraph_sdk/client.py](/Users/brianmeyer/mollygraph/sdk/mollygraph_sdk/client.py)

## Default data planes

- graph: Ladybug via [service/memory/graph/ladybug.py](/Users/brianmeyer/mollygraph/service/memory/graph/ladybug.py)
- vector: Ladybug via [service/memory/vector_store.py](/Users/brianmeyer/mollygraph/service/memory/vector_store.py)
- queue: SQLite WAL via [service/extraction/queue.py](/Users/brianmeyer/mollygraph/service/extraction/queue.py)
- runtime state: `~/.graph-memory/`

## Default processing flow

1. `POST /ingest` enqueues an `ExtractionJob`.
2. The queue worker runs [ExtractionPipeline](/Users/brianmeyer/mollygraph/service/extraction/pipeline.py).
3. `GLiNER2` extracts entities and relations.
4. Speaker anchoring and relation gating reduce graph pollution before write.
5. Entities and relationships are written to the active graph backend.
6. Entities are embedded and written to the active vector backend.
7. Query surfaces merge graph lookup and vector similarity results.

## Why the current Ladybug runtime uses two files

At the moment the graph and vector layers use separate Ladybug database files by default:

- graph: `MOLLYGRAPH_LADYBUG_GRAPH_DB`
- vector: `MOLLYGRAPH_LADYBUG_VECTOR_DB`

This keeps the runtime simple and safe while we finish the shared-runtime refactor. The long-term goal is a cleaner unified Ladybug-backed memory core, but the current setup is already fully local-first.

## Experimental layers

These remain in the codebase but are not the default product path:

- decision traces
- audit chains
- GLiNER training loops
- GLiREL and spaCy enrichment
- richer extractor registry flows

They should be treated as optional and backend-dependent, not as assumptions of the base runtime.
