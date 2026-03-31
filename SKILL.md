# MollyGraph Skill

**Name:** `mollygraph`  
**Description:** Local-first graph memory for AI agents  
**Version:** `3.0.0`

## Overview

Use these docs as the source of truth:

- [README.md](/Users/brianmeyer/mollygraph/README.md) for the product story and quick start
- [service/README.md](/Users/brianmeyer/mollygraph/service/README.md) for the default runtime
- [docs/ARCHITECTURE.md](/Users/brianmeyer/mollygraph/docs/ARCHITECTURE.md) for the current system shape

MollyGraph's default product path is:

- `Ladybug` for local graph storage
- `Ladybug` for local vector storage
- `GLiNER2` for entity and relation extraction
- `Snowflake/snowflake-arctic-embed-s` for default local embeddings
- `SQLite` for the async extraction queue
- `HTTP + MCP + Python SDK` for the user-facing surface

Neo4j, audit flows, training loops, and decision traces still exist as optional compatibility or later-phase surfaces, but they are not part of the default runtime.

## Installation

```bash
cd /Users/brianmeyer/mollygraph
./scripts/install.sh
./scripts/start.sh
```

`./scripts/install.sh` creates `service/.env` if needed and installs the runtime into `service/.venv`.
This starts the HTTP API on port `7422`. The default API key is `dev-key-change-in-production`.

## Default Runtime

```env
MOLLYGRAPH_GRAPH_BACKEND=ladybug
MOLLYGRAPH_VECTOR_BACKEND=ladybug
MOLLYGRAPH_EMBEDDING_ST_MODEL=Snowflake/snowflake-arctic-embed-s
```

Graph and vector use separate Ladybug database files by default.

## Working Guidance

- Prefer the default surfaces: `ingest`, `query`, `get_entity`, queue status, health, and stats.
- Treat audit, training, decision, and registry surfaces as explicitly optional.
- Keep docs aligned with the default runtime before documenting later-phase features.
- Use the service health capabilities to decide whether optional MCP tools should be exposed.
- Default local data lives under `~/.graph-memory`.
