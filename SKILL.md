# MollyGraph Skill

**Name:** `mollygraph`  
**Description:** Local-first graph memory for AI agents  
**Version:** `3.0.0`

## Overview

MollyGraph's default product path is now:

- `Ladybug` for local graph storage
- `Ladybug` for local vector storage
- `GLiNER2` for entity and relation extraction
- `Snowflake/snowflake-arctic-embed-s` for default local embeddings
- `SQLite` for the async extraction queue
- `HTTP + MCP + Python SDK` for the user-facing surface

Neo4j, audit flows, training loops, and decision traces still exist as experimental or legacy surfaces, but they are not the default runtime.

## Installation

```bash
cd /Users/brianmeyer/mollygraph
./scripts/install.sh
cp service/.env.example service/.env
./scripts/start.sh
```

This starts the HTTP API on port `7422`. The default API key is `dev-key-change-in-production`.

## Default Runtime

The intended local-first defaults are:

```env
MOLLYGRAPH_GRAPH_BACKEND=ladybug
MOLLYGRAPH_VECTOR_BACKEND=ladybug
MOLLYGRAPH_EMBEDDING_ST_MODEL=Snowflake/snowflake-arctic-embed-s
```

Graph and vector currently use separate Ladybug database files by default while the shared-runtime refactor continues.

## MCP Integration

Add to your MCP client config:

```json
{
  "mcpServers": {
    "mollygraph": {
      "command": "mollygraph-mcp",
      "args": [
        "--base-url",
        "http://127.0.0.1:7422",
        "--api-key",
        "dev-key-change-in-production"
      ]
    }
  }
}
```

## Default MCP Tools

| Tool | Description |
|------|-------------|
| `add_episode` | Queue text for extraction |
| `search_facts` | Search merged graph and memory facts |
| `search_nodes` | Find graph entities by name |
| `get_entity_context` | Get entity context and neighborhood |
| `get_queue_status` | Check extraction queue health |
| `delete_entity` | Remove an entity from graph and vector storage |
| `prune_entities` | Clean up entities in bulk |

Experimental tools like audit and training are only exposed when the configured backend supports them.

## HTTP API

Core endpoints:

- `GET /health`
- `GET /stats`
- `POST /ingest`
- `GET /query`
- `POST /query`
- `GET /entity/{name}`
- `DELETE /entity/{name}`
- `POST /entities/prune`

## Authentication

Protected endpoints require:

```text
Authorization: Bearer dev-key-change-in-production
```

Override with `MOLLYGRAPH_API_KEY`.

## Data Storage

- Graph: `Ladybug` (`~/.graph-memory/graph.lbug` by default)
- Vectors: `Ladybug` (`~/.graph-memory/vectors.lbug` by default)
- Queue: `SQLite` (`~/.graph-memory/extraction_queue.db`)

## Architecture

```text
User Input
  -> Queue (SQLite)
  -> GLiNER2 extraction
  -> Speaker anchoring + relation gate
  -> Ladybug graph + Ladybug vector
  -> MCP / HTTP / SDK query surface
```
