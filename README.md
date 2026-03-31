<p align="center">
  <h1 align="center">MollyGraph</h1>
  <p align="center"><strong>Embedded local-first graph memory for AI agents.</strong></p>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> · <a href="#default-stack">Default Stack</a> · <a href="#how-it-works">How It Works</a> · <a href="#mcp">MCP</a> · <a href="#http-api">HTTP API</a> · <a href="#configuration">Config</a>
</p>

MollyGraph is a local graph-and-vector memory service for agents.

The default path is simple:
- ingest text
- extract entities and relationships with `GLiNER2`
- store graph and vectors locally in `Ladybug`
- query through `MCP`, `HTTP`, or the Python `SDK`

The goal is not “a vector store with a wrapper.” The goal is structured local memory that agents can actually use.

## Quick Start

```bash
git clone https://github.com/brianmeyer/mollygraph.git
cd mollygraph
./scripts/install.sh
./scripts/start.sh
```

The canonical local runtime is Python `3.12`.
`./scripts/install.sh` will use `python3.12` when it is available, or `uv` to provision Python `3.12` automatically.

The install script:
- creates `service/.env` from `service/.env.example` if needed
- creates the runtime venv at `service/.venv`
- preloads the default `GLiNER2` and embedding models when possible

Default local API:
- base URL: `http://127.0.0.1:7422`
- auth: `Bearer dev-key-change-in-production`

Production-style smoke test:

```bash
service/.venv/bin/python scripts/production_smoke.py --json
```

## Default Stack

- graph storage: `Ladybug`
- vector storage: `Ladybug`
- extraction: `GLiNER2`
- default embeddings: `Snowflake/snowflake-arctic-embed-s`
- queue: `SQLite`
- client surfaces: `MCP`, `HTTP`, `SDK`

The default runtime does not require Neo4j.
Neo4j-backed audit, training, and decision-oriented flows are optional compatibility surfaces, not the base product.

## How It Works

```text
text in
  -> SQLite extraction queue
  -> GLiNER2 extraction
  -> speaker anchoring + relation gate
  -> Ladybug graph + Ladybug vector storage
  -> MCP / HTTP / SDK query surface
```

Default retrieval path:
- graph exact-match and vector similarity run in parallel
- results are merged into one response

The current graph and vector layers use separate `.lbug` files by default.
That keeps the runtime predictable while the shared-Ladybug storage path is evaluated.

## MCP

Example MCP config:

```json
{
  "mcpServers": {
    "mollygraph": {
      "command": "mollygraph-mcp",
      "args": ["--base-url", "http://localhost:7422", "--api-key", "YOUR_KEY"]
    }
  }
}
```

Default tools:
- `add_episode`
- `search_facts`
- `search_nodes`
- `get_entity_context`
- `get_queue_status`
- `delete_entity`
- `prune_entities`

Optional audit or training tools are only exposed when the running backend reports support for them.

## Python SDK

```bash
pip install "git+https://github.com/brianmeyer/mollygraph.git#subdirectory=sdk"
```

```python
from mollygraph_sdk import MollyGraphClient

client = MollyGraphClient(
    base_url="http://localhost:7422",
    api_key="YOUR_KEY",
)

client.ingest("Sarah joined the ML team at Acme Corp.", source="slack")
result = client.query("What do we know about Sarah?")
entity = client.get_entity("Sarah")
client.close()
```

## HTTP API

Core endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health and graph capability report |
| `/stats` | GET | Graph, queue, and vector stats |
| `/ingest` | POST | Queue text for extraction |
| `/query` | GET/POST | Combined graph + vector retrieval |
| `/entity/{name}` | GET | Entity context |
| `/entity/{name}` | DELETE | Delete one entity |
| `/entities/prune` | POST | Bulk prune entities |

Operator utilities still exist for quality checks, vector reconciliation, and embedding refresh, but they are not the main onboarding path.

## Configuration

Everything is env-driven.
See [service/.env.example](/Users/brianmeyer/mollygraph/service/.env.example) for the full configuration surface.

Key defaults:

```env
MOLLYGRAPH_GRAPH_BACKEND=ladybug
MOLLYGRAPH_VECTOR_BACKEND=ladybug
MOLLYGRAPH_EXTRACTOR_BACKEND=gliner2
MOLLYGRAPH_EMBEDDING_ST_MODEL=Snowflake/snowflake-arctic-embed-s
MOLLYGRAPH_EMBEDDING_VECTOR_DIMENSION=384
```

Optional local alternate:

```env
MOLLYGRAPH_EMBEDDING_OLLAMA_MODEL=nomic-embed-text
```

## Docs

Current docs:
- [docs/ARCHITECTURE.md](/Users/brianmeyer/mollygraph/docs/ARCHITECTURE.md)
- [docs/DOCS_MAP.md](/Users/brianmeyer/mollygraph/docs/DOCS_MAP.md)
- [docs/PRODUCTION_TEST_CHECKLIST.md](/Users/brianmeyer/mollygraph/docs/PRODUCTION_TEST_CHECKLIST.md)
- [service/README.md](/Users/brianmeyer/mollygraph/service/README.md)
- [sdk/README.md](/Users/brianmeyer/mollygraph/sdk/README.md)
- [service/BACKLOG.md](/Users/brianmeyer/mollygraph/service/BACKLOG.md)

Later-phase plan:
- [service/DECISION_TRACES_PLAN.md](/Users/brianmeyer/mollygraph/service/DECISION_TRACES_PLAN.md)

## License

MIT
