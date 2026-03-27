<p align="center">
  <h1 align="center">🧠 MollyGraph</h1>
  <p align="center"><strong>Local-first graph memory for AI agents.</strong></p>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> · <a href="#how-it-works">How It Works</a> · <a href="#mcp-integration">MCP</a> · <a href="#http-api">API</a> · <a href="#configuration">Config</a>
</p>

---

Most agent memory is a vector store with a wrapper. Same extraction quality forever. No structure. No relationships. No local-first story.

MollyGraph is a **local graph + vector memory core** for AI agents. Ingest text -> extract entities and relationships with GLiNER2 -> store them locally in Ladybug -> serve them through MCP, HTTP, and the SDK.

---

## What Makes This Different

- **Local-first memory core** — Embedded Ladybug graph storage plus local vector storage by default. No mandatory database daemon for the main path.
- **GLiNER2-first extraction** — The default product path is built around local structured extraction, not cloud reasoning.
- **Speaker-anchored ingestion** — Each message is processed individually with the speaker as anchor entity, which is the main defense against bad chat graphs.
- **Parallel retrieval** — Graph exact-match and vector similarity run together, then merge into one response.
- **Simple local embeddings** — Default embedder is `Snowflake/snowflake-arctic-embed-s`, with `nomic-embed-text` via Ollama as the optional local alternative.
- **Experimental features still available** — audit chains, GLiREL, training loops, and decision traces are being kept behind explicit flags instead of defining the core product.

---

## Quick Start

```bash
git clone https://github.com/brianmeyer/mollygraph.git
cd mollygraph
./scripts/install.sh
./scripts/start.sh
```

API at `http://127.0.0.1:7422`. Auth: `Bearer dev-key-change-in-production`.

`./scripts/install.sh` creates `service/.env` from `service/.env.example` if it does not exist yet, and installs the runtime into `service/.venv`.

Default local stack:
- `MOLLYGRAPH_GRAPH_BACKEND=ladybug`
- `MOLLYGRAPH_VECTOR_BACKEND=ladybug`
- `MOLLYGRAPH_EMBEDDING_ST_MODEL=Snowflake/snowflake-arctic-embed-s`

Neo4j is still available for legacy and experimental workflows, but it is no longer the intended default.

---

## How It Works

```
  ingest text
       │
       ▼
  Speaker-anchored extraction (GLiNER2)
  Per-source confidence gates
       │
       ├──▶ Ladybug graph
       ├──▶ Ladybug vector index
       └──▶ MCP / HTTP / SDK query surface
```

**Default query path:** graph exact + vector similarity fire in parallel -> merge -> serve.

**Experimental path:** audit, training, decision traces, and extra enrichment layers are still present, but they are not part of the default stripped-down runtime.

---

## MCP Integration

Works with Claude, OpenClaw, Cursor, Ollama-adjacent local agents, or any MCP client.

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

Core tools: `add_episode` · `search_facts` · `search_nodes` · `get_entity_context` · `delete_entity` · `prune_entities` · `get_queue_status`

Legacy tools like audit and training remain available only when the active backend supports them.

---

## Python SDK

```bash
pip install "git+https://github.com/brianmeyer/mollygraph.git#subdirectory=sdk"
```

```python
from mollygraph_sdk import MollyGraphClient

client = MollyGraphClient(base_url="http://localhost:7422", api_key="YOUR_KEY")
client.ingest("Sarah joined the ML team at Acme Corp.", source="slack", speaker="Brian")
result = client.query("What do we know about Sarah?")
```

---

## HTTP API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health |
| `/ingest` | POST | Ingest text (supports `speaker` field) |
| `/query` | GET | Parallel graph+vector search |
| `/entity/{name}` | GET | Entity context (2-hop neighborhood) |
| `/entity/{name}` | DELETE | Delete entity + relationships |
| `/entities/prune` | POST | Bulk prune + orphan detection |
| `/metrics/dashboard` | GET | Unified health JSON |
| `/stats` | GET | Graph/vector/runtime stats |

Neo4j-oriented endpoints like audit, training, and the legacy maintenance cycle remain in the codebase but are intentionally gated when the Ladybug backend is active.

---

## Configuration

100% env-var driven. See [service/.env.example](/Users/brianmeyer/mollygraph/service/.env.example) for all options. Key settings:

```env
# Local-first defaults
MOLLYGRAPH_GRAPH_BACKEND=ladybug
MOLLYGRAPH_VECTOR_BACKEND=ladybug
MOLLYGRAPH_EMBEDDING_ST_MODEL=Snowflake/snowflake-arctic-embed-s

# Optional local alternate
MOLLYGRAPH_EMBEDDING_OLLAMA_MODEL=nomic-embed-text

# Extraction confidence (per source)
MOLLYGRAPH_EXTRACTION_CONFIDENCE_SESSION=0.55
MOLLYGRAPH_EXTRACTION_CONFIDENCE_EMAIL=0.45
MOLLYGRAPH_EXTRACTION_CONFIDENCE_DEFAULT=0.4

# Neo4j remains available for legacy/full workflows
# NEO4J_URI=bolt://localhost:7687
```

---

## Roadmap

The authoritative roadmap lives in `service/BACKLOG.md`.
`service/DECISION_TRACES_PLAN.md` is a later-phase experimental plan, not part of the default product path.

**Top priorities right now:**
- Finish the Ladybug local core for the default ingest/query/vector path
- Keep the default runtime small and trustworthy for non-developers
- Harden MCP, HTTP, and SDK behavior around the simplified memory core
- Add source-routed extraction where it improves graph quality without re-complexifying the product
- Keep advanced audit/training/decision-trace work clearly optional

**Current feature status:**
- Ladybug graph backend — active default
- Ladybug vector backend — active
- GLiNER2 local extraction — active
- Snowflake local embeddings — active default
- Audit, training, and decision surfaces — legacy/experimental, being gated behind backend capability

---

## License

MIT

---

<p align="center">
  <strong>Structured local memory for agents.</strong>
</p>
