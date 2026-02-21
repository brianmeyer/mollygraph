# MollyGraph

Local-first memory layer for agents and RAG applications.

MollyGraph can be used as:
- an MCP server adapter (`mollygraph-mcp`)
- a Python SDK (`mollygraph-sdk`)
- a self-hosted HTTP memory service

## Why MollyGraph

- Local-first by default (no cloud dependency required).
- Model-agnostic runtime configuration.
- Optional local model support (Ollama, sentence-transformers).
- Optional LLM audit pipeline (disabled by default).
- Stable API contract with legacy aliases preserved.

## Quick Start (Local Service)

```bash
cd /Users/brianmeyer/mollygraph
cp .env.example .env
docker compose -f docker-compose.neo4j.yml up -d
./scripts/install.sh
./scripts/start.sh
```

Service defaults:
- API: `http://127.0.0.1:7422`
- Runtime state: `~/.graph-memory`
- API key: `dev-key-change-in-production`

## Use as MCP

Install:

```bash
pip install "git+https://github.com/brianmeyer/mollygraph.git#subdirectory=sdk[mcp]"
```

Run:

```bash
mollygraph-mcp --base-url http://localhost:7422 --api-key dev-key-change-in-production
```

Example MCP config:

```json
{
  "mcpServers": {
    "mollygraph": {
      "command": "mollygraph-mcp",
      "args": ["--base-url", "http://localhost:7422"],
      "env": {
        "MOLLYGRAPH_API_KEY": "dev-key-change-in-production"
      }
    }
  }
}
```

MCP tools:
- `add_episode`
- `search_facts`
- `search_nodes`
- `get_entity_context`
- `get_queue_status`
- `run_audit`
- `get_training_status`

## Use as Python SDK

Install:

```bash
pip install "git+https://github.com/brianmeyer/mollygraph.git#subdirectory=sdk"
```

Example:

```python
from mollygraph_sdk import MollyGraphClient

client = MollyGraphClient(
    base_url="http://localhost:7422",
    api_key="dev-key-change-in-production",
)

client.ingest("Brian works at Databricks.", source="manual")
print(client.query("What do we know about Brian?"))
client.close()
```

## Local Model Configuration

Embeddings:
- `MOLLYGRAPH_EMBEDDING_BACKEND=hash` (default, no extra model runtime)
- `MOLLYGRAPH_EMBEDDING_BACKEND=sentence-transformers`
- `MOLLYGRAPH_EMBEDDING_BACKEND=ollama`

Embedding model vars:
- `MOLLYGRAPH_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`
- `MOLLYGRAPH_OLLAMA_EMBED_MODEL=nomic-embed-text`

Optional local LLM audit with Ollama:

```env
AUDIT_LLM_ENABLED=true
AUDIT_PROVIDER_ORDER=ollama,none
AUDIT_MODEL_LOCAL=llama3.1:8b
OLLAMA_CHAT_BASE_URL=http://127.0.0.1:11434/v1
```

## Optional Cloud LLM Add-ons

Cloud providers are optional and off-path by default.
Set `AUDIT_LLM_ENABLED=true`, choose `AUDIT_PROVIDER_ORDER`, then provide relevant API keys.

## HTTP API Contract

Canonical endpoints:
- `GET /health`
- `GET /stats`
- `POST /ingest`
- `GET /entity/{name}`
- `GET /query`
- `POST /audit`
- `GET /suggestions/digest`
- `POST /train/gliner`
- `GET /train/status`
- `POST /maintenance/run`

Legacy aliases retained:
- `POST /extract` -> `POST /ingest`
- `POST /audit/run` -> `POST /audit`
- `POST /maintenance/audit` -> `POST /audit`
- `GET /suggestions_digest` -> `GET /suggestions/digest`
- `POST /training/gliner` -> `POST /train/gliner`
- `GET /training/status` -> `GET /train/status`

## Development

Run tests:

```bash
MOLLYGRAPH_TEST_MODE=1 ~/.graph-memory/venv/bin/pytest -q
```

## Repository Layout

- `service/`: FastAPI service and core memory pipeline
- `sdk/`: Python SDK + MCP adapter package
- `scripts/`: install/start scripts
- `tests/`: smoke + integration API contract tests
- `docs/`: architecture docs
