# MollyGraph

Local-first memory for agents and RAG solutions, exposed as MCP, Python SDK, and HTTP API.

## Product Stance

- Local by default.
- Model agnostic.
- LLM audit is optional (off by default).
- Works with local models (for example Ollama and sentence-transformers).

## Integration Modes

1. MCP adapter for agent runtimes and MCP clients.
2. Python SDK for RAG pipelines and app code.
3. HTTP API for self-hosted deployments.

## MCP Usage

Install SDK + MCP adapter:

```bash
pip install "git+https://github.com/brianmeyer/mollygraph.git#subdirectory=sdk[mcp]"
```

Run MCP adapter:

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

## Python SDK Usage

Install:

```bash
pip install "git+https://github.com/brianmeyer/mollygraph.git#subdirectory=sdk"
```

Use:

```python
from mollygraph_sdk import MollyGraphClient

client = MollyGraphClient(base_url="http://localhost:7422", api_key="dev-key-change-in-production")
print(client.ingest("Brian works at Databricks."))
print(client.query("What do we know about Brian?"))
client.close()
```

## Self-Host Service

```bash
cd /Users/brianmeyer/mollygraph
cp .env.example .env
docker compose -f docker-compose.neo4j.yml up -d
./scripts/install.sh
./scripts/start.sh
```

Defaults:
- API: `http://127.0.0.1:7422`
- Runtime state: `~/.graph-memory`
- Override state root: `MOLLYGRAPH_HOME_DIR=/custom/path`

## Local Model Options

Embedding backends:
- `MOLLYGRAPH_EMBEDDING_BACKEND=hash` (default, zero dependencies)
- `MOLLYGRAPH_EMBEDDING_BACKEND=sentence-transformers`
- `MOLLYGRAPH_EMBEDDING_BACKEND=ollama`

Embedding model settings:
- `MOLLYGRAPH_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`
- `MOLLYGRAPH_OLLAMA_EMBED_MODEL=nomic-embed-text`

Optional LLM audit:
- `AUDIT_LLM_ENABLED=false` by default
- Local provider example:
  - `AUDIT_LLM_ENABLED=true`
  - `AUDIT_PROVIDER_ORDER=ollama,none`
  - `AUDIT_MODEL_LOCAL=llama3.1:8b`
  - `OLLAMA_CHAT_BASE_URL=http://127.0.0.1:11434/v1`

## HTTP API Contract

Canonical endpoints are stable, with legacy aliases kept for compatibility:
- `POST /ingest` (`POST /extract`)
- `POST /audit` (`POST /audit/run`, `POST /maintenance/audit`)
- `GET /suggestions/digest` (`GET /suggestions_digest`)
- `POST /train/gliner` (`POST /training/gliner`)
- `GET /train/status` (`GET /training/status`)
