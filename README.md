# MollyGraph

MCP-first memory layer for agents and RAG applications.

## Integration Modes

1. MCP adapter: connect MollyGraph tools to agent frameworks and MCP clients.
2. Python SDK: call MollyGraph from RAG pipelines and application code.
3. HTTP API: self-host the memory service for local or team deployments.

## MCP Usage (Recommended for Agents)

Install from source today:

```bash
pip install "git+https://github.com/brianmeyer/mollygraph.git#subdirectory=sdk[mcp]"
```

Run MCP adapter:

```bash
mollygraph-mcp --base-url http://localhost:7422 --api-key dev-key-change-in-production
```

Example MCP server config:

```json
{
  "mcpServers": {
    "mollygraph": {
      "command": "mollygraph-mcp",
      "args": [
        "--base-url",
        "http://localhost:7422"
      ],
      "env": {
        "MOLLYGRAPH_API_KEY": "dev-key-change-in-production"
      }
    }
  }
}
```

Tools exposed by MCP adapter:
- `add_episode`
- `search_facts`
- `search_nodes`
- `get_entity_context`
- `get_queue_status`
- `run_audit`
- `get_training_status`

## Python SDK Usage (RAG + App Code)

Install:

```bash
pip install "git+https://github.com/brianmeyer/mollygraph.git#subdirectory=sdk"
```

Use:

```python
from mollygraph_sdk import MollyGraphClient

client = MollyGraphClient(
    base_url="http://localhost:7422",
    api_key="dev-key-change-in-production",
)

print(client.ingest("Brian works at Databricks."))
print(client.query("What do we know about Brian?"))
client.close()
```

## Self-Host API Service

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
- Override state root with `MOLLYGRAPH_HOME_DIR=/custom/path`

## HTTP API Contract

Canonical endpoints are stable and legacy aliases are maintained for compatibility:
- `POST /ingest` (`POST /extract`)
- `POST /audit` (`POST /audit/run`, `POST /maintenance/audit`)
- `GET /suggestions/digest` (`GET /suggestions_digest`)
- `POST /train/gliner` (`POST /training/gliner`)
- `GET /train/status` (`GET /training/status`)
