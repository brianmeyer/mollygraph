# MollyGraph

Memory that gets better as your agents run.

MollyGraph is a local-first memory layer for agentic systems and RAG apps.  
Use it as MCP, Python SDK, or a self-hosted API.

## Why It Hits

- Drop-in memory for agents and tools via MCP.
- Local by default. No cloud dependency required.
- Model-agnostic configuration (local + cloud optional).
- Self-improving extraction pipeline with benchmark-gated model upgrades.
- Continuous graph-quality suggestions for better entities + relationships.

## The Good Stuff

### 1) Self-evolving extraction (GLiNER)

MollyGraph continuously accumulates real episodes into training data, then runs a fine-tune cycle:

- accumulates examples from your own graph history
- chooses training strategy (`lora` vs `full`) based on data and prior performance
- benchmarks candidate vs active model on held-out eval data
- deploys only if improvement clears threshold (`GLINER_FINETUNE_BENCHMARK_THRESHOLD`)

Endpoints:
- `POST /train/gliner`
- `GET /train/status`

### 2) Entity + relationship improvement suggestions

MollyGraph tracks schema misses and relationship uncertainty in production:

- relationship fallback events (unknown relation -> `RELATED_TO`)
- `RELATED_TO` hotspots that should be upgraded to precise relation types
- suggestion digests for nightly review
- optional auto-adoption rules over repeated patterns

Endpoint:
- `GET /suggestions/digest`

### 3) Local-first quality loop

Deterministic cleanup + optional LLM audit:

- orphan/self-reference cleanup
- strength decay maintenance
- optional LLM audit pipeline (`AUDIT_LLM_ENABLED=false` by default)
- provider chain can be local (`ollama`) or cloud add-ons

Endpoint:
- `POST /audit`

## Integration Modes

1. MCP adapter (`mollygraph-mcp`) for agents
2. Python SDK (`mollygraph-sdk`) for app/RAG code
3. HTTP API for self-hosting

## Quick Start (Self-Hosted)

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
- Embedding registry file: `~/.graph-memory/embedding_config.json`
- API key: `dev-key-change-in-production`

## Use As MCP

Install:

```bash
pip install "git+https://github.com/brianmeyer/mollygraph.git#subdirectory=sdk[mcp]"
```

Run:

```bash
mollygraph-mcp --base-url http://localhost:7422 --api-key dev-key-change-in-production
```

MCP tools:
- `add_episode`
- `search_facts`
- `search_nodes`
- `get_entity_context`
- `get_queue_status`
- `run_audit`
- `get_training_status`

## Use As Python SDK

Install:

```bash
pip install "git+https://github.com/brianmeyer/mollygraph.git#subdirectory=sdk"
```

Example:

```python
from mollygraph_sdk import MollyGraphClient

client = MollyGraphClient(base_url="http://localhost:7422", api_key="dev-key-change-in-production")
client.ingest("Brian works at Databricks.", source="manual")
print(client.query("What do we know about Brian?"))
client.close()
```

## Local Model Options

Embeddings:
- `MOLLYGRAPH_EMBEDDING_BACKEND=hash` (default, dependency-free)
- `MOLLYGRAPH_EMBEDDING_BACKEND=sentence-transformers`
- `MOLLYGRAPH_EMBEDDING_BACKEND=ollama`

Models:
- `MOLLYGRAPH_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`
- `MOLLYGRAPH_OLLAMA_EMBED_MODEL=nomic-embed-text`

Optional local audit with Ollama:

```env
AUDIT_LLM_ENABLED=true
AUDIT_PROVIDER_ORDER=ollama,none
AUDIT_MODEL_LOCAL=llama3.1:8b
OLLAMA_CHAT_BASE_URL=http://127.0.0.1:11434/v1
```

Configure embedding providers/models at runtime:

```bash
# Register additional models
curl -X POST http://127.0.0.1:7422/embeddings/models \
  -H "Authorization: Bearer dev-key-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{"provider":"huggingface","model":"BAAI/bge-small-en-v1.5"}'

curl -X POST http://127.0.0.1:7422/embeddings/models \
  -H "Authorization: Bearer dev-key-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{"provider":"ollama","model":"nomic-embed-text","activate":true}'

# Switch active provider/model
curl -X POST http://127.0.0.1:7422/embeddings/config \
  -H "Authorization: Bearer dev-key-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{"provider":"huggingface","model":"BAAI/bge-small-en-v1.5"}'

# Check provider/model readiness (HF deps, Ollama reachability/models)
curl -H "Authorization: Bearer dev-key-change-in-production" \
  http://127.0.0.1:7422/embeddings/status

# Reindex existing entities after switching model/provider
curl -X POST http://127.0.0.1:7422/embeddings/reindex \
  -H "Authorization: Bearer dev-key-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{"limit":5000,"dry_run":false}'
```

## Optional Cloud Add-ons

Cloud providers are optional, not required for core operation.
Enable only if you want them:

- set `AUDIT_LLM_ENABLED=true`
- set `AUDIT_PROVIDER_ORDER`
- configure provider keys in `.env`

## HTTP API Contract

Canonical:
- `GET /health`
- `GET /stats`
- `POST /ingest`
- `GET /entity/{name}`
- `GET /query`
- `POST /audit`
- `GET /suggestions/digest`
- `POST /train/gliner`
- `GET /train/status`
- `GET /embeddings/config`
- `GET /embeddings/status`
- `POST /embeddings/config`
- `POST /embeddings/models`
- `POST /embeddings/reindex`
- `POST /maintenance/run`

Legacy aliases retained:
- `POST /extract` -> `POST /ingest`
- `POST /audit/run` -> `POST /audit`
- `POST /maintenance/audit` -> `POST /audit`
- `GET /suggestions_digest` -> `GET /suggestions/digest`
- `POST /training/gliner` -> `POST /train/gliner`
- `GET /training/status` -> `GET /train/status`
