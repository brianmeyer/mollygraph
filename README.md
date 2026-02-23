<p align="center">
  <h1 align="center">🧠 MollyGraph</h1>
  <p align="center"><strong>Memory that gets smarter every time your agent runs.</strong></p>
</p>

<p align="center">
  <em>The only local-first memory layer that fine-tunes its own extraction model on your data.</em>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> · <a href="#use-as-mcp">MCP</a> · <a href="#use-as-python-sdk">Python SDK</a> · <a href="#http-api">HTTP API</a>
</p>

---

Most agent memory systems store what you tell them. MollyGraph **learns how to extract better** from every conversation — automatically.

- 🔁 **Self-evolving extraction** — LoRA fine-tunes GLiNER on your real graph data. Benchmark-gated. Shadow-evaluated. Auto-deployed.
- 🏠 **Local-first** — Neo4j + local embeddings + local models. No cloud required.
- 🔌 **Drop-in** — MCP server, Python SDK, or HTTP API. Pick your integration.
- 📈 **Measurably better over time** — Every training cycle A/B tests the candidate model against the current one. Ships only if it wins.

> **First LoRA cycle on real data: Entity F1 +15%, Relation F1 +13%.** No manual labeling. No prompt engineering. Just run your agents.

## What Makes This Different

Other memory layers are **static pipelines** — same extraction quality on day 1 as day 100.

MollyGraph runs a **continuous improvement loop**:

```
ingest episodes → extract entities & relations → accumulate training data
       ↓                                                    ↓
  serve queries ← deploy if better ← benchmark A/B ← fine-tune GLiNER
                        ↑
                  shadow evaluation
                  (test on live data before promoting)
```

The extraction model that parses your text today is better than the one from last week. And next week's will be better than today's.

### The Pipeline

| Stage | What Happens |
|-------|-------------|
| **Accumulate** | Every ingested episode becomes a training example |
| **Train** | LoRA or full fine-tune, chosen automatically based on data volume |
| **Benchmark** | Candidate vs. active model on held-out eval split |
| **Shadow** | Run both models on live episodes, check for regressions |
| **Deploy** | Hot-reload — zero downtime, automatic rollback if needed |

### Quality Signals

MollyGraph doesn't just store — it **tracks what it doesn't know**:

- Relationship fallbacks → `RELATED_TO` (something was extracted but the type was uncertain)
- Schema suggestions → new relation types auto-proposed after repeated patterns
- Nightly audits → LLM-powered review catches drift and inconsistencies
- Strength decay → stale entities fade, active ones stay prominent

## Quick Start

```bash
git clone https://github.com/brianmeyer/mollygraph.git
cd mollygraph
cp .env.example .env
docker compose -f docker-compose.neo4j.yml up -d
./scripts/install.sh
./scripts/start.sh
```

That's it. API at `http://127.0.0.1:7422`. State lives in `~/.graph-memory`.

## Use As MCP

The fastest way to give your agent memory.

```bash
pip install "git+https://github.com/brianmeyer/mollygraph.git#subdirectory=sdk[mcp]"
mollygraph-mcp --base-url http://localhost:7422 --api-key YOUR_KEY
```

**7 tools, zero config:**

| Tool | What It Does |
|------|-------------|
| `add_episode` | Ingest text → entities + relations extracted automatically |
| `search_facts` | Semantic search across the knowledge graph |
| `search_nodes` | Find entities by name or type |
| `get_entity_context` | Full context for a specific entity |
| `get_queue_status` | Check async processing status |
| `run_audit` | Trigger quality audit on demand |
| `get_training_status` | Check GLiNER training pipeline status |

## Use As Python SDK

```bash
pip install "git+https://github.com/brianmeyer/mollygraph.git#subdirectory=sdk"
```

```python
from mollygraph_sdk import MollyGraphClient

client = MollyGraphClient(base_url="http://localhost:7422", api_key="YOUR_KEY")

# Ingest — extraction happens automatically
client.ingest("Sarah joined the ML team at Acme Corp last Tuesday.", source="slack")

# Query — semantic + graph traversal
result = client.query("What do we know about Sarah?")
print(result)

client.close()
```

## Local Models

MollyGraph runs entirely on local hardware. No API keys needed for core functionality.

**Embeddings** (pick one):
```env
MOLLYGRAPH_EMBEDDING_BACKEND=sentence-transformers   # recommended
MOLLYGRAPH_EMBEDDING_MODEL=google/embeddinggemma-300m
# or
MOLLYGRAPH_EMBEDDING_BACKEND=ollama
MOLLYGRAPH_OLLAMA_EMBED_MODEL=nomic-embed-text
```

**Extraction** — GLiNER2 runs locally on CPU or MPS (Apple Silicon):
```env
# Starts with base model, self-improves over time
# No configuration needed — it just gets better
```

**Audits** (optional LLM review):
```env
AUDIT_LLM_ENABLED=true
AUDIT_PROVIDER_ORDER=ollama,none
AUDIT_MODEL_LOCAL=llama3.1:8b
```

## HTTP API

Full REST API for custom integrations:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health + vector stats |
| `/stats` | GET | Graph statistics |
| `/ingest` | POST | Ingest episode text |
| `/query` | GET | Semantic + graph search |
| `/entity/{name}` | GET | Entity context |
| `/audit` | POST | Run quality audit |
| `/train/gliner` | POST | Trigger training cycle |
| `/train/status` | GET | Training pipeline status |
| `/suggestions/digest` | GET | Schema improvement suggestions |
| `/embeddings/config` | GET/POST | Embedding provider config |
| `/embeddings/status` | GET | Provider readiness check |
| `/embeddings/reindex` | POST | Reindex after model switch |
| `/maintenance/nightly` | POST | Run full maintenance cycle |

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  MCP Client │────▶│  MollyGraph  │────▶│   Neo4j     │
│  or SDK     │     │   Service    │     │  (entities, │
│  or HTTP    │     │  :7422       │     │  relations) │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
                    ┌──────┴───────┐
                    │   Zvec       │
                    │  (vectors,   │
                    │  embeddings) │
                    └──────┬───────┘
                           │
                    ┌──────┴───────┐
                    │  GLiNER2     │
                    │  (extraction,│
                    │  self-tuning)│
                    └──────────────┘
```

## Roadmap

- [x] Self-evolving GLiNER extraction with LoRA
- [x] Benchmark-gated deployment with shadow evaluation
- [x] MCP server + Python SDK
- [x] Local embedding support (sentence-transformers, Ollama)
- [x] Relationship suggestion system
- [x] Hot-reload model deployment
- [x] Bi-temporal graph (valid_time vs observed_time tracking)
- [x] Strength decay — stale knowledge fades, active knowledge stays prominent
- [ ] Multi-agent memory isolation
- [ ] Web UI for graph exploration
- [ ] Plugin system for custom extractors

## License

MIT

---

<p align="center">
  <strong>Your agents deserve memory that learns. Give them MollyGraph.</strong>
</p>
