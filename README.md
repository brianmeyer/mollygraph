<p align="center">
  <h1 align="center">🧠 MollyGraph</h1>
  <p align="center"><strong>The AI memory layer that rewrites its own brain.</strong></p>
</p>

<p align="center">
  <em>Local-first knowledge graph with self-improving extraction, parallel graph+vector retrieval, and a LoRA training loop that ships only when it wins.</em>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> · <a href="#mcp-integration">MCP</a> · <a href="#python-sdk">Python SDK</a> · <a href="#http-api">HTTP API</a> · <a href="#roadmap">Roadmap</a>
</p>

---

Most memory layers are **static pipes** — same extraction quality on day 1 as day 1000. MollyGraph is different: it **fine-tunes its own extraction model on your data**, runs a benchmark A/B test, and auto-promotes only if it wins. Your memory gets better the more you use it.

Oh, and queries now run graph exact + vector similarity **in parallel** and merge results. No more waterfall cascades.

---

## ✨ What Makes This Different

| Feature | Other memory systems | MollyGraph |
|---------|---------------------|------------|
| Extraction quality | Static, day 1 = day 1000 | Self-improving LoRA loop |
| Retrieval | Sequential fallback | Parallel graph + vector, merged |
| Embeddings | Hardcoded model | Env-var swappable, tiered fallback |
| Training gating | None | Benchmark A/B + shadow eval |
| Graph health | You figure it out | `/metrics/dashboard` unified JSON |
| Memory management | Append-only | Delete, prune, orphan detection |
| Schema drift | Silently explodes | Alarm at >5%/day growth |

---

## 🚀 Key Features

### 🔁 Self-Evolving Extraction (The Killer Feature)
Every ingested episode becomes a training example. Periodically, MollyGraph fine-tunes GLiNER on your accumulated real-world graph data — not synthetic examples, not pre-baked benchmarks. The candidate model gets A/B tested against the active model on a held-out eval split, shadow-evaluated on live traffic, and hot-reloaded with zero downtime only if it **wins**. Loses? Automatic rollback. No manual labeling. No prompt engineering. Just run your agents.

> **From last LoRA run: Entity F1 +7%, Relation F1 +1.6%**  
> **Training set: 1,798 real examples, zero human annotation**

### ⚡ Parallel Graph + Vector Retrieval (New)
Old system: try graph exact → if miss, try fuzzy → if miss, try vector. Waterfalls are slow and lossy.  
New system: graph exact AND vector similarity fire simultaneously. Results are merged and deduped. The `graph_lift_pct` metric proves graph is doing real work on top of vector — it's not just a slow wrapper around semantic search.

> **Current stats: 2.3% RELATED_TO fallback rate** (down from 8.4%)

### 🧬 Live Graph — Right Now
```
907 entities   ·   2,196 relationships   ·   28 relationship types
2.42 density (relationships per entity)
```

### 🎛️ Jina v5-nano Embeddings (Feb 18 2026)
Upgraded from `embeddingGemma-300m` to `jinaai/jina-embeddings-v5-text-nano` — 71.0 MTEB score, 239M params. Still fully local, still swappable via env var. Tiered fallback: sentence-transformers → ollama → cloud → hash. Never breaks.

### 🔍 Configurable Reranker
Jina reranker v2 (same embedding family). Off by default. One env var to enable. Reranks merged graph+vector results for max precision.

### 🗑️ Memory Management
Delete entities. Delete relationships. Bulk prune with orphan detection. The nightly audit can auto-execute its own suggestions (opt-in). Your graph stays lean.

### 📊 Unified Metrics Dashboard
`GET /metrics/dashboard` — single JSON endpoint with graph health, retrieval stats, embedding config, training status, uptime, and lift metrics. Know your memory's health at a glance.

### 🤖 MCP Tools (9 tools)
`search_facts`, `search_nodes`, `get_entity_context`, `add_episode`, `delete_entity`, `prune_entities`, `run_audit`, `get_training_status`, `get_queue_status`

---

## 📈 Live Metrics Snapshot

```
Graph Health
  Entities:          907
  Relationships:    2,196
  Relationship types:  28
  Graph density:      2.42 (rels/entity)

Retrieval Quality
  RELATED_TO fallback:  2.3%  (was 8.4%)
  graph_lift_pct:       tracked per query

Extraction (GLiNER LoRA)
  Training examples:   1,798
  Last cycle delta:     Entity F1 +7.0%, Relation F1 +1.6%

Embeddings
  Model:  jinaai/jina-embeddings-v5-text-nano
  MTEB:   71.0
  Params: 239M
```

---

## 🔄 The Loop

```
┌─────────────────────────────────────────────────────────────┐
│                    The Self-Improvement Loop                 │
└─────────────────────────────────────────────────────────────┘

  ingest episode
       │
       ▼
  GLiNER2 extraction ──────────────────────────────────┐
  + SpaCy enrichment                                    │
       │                                                │
       ▼                                                ▼
  Neo4j graph write                            training example
  + vector embed (Jina v5-nano)                accumulated in queue
       │                                                │
       ▼                                    (threshold reached)
  ┌─────────────────┐                                   │
  │  PARALLEL QUERY │                                   ▼
  │  graph exact ───┼──┐                        LoRA fine-tune GLiNER
  │  vector sim  ───┼──┤                                │
  └─────────────────┘  │                                ▼
                        ▼                       benchmark A/B test
                   merge + rerank               (candidate vs active)
                   → serve results                       │
                                              ┌──────────┴──────────┐
                                              │ wins?                │ loses?
                                              ▼                      ▼
                                         hot-reload             rollback
                                       (zero downtime)       (nothing changes)
```

---

## ⚡ Quick Start

```bash
git clone https://github.com/brianmeyer/mollygraph.git
cd mollygraph
cp .env.example .env
docker compose -f docker-compose.neo4j.yml up -d
./scripts/install.sh && ./scripts/start.sh
```

API at `http://127.0.0.1:7422`. State in `~/.graph-memory`.

Default auth: `Bearer dev-key-change-in-production` (change this in `.env` before production).

---

## 🔌 MCP Integration

The fastest path to agent memory. Works with Claude, OpenClaw, Cursor, or any MCP-compatible client.

```bash
pip install "git+https://github.com/brianmeyer/mollygraph.git#subdirectory=sdk[mcp]"
mollygraph-mcp --base-url http://localhost:7422 --api-key YOUR_KEY
```

**9 tools, zero config:**

| Tool | What It Does |
|------|-------------|
| `add_episode` | Ingest text → entities + relations extracted automatically |
| `search_facts` | Parallel graph+vector search across the knowledge graph |
| `search_nodes` | Find entities by name or type |
| `get_entity_context` | Full neighborhood context for a specific entity |
| `delete_entity` | Remove an entity and its relationships |
| `prune_entities` | Bulk prune + orphan detection |
| `run_audit` | Trigger LLM quality audit on demand |
| `get_training_status` | Check GLiNER training pipeline status |
| `get_queue_status` | Check async processing queue |

**claude_desktop_config.json:**
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

---

## 🐍 Python SDK

```bash
pip install "git+https://github.com/brianmeyer/mollygraph.git#subdirectory=sdk"
```

```python
from mollygraph_sdk import MollyGraphClient

client = MollyGraphClient(base_url="http://localhost:7422", api_key="YOUR_KEY")

# Ingest — extraction happens automatically, graph is updated
client.ingest("Sarah joined the ML team at Acme Corp last Tuesday.", source="slack")

# Query — parallel graph+vector, merged results
result = client.query("What do we know about Sarah?")
print(result)

# Entity management
client.delete_entity("old-entity-name")
client.prune_entities(min_strength=0.1)  # remove weak nodes

# Check the loop
status = client.get_training_status()
print(f"Training examples: {status['queue_size']}")

client.close()
```

---

## 🌐 HTTP API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health + vector stats |
| `/stats` | GET | Graph statistics |
| `/ingest` | POST | Ingest episode text |
| `/query` | GET | Parallel graph+vector search |
| `/entity/{name}` | GET | Entity context |
| `/entity/{name}` | DELETE | Delete entity + relationships |
| `/relationship` | DELETE | Delete specific relationship |
| `/entities/prune` | POST | Bulk prune + orphan detection |
| `/audit` | POST | Run quality audit |
| `/train/gliner` | POST | Trigger LoRA training cycle |
| `/train/status` | GET | Training pipeline status |
| `/metrics/dashboard` | GET | Unified health JSON (new) |
| `/metrics/retrieval` | GET | graph_exact / vector / combined breakdown |
| `/metrics/evolution` | GET | Graph stats over time, training history |
| `/metrics/schema-drift` | GET | Ontology growth alarm (>5%/day threshold) |
| `/metrics/summary` | GET | Daily request counts + latency percentiles |
| `/suggestions/digest` | GET | Auto-proposed schema improvements |
| `/embeddings/config` | GET/POST | Swap embedding provider live |
| `/embeddings/status` | GET | Provider readiness |
| `/embeddings/reindex` | POST | Reindex after model switch |
| `/maintenance/nightly` | POST | Full maintenance cycle |

---

## ⚙️ Configuration

100% env-var driven. Copy `.env.example` and edit. No hardcoded values.

### Embeddings

```env
# Model (swappable at runtime — will auto-reindex)
MOLLYGRAPH_EMBEDDING_MODEL=jinaai/jina-embeddings-v5-text-nano
MOLLYGRAPH_EMBEDDING_BACKEND=sentence-transformers  # or: ollama | cloud

# Tiered fallback order: sentence-transformers → ollama → cloud → hash
# If your primary fails, it falls through automatically

# Reranker (off by default)
MOLLYGRAPH_RERANKER_ENABLED=false
MOLLYGRAPH_RERANKER_MODEL=jinaai/jina-reranker-v2-base-multilingual

# Ollama fallback
MOLLYGRAPH_OLLAMA_BASE_URL=http://localhost:11434
MOLLYGRAPH_OLLAMA_EMBED_MODEL=nomic-embed-text
```

### Extraction

```env
# GLiNER2 — self-improving LoRA loop
# Starts with base model, accumulates training data, fine-tunes on your graph
GLINER_MODEL=urchade/gliner_medium-v2.1
SPACY_ENRICHMENT_ENABLED=true  # SpaCy enrichment pass (on by default)

# Training
TRAINING_MIN_EXAMPLES=100      # examples before first fine-tune
TRAINING_EVAL_SPLIT=0.2        # held-out benchmark fraction
```

### Audit

```env
# Tiered LLM audit: deterministic → local → primary → fallback
AUDIT_LLM_ENABLED=true
AUDIT_AUTO_DELETE=false        # opt-in: audit auto-executes its delete suggestions

# Provider order (first available wins)
AUDIT_PROVIDER_ORDER=ollama,anthropic,openai

# Supported: moonshot, groq, ollama, openai, openrouter, together, fireworks, anthropic, gemini
AUDIT_MODEL_LOCAL=llama3.1:8b
AUDIT_MODEL_PRIMARY=claude-3-5-haiku-20241022
AUDIT_MODEL_FALLBACK=gpt-4o-mini
```

### Core

```env
MOLLYGRAPH_API_KEY=dev-key-change-in-production
MOLLYGRAPH_HOST=127.0.0.1
MOLLYGRAPH_PORT=7422

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password

MOLLYGRAPH_STATE_DIR=~/.graph-memory
```

---

## 🗺️ Roadmap

### Enrichment Pipeline
- [ ] **Specialized GLiNER passes** — domain-specific models as additive passes: biomedical (`gliner-biomed`), PII detection (`nvidia/gliner-PII`), multilingual (`gliner_multi-v2.1`)
- [ ] **GLiREL integration** — `jackboyla/glirel-large-v0` as second-pass relation extractor; targets the residual RELATED_TO fallback
- [ ] **Entity merge & dedup** — when multiple models extract the same entity, merge by confidence score

### Model Infrastructure
- [ ] **Model hot-swap with unload** — config change → unload old model → load new one, no restart
- [ ] **GGUF support** — quantized models via llama-cpp-python for lower memory
- [ ] **Model download management** — pre-download, disk usage tracking, cleanup

### Known Engineering Gaps
- [ ] Partial graph writes — no rollback on entity+relationship step failures
- [ ] Duplicate edges under concurrency — relationship upsert is read-then-write
- [ ] Queue worker silent death — health says "healthy" even if worker crashed
- [ ] Training cursor skips at timestamp boundaries
- [ ] Deployment fallback can delete active model before copy succeeds

### Future
- [ ] Multi-agent memory isolation
- [ ] Web UI for graph exploration
- [ ] Plugin system for custom extractors
- [ ] MLX embeddings (Apple Silicon)

### Done ✅
- [x] Self-evolving GLiNER extraction with LoRA
- [x] Benchmark-gated deployment with shadow evaluation
- [x] **Parallel graph + vector retrieval** (Feb 2026)
- [x] **Jina v5-nano embeddings** (Feb 2026)
- [x] **Tiered embedding fallback** (sentence-transformers → ollama → cloud → hash)
- [x] **Configurable Jina reranker v2** (off by default)
- [x] **Entity + relationship delete endpoints**
- [x] **POST /entities/prune** (bulk + orphan detection)
- [x] **GET /metrics/dashboard** unified health JSON
- [x] **Retrieval lift metrics** (graph_lift_pct, vector_lift_pct)
- [x] **Audit auto-delete** (opt-in nightly auto-execute)
- [x] MCP server + Python SDK
- [x] Local embedding support (sentence-transformers, Ollama)
- [x] Relationship suggestion system
- [x] Hot-reload model deployment + automatic rollback
- [x] Strength decay — stale knowledge fades, active stays prominent
- [x] Bi-temporal graph (valid_from, valid_to, observed_at, last_seen)
- [x] Schema drift alarm (>5%/day growth threshold)
- [x] Retrieval source tracking — Graph Lift metric

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issues and PRs welcome.

**Before you open a PR:** run the test suite, check that new env vars are documented in `.env.example`, and note if your change affects the training pipeline or retrieval path — those need benchmarks.

## 📄 License

MIT

---

<p align="center">
  <strong>Your agents deserve memory that learns. Give them MollyGraph.</strong>
</p>
