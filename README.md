<p align="center">
  <h1 align="center">🧠 MollyGraph</h1>
  <p align="center"><strong>The self-improving context graph for AI agents.</strong></p>
</p>

<p align="center">
  <em>Local-first knowledge graph that fine-tunes its own extraction model, runs three-layer NER+relation extraction, and proves graph+vector beats vector-alone — with metrics to back it up.</em>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> · <a href="#mcp-integration">MCP</a> · <a href="#python-sdk">Python SDK</a> · <a href="#http-api">HTTP API</a> · <a href="#-the-loop">The Loop</a> · <a href="#roadmap">Roadmap</a>
</p>

---

Most memory layers are **static vector stores with a fancy wrapper**. Same extraction quality on day 1 as day 1000. Same cosine similarity search. No structure. No relationships. No learning.

MollyGraph is a **context graph** — it extracts entities and relationships, builds a knowledge graph, fine-tunes its own extraction model on your data via LoRA, and auto-promotes the new model only if it wins a benchmark A/B test. Three extraction layers (GLiNER2 → spaCy → GLiREL) catch what single-model systems miss. Queries run graph exact + vector similarity **in parallel** and merge results.

**The result: 92% combined retrieval hits. Graph finds things vector misses. Vector finds things graph misses. Together they cover everything.**

---

## ✨ What Makes This Different

| Feature | Vector-only memory | MollyGraph |
|---------|-------------------|------------|
| Extraction | Static NER, never improves | Self-improving GLiNER2 LoRA loop |
| Relation extraction | None or rule-based | Three-layer: GLiNER2 → spaCy → GLiREL |
| Retrieval | Cosine similarity only | Parallel graph + vector, merged + reranked |
| Structure | Flat chunks | Knowledge graph with typed relationships |
| Embeddings | Hardcoded model | Env-var swappable, tiered fallback chain |
| Training gating | None | Benchmark A/B + shadow eval, ships only if it wins |
| Graph health | You figure it out | `/metrics/dashboard` — unified health JSON |
| Memory management | Append-only | Delete, prune, orphan detection, strength decay |
| Schema evolution | Manual | Auto-adoption pipeline with frequency gates |

---

## 🚀 Key Features

### 🔁 Self-Evolving Extraction
Every ingested episode becomes a training example. MollyGraph fine-tunes GLiNER2 on your accumulated real-world graph data — not synthetic examples, not pre-baked benchmarks. The candidate model gets A/B tested against the active model on a held-out eval split, shadow-evaluated on live traffic, and hot-reloaded with zero downtime only if it **wins**. No manual labeling. No prompt engineering. Just run your agents.

> **Last LoRA run: Entity F1 +7%, Relation F1 +1.6%**  
> **1,798 real training examples, zero human annotation**

### 🧬 Three-Layer Extraction Pipeline
Single-model extraction misses things. MollyGraph runs three layers:

1. **GLiNER2** — Primary NER + relation extraction with self-improving LoRA
2. **spaCy** — Additive NER enrichment (catches entities GLiNER2 misses)
3. **GLiREL** — Dedicated relation extraction with 35-entry synonym map and confidence-gated training data generation

GLiREL's high-confidence extractions (>0.4) become silver-label training data for the LoRA loop. The system teaches itself.

### ⚡ Parallel Graph + Vector Retrieval
Old way: try graph → if miss, try vector. Waterfalls are slow and lossy.  
MollyGraph: graph exact AND vector similarity fire **simultaneously**. Results merge and dedup. Retrieval lift metrics prove each system catches what the other misses.

> **100% hit rate · 92% combined retrieval · 3.8% RELATED_TO fallback (down from 8.4%)**

### 📊 Live Graph
```
975 entities  ·  2,701 relationships  ·  28 types  ·  2.77 density
1,798 training examples  ·  5 LoRA runs  ·  Last F1 gain: +4.32%
```

### 🎛️ Jina v5-nano Embeddings
`jinaai/jina-embeddings-v5-text-nano` — 71.0 MTEB, 239M params, 8192 context window. Fully local, swappable via env var. Tiered fallback: sentence-transformers → ollama → cloud → hash. Never breaks.

### 🔍 Configurable Reranker
Jina reranker v2 (same embedding family). Off by default — one env var to enable at scale. Reranks merged graph+vector results for max precision.

### 🗑️ Memory Management
Delete entities. Delete relationships. Bulk prune with orphan detection. Strength decay fades stale knowledge. Nightly LLM audit with opt-in auto-execute. Your graph stays lean.

### 📊 Unified Metrics Dashboard
`GET /metrics/dashboard` — graph health, retrieval stats (hit rate, lift %, latency percentiles), embedding config, extraction yields, training status, uptime. One endpoint, full picture.

### 🤖 MCP Tools (9 tools)
`add_episode` · `search_facts` · `search_nodes` · `get_entity_context` · `delete_entity` · `prune_entities` · `run_audit` · `get_training_status` · `get_queue_status`

---

## 📈 Live Metrics

```
Graph                              Retrieval
  Entities:        975               Hit rate:           100%
  Relationships:  2,701              Combined hits:       92%
  Types:            28               Avg latency:        244ms
  Density:         2.77              Graph lift:         100%
  RELATED_TO rate: 3.8%              Vector lift:        100%

Extraction                         Training
  Backend:     GLiNER2 + LoRA        Examples:          1,798
  + spaCy enrichment                 LoRA runs:             5
  + GLiREL relations                 Last F1 delta:    +4.32%
  Avg entity yield:  5.56            Cooldown:           48h
  Unique entity rate: 51%

Embeddings
  Model:   jinaai/jina-embeddings-v5-text-nano (71.0 MTEB, 239M params)
  Vectors: 954 (tiered fallback: sentence-transformers → ollama → cloud → hash)
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
  + spaCy enrichment                                    │
  + GLiREL relation extraction                          │
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

### 🛠️ MCP Tool Reference

#### `add_episode`
Ingest a piece of text. GLiNER2 + GLiREL extract entities and relationships automatically; the graph and vector store are updated asynchronously.

```
add_episode(content: str, source: str = "mcp", priority: int = 1) -> str
```

- **content** — text to ingest (conversation turn, document excerpt, etc.)
- **source** — label for the origin (e.g. `"slack"`, `"email"`, `"mcp"`)
- **priority** — queue priority; higher = processed sooner (default `1`)

Returns: `"queued <job_id> (depth=<n>)"`

---

#### `search_facts`
Run a natural-language query against the knowledge graph using **parallel graph + vector search**. Graph exact-match runs simultaneously with vector similarity; results are merged and deduplicated.

```
search_facts(query: str) -> str
```

- **query** — natural-language question or keyword string

Returns: up to 5 matching entities with their top-4 relationships, formatted as plain text.

---

#### `search_nodes`
Search Neo4j nodes (entities) directly by name substring and optional type. Uses the `/entities` endpoint rather than the query pipeline — faster for browsing the graph than for semantic search.

```
search_nodes(query: str, node_type: str = "", limit: int = 20) -> str
```

- **query** — substring to match against entity names (case-insensitive)
- **node_type** — optional type filter: `"Person"`, `"Technology"`, `"Organization"`, `"Project"`, `"Place"`, `"Concept"`, or any custom type
- **limit** — max results (1–50, default 20)

Returns: newline-delimited list of `"Name [Type]"` strings.

---

#### `get_entity_context`
Retrieve the full 2-hop neighborhood for a specific entity: all direct facts + second-degree connections.

```
get_entity_context(name: str) -> str
```

- **name** — exact or approximate entity name

Returns: entity name, facts (relationship → target), and direct connections summary.

---

#### `delete_entity`
Remove a single entity from Neo4j (DETACH DELETE, removes all attached relationships) and from the vector store.

```
delete_entity(name: str) -> str
```

- **name** — exact entity name to delete

Returns: `"deleted entity=<name> relationships_removed=<n> vector_removed=<bool>"`

---

#### `prune_entities`
Bulk-delete a list of entities. Each entity is removed from Neo4j (with all relationships) and the vector store. Useful for cleaning up noise or stale data.

```
prune_entities(names: list[str]) -> str
```

- **names** — list of entity names to remove

Returns: `"pruned=<n> vectors_removed=<n> entities=[...]"`

---

#### `run_audit`
Trigger an on-demand LLM quality audit of relationship data. The audit scores relationships deterministically (rule-based signals) and optionally with an LLM, then flags weak/incorrect relationships.

```
run_audit(limit: int = 500, dry_run: bool = False, schedule: str = "nightly") -> str
```

- **limit** — max relationships to audit (1–5000, default 500)
- **dry_run** — if `True`, score without writing changes back
- **schedule** — `"nightly"` (focused audit) or `"weekly"` (broader sweep)

Returns: `"audit status=<s> scanned=<n> verified=<n> autofixed=<n>"`

---

#### `get_training_status`
Check the state of the GLiNER self-improvement pipeline: how many examples have accumulated, the last fine-tune timestamp, and whether the last cycle passed benchmark.

```
get_training_status() -> str
```

Returns: `"examples=<n> last=<iso_ts> status=<cycle_status>"`

---

#### `get_queue_status`
Inspect the async extraction queue and vector store counters.

```
get_queue_status() -> str
```

Returns: `"pending=<n> processing=<n> vector={...}"`

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
GLINER_MODEL=fastino/gliner2-large-v1
SPACY_ENRICHMENT_ENABLED=true       # spaCy NER enrichment (on by default)

# GLiREL — dedicated relation extraction (second pass)
MOLLYGRAPH_GLIREL_ENABLED=true
MOLLYGRAPH_GLIREL_MODEL=jackboyla/glirel-large-v0
MOLLYGRAPH_GLIREL_CONFIDENCE=0.15   # minimum confidence to surface
MOLLYGRAPH_GLIREL_TRAINING_THRESHOLD=0.4  # confidence gate for silver-label training data
MOLLYGRAPH_GLIREL_LLM_SYNONYMS=true # LLM-assisted synonym generation on type adoption

# Training
GLINER_MIN_NEW_EXAMPLES=500    # new examples since last run to trigger training
GLINER_MAX_TRAINING_EXAMPLES=10000  # sliding window cap
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

### Next Up
- [ ] **Graph-aware reranking** — score results by graph neighborhood density, path distance, and relationship relevance (queued for overnight build)
- [ ] **Per-source quality metrics** — entity yield, unique rate, and quality score per ingestion source (queued for overnight build)
- [ ] **Decision traces** — capture reasoning chains as first-class graph nodes for audit trails and precedent search

### Future
- [ ] Specialized GLiNER passes — domain-specific models (biomedical, PII, multilingual) as additive extraction layers
- [ ] Multi-agent memory isolation
- [ ] Web UI for graph exploration
- [ ] Plugin system for custom extractors

### Done ✅
- [x] Self-evolving GLiNER2 extraction with LoRA fine-tuning
- [x] Benchmark-gated deployment with shadow evaluation
- [x] **Three-layer extraction: GLiNER2 → spaCy → GLiREL** (Feb 2026)
- [x] **GLiREL relation extraction** with 35-entry synonym map + auto-synonym generation
- [x] **Parallel graph + vector retrieval** (Feb 2026)
- [x] **Jina v5-nano embeddings** — 71.0 MTEB, tiered fallback chain
- [x] **Configurable Jina reranker v2** (off by default, same embedding family)
- [x] **Entity + relationship delete** + bulk prune with orphan detection
- [x] **Unified metrics dashboard** — graph, retrieval, extraction, training, uptime
- [x] **Retrieval lift metrics** — graph_lift_pct, vector_lift_pct, combined_hits
- [x] **Tiered LLM audit** — deterministic → local → primary → fallback (8 providers)
- [x] **Audit auto-delete** (opt-in, blast radius capped at 5% per run)
- [x] **Schema auto-adoption** — frequency-gated with GLiREL synonym generation
- [x] MCP server (9 tools) + Python SDK
- [x] PID file + stuck job recovery + Neo4j connection pooling
- [x] Strength decay — stale knowledge fades, active stays prominent
- [x] Bi-temporal graph (valid_from, valid_to, observed_at, last_seen)
- [x] Schema drift alarm (>5%/day growth threshold)

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
