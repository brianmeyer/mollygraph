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

MollyGraph is a **context graph** — it extracts entities and relationships, builds a knowledge graph, fine-tunes its own extraction model on your data via LoRA, and auto-promotes the new model only if it wins a benchmark A/B test. Speaker-anchored extraction processes each message individually with per-source confidence thresholds. Queries run graph exact + vector similarity **in parallel** and merge results. A tiered LLM audit (Moonshot/Kimi → Gemini Flash → deterministic) reviews relationships and auto-cleans garbage.

**The result: 251 verified entities, 1,419 relationships, zero noise. Graph finds things vector misses. Vector finds things graph misses. Together they cover everything.**

---

## ✨ What Makes This Different

| Feature | Vector-only memory | MollyGraph |
|---------|-------------------|------------|
| Extraction | Static NER, never improves | Self-improving GLiNER2 LoRA loop |
| Relation extraction | None or rule-based | Three-layer: GLiNER2 → spaCy → GLiREL |
| Chat extraction | Dumps everything, hopes for the best | Speaker-anchored, per-source confidence gates |
| Retrieval | Cosine similarity only | Parallel graph + vector, merged + reranked |
| Quality control | None | Tiered LLM audit (Kimi k2.5 → Gemini Flash → rules) |
| Structure | Flat chunks | Knowledge graph with typed relationships |
| Embeddings | Hardcoded model | Env-var swappable, tiered fallback chain |
| Training gating | None | Benchmark A/B + shadow eval, ships only if it wins |
| Memory management | Append-only | Delete, prune, LLM audit, strength decay |

---

## 🚀 Key Features

### 🔁 Self-Evolving Extraction
Every ingested episode becomes a training example. MollyGraph fine-tunes GLiNER2 on your accumulated real-world graph data — not synthetic examples, not pre-baked benchmarks. The candidate model gets A/B tested against the active model on a held-out eval split, shadow-evaluated on live traffic, and hot-reloaded with zero downtime only if it **wins**. No manual labeling. No prompt engineering. Just run your agents.

> **Last LoRA run: Entity F1 +4.32%**  
> **Training from clean, verified graph data only**

### 🎯 Speaker-Anchored Extraction
Most graph memory systems dump 50 messages into one chunk and hope for the best. Every speaker name becomes an entity. Every co-occurrence becomes a relationship. Garbage in, garbage out.

MollyGraph processes each message individually with the **speaker as the anchor entity**. Per-source confidence thresholds (0.55 for chat, 0.45 for email, 0.40 for manual) gate what enters the graph. Chat noise stays out. Real relationships get in.

### 🧬 Three-Layer Extraction Pipeline
Single-model extraction misses things. MollyGraph runs three layers:

1. **GLiNER2** — Primary NER + relation extraction with self-improving LoRA
2. **spaCy** — Additive NER enrichment (catches entities GLiNER2 misses)
3. **GLiREL** — Dedicated relation extraction with 35-entry synonym map and confidence-gated training data generation

GLiREL's high-confidence extractions (>0.4) become silver-label training data for the LoRA loop. The system teaches itself.

### ⚡ Parallel Graph + Vector Retrieval
Old way: try graph → if miss, try vector. Waterfalls are slow and lossy.  
MollyGraph: graph exact AND vector similarity fire **simultaneously**. Results merge and dedup. Retrieval lift metrics prove each system catches what the other misses.

> **100% hit rate · Parallel graph + vector · Merged + deduped**

### 🔍 Tiered LLM Audit
Relationships get reviewed by an LLM audit chain: Moonshot/Kimi k2.5 (instant mode) → Gemini 2.5 Flash (fallback) → deterministic rules. Each relationship is scored with context snippets from the original conversation. Verdicts: verify, reclassify, quarantine, or delete. Auto-executes with blast radius caps.

### 📊 Live Graph (post-purge, clean slate)
```
251 entities  ·  1,419 relationships  ·  28 types
183 episodes  ·  100% verified data  ·  Zero legacy noise
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
Graph                              Audit
  Entities:        251               Provider:     Kimi k2.5 (instant)
  Relationships:  1,419              Fallback:     Gemini 2.5 Flash
  Types:            28               Parse failures:     0
  Episodes:        183               Auto-delete:     enabled

Extraction                         Training
  Backend:     GLiNER2               Examples:    rebuilding from clean graph
  + GLiREL relations                 LoRA:        base model (post-purge)
  Speaker-anchored:   ✅             Cooldown:    48h between runs
  Per-source thresholds: ✅

Embeddings
  Model:   jinaai/jina-embeddings-v5-text-nano (71.0 MTEB, 239M params)
  Tiered fallback: sentence-transformers → ollama → cloud → hash
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
# Tiered LLM audit: primary → fallback → deterministic
AUDIT_LLM_ENABLED=true
MOLLYGRAPH_AUDIT_PROVIDER_TIERS=primary,fallback,deterministic

# Primary: Moonshot/Kimi k2.5 (instant mode — no thinking overhead)
MOLLYGRAPH_AUDIT_TIER_PRIMARY=moonshot
MOLLYGRAPH_AUDIT_MODEL_PRIMARY=kimi-k2.5

# Fallback: Gemini 2.5 Flash
MOLLYGRAPH_AUDIT_TIER_FALLBACK=gemini
MOLLYGRAPH_AUDIT_MODEL_FALLBACK=gemini-2.5-flash

# Supported providers: moonshot, groq, ollama, openai, openrouter, together, fireworks, anthropic, gemini
```

### Extraction Confidence

```env
# Per-source confidence thresholds (speaker-anchored extraction)
MOLLYGRAPH_EXTRACTION_CONFIDENCE_DEFAULT=0.4
MOLLYGRAPH_EXTRACTION_CONFIDENCE_SESSION=0.55
MOLLYGRAPH_EXTRACTION_CONFIDENCE_WHATSAPP=0.55
MOLLYGRAPH_EXTRACTION_CONFIDENCE_IMESSAGE=0.55
MOLLYGRAPH_EXTRACTION_CONFIDENCE_EMAIL=0.45
MOLLYGRAPH_EXTRACTION_CONFIDENCE_VOICE=0.50
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
- [x] **Speaker-anchored extraction** — Graphiti-style per-message processing with speaker as anchor entity (Feb 2026)
- [x] **Per-source confidence thresholds** — chat 0.55, email 0.45, manual 0.40 (Feb 2026)
- [x] **Tiered LLM audit** — Kimi k2.5 instant → Gemini 2.5 Flash → deterministic rules (Feb 2026)
- [x] **Graph purge tooling** — bulk delete old/noisy data, reset LoRA, rebuild clean (Feb 2026)
- [x] Self-evolving GLiNER2 extraction with LoRA fine-tuning
- [x] Benchmark-gated deployment with shadow evaluation
- [x] Three-layer extraction: GLiNER2 → spaCy → GLiREL
- [x] GLiREL relation extraction with 35-entry synonym map
- [x] Parallel graph + vector retrieval
- [x] Jina v5-nano embeddings — 71.0 MTEB, tiered fallback chain
- [x] Configurable Jina reranker v2
- [x] Unified metrics dashboard
- [x] MCP server (9 tools) + Python SDK
- [x] Bi-temporal graph + strength decay + schema drift alarm

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
