<p align="center">
  <h1 align="center">🧠 MollyGraph</h1>
  <p align="center"><strong>Self-improving knowledge graph for AI agents.</strong></p>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> · <a href="#how-it-works">How It Works</a> · <a href="#mcp-integration">MCP</a> · <a href="#http-api">API</a> · <a href="#configuration">Config</a>
</p>

---

Most agent memory is a vector store with a wrapper. Same extraction quality forever. No structure. No relationships. No learning.

MollyGraph builds a **knowledge graph that improves its own extraction model**. Ingest text → extract entities and relationships → fine-tune the extractor on your data → deploy only if it beats the current model. The graph gets smarter the more you use it.

---

## What Makes This Different

- **Self-improving extraction** — GLiNER2 fine-tunes itself via LoRA on your real graph data. Candidate models must win an A/B benchmark to deploy. No manual labeling.
- **Speaker-anchored ingestion** — Each message processed individually with the speaker as anchor entity. Per-source confidence thresholds prevent chat noise from polluting the graph.
- **Parallel retrieval** — Graph exact-match and vector similarity run simultaneously. Results merge and dedup. Each catches what the other misses.
- **LLM audit chain** — Relationships reviewed by Kimi k2.5 (instant) → Gemini Flash → deterministic rules. Verdicts: verify, reclassify, quarantine, delete.
- **Three-layer NER** — GLiNER2 → spaCy enrichment → GLiREL relation extraction with synonym maps.
- **100% env-var config** — Swap embedding models, audit providers, confidence thresholds, everything. No code changes.

---

## Quick Start

```bash
git clone https://github.com/brianmeyer/mollygraph.git
cd mollygraph
cp .env.example .env    # edit with your Neo4j creds
docker compose -f docker-compose.neo4j.yml up -d
./scripts/install.sh && ./scripts/start.sh
```

API at `http://127.0.0.1:7422`. Auth: `Bearer dev-key-change-in-production`.

---

## How It Works

```
  ingest text
       │
       ▼
  Speaker-anchored extraction (GLiNER2 + spaCy + GLiREL)
  Per-source confidence gates (chat: 0.55, email: 0.45)
       │
       ├──▶ Neo4j graph + Jina v5-nano vectors
       └──▶ Training examples accumulated
                    │
              (threshold reached)
                    ▼
              LoRA fine-tune → A/B benchmark → deploy if wins
```

**Query path:** graph exact + vector similarity fire in parallel → merge → rerank → serve.

**Audit path:** LLM reviews relationships with original context snippets → verify/reclassify/quarantine/delete.

---

## MCP Integration

Works with Claude, OpenClaw, Cursor, or any MCP client.

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

**9 tools:** `add_episode` · `search_facts` · `search_nodes` · `get_entity_context` · `delete_entity` · `prune_entities` · `run_audit` · `get_training_status` · `get_queue_status`

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
| `/audit/run` | POST | Trigger LLM audit |
| `/training/status` | GET | LoRA pipeline status |
| `/metrics/dashboard` | GET | Unified health JSON |
| `/maintenance/nightly` | POST | Full maintenance cycle |

---

## Configuration

100% env-var driven. See `.env.example` for all options. Key settings:

```env
# Extraction confidence (per source)
MOLLYGRAPH_EXTRACTION_CONFIDENCE_SESSION=0.55
MOLLYGRAPH_EXTRACTION_CONFIDENCE_EMAIL=0.45
MOLLYGRAPH_EXTRACTION_CONFIDENCE_DEFAULT=0.4

# LLM audit tiers
MOLLYGRAPH_AUDIT_PROVIDER_TIERS=primary,fallback,deterministic
MOLLYGRAPH_AUDIT_TIER_PRIMARY=moonshot        # Kimi k2.5 instant
MOLLYGRAPH_AUDIT_TIER_FALLBACK=gemini         # Gemini 2.5 Flash

# Embeddings (swappable, tiered fallback)
MOLLYGRAPH_EMBEDDING_MODEL=jinaai/jina-embeddings-v5-text-nano
```

---

## Roadmap

- [ ] Quarantine review system — LLM generates yes/no questions for human review of ambiguous relationships
- [ ] Local-first audit — get ollama models producing reliable structured JSON for zero-cost audits
- [ ] Decision traces — reasoning chains as first-class graph nodes
- [ ] Multi-agent memory isolation
- [ ] Web UI for graph exploration

**Implemented (behind feature flags):**
- Graph-aware reranking — neighborhood density, shortest-path distance, relationship-type relevance (`GRAPH_RERANK_ENABLED`)
- Jina reranker v2 (`MOLLYGRAPH_RERANKER_ENABLED`)
- Both designed to activate at scale (5K+ entities)

---

## License

MIT

---

<p align="center">
  <strong>Memory that learns. Not memory that stores.</strong>
</p>
