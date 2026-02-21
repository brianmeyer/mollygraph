# MollyGraph v1 (OpenClaw-First)

**Status:** Active development  
**Location:** `~/.openclaw/workspace/skills/graph-memory/`

## Quick Start

```bash
cd ~/.openclaw/workspace/skills/graph-memory
./scripts/install.sh  # requires python3.12
./scripts/start.sh    # starts HTTP API on port 7422
```

## Architecture

| Component | Technology | Port |
|-----------|-----------|------|
| HTTP API | FastAPI | 7422 |
| MCP Server | Model Context Protocol | 7423 (optional) |
| Graph DB | Neo4j | 7687 (Docker) |
| Vector DB | Zvec (Alibaba) | Embedded |
| Queue | SQLite | Embedded |

## API Endpoints

| Endpoint | Auth | Description |
|----------|------|-------------|
| `GET /health` | No | Health check |
| `GET /stats` | Bearer | Graph statistics |
| `POST /ingest` | Bearer | Queue text for extraction |
| `GET /entity/{name}` | Bearer | Get entity facts |
| `GET /query` | Bearer | Natural language query |
| `POST /audit` | Bearer | Run nightly/weekly relationship audit |
| `GET /suggestions/digest` | Bearer | View schema/hotspot suggestions |
| `POST /train/gliner` | Bearer | Trigger GLiNER training cycle |
| `GET /train/status` | Bearer | GLiNER training status |
| `POST /maintenance/run` | Bearer | Trigger full maintenance cycle |

## MCP Tools (like Graphiti/Mem0)

- `add_episode` - Queue text for extraction
- `search_facts` - Search graph relationships
- `search_nodes` - Find entities by name
- `get_entity_context` - Get entity with 2-hop context
- `get_queue_status` - Check queue depth
- `run_audit` - Trigger graph audit from MCP
- `get_training_status` - GLiNER training status snapshot

## Files

### Service (`service/`)
- `main.py` - FastAPI HTTP service
- `mcp_server.py` - Standalone MCP server
- `config.py` - Configuration
- `memory/` - Graph and vector operations
- `extraction/` - GLiNER2 pipeline and queue
- `maintenance/` - Audit and cleanup

### Documentation
- `SKILL.md` - Skill documentation
- `README.md` - This file
- `.env.example` - Runtime env template
- `docs/ARCHITECTURE.md` - System architecture and flow
- `CONTRIBUTING.md` - Setup and contribution guide

### SDK
- `sdk/mollygraph_sdk/` - Thin Python SDK (`ingest/query/get_entity/run_audit/train_gliner`)

### Plans (Archived)
- `archive/old-plans/` - Old design documents

## Current State

- ✅ Bi-temporal graph (observed_at, valid_at)
- ✅ Zvec vector search (HNSW index)
- ✅ GLiNER2 extraction (local, no API cost)
- ✅ Optional spaCy enrichment fallback (`MOLLYGRAPH_SPACY_ENRICHMENT=true`)
- ✅ API authentication (Bearer tokens)
- ✅ Async job queue (SQLite WAL)
- ✅ LLM-backed audit with deterministic cleanup
- ✅ Suggestion digest + auto-adoption loop
- ✅ GLiNER training pipeline with benchmark gate

## Next Steps

1. Migration from old Molly data
2. iMessage integration
3. Voice skill (Pipecat)
