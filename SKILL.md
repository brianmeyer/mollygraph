# MollyGraph Skill

**Name:** mollygraph  
**Description:** Bi-temporal knowledge graph with local-first entity extraction  
**Version:** 2.0.0

## Overview

MollyGraph is a local knowledge graph system using:
- **Neo4j** for graph storage (bi-temporal relationships)
- **Zvec** for vector similarity search (Alibaba, embedded)
- **GLiNER2** for entity/relationship extraction (local, no API cost)
- **SQLite** for async job queue

## Installation

```bash
cd /Users/brianmeyer/mollygraph
./scripts/install.sh   # runtime venv at ~/.graph-memory/venv (Python 3.12)
./scripts/start.sh     # HTTP API on port 7422
MOLLYGRAPH_SPACY_ENRICHMENT=true ~/.graph-memory/venv/bin/python service/mcp_server.py  # optional MCP on 7423
```

## MCP Integration

Add to OpenClaw config:

```json
{
  "mcpServers": {
    "mollygraph": {
      "command": "python3",
      "args": [
        "/Users/brianmeyer/mollygraph/service/mcp_server.py"
      ],
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_PASSWORD": "mollygraph"
      }
    }
  }
}
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `add_episode` | Queue text for entity extraction |
| `search_facts` | Search relationships in graph |
| `search_nodes` | Find entities by name |
| `get_entity_context` | Get entity with 2-hop context |
| `get_queue_status` | Check extraction queue |
| `run_audit` | Run nightly/weekly graph audit |
| `get_training_status` | Get GLiNER training status |

## HTTP API (Alternative)

If MCP is unavailable, use direct HTTP:

```bash
# Health
curl http://localhost:7422/health

# Ingest (auth required)
curl -X POST -H "Authorization: Bearer dev-key-change-in-production" \
  "http://localhost:7422/ingest?content=Text&source=manual&priority=0"

# Query (auth required)
curl -H "Authorization: Bearer dev-key-change-in-production" \
  "http://localhost:7422/entity/Brian%20Meyer"
```

## Authentication

Default API key: `dev-key-change-in-production`  
Set via env: `MOLLYGRAPH_API_KEY=your-key`

## Ports

- HTTP API: `7422`
- MCP SSE: `7423` (if running mcp_server.py separately)

## Data Storage

- Graph: Neo4j (Docker, port 7687)
- Vectors: Zvec (embedded, `~/.graph-memory/zvec_collection`)
- Queue: SQLite (`~/.graph-memory/extraction_queue.db`)

## Architecture

```
User Input → Queue (SQLite) → Worker → GLiNER2 Extraction → Neo4j + Zvec
                ↑                                              ↓
         Async Processing                              Query/Retrieval
```
