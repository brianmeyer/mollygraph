# MollyGraph v1 Architecture

## Runtime surfaces
- HTTP service: `service/main.py` on port `7422`
- MCP proxy: `service/mcp_server.py` via stdio transport

## Data planes
- Graph: Neo4j (`service/memory/bitemporal_graph.py` for ingest/query and `service/memory/graph.py` for audit/training ops)
- Vector: Zvec preferred with sqlite-vec fallback (`service/memory/vector_store.py`)
- Queue: SQLite WAL (`service/extraction/queue.py`)
- State: `~/.graph-memory/state.json` (override root via `MOLLYGRAPH_HOME_DIR`)

## Processing flow
1. `POST /ingest` enqueues an `ExtractionJob`.
2. Queue worker runs `ExtractionPipeline`.
3. GLiNER2 extracts entities/relationships.
4. Entities and relationships are written to Neo4j with temporal metadata.
5. Entities are indexed in vector storage for retrieval.
6. Suggestions are logged when relation labels fall back.

## Nightly maintenance flow
1. Deterministic cleanup: decay + orphan/self-ref cleanup.
2. LLM-backed audit with provider fallback chain.
3. Suggestion digest and auto-adoption update.
4. GLiNER accumulation and optional benchmark-gated deploy.

## Audit provider fallback
Default order is env-configurable via `AUDIT_PROVIDER_ORDER`.
Default chain:
- Gemini Flash Lite
- Kimi 2.5
- Groq gpt-oss-120b
