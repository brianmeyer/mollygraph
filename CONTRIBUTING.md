# Contributing

## Setup

1. Create env and install deps:
```bash
./scripts/install.sh
```
2. Configure env:
```bash
cp .env.example .env
```
3. Start Neo4j:
```bash
docker compose -f docker-compose.neo4j.yml up -d
```
4. Start service:
```bash
./scripts/start.sh
```

## Test

Use the runtime venv (Python 3.12):

```bash
MOLLYGRAPH_TEST_MODE=1 ~/.graph-memory/venv/bin/pytest -q -m smoke
MOLLYGRAPH_TEST_MODE=1 ~/.graph-memory/venv/bin/pytest -q -m integration
```

Run startup script preflight checks:

```bash
bash -n scripts/install.sh scripts/start.sh
```

Validate SDK + MCP adapter packaging:

```bash
pip install -e "sdk[mcp]"
mollygraph-mcp --help
python -c "from mollygraph_sdk import MollyGraphClient; print('sdk ok')"
```

## API key

Protected endpoints require bearer auth.
Default local key: `dev-key-change-in-production`.

## Scope rules

- Keep graph writes deterministic.
- Keep audit fallback order configurable in `/Users/brianmeyer/mollygraph/service/config.py`.
- Do not introduce cloud-only runtime dependencies for core ingestion.
