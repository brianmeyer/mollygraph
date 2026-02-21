# Contributing

## Setup
1. Create env and install deps:
```bash
./scripts/install.sh
```
2. Start Neo4j:
```bash
docker compose -f docker-compose.neo4j.yml up -d
```
3. Start service:
```bash
./scripts/start.sh
```

## Test
Use runtime venv (Python 3.12):
```bash
~/.graph-memory/venv/bin/pytest -q tests/test_basic.py
```

## API key
Protected endpoints require bearer auth.
Default local key: `dev-key-change-in-production`.

## Scope rules
- Keep graph writes deterministic.
- Keep audit fallback order configurable in `service/config.py`.
- Do not introduce cloud-only runtime dependencies for core ingestion.
