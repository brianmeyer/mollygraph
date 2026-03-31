# Contributing

Use [docs/DOCS_MAP.md](/Users/brianmeyer/mollygraph/docs/DOCS_MAP.md) first so you know which markdown files are current and which are experimental plans.

## Setup

1. Install dependencies:

```bash
./scripts/install.sh
```

This creates `service/.venv`, pins it to Python `3.12`, and bootstraps `service/.env` from the example file if needed.
If `python3.12` is not already installed, the script can use `uv` to provision it.

2. Configure the service if you want to change defaults:

```bash
$EDITOR service/.env
```

3. Start the default local-first stack:

```bash
./scripts/start.sh
```

Neo4j is optional and only needed if you are explicitly working on legacy or experimental surfaces that still depend on the old backend.

## Test

Use the project service venv:

```bash
service/.venv/bin/python -m pytest -q
```

That root command runs both `tests/` and `service/tests/`.

Useful focused runs:

```bash
service/.venv/bin/python -m pytest service/tests/test_ladybug_core_flow.py -q
service/.venv/bin/python -m pytest service/tests/test_graph_quality.py -q
service/.venv/bin/python -m pytest service/tests/test_ladybug_graph.py service/tests/test_ladybug_vector_backend.py -q
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

- Keep the default runtime local-first.
- Keep the core product path small: ingest, query, entity context, cleanup, health, and stats.
- Treat audit, training, decision traces, and extra enrichment layers as optional unless the task is explicitly about them.
- Do not reintroduce cloud-only requirements into the default product path.
