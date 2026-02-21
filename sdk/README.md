# mollygraph-sdk

Python SDK and MCP adapter for MollyGraph.

## Install

```bash
pip install mollygraph-sdk
```

From source:

```bash
pip install "git+https://github.com/brianmeyer/mollygraph.git#subdirectory=sdk"
```

## Python SDK

```python
from mollygraph_sdk import MollyGraphClient

client = MollyGraphClient(
    base_url="http://localhost:7422",
    api_key="dev-key-change-in-production",
)

print(client.health())
print(client.query("What do we know about Brian?"))
client.close()
```

## MCP Adapter

Install MCP extra:

```bash
pip install "mollygraph-sdk[mcp]"
```

Run MCP server over stdio:

```bash
mollygraph-mcp --base-url http://localhost:7422 --api-key dev-key-change-in-production
```

The adapter exposes tools:
- `add_episode`
- `search_facts`
- `search_nodes`
- `get_entity_context`
- `get_queue_status`
- `run_audit`
- `get_training_status`
