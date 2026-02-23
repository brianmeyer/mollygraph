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
print(client.get_embedding_config())
print(client.get_embedding_status())
client.close()
```

Switch embedding providers/models:

```python
client.add_embedding_model("huggingface", "BAAI/bge-small-en-v1.5")
client.set_embedding_provider("huggingface", "BAAI/bge-small-en-v1.5")
client.add_embedding_model("ollama", "nomic-embed-text", activate=True)
client.reindex_embeddings(limit=5000, dry_run=False)
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
