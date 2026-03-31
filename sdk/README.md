# mollygraph-sdk

Thin Python SDK and MCP adapter for MollyGraph.

The default product path is a local-first graph memory service:

- `Ladybug` graph
- `Ladybug` vector storage
- `GLiNER2` extraction
- `Snowflake/snowflake-arctic-embed-s` local embeddings

## Install

```bash
pip install "git+https://github.com/brianmeyer/mollygraph.git#subdirectory=sdk"
```

The package is currently documented for source installation. If you are working from a local clone, you can also install it from the repo with:

```bash
pip install -e sdk
```

## Python SDK quickstart

```python
from mollygraph_sdk import MollyGraphClient

client = MollyGraphClient(
    base_url="http://localhost:7422",
    api_key="dev-key-change-in-production",
)

print(client.health())
print(client.ingest("Alice works at Acme.", source="manual"))
print(client.query("What do we know about Alice?"))
print(client.get_entity("Alice"))
client.close()
```

## MCP adapter

Install the MCP extra:

```bash
pip install -e "sdk[mcp]"
```

Run the MCP server over stdio:

```bash
mollygraph-mcp --base-url http://localhost:7422 --api-key dev-key-change-in-production
```

Default MCP tools:

- `add_episode`
- `search_facts`
- `search_nodes`
- `get_entity_context`
- `get_queue_status`
- `delete_entity`
- `prune_entities`

Optional audit and training tools are only exposed when the running backend reports support for them.

## Advanced operations

Embedding-provider switching and reindexing are available, but they are operator tasks rather than the default onboarding path:

```python
client.get_embedding_config()
client.get_embedding_status()
client.add_embedding_model("sentence-transformers", "Snowflake/snowflake-arctic-embed-s")
client.set_embedding_provider("sentence-transformers", "Snowflake/snowflake-arctic-embed-s")
client.reindex_embeddings(limit=5000, dry_run=False)
```

The recommended path is the local core: ingest, query, inspect entity context, and keep the runtime healthy. Experimental admin surfaces remain available through the service only when you are explicitly working on those flows.
