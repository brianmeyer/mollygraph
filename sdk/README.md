# mollygraph-sdk

Thin Python SDK and MCP adapter for MollyGraph.

The default product path is a local-first graph memory service:

- `Ladybug` graph
- `Ladybug` vector storage
- `GLiNER2` extraction
- `Snowflake/snowflake-arctic-embed-s` local embeddings

## Install

```bash
pip install mollygraph-sdk
```

From source:

```bash
pip install "git+https://github.com/brianmeyer/mollygraph.git#subdirectory=sdk"
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
pip install "mollygraph-sdk[mcp]"
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

Experimental MCP tools like audit and training are only exposed when the runtime is configured for the older experimental surface.

## Advanced operations

Embedding-provider switching and reindexing are available, but they are advanced operations rather than the default onboarding path:

```python
client.get_embedding_config()
client.get_embedding_status()
client.add_embedding_model("sentence-transformers", "Snowflake/snowflake-arctic-embed-s")
client.set_embedding_provider("sentence-transformers", "Snowflake/snowflake-arctic-embed-s")
client.reindex_embeddings(limit=5000, dry_run=False)
```

The recommended path is still the local core: ingest, query, inspect entity context, and keep the runtime healthy. Experimental admin surfaces remain available through the service when you are explicitly working on those flows.
