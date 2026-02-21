# mollygraph-sdk

```python
from mollygraph_sdk import MollyGraphClient

client = MollyGraphClient(base_url="http://localhost:7422", api_key="dev-key-change-in-production")
print(client.health())
print(client.query("What do we know about Brian?"))
client.close()
```
