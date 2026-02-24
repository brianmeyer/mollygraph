"""MollyGraph MCP server (HTTP proxy)."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

import httpx
from mcp.server.fastmcp import FastMCP

MOLLYGRAPH_URL = os.getenv("MOLLYGRAPH_URL", "http://localhost:7422")
API_KEY = os.getenv("MOLLYGRAPH_API_KEY", "dev-key-change-in-production")

http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def app_lifespan(_server: FastMCP):
    global http_client
    http_client = httpx.AsyncClient(
        base_url=MOLLYGRAPH_URL,
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=45.0,
    )
    yield
    await http_client.aclose()


mcp = FastMCP("mollygraph", lifespan=app_lifespan)


def _ensure_ok(resp: httpx.Response) -> None:
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:400]}")


@mcp.tool()
async def add_episode(content: str, source: str = "mcp", priority: int = 1) -> str:
    resp = await http_client.post(
        "/ingest",
        params={"content": content, "source": source, "priority": priority},
    )
    _ensure_ok(resp)
    data = resp.json()
    return f"queued {data.get('job_id', '')[:8]} (depth={data.get('queue_depth', 0)})"


@mcp.tool()
async def search_facts(query: str) -> str:
    resp = await http_client.get("/query", params={"q": query})
    _ensure_ok(resp)
    data = resp.json()
    results = data.get("results", [])
    if not results:
        return f"no results for {query!r}"

    lines = [f"results for {query!r}:"]
    for item in results[:5]:
        entity = item.get("entity", "unknown")
        lines.append(f"- {entity}")
        for fact in item.get("facts", [])[:4]:
            rel = fact.get("rel_type", "RELATED_TO")
            target = fact.get("target_name", "?")
            lines.append(f"  {rel} -> {target}")
    return "\n".join(lines)


@mcp.tool()
async def search_nodes(query: str, limit: int = 20) -> str:
    # Backward compatible alias over /query response entities.
    resp = await http_client.get("/query", params={"q": query})
    _ensure_ok(resp)
    data = resp.json()

    names: list[str] = []
    for item in data.get("results", []):
        name = str(item.get("entity") or "").strip()
        if name:
            names.append(name)

    if not names:
        return f"no entities matched {query!r}"
    return "\n".join(names[: max(1, limit)])


@mcp.tool()
async def get_entity_context(name: str) -> str:
    resp = await http_client.get(f"/entity/{name}")
    if resp.status_code == 404:
        return f"entity {name!r} not found"
    _ensure_ok(resp)

    data = resp.json()
    facts = data.get("facts", [])
    context = data.get("context", {})
    lines = [f"entity: {name}", "facts:"]
    for fact in facts[:10]:
        lines.append(f"- {fact.get('rel_type', 'RELATED_TO')} -> {fact.get('target_name', '?')}")

    direct = context.get("direct_connections", []) if isinstance(context, dict) else []
    if direct:
        lines.append("direct:")
        for row in direct[:8]:
            lines.append(f"- {row.get('rel_type', 'RELATED_TO')} {row.get('target_name', '?')}")

    return "\n".join(lines)


@mcp.tool()
async def get_queue_status() -> str:
    resp = await http_client.get("/stats")
    _ensure_ok(resp)
    data = resp.json()
    queue = data.get("queue", {})
    vector = data.get("vector_store", {})
    return (
        f"pending={queue.get('pending', 0)} "
        f"processing={queue.get('processing', 0)} "
        f"vector={vector}"
    )


@mcp.tool()
async def delete_entity(name: str) -> str:
    """Delete an entity by name from Neo4j and the vector store.

    All relationships attached to the entity are also removed (DETACH DELETE).
    """
    resp = await http_client.request("DELETE", f"/entity/{name}")
    if resp.status_code == 404:
        return f"entity {name!r} not found"
    _ensure_ok(resp)
    data = resp.json()
    return (
        f"deleted entity={data.get('entity')} "
        f"relationships_removed={data.get('relationships_removed', 0)} "
        f"vector_removed={data.get('vector_removed', False)}"
    )


@mcp.tool()
async def prune_entities(names: list[str]) -> str:
    """Bulk delete entities by name from Neo4j and the vector store.

    Pass a list of entity names; each is removed along with all its relationships.
    """
    resp = await http_client.post("/entities/prune", json={"names": names})
    _ensure_ok(resp)
    data = resp.json()
    pruned = data.get("pruned", 0)
    entities = data.get("entities", [])
    vectors_removed = data.get("vectors_removed", 0)
    return (
        f"pruned={pruned} vectors_removed={vectors_removed} entities={entities}"
    )


@mcp.tool()
async def run_audit(limit: int = 500, dry_run: bool = False, schedule: str = "nightly") -> str:
    resp = await http_client.post(
        "/audit",
        json={
            "limit": max(1, int(limit)),
            "dry_run": bool(dry_run),
            "schedule": schedule,
        },
    )
    _ensure_ok(resp)
    data = resp.json()
    return (
        f"audit status={data.get('status', 'ok')} "
        f"scanned={data.get('relationships_scanned', 0)} "
        f"verified={data.get('verified', 0)} "
        f"autofixed={data.get('auto_fixed', 0)}"
    )


@mcp.tool()
async def get_training_status() -> str:
    resp = await http_client.get("/train/status")
    _ensure_ok(resp)
    data = resp.json().get("gliner", {})
    return (
        f"examples={data.get('examples_accumulated', 0)} "
        f"last={data.get('last_finetune_at', '') or data.get('last_finetune', '')} "
        f"status={data.get('last_cycle_status', '')}"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
