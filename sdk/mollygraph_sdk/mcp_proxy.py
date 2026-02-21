"""MollyGraph MCP adapter over MollyGraph HTTP API."""
from __future__ import annotations

import argparse
import os
from contextlib import asynccontextmanager

import httpx

try:
    from mcp.server.fastmcp import FastMCP
except Exception:  # pragma: no cover - optional dependency path
    FastMCP = None


def _build_server(base_url: str, api_key: str):
    if FastMCP is None:
        raise RuntimeError(
            "MCP dependency is not installed. Install with: pip install 'mollygraph-sdk[mcp]'"
        )

    http_client: httpx.AsyncClient | None = None

    @asynccontextmanager
    async def app_lifespan(_server: FastMCP):
        nonlocal http_client
        http_client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=45.0,
        )
        try:
            yield
        finally:
            await http_client.aclose()

    mcp = FastMCP("mollygraph", lifespan=app_lifespan)

    def _client() -> httpx.AsyncClient:
        if http_client is None:
            raise RuntimeError("MollyGraph MCP HTTP client is not initialized")
        return http_client

    def _ensure_ok(resp: httpx.Response) -> None:
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:400]}")

    @mcp.tool()
    async def add_episode(content: str, source: str = "mcp", priority: int = 1) -> str:
        resp = await _client().post(
            "/ingest",
            params={"content": content, "source": source, "priority": priority},
        )
        _ensure_ok(resp)
        data = resp.json()
        return f"queued {data.get('job_id', '')[:8]} (depth={data.get('queue_depth', 0)})"

    @mcp.tool()
    async def search_facts(query: str) -> str:
        resp = await _client().get("/query", params={"q": query})
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
        resp = await _client().get("/query", params={"q": query})
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
        resp = await _client().get(f"/entity/{name}")
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
        resp = await _client().get("/stats")
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
    async def run_audit(limit: int = 500, dry_run: bool = False, schedule: str = "nightly") -> str:
        resp = await _client().post(
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
        resp = await _client().get("/train/status")
        _ensure_ok(resp)
        data = resp.json().get("gliner", {})
        return (
            f"examples={data.get('examples_accumulated', 0)} "
            f"last={data.get('last_finetune_at', '') or data.get('last_finetune', '')} "
            f"status={data.get('last_cycle_status', '')}"
        )

    return mcp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MollyGraph MCP adapter")
    parser.add_argument(
        "--base-url",
        default=os.getenv("MOLLYGRAPH_URL", "http://localhost:7422"),
        help="MollyGraph HTTP API base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("MOLLYGRAPH_API_KEY", "dev-key-change-in-production"),
        help="MollyGraph bearer token.",
    )
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=("stdio", "sse", "streamable-http"),
        help="MCP transport mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    server = _build_server(base_url=args.base_url, api_key=args.api_key)
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
