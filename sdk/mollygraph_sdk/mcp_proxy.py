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

_PROBE_TIMEOUT = httpx.Timeout(3.0, connect=1.5)


def _probe_server_capabilities(base_url: str, api_key: str) -> set[str]:
    try:
        with httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=_PROBE_TIMEOUT,
        ) as client:
            response = client.get("/health")
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError):
        return set()

    raw_capabilities = payload.get("graph_capabilities", [])
    if not isinstance(raw_capabilities, list):
        return set()
    return {str(item).strip() for item in raw_capabilities if str(item).strip()}


def _build_server(base_url: str, api_key: str):
    if FastMCP is None:
        raise RuntimeError(
            "MCP dependency is not installed. Install with: pip install 'mollygraph-sdk[mcp]'"
        )

    detected_capabilities = _probe_server_capabilities(base_url, api_key)
    show_audit_tools = "audit" in detected_capabilities
    show_training_tools = "training" in detected_capabilities

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
    async def search_nodes(query: str, node_type: str = "", limit: int = 20) -> str:
        effective_limit = max(1, min(int(limit), 50))
        params: dict[str, object] = {"limit": effective_limit, "offset": 0}
        if node_type:
            params["type"] = node_type

        resp = await _client().get("/entities", params=params)
        _ensure_ok(resp)
        data = resp.json()
        entities: list[dict] = data.get("entities", [])
        query_lower = query.strip().lower()
        matches: list[str] = []
        for ent in entities:
            name = str(ent.get("name") or "").strip()
            if not name:
                continue
            if query_lower and query_lower not in name.lower():
                continue
            entity_type = str(ent.get("entity_type") or "").strip()
            matches.append(f"{name} [{entity_type}]" if entity_type else name)
            if len(matches) >= effective_limit:
                break

        if matches:
            return "\n".join(matches)
        return f"no entities matched {query!r}" + (f" (type={node_type!r})" if node_type else "")

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
    async def delete_entity(name: str) -> str:
        resp = await _client().request("DELETE", f"/entity/{name}")
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
        resp = await _client().post("/entities/prune", json={"names": names})
        _ensure_ok(resp)
        data = resp.json()
        return (
            f"pruned={data.get('pruned', 0)} "
            f"vectors_removed={data.get('vectors_removed', 0)} "
            f"entities={data.get('entities', [])}"
        )

    if show_audit_tools:
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

    if show_training_tools:
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
