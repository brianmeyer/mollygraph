from __future__ import annotations

from pathlib import Path
import sys

_HERE = Path(__file__).parent
_SERVICE_ROOT = _HERE.parent
_REPO_ROOT = _SERVICE_ROOT.parent
if str(_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVICE_ROOT))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import mcp_server as service_mcp
from sdk.mollygraph_sdk import mcp_proxy as sdk_mcp


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, path: str):
        assert path == "/health"
        return _FakeResponse(self._payload)


def test_service_mcp_probe_reads_runtime_capabilities(monkeypatch):
    monkeypatch.setattr(
        service_mcp.httpx,
        "Client",
        lambda *args, **kwargs: _FakeClient({"graph_capabilities": ["core_memory", "audit", "training"]}),
    )

    capabilities = service_mcp._probe_server_capabilities("http://example.test", "dev-key")

    assert capabilities == {"core_memory", "audit", "training"}


def test_service_mcp_probe_returns_empty_set_on_invalid_health_payload(monkeypatch):
    monkeypatch.setattr(
        service_mcp.httpx,
        "Client",
        lambda *args, **kwargs: _FakeClient({"graph_capabilities": "not-a-list"}),
    )

    capabilities = service_mcp._probe_server_capabilities("http://example.test", "dev-key")

    assert capabilities == set()


def test_sdk_mcp_probe_reads_runtime_capabilities(monkeypatch):
    monkeypatch.setattr(
        sdk_mcp.httpx,
        "Client",
        lambda *args, **kwargs: _FakeClient({"graph_capabilities": ["core_memory", "audit"]}),
    )

    capabilities = sdk_mcp._probe_server_capabilities("http://example.test", "dev-key")

    assert capabilities == {"core_memory", "audit"}
