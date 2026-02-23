"""Thin Python SDK for MollyGraph HTTP API."""
from __future__ import annotations

from typing import Any

import httpx


class MollyGraphClient:
    def __init__(self, base_url: str = "http://localhost:7422", api_key: str = "dev-key-change-in-production", timeout: float = 30.0):
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def close(self) -> None:
        self._client.close()

    def health(self) -> dict[str, Any]:
        return self._client.get("/health").json()

    def ingest(self, content: str, source: str = "manual", priority: int = 1) -> dict[str, Any]:
        resp = self._client.post(
            "/ingest",
            params={"content": content, "source": source, "priority": priority},
        )
        resp.raise_for_status()
        return resp.json()

    def query(self, q: str) -> dict[str, Any]:
        resp = self._client.get("/query", params={"q": q})
        resp.raise_for_status()
        return resp.json()

    def get_entity(self, name: str) -> dict[str, Any]:
        resp = self._client.get(f"/entity/{name}")
        resp.raise_for_status()
        return resp.json()

    def run_audit(self, limit: int = 500, dry_run: bool = False, schedule: str = "nightly") -> dict[str, Any]:
        resp = self._client.post(
            "/audit",
            json={"limit": limit, "dry_run": dry_run, "schedule": schedule},
        )
        resp.raise_for_status()
        return resp.json()

    def train_gliner(self, force: bool = False) -> dict[str, Any]:
        resp = self._client.post("/train/gliner", json={"force": force})
        resp.raise_for_status()
        return resp.json()

    def get_embedding_config(self) -> dict[str, Any]:
        resp = self._client.get("/embeddings/config")
        resp.raise_for_status()
        return resp.json()

    def get_embedding_status(self) -> dict[str, Any]:
        resp = self._client.get("/embeddings/status")
        resp.raise_for_status()
        return resp.json()

    def set_embedding_provider(self, provider: str, model: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"provider": provider}
        if model is not None:
            payload["model"] = model
        resp = self._client.post("/embeddings/config", json=payload)
        resp.raise_for_status()
        return resp.json()

    def add_embedding_model(self, provider: str, model: str, activate: bool = False) -> dict[str, Any]:
        resp = self._client.post(
            "/embeddings/models",
            json={"provider": provider, "model": model, "activate": activate},
        )
        resp.raise_for_status()
        return resp.json()

    def reindex_embeddings(self, limit: int = 5000, dry_run: bool = False) -> dict[str, Any]:
        resp = self._client.post("/embeddings/reindex", json={"limit": limit, "dry_run": dry_run})
        resp.raise_for_status()
        return resp.json()
