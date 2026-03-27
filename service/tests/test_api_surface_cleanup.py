from __future__ import annotations

from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient

_HERE = Path(__file__).parent
_SERVICE_ROOT = _HERE.parent
if str(_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVICE_ROOT))

from api import admin, ingest
import main


def test_legacy_alias_routes_are_removed_from_http_surface():
    app = FastAPI()
    app.include_router(ingest.router)
    app.include_router(admin.router)
    client = TestClient(app)

    assert client.post("/extract").status_code == 404
    assert client.post("/audit/run").status_code == 404
    assert client.post("/maintenance/audit").status_code == 404
    assert client.get("/suggestions_digest").status_code == 404
    assert client.post("/training/gliner").status_code == 404
    assert client.get("/training/status").status_code == 404


def test_default_openapi_shows_core_surface_and_hides_experimental_routes():
    main.app.openapi_schema = None
    schema = main.app.openapi()
    paths = schema["paths"]

    assert "/ingest" in paths
    assert "/query" in paths
    assert "/entity/{name}" in paths
    assert "/stats" in paths

    assert "/decisions" not in paths
    assert "/audit" not in paths
    assert "/train/gliner" not in paths
    assert "/training/runs" not in paths
    assert "/maintenance/run" not in paths
    assert "/maintenance/nightly" not in paths
    assert "/extractors/config" not in paths
