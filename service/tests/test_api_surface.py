from __future__ import annotations

from pathlib import Path
import sys

from fastapi import FastAPI

_HERE = Path(__file__).parent
_SERVICE_ROOT = _HERE.parent
if str(_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVICE_ROOT))

from api import admin, ingest
import main


def test_legacy_alias_routes_removed_from_core_routers():
    app = FastAPI()
    app.include_router(ingest.router)
    app.include_router(admin.router)
    paths = {route.path for route in app.routes}

    assert "/extract" not in paths
    assert "/audit/run" not in paths
    assert "/maintenance/audit" not in paths
    assert "/suggestions_digest" not in paths
    assert "/training/gliner" not in paths
    assert "/training/status" not in paths


def test_default_app_surface_excludes_decisions_routes():
    paths = {route.path for route in main.app.routes}

    assert "/decisions" not in paths
    assert "/decisions/{decision_id}" not in paths


def test_default_app_reports_graph_capabilities():
    health = main.asyncio.run(main.health())

    assert health["graph_backend"] == "ladybug"
    assert "core_memory" in health["graph_capabilities"]
    assert "decisions" not in health["graph_capabilities"]
