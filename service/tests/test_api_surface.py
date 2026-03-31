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


def test_default_app_exposes_operator_advisories_for_busy_queue(monkeypatch):
    class _BusyQueue:
        def get_pending_count(self):
            return 2

        def get_processing_count(self):
            return 1

        def get_stuck_count(self):
            return 1

        def get_dead_count(self):
            return 0

    class _HealthyTask:
        def done(self):
            return False

    monkeypatch.setattr(main, "get_queue_instance", lambda: _BusyQueue())
    monkeypatch.setattr(main, "_worker_task", _HealthyTask())
    monkeypatch.setattr(main.config, "QUEUE_MAX_CONCURRENT", 1)

    health = main.asyncio.run(main.health())

    assert health["operator_advisories"]
    assert any("starting another" in advisory.lower() for advisory in health["operator_advisories"])
