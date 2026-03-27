from __future__ import annotations

from pathlib import Path
import sys

_HERE = Path(__file__).parent
_SERVICE_ROOT = _HERE.parent
if str(_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVICE_ROOT))

import config
from memory.graph import LadybugGraph
import runtime_graph


def test_require_graph_instance_selects_ladybug_backend(monkeypatch, tmp_path):
    runtime_graph.set_graph_instance(None)
    monkeypatch.setattr(config, "GRAPH_BACKEND", "ladybug")
    monkeypatch.setattr(config, "LADYBUG_GRAPH_DB_PATH", tmp_path / "graph.lbug")

    graph = runtime_graph.require_graph_instance()

    assert isinstance(graph, LadybugGraph)
    assert runtime_graph.get_graph_backend_name(graph) == "ladybug"
    assert "core_memory" in runtime_graph.get_graph_capabilities(graph)
    assert "decisions" not in runtime_graph.get_graph_capabilities(graph)

    runtime_graph.set_graph_instance(None)


def test_graph_capability_infers_decision_support_from_object_methods():
    class _DecisionGraph:
        def create_decision(self):
            return {}

        def list_decisions(self):
            return []

        def get_decision(self):
            return None

    graph = _DecisionGraph()
    assert runtime_graph.graph_supports_capability("decisions", graph) is True
