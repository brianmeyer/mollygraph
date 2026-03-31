from __future__ import annotations

import json
from pathlib import Path
import os
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "production_smoke.py"

pytestmark = [pytest.mark.integration, pytest.mark.runtime]


def _last_nonempty_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        if line.strip():
            return line
    return ""


def test_runtime_smoke_runner() -> None:
    if os.environ.get("MOLLYGRAPH_RUN_RUNTIME_SMOKE") != "1":
        pytest.skip("Set MOLLYGRAPH_RUN_RUNTIME_SMOKE=1 to run the real runtime smoke test.")

    result = subprocess.run(
        [sys.executable, str(SMOKE_SCRIPT), "--json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    if result.returncode != 0:
        raise AssertionError(
            "production smoke runner failed\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    payload = json.loads(_last_nonempty_line(result.stdout))
    assert payload["status"] == "ok"
    assert payload["startup_health"]["status"] == "healthy"
    assert payload["startup_health"]["operator_advisories"] == []
    assert payload["ingest"]["status"] == 200
    assert payload["query"]["status"] == 200
