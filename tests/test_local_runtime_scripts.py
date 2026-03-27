from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_install_script_uses_service_venv_and_service_env() -> None:
    content = _read("scripts/install.sh")

    assert 'VENV_DIR="${MOLLYGRAPH_VENV_DIR:-$SERVICE_DIR/.venv}"' in content
    assert 'ENV_FILE="$SERVICE_DIR/.env"' in content
    assert 'cp "$ENV_EXAMPLE_FILE" "$ENV_FILE"' in content


def test_start_script_prefers_service_env_and_service_venv() -> None:
    content = _read("scripts/start.sh")

    assert 'CANONICAL_VENV_DIR="${MOLLYGRAPH_VENV_DIR:-$SERVICE_DIR/.venv}"' in content
    assert 'ENV_FILE="$SERVICE_DIR/.env"' in content
    assert 'LEGACY_ENV_FILE="$ROOT_DIR/.env"' in content
    assert 'Using legacy env file at $LEGACY_ENV_FILE' in content


def test_env_example_keeps_neo4j_commented_by_default() -> None:
    content = _read("service/.env.example")

    assert "# NEO4J_URI=bolt://localhost:7687" in content
    assert "# NEO4J_USER=neo4j" in content
    assert "# NEO4J_PASSWORD=mollygraph" in content
