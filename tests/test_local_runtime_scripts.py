from __future__ import annotations

from pathlib import Path
import shlex
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_install_script_uses_service_venv_and_service_env() -> None:
    content = _read("scripts/install.sh")

    assert 'VENV_DIR="${MOLLYGRAPH_VENV_DIR:-$SERVICE_DIR/.venv}"' in content
    assert 'ENV_FILE="$SERVICE_DIR/.env"' in content
    assert 'TARGET_PYTHON="3.12"' in content
    assert 'cp "$ENV_EXAMPLE_FILE" "$ENV_FILE"' in content
    assert 'source "$ROOT_DIR/scripts/lib/env.sh"' in content
    assert 'load_env_file "$ENV_FILE"' in content
    assert 'uv python install "$TARGET_PYTHON"' in content
    assert 'uv venv --python "$TARGET_PYTHON" --seed --clear "$VENV_DIR"' in content
    assert 'python3.12 -m venv "$VENV_DIR"' in content


def test_repo_pins_python_312() -> None:
    content = _read(".python-version")

    assert content.strip() == "3.12"


def test_start_script_prefers_service_env_and_service_venv() -> None:
    content = _read("scripts/start.sh")

    assert 'CANONICAL_VENV_DIR="${MOLLYGRAPH_VENV_DIR:-$SERVICE_DIR/.venv}"' in content
    assert 'TARGET_PYTHON="3.12"' in content
    assert 'ENV_FILE="$SERVICE_DIR/.env"' in content
    assert 'LEGACY_ENV_FILE="$ROOT_DIR/.env"' in content
    assert 'Using legacy env file at $LEGACY_ENV_FILE' in content
    assert 'source "$ROOT_DIR/scripts/lib/env.sh"' in content
    assert 'load_env_file "$ENV_FILE"' in content
    assert "service runtime requires Python $TARGET_PYTHON" in content
    assert "Another MollyGraph process is already using this local runtime" in content
    assert "MOLLYGRAPH_QUEUE_MAX_CONCURRENT=1" in content
    assert "warn_if_concurrent_local_runs()" in content
    assert "MOLLYGRAPH_ALLOW_CONCURRENT_LOCAL_RUNS" in content
    assert 'ps -Ao pid=,command=' in content
    assert "another local memory/model-heavy Python workload appears to be running" in content


def test_env_example_keeps_neo4j_commented_by_default() -> None:
    content = _read("service/.env.example")

    assert "# NEO4J_URI=bolt://localhost:7687" in content
    assert "# NEO4J_USER=neo4j" in content
    assert "# NEO4J_PASSWORD=mollygraph" in content


def test_install_script_calls_out_single_run_safety() -> None:
    content = _read("scripts/install.sh")

    assert "MOLLYGRAPH_QUEUE_MAX_CONCURRENT=1" in content
    assert "scripts/start.sh will warn and stop before launching a second copy" in content


def test_env_loader_handles_unquoted_values_with_spaces(tmp_path: Path) -> None:
    env_file = tmp_path / "sample.env"
    env_file.write_text(
        "\n".join(
            [
                "# comment",
                "MOLLYGRAPH_OWNER_NAME=Brian Meyer",
                'MOLLYGRAPH_SCHOOL_CAMPUS_ALIASES={"Miami": ["miami campus"]}',
                "",
            ]
        ),
        encoding="utf-8",
    )

    helper = REPO_ROOT / "scripts/lib/env.sh"
    command = (
        f"source {shlex.quote(str(helper))}; "
        f"load_env_file {shlex.quote(str(env_file))}; "
        'printf "%s\\n%s\\n" "$MOLLYGRAPH_OWNER_NAME" "$MOLLYGRAPH_SCHOOL_CAMPUS_ALIASES"'
    )
    result = subprocess.run(
        ["bash", "-lc", command],
        check=True,
        capture_output=True,
        text=True,
    )

    lines = result.stdout.strip().splitlines()
    assert lines == [
        "Brian Meyer",
        '{"Miami": ["miami campus"]}',
    ]


def test_env_loader_strips_matching_outer_quotes(tmp_path: Path) -> None:
    env_file = tmp_path / "quoted.env"
    env_file.write_text(
        "\n".join(
            [
                'MOLLYGRAPH_OWNER_NAME="Brian Meyer"',
                "MOLLYGRAPH_CITY='Chicago'",
                "",
            ]
        ),
        encoding="utf-8",
    )

    helper = REPO_ROOT / "scripts/lib/env.sh"
    command = (
        f"source {shlex.quote(str(helper))}; "
        f"load_env_file {shlex.quote(str(env_file))}; "
        'printf "%s\\n%s\\n" "$MOLLYGRAPH_OWNER_NAME" "$MOLLYGRAPH_CITY"'
    )
    result = subprocess.run(
        ["bash", "-lc", command],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip().splitlines() == [
        "Brian Meyer",
        "Chicago",
    ]
