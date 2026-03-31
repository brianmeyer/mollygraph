#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_DIR="$ROOT_DIR/service"
ENV_FILE="$SERVICE_DIR/.env"
LEGACY_ENV_FILE="$ROOT_DIR/.env"
ACTIVE_ENV_FILE="$ENV_FILE"

# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/lib/env.sh"

if [[ -f "$ENV_FILE" ]]; then
  load_env_file "$ENV_FILE"
elif [[ -f "$LEGACY_ENV_FILE" ]]; then
  ACTIVE_ENV_FILE="$LEGACY_ENV_FILE"
  echo "Using legacy env file at $LEGACY_ENV_FILE. Move your config to $ENV_FILE." >&2
  load_env_file "$LEGACY_ENV_FILE"
else
  echo "Missing $ENV_FILE. Run scripts/install.sh first or copy service/.env.example to service/.env." >&2
  exit 1
fi

STATE_DIR="${MOLLYGRAPH_HOME_DIR:-$HOME/.graph-memory}"
CANONICAL_VENV_DIR="${MOLLYGRAPH_VENV_DIR:-$SERVICE_DIR/.venv}"
LEGACY_VENV_DIR="$STATE_DIR/venv"
VENV_DIR="$CANONICAL_VENV_DIR"
LOG_FILE="$STATE_DIR/logs/service.log"
TARGET_PYTHON="3.12"

if [[ ! -f "$VENV_DIR/bin/activate" && -f "$LEGACY_VENV_DIR/bin/activate" ]]; then
  echo "Using legacy virtualenv at $LEGACY_VENV_DIR. Re-run scripts/install.sh to migrate to $CANONICAL_VENV_DIR." >&2
  VENV_DIR="$LEGACY_VENV_DIR"
fi

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "Missing virtualenv at $VENV_DIR. Run scripts/install.sh first." >&2
  exit 1
fi

RUNTIME_PYTHON_VERSION="$("$VENV_DIR/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$RUNTIME_PYTHON_VERSION" != "$TARGET_PYTHON" ]]; then
  echo "MollyGraph's service runtime requires Python $TARGET_PYTHON." >&2
  echo "Found Python $RUNTIME_PYTHON_VERSION in $VENV_DIR. Rerun scripts/install.sh to rebuild the runtime." >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG_FILE")"

warn_if_concurrent_local_runs() {
  local allow="${MOLLYGRAPH_ALLOW_CONCURRENT_LOCAL_RUNS:-}"
  local current_pid="$$"
  local matches=()

  case "${allow,,}" in
    1|true|yes|on)
      return 0
      ;;
  esac

  while IFS= read -r line; do
    [[ -n "$line" ]] || continue

    local pid="${line%% *}"
    local cmd="${line#* }"
    local lower_cmd="${cmd,,}"

    [[ "$pid" == "$current_pid" ]] && continue

    if [[ "$lower_cmd" == *"uvicorn main:app"* ]] || \
       ([[ "$lower_cmd" == *"mollygraph"* || "$lower_cmd" == *"recallforge"* ]] && \
        [[ "$lower_cmd" == *"python"* || "$lower_cmd" == *"pytest"* ]]); then
      matches+=("  pid=$pid  $cmd")
    fi
  done < <(ps -Ao pid=,command=)

  if [[ ${#matches[@]} -eq 0 ]]; then
    return 0
  fi

  echo "Warning: another local memory/model-heavy Python workload appears to be running." >&2
  echo "Starting an additional MollyGraph process can spike CPU, RAM, and model warm-up pressure on laptops." >&2
  printf '%s\n' "${matches[@]}" >&2
  echo "Starting anyway. Set MOLLYGRAPH_ALLOW_CONCURRENT_LOCAL_RUNS=1 to suppress this warning." >&2
}

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
cd "$SERVICE_DIR"

python - <<'PY'
import os
import sqlite3
import sys
from pathlib import Path

try:
    import config

    if getattr(config, "GRAPH_BACKEND", "ladybug") == "ladybug":
        import real_ladybug  # noqa: F401

    if getattr(config, "VECTOR_BACKEND", "ladybug") == "ladybug":
        import real_ladybug  # noqa: F401

    state_dir = Path(os.environ.get("MOLLYGRAPH_HOME_DIR", str(Path.home() / ".graph-memory"))).expanduser()
    pid_file = state_dir / "mollygraph.pid"
    queue_db = Path(os.environ.get("MOLLYGRAPH_QUEUE_DB", str(state_dir / "extraction_queue.db"))).expanduser()
    queue_max_concurrent = max(1, int(os.environ.get("MOLLYGRAPH_QUEUE_MAX_CONCURRENT", "1")))

    if pid_file.exists():
        try:
            saved_pid = int(pid_file.read_text().strip())
            os.kill(saved_pid, 0)
        except (ValueError, OSError, ProcessLookupError):
            pass
        else:
            print(
                "Another MollyGraph process is already using this local runtime "
                f"(pid {saved_pid}). Starting a second copy on the same data "
                "directory can double model load and queue pressure.",
                file=sys.stderr,
            )
            raise SystemExit(1)

    advisories: list[str] = []
    if queue_max_concurrent > 1:
        advisories.append(
            f"Queue concurrency is set to {queue_max_concurrent}. Keep "
            "MOLLYGRAPH_QUEUE_MAX_CONCURRENT=1 for safer laptop runs."
        )

    if queue_db.exists():
        try:
            with sqlite3.connect(queue_db) as conn:
                pending, processing = conn.execute(
                    """
                    SELECT
                        COALESCE(SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END), 0) AS pending,
                        COALESCE(SUM(CASE WHEN status = 'processing' THEN 1 ELSE 0 END), 0) AS processing
                    FROM jobs
                    WHERE status IN ('pending', 'processing')
                    """
                ).fetchone()
            if pending or processing:
                advisories.append(
                    f"The queue already has {pending} pending and {processing} "
                    "processing job(s). Starting another local model-heavy run "
                    "will compound CPU/RAM pressure."
                )
        except Exception:
            pass

    for advisory in advisories:
        print(f"Operator advisory: {advisory}", file=sys.stderr)
except Exception as exc:
    print(
        "Startup preflight failed: missing runtime dependency or invalid configuration.\n"
        f"Reason: {exc}\n"
        "Rerun ./scripts/install.sh to refresh the runtime venv before starting the service.",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY

warn_if_concurrent_local_runs

echo "Starting MollyGraph from $SERVICE_DIR"
echo "Using venv: $VENV_DIR"
echo "Using python: $RUNTIME_PYTHON_VERSION"
echo "Using env:  $ACTIVE_ENV_FILE"
echo "Logs:       $LOG_FILE"
echo "URL:        http://${GRAPH_MEMORY_HOST:-127.0.0.1}:${GRAPH_MEMORY_PORT:-7422}"

exec python -m uvicorn main:app \
  --host "${GRAPH_MEMORY_HOST:-127.0.0.1}" \
  --port "${GRAPH_MEMORY_PORT:-7422}" \
  --workers 1 \
  --log-level info \
  >> "$LOG_FILE" 2>&1
