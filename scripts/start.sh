#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_DIR="$ROOT_DIR/service"
ENV_FILE="$SERVICE_DIR/.env"
LEGACY_ENV_FILE="$ROOT_DIR/.env"
ACTIVE_ENV_FILE="$ENV_FILE"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
elif [[ -f "$LEGACY_ENV_FILE" ]]; then
  ACTIVE_ENV_FILE="$LEGACY_ENV_FILE"
  echo "Using legacy env file at $LEGACY_ENV_FILE. Move your config to $ENV_FILE." >&2
  set -a
  # shellcheck disable=SC1090
  source "$LEGACY_ENV_FILE"
  set +a
else
  echo "Missing $ENV_FILE. Run scripts/install.sh first or copy service/.env.example to service/.env." >&2
  exit 1
fi

STATE_DIR="${MOLLYGRAPH_HOME_DIR:-$HOME/.graph-memory}"
CANONICAL_VENV_DIR="${MOLLYGRAPH_VENV_DIR:-$SERVICE_DIR/.venv}"
LEGACY_VENV_DIR="$STATE_DIR/venv"
VENV_DIR="$CANONICAL_VENV_DIR"
LOG_FILE="$STATE_DIR/logs/service.log"

if [[ ! -f "$VENV_DIR/bin/activate" && -f "$LEGACY_VENV_DIR/bin/activate" ]]; then
  echo "Using legacy virtualenv at $LEGACY_VENV_DIR. Re-run scripts/install.sh to migrate to $CANONICAL_VENV_DIR." >&2
  VENV_DIR="$LEGACY_VENV_DIR"
fi

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "Missing virtualenv at $VENV_DIR. Run scripts/install.sh first." >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG_FILE")"

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
cd "$SERVICE_DIR"

python - <<'PY'
import sys

try:
    import config

    if getattr(config, "GRAPH_BACKEND", "ladybug") == "ladybug":
        import real_ladybug  # noqa: F401

    if getattr(config, "VECTOR_BACKEND", "ladybug") == "ladybug":
        import real_ladybug  # noqa: F401
except Exception as exc:
    print(
        "Startup preflight failed: missing runtime dependency or invalid configuration.\n"
        f"Reason: {exc}\n"
        "Rerun ./scripts/install.sh to refresh the runtime venv before starting the service.",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY

echo "Starting MollyGraph from $SERVICE_DIR"
echo "Using venv: $VENV_DIR"
echo "Using env:  $ACTIVE_ENV_FILE"
echo "Logs:       $LOG_FILE"
echo "URL:        http://${GRAPH_MEMORY_HOST:-127.0.0.1}:${GRAPH_MEMORY_PORT:-7422}"

exec python -m uvicorn main:app \
  --host "${GRAPH_MEMORY_HOST:-127.0.0.1}" \
  --port "${GRAPH_MEMORY_PORT:-7422}" \
  --workers 1 \
  --log-level info \
  >> "$LOG_FILE" 2>&1
