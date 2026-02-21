#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_DIR="$ROOT_DIR/service"
STATE_DIR="${MOLLYGRAPH_HOME_DIR:-$HOME/.graph-memory}"
VENV_DIR="$STATE_DIR/venv"
LOG_FILE="$STATE_DIR/logs/service.log"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ROOT_DIR/.env"
  set +a
fi

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "Missing virtualenv at $VENV_DIR. Run scripts/install.sh first." >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG_FILE")"

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
cd "$SERVICE_DIR"

exec uvicorn main:app \
  --host "${GRAPH_MEMORY_HOST:-127.0.0.1}" \
  --port "${GRAPH_MEMORY_PORT:-7422}" \
  --workers 1 \
  --log-level info \
  >> "$LOG_FILE" 2>&1
