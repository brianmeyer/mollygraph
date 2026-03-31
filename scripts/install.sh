#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_DIR="$ROOT_DIR/service"
ENV_FILE="$SERVICE_DIR/.env"
ENV_EXAMPLE_FILE="$SERVICE_DIR/.env.example"
ACTIVE_ENV_FILE="$ENV_FILE"
TARGET_PYTHON="3.12"

# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/lib/env.sh"

if [[ ! -f "$ENV_FILE" && -f "$ENV_EXAMPLE_FILE" ]]; then
  cp "$ENV_EXAMPLE_FILE" "$ENV_FILE"
  echo "Created $ENV_FILE from $ENV_EXAMPLE_FILE"
fi

if [[ -f "$ENV_FILE" ]]; then
  load_env_file "$ENV_FILE"
elif [[ -f "$ROOT_DIR/.env" ]]; then
  ACTIVE_ENV_FILE="$ROOT_DIR/.env"
  echo "Using legacy env file at $ACTIVE_ENV_FILE. Move your config to $ENV_FILE." >&2
  load_env_file "$ROOT_DIR/.env"
fi

STATE_DIR="${MOLLYGRAPH_HOME_DIR:-$HOME/.graph-memory}"
VENV_DIR="${MOLLYGRAPH_VENV_DIR:-$SERVICE_DIR/.venv}"

mkdir -p "$STATE_DIR/models" "$STATE_DIR/training/gliner" "$STATE_DIR/logs" "$STATE_DIR/logs/maintenance" "$STATE_DIR/suggestions"

recreate_venv=0
if [[ -x "$VENV_DIR/bin/python" ]]; then
  EXISTING_PY="$("$VENV_DIR/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "$EXISTING_PY" != "$TARGET_PYTHON" ]]; then
    echo "Existing venv uses Python $EXISTING_PY at $VENV_DIR." >&2
    echo "Rebuilding the local runtime so MollyGraph uses Python $TARGET_PYTHON." >&2
    recreate_venv=1
  fi
else
  recreate_venv=1
fi

if [[ "$recreate_venv" -eq 1 ]]; then
  if command -v python3.12 >/dev/null 2>&1; then
    rm -rf "$VENV_DIR"
    python3.12 -m venv "$VENV_DIR"
  elif command -v uv >/dev/null 2>&1; then
    uv python install "$TARGET_PYTHON"
    uv venv --python "$TARGET_PYTHON" --seed --clear "$VENV_DIR"
  else
    echo "MollyGraph requires Python $TARGET_PYTHON." >&2
    echo "Install python3.12 or install uv so the runtime can provision Python $TARGET_PYTHON automatically." >&2
    exit 1
  fi
fi

PYTHON_BIN="$VENV_DIR/bin/python"
RUNTIME_PY="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$RUNTIME_PY" != "$TARGET_PYTHON" ]]; then
  echo "Expected Python $TARGET_PYTHON at $VENV_DIR, found $RUNTIME_PY." >&2
  echo "Remove $VENV_DIR and rerun scripts/install.sh." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "$ROOT_DIR/requirements.txt"

python - <<'PY'
try:
    from gliner2 import GLiNER2
    GLiNER2.from_pretrained("fastino/gliner2-large-v1")
    print("Downloaded base GLiNER2 model")
except Exception as exc:
    print(f"Skipped GLiNER2 pre-download: {exc}")
PY

python - <<'PY'
try:
    from sentence_transformers import SentenceTransformer
    SentenceTransformer(
        "Snowflake/snowflake-arctic-embed-s",
        trust_remote_code=True,
    )
    print("Downloaded default local embedding model")
except Exception as exc:
    print(f"Skipped embedding model pre-download: {exc}")
PY

echo "Install complete."
echo "Canonical local runtime:"
echo "  venv: $VENV_DIR"
echo "  env:  $ACTIVE_ENV_FILE"
echo "  data: ${MOLLYGRAPH_HOME_DIR:-$HOME/.graph-memory}"
echo "  python: $TARGET_PYTHON"
echo "Default local stack:"
echo "  MOLLYGRAPH_GRAPH_BACKEND=ladybug"
echo "  MOLLYGRAPH_VECTOR_BACKEND=ladybug"
echo "  MOLLYGRAPH_EMBEDDING_ST_MODEL=Snowflake/snowflake-arctic-embed-s"
echo "  MOLLYGRAPH_QUEUE_MAX_CONCURRENT=1"
echo "  If another MollyGraph process is already running, scripts/start.sh will warn and stop before launching a second copy."
echo "Neo4j is optional and only needed for legacy or experimental surfaces."
echo "Optional: enable Ollama embeddings with MOLLYGRAPH_EMBEDDING_OLLAMA_MODEL=nomic-embed-text."
echo "Start service with: $ROOT_DIR/scripts/start.sh"
