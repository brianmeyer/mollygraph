#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_DIR="$ROOT_DIR/service"
ENV_FILE="$SERVICE_DIR/.env"
ENV_EXAMPLE_FILE="$SERVICE_DIR/.env.example"
ACTIVE_ENV_FILE="$ENV_FILE"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
elif [[ -f "$ROOT_DIR/.env" ]]; then
  ACTIVE_ENV_FILE="$ROOT_DIR/.env"
  echo "Using legacy env file at $ACTIVE_ENV_FILE. Move your config to $ENV_FILE." >&2
  set -a
  # shellcheck disable=SC1090
  source "$ROOT_DIR/.env"
  set +a
fi

if [[ ! -f "$ENV_FILE" && -f "$ENV_EXAMPLE_FILE" ]]; then
  cp "$ENV_EXAMPLE_FILE" "$ENV_FILE"
  echo "Created $ENV_FILE from $ENV_EXAMPLE_FILE"
fi

STATE_DIR="${MOLLYGRAPH_HOME_DIR:-$HOME/.graph-memory}"
VENV_DIR="${MOLLYGRAPH_VENV_DIR:-$SERVICE_DIR/.venv}"

mkdir -p "$STATE_DIR/models" "$STATE_DIR/training/gliner" "$STATE_DIR/logs" "$STATE_DIR/logs/maintenance" "$STATE_DIR/suggestions"

if [[ -x "$VENV_DIR/bin/python" ]]; then
  EXISTING_PY="$("$VENV_DIR/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  case "$EXISTING_PY" in
    3.12|3.13|3.14)
      PYTHON_BIN="$VENV_DIR/bin/python"
      ;;
    *)
      echo "Existing venv uses unsupported Python $EXISTING_PY at $VENV_DIR." >&2
      echo "Remove it and rerun install.sh so the local runtime uses Python 3.12+." >&2
      exit 1
      ;;
  esac
else
  PYTHON_BIN=""
  for candidate in python3.12 python3.13 python3.14 python3; do
    if ! command -v "$candidate" >/dev/null 2>&1; then
      continue
    fi
    CANDIDATE_VERSION="$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    case "$CANDIDATE_VERSION" in
      3.12|3.13|3.14)
        PYTHON_BIN="$candidate"
        break
        ;;
    esac
  done

  if [[ -z "$PYTHON_BIN" ]]; then
    echo "Python 3.12+ is required. Install Python 3.12, 3.13, or 3.14 and rerun." >&2
    exit 1
  fi

  "$PYTHON_BIN" -m venv "$VENV_DIR"
  PYTHON_BIN="$VENV_DIR/bin/python"
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
echo "Default local stack:"
echo "  MOLLYGRAPH_GRAPH_BACKEND=ladybug"
echo "  MOLLYGRAPH_VECTOR_BACKEND=ladybug"
echo "  MOLLYGRAPH_EMBEDDING_ST_MODEL=Snowflake/snowflake-arctic-embed-s"
echo "Neo4j is optional and only needed for legacy or experimental surfaces."
echo "Optional: enable Ollama embeddings with MOLLYGRAPH_EMBEDDING_OLLAMA_MODEL=nomic-embed-text."
echo "Start service with: $ROOT_DIR/scripts/start.sh"
