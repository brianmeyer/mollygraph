#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${MOLLYGRAPH_HOME_DIR:-$HOME/.graph-memory}"
VENV_DIR="$STATE_DIR/venv"

mkdir -p "$STATE_DIR/models" "$STATE_DIR/training/gliner" "$STATE_DIR/logs" "$STATE_DIR/logs/maintenance" "$STATE_DIR/suggestions"

if command -v python3.12 >/dev/null 2>&1; then
  PYTHON_BIN="python3.12"
else
  echo "Python 3.12 is required (spacy/zvec compatibility). Install python3.12 and rerun." >&2
  exit 1
fi

if [[ -x "$VENV_DIR/bin/python" ]]; then
  EXISTING_PY="$("$VENV_DIR/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "$EXISTING_PY" != "3.12" ]]; then
    echo "Existing venv uses Python $EXISTING_PY at $VENV_DIR." >&2
    echo "Remove it and rerun install.sh so runtime is pinned to Python 3.12." >&2
    exit 1
  fi
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install -r "$ROOT_DIR/requirements.txt"

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
    import spacy
    try:
        spacy.load("en_core_web_sm")
        print("spaCy model en_core_web_sm already present")
    except Exception:
        from spacy.cli import download
        download("en_core_web_sm")
        print("Downloaded spaCy model en_core_web_sm")
except Exception as exc:
    print(f"Skipped spaCy setup: {exc}")
PY

echo "Install complete."
echo "Set NEO4J_* in $ROOT_DIR/.env (or shell env)."
echo "Optional: enable LLM audit with AUDIT_LLM_ENABLED=true and a provider configuration."
echo "Optional: set MOLLYGRAPH_HOME_DIR to isolate runtime state from ~/.graph-memory."
echo "Optional: set MOLLYGRAPH_SPACY_ENRICHMENT=true to enable spaCy fallback enrichment."
echo "Start service with: $ROOT_DIR/scripts/start.sh"
