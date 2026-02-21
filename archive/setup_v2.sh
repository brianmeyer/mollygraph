#!/bin/bash
set -e

echo "📦 Setting up Python environment..."
cd ~/.openclaw/workspace/skills/graph-memory/service
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

echo "⬇️ Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "⬇️ Downloading spaCy model..."
if ! python3 -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null; then
    python3 -m spacy download en_core_web_sm
fi

echo "✅ Environment ready for MollyGraph V2"
