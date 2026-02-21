"""Service configuration — all settings from environment variables."""
from __future__ import annotations

import os
from pathlib import Path

# ── Core paths ────────────────────────────────────────────────────────────────
GRAPH_MEMORY_DIR = Path(
    os.environ.get("MOLLYGRAPH_HOME_DIR", str(Path.home() / ".graph-memory"))
).expanduser()
MODELS_DIR        = GRAPH_MEMORY_DIR / "models"
TRAINING_DIR      = GRAPH_MEMORY_DIR / "training" / "gliner"
LOGS_DIR          = GRAPH_MEMORY_DIR / "logs"
MAINTENANCE_DIR   = LOGS_DIR / "maintenance"
SUGGESTIONS_DIR   = GRAPH_MEMORY_DIR / "suggestions"
STATE_FILE        = GRAPH_MEMORY_DIR / "state.json"
QUEUE_DB_PATH     = Path(
    os.environ.get("MOLLYGRAPH_QUEUE_DB", str(GRAPH_MEMORY_DIR / "extraction_queue.db"))
).expanduser()
SQLITE_VEC_DB_PATH = Path(
    os.environ.get("MOLLYGRAPH_SQLITE_VEC_DB", str(GRAPH_MEMORY_DIR / "vectors.db"))
).expanduser()
ZVEC_COLLECTION_DIR = Path(
    os.environ.get("MOLLYGRAPH_ZVEC_DIR", str(GRAPH_MEMORY_DIR / "zvec_collection"))
).expanduser()

# User-extensible relation schema (optional YAML file)
RELATION_SCHEMA_FILE: Path | None = (
    Path(p) if (p := os.environ.get("RELATION_SCHEMA_FILE")) else None
)

# ── Neo4j ─────────────────────────────────────────────────────────────────────
NEO4J_URI      = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "mollygraph")

# ── LLM API keys ──────────────────────────────────────────────────────────────
GOOGLE_API_KEY      = os.environ.get("GOOGLE_API_KEY", "")
ANTHROPIC_API_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")
MOONSHOT_API_KEY    = os.environ.get("MOONSHOT_API_KEY", "")
GROQ_API_KEY        = os.environ.get("GROQ_API_KEY", "")
OLLAMA_API_KEY      = os.environ.get("OLLAMA_API_KEY", "")

# ── Audit models ──────────────────────────────────────────────────────────────
# Local-first defaults: LLM audit is optional and disabled by default.
AUDIT_LLM_ENABLED = os.environ.get("AUDIT_LLM_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
AUDIT_MODEL_NIGHTLY = os.environ.get("AUDIT_MODEL_NIGHTLY", "llama3.1:8b")
AUDIT_MODEL_PRETRAIN = os.environ.get("AUDIT_MODEL_PRETRAIN", "llama3.1:8b")
AUDIT_MODEL_WEEKLY = os.environ.get("AUDIT_MODEL_WEEKLY", "llama3.1:8b")
AUDIT_MODEL_LOCAL = os.environ.get("AUDIT_MODEL_LOCAL", "llama3.1:8b")

# Provider/model chain used by audit fallback registry.
AUDIT_MODEL_PRIMARY = os.environ.get("AUDIT_MODEL_PRIMARY", AUDIT_MODEL_NIGHTLY)
AUDIT_MODEL_SECONDARY = os.environ.get("AUDIT_MODEL_SECONDARY", "llama3.1:8b")
AUDIT_MODEL_TERTIARY = os.environ.get("AUDIT_MODEL_TERTIARY", "gpt-oss-120b")
AUDIT_PROVIDER_ORDER = os.environ.get("AUDIT_PROVIDER_ORDER", "none")

GEMINI_BASE_URL    = "https://generativelanguage.googleapis.com/v1beta/openai"
MOONSHOT_BASE_URL  = os.environ.get("MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1")
GROQ_BASE_URL      = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
OLLAMA_BASE_URL    = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_CHAT_BASE_URL = os.environ.get("OLLAMA_CHAT_BASE_URL", f"{OLLAMA_BASE_URL}/v1")

# ── Service ───────────────────────────────────────────────────────────────────
HOST = os.environ.get("GRAPH_MEMORY_HOST", "127.0.0.1")
PORT = int(os.environ.get("GRAPH_MEMORY_PORT", "7422"))
API_KEY = os.environ.get("MOLLYGRAPH_API_KEY", "dev-key-change-in-production")
TEST_MODE = os.environ.get("MOLLYGRAPH_TEST_MODE", "0").strip().lower() in {"1", "true", "yes"}
VECTOR_BACKEND = os.environ.get("MOLLYGRAPH_VECTOR_BACKEND", "zvec")
SPACY_ENRICHMENT = os.environ.get("MOLLYGRAPH_SPACY_ENRICHMENT", "0").strip().lower() in {"1", "true", "yes", "on"}
SPACY_MODEL = os.environ.get("MOLLYGRAPH_SPACY_MODEL", "en_core_web_sm")
SPACY_MIN_GLINER_ENTITIES = int(os.environ.get("MOLLYGRAPH_SPACY_MIN_GLINER_ENTITIES", "2"))
EMBEDDING_BACKEND = os.environ.get("MOLLYGRAPH_EMBEDDING_BACKEND", "hash").strip().lower()
EMBEDDING_MODEL = os.environ.get("MOLLYGRAPH_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_EMBED_MODEL = os.environ.get("MOLLYGRAPH_OLLAMA_EMBED_MODEL", "nomic-embed-text")

# ── GLiNER2 training ──────────────────────────────────────────────────────────
GLINER_BASE_MODEL                  = os.environ.get("GLINER_BASE_MODEL", "fastino/gliner2-large-v1")
GLINER_FINETUNE_MIN_EXAMPLES       = int(os.environ.get("GLINER_FINETUNE_MIN_EXAMPLES", "500"))
GLINER_FINETUNE_COOLDOWN_DAYS      = int(os.environ.get("GLINER_FINETUNE_COOLDOWN_DAYS", "7"))
GLINER_FINETUNE_BENCHMARK_THRESHOLD= float(os.environ.get("GLINER_FINETUNE_BENCHMARK_THRESHOLD", "0.05"))
GLINER_FULL_FINETUNE_MIN_EXAMPLES  = int(os.environ.get("GLINER_FULL_FINETUNE_MIN_EXAMPLES", "2000"))
GLINER_LORA_PLATEAU_WINDOW         = int(os.environ.get("GLINER_LORA_PLATEAU_WINDOW", "3"))
GLINER_LORA_PLATEAU_EPSILON        = float(os.environ.get("GLINER_LORA_PLATEAU_EPSILON", "0.01"))
GLINER_TRAINING_SCAN_LIMIT         = int(os.environ.get("GLINER_TRAINING_SCAN_LIMIT", "4000"))
GLINER_BENCHMARK_SEED              = 1337
GLINER_BENCHMARK_EVAL_RATIO        = 0.2
GLINER_BENCHMARK_THRESHOLD         = 0.4

# ── Ensure runtime dirs exist ────────────────────────────────────────────────
for _d in (
    GRAPH_MEMORY_DIR,
    MODELS_DIR,
    TRAINING_DIR,
    LOGS_DIR,
    MAINTENANCE_DIR,
    SUGGESTIONS_DIR,
    QUEUE_DB_PATH.parent,
    SQLITE_VEC_DB_PATH.parent,
    ZVEC_COLLECTION_DIR.parent,
):
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Some execution sandboxes cannot write outside the workspace.
        # Keep defaults unchanged for normal runtime; callers can still override
        # paths via env if needed.
        pass
