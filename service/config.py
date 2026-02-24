"""Service configuration — all settings from environment variables."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=False)

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
OPENAI_API_KEY      = os.environ.get("OPENAI_API_KEY", "")
OPENROUTER_API_KEY  = os.environ.get("OPENROUTER_API_KEY", "")
TOGETHER_API_KEY    = os.environ.get("TOGETHER_API_KEY", "")
FIREWORKS_API_KEY   = os.environ.get("FIREWORKS_API_KEY", "")

# ── Audit models ──────────────────────────────────────────────────────────────
# Local-first defaults: LLM audit is optional and disabled by default.
AUDIT_LLM_ENABLED    = os.environ.get("AUDIT_LLM_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
AUDIT_MODEL_NIGHTLY  = os.environ.get("AUDIT_MODEL_NIGHTLY", "llama3.1:8b")
AUDIT_MODEL_WEEKLY   = os.environ.get("AUDIT_MODEL_WEEKLY", "llama3.1:8b")
AUDIT_MODEL_PRETRAIN = os.environ.get("AUDIT_MODEL_PRETRAIN", "llama3.1:8b")  # used by gliner pre-training validation
AUDIT_PROVIDER_ORDER = os.environ.get("AUDIT_PROVIDER_ORDER", "none")  # legacy single-provider override

# Audit LLM tier chain (deterministic always runs; LLM tiers fall through on failure)
AUDIT_PROVIDER_TIERS = os.environ.get(
    "MOLLYGRAPH_AUDIT_PROVIDER_TIERS",
    "deterministic,local,primary,fallback",
).split(",")
# Per-tier provider mapping
AUDIT_TIER_LOCAL    = os.environ.get("MOLLYGRAPH_AUDIT_TIER_LOCAL", "ollama")   # ollama, llama-cpp, etc.
AUDIT_TIER_PRIMARY  = os.environ.get("MOLLYGRAPH_AUDIT_TIER_PRIMARY", "")       # moonshot, openai, groq, etc.
AUDIT_TIER_FALLBACK = os.environ.get("MOLLYGRAPH_AUDIT_TIER_FALLBACK", "")      # different provider as backup
# Per-tier model
AUDIT_MODEL_LOCAL    = os.environ.get("MOLLYGRAPH_AUDIT_MODEL_LOCAL", "llama3.1:8b")
AUDIT_MODEL_PRIMARY  = os.environ.get("MOLLYGRAPH_AUDIT_MODEL_PRIMARY", "")
AUDIT_MODEL_FALLBACK = os.environ.get("MOLLYGRAPH_AUDIT_MODEL_FALLBACK", "")

# ── LLM provider base URLs ────────────────────────────────────────────────────
GEMINI_BASE_URL    = "https://generativelanguage.googleapis.com/v1beta/openai"
MOONSHOT_BASE_URL  = os.environ.get("MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1")
GROQ_BASE_URL      = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
OLLAMA_BASE_URL    = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_CHAT_BASE_URL = os.environ.get("OLLAMA_CHAT_BASE_URL", f"{OLLAMA_BASE_URL}/v1")
OPENAI_BASE_URL      = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENROUTER_BASE_URL  = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
TOGETHER_BASE_URL    = os.environ.get("TOGETHER_BASE_URL", "https://api.together.xyz/v1")
FIREWORKS_BASE_URL   = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")

# ── Service ───────────────────────────────────────────────────────────────────
HOST = os.environ.get("GRAPH_MEMORY_HOST", "127.0.0.1")
PORT = int(os.environ.get("GRAPH_MEMORY_PORT", "7422"))
API_KEY = os.environ.get("MOLLYGRAPH_API_KEY", "dev-key-change-in-production")
TEST_MODE = os.environ.get("MOLLYGRAPH_TEST_MODE", "0").strip().lower() in {"1", "true", "yes"}
VECTOR_BACKEND = os.environ.get("MOLLYGRAPH_VECTOR_BACKEND", "zvec")
RUNTIME_PROFILE = os.environ.get("MOLLYGRAPH_RUNTIME_PROFILE", "hybrid").strip().lower()
STRICT_AI = (
    RUNTIME_PROFILE == "strict_ai"
    or os.environ.get("MOLLYGRAPH_STRICT_AI", "0").strip().lower() in {"1", "true", "yes", "on"}
)
SPACY_ENRICHMENT = os.environ.get("MOLLYGRAPH_SPACY_ENRICHMENT", "1").strip().lower() in {"1", "true", "yes", "on"}
SPACY_MODEL = os.environ.get("MOLLYGRAPH_SPACY_MODEL", "en_core_web_sm")
SPACY_MIN_GLINER_ENTITIES = int(os.environ.get("MOLLYGRAPH_SPACY_MIN_GLINER_ENTITIES", "2"))
EMBEDDING_BACKEND = os.environ.get("MOLLYGRAPH_EMBEDDING_BACKEND", "").strip().lower()  # legacy override; empty = use tier chain
# Embedding tier chain (falls through on failure; hash is always last resort)
EMBEDDING_TIER_ORDER = os.environ.get(
    "MOLLYGRAPH_EMBEDDING_TIER_ORDER",
    "sentence-transformers,ollama,cloud,hash",
).split(",")
# Per-tier embedding config
EMBEDDING_MODEL = os.environ.get("MOLLYGRAPH_EMBEDDING_MODEL", "")  # empty = use tier default
EMBEDDING_ST_MODEL = os.environ.get("MOLLYGRAPH_EMBEDDING_ST_MODEL", "")  # sentence-transformers model
EMBEDDING_OLLAMA_MODEL = os.environ.get("MOLLYGRAPH_EMBEDDING_OLLAMA_MODEL",
    os.environ.get("MOLLYGRAPH_OLLAMA_EMBED_MODEL", "nomic-embed-text"))  # also accepts old var
EMBEDDING_CLOUD_PROVIDER = os.environ.get("MOLLYGRAPH_EMBEDDING_CLOUD_PROVIDER", "openai")  # openai, google, etc.
EMBEDDING_CLOUD_MODEL = os.environ.get("MOLLYGRAPH_EMBEDDING_CLOUD_MODEL", "text-embedding-3-small")
# Legacy alias kept for backward compat
OLLAMA_EMBED_MODEL = EMBEDDING_OLLAMA_MODEL
_extractor_backend_raw = os.environ.get("MOLLYGRAPH_EXTRACTOR_BACKEND", "gliner2").strip().lower()
if _extractor_backend_raw not in {"gliner2", "gliner"}:
    import logging as _logging
    _logging.getLogger(__name__).error(
        "Unsupported MOLLYGRAPH_EXTRACTOR_BACKEND=%r — GLiNER2 is the only supported backend. "
        "Overriding to 'gliner2'.",
        _extractor_backend_raw,
    )
EXTRACTOR_BACKEND = "gliner2"
EXTRACTOR_MODEL = os.environ.get("MOLLYGRAPH_EXTRACTOR_MODEL", "").strip()

# ── Reranker ──────────────────────────────────────────────────────────────────
RERANKER_ENABLED = os.environ.get("MOLLYGRAPH_RERANKER_ENABLED", "false").lower() == "true"
RERANKER_MODEL = os.environ.get("MOLLYGRAPH_RERANKER_MODEL", "jinaai/jina-reranker-v2-base-multilingual")

# ── GLiREL enrichment (second-pass relation extraction) ──────────────────────
GLIREL_ENABLED = os.environ.get("MOLLYGRAPH_GLIREL_ENABLED", "false").lower() == "true"
GLIREL_MODEL = os.environ.get("MOLLYGRAPH_GLIREL_MODEL", "jackboyla/glirel-large-v0")
GLIREL_CONFIDENCE_THRESHOLD = float(os.environ.get("MOLLYGRAPH_GLIREL_CONFIDENCE", "0.15"))  # GLiREL scores rarely exceed 0.3; 0.15 is a reasonable threshold
GLIREL_TRAINING_THRESHOLD = float(os.environ.get("MOLLYGRAPH_GLIREL_TRAINING_THRESHOLD", "0.8"))

# ── GLiNER2 training ──────────────────────────────────────────────────────────
GLINER_BASE_MODEL                  = os.environ.get("GLINER_BASE_MODEL", "fastino/gliner2-large-v1")
GLINER_FINETUNE_MIN_EXAMPLES       = int(os.environ.get("GLINER_FINETUNE_MIN_EXAMPLES", "500"))
GLINER_FINETUNE_COOLDOWN_DAYS      = int(os.environ.get("GLINER_FINETUNE_COOLDOWN_DAYS", "7"))
GLINER_LORA_COOLDOWN_DAYS          = int(os.environ.get("GLINER_LORA_COOLDOWN_DAYS", "2"))
GLINER_FINETUNE_BENCHMARK_THRESHOLD= float(os.environ.get("GLINER_FINETUNE_BENCHMARK_THRESHOLD", "0.03"))
GLINER_FULL_FINETUNE_MIN_EXAMPLES  = int(os.environ.get("GLINER_FULL_FINETUNE_MIN_EXAMPLES", "2000"))
GLINER_LORA_PLATEAU_WINDOW         = int(os.environ.get("GLINER_LORA_PLATEAU_WINDOW", "3"))
GLINER_LORA_PLATEAU_EPSILON        = float(os.environ.get("GLINER_LORA_PLATEAU_EPSILON", "0.01"))
GLINER_TRAINING_SCAN_LIMIT         = int(os.environ.get("GLINER_TRAINING_SCAN_LIMIT", "4000"))
GLINER_BENCHMARK_SEED              = 1337
GLINER_BENCHMARK_EVAL_RATIO        = 0.2
GLINER_BENCHMARK_THRESHOLD         = 0.4
GLINER_SHADOW_EPISODES             = int(os.environ.get("GLINER_SHADOW_EPISODES", "20"))
GLINER_SHADOW_ENABLED              = os.environ.get("GLINER_SHADOW_ENABLED", "true").lower() == "true"

# ── LoRA benchmark tolerance ──────────────────────────────────────────────────
# Shadow benchmark: candidate model is rejected if fallback_rate > base_rate * TOLERANCE.
# 1.20 means the candidate may be up to 20% worse than the baseline before it's rejected.
LORA_BENCHMARK_TOLERANCE = float(os.environ.get("MOLLYGRAPH_LORA_BENCHMARK_TOLERANCE", "1.20"))

# ── Schema drift alarm thresholds ─────────────────────────────────────────────
# Percentage growth in 24 h that triggers the schema-drift alarm.
# Defaults: +5 % for relation types, +10 % for entity types.
SCHEMA_DRIFT_ALARM_REL_THRESHOLD = float(os.environ.get("MOLLYGRAPH_DRIFT_ALARM_REL", "5.0"))
SCHEMA_DRIFT_ALARM_ENT_THRESHOLD = float(os.environ.get("MOLLYGRAPH_DRIFT_ALARM_ENT", "10.0"))

# ── Maintenance lock timeout ───────────────────────────────────────────────────
# Maximum age (seconds) before a maintenance lock is considered stale.
MAINTENANCE_LOCK_TIMEOUT_SECONDS = int(os.environ.get("MOLLYGRAPH_LOCK_TIMEOUT", str(30 * 60)))

# ── Schema auto-adoption caps ─────────────────────────────────────────────────
# Maximum number of new relation/entity types that can be auto-adopted per nightly cycle.
SCHEMA_MAX_NEW_RELATIONS = int(os.environ.get("MOLLYGRAPH_MAX_NEW_RELATIONS", "3"))
SCHEMA_MAX_NEW_ENTITIES  = int(os.environ.get("MOLLYGRAPH_MAX_NEW_ENTITIES",  "2"))

# ── Model degradation detection ───────────────────────────────────────────────
# Rolling window size (number of extractions) used for continuous degradation monitoring.
MODEL_DEGRADATION_WINDOW_SIZE = int(os.environ.get("MOLLYGRAPH_DEGRADATION_WINDOW", "100"))
# Fallback-rate increase above baseline that triggers a degradation WARNING.
# 0.15 = if the rolling fallback rate rises more than 15 percentage points above baseline.
MODEL_DEGRADATION_THRESHOLD = float(os.environ.get("MOLLYGRAPH_DEGRADATION_THRESHOLD", "0.15"))

# ── Audit auto-delete ─────────────────────────────────────────────────────────
# When True, the nightly maintenance pipeline will auto-execute delete suggestions
# generated by the LLM audit (action="delete").  Default: off for safety.
AUDIT_AUTO_DELETE = os.environ.get("MOLLYGRAPH_AUDIT_AUTO_DELETE", "false").lower() == "true"

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
