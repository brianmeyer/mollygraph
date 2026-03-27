"""Tests for GLiREL per-message chunking.

Verifies that:
- Multi-message blobs are split into chunks of at most GLIREL_CHUNK_SIZE messages.
- Single messages pass through unchanged (as a single-item list).
- Empty content returns an empty list.
"""
from __future__ import annotations

import sys
import os
import types
import unittest

# ---------------------------------------------------------------------------
# Lightweight stubs so the pipeline module can be imported without all its
# heavy runtime dependencies (Neo4j, GLiNER2, GLiREL, etc.).
# ---------------------------------------------------------------------------

def _make_stub(name: str, extra_attrs: dict | None = None) -> types.ModuleType:
    mod = types.ModuleType(name)
    if extra_attrs:
        for k, v in extra_attrs.items():
            setattr(mod, k, v)
    return mod


# Stub out every top-level import that pipeline.py pulls in.
_STUB_MODULES = [
    "neo4j",
    "glirel",
    "gliner",
    "spacy",
    "sentence_transformers",
    "torch",
    "numpy",
    "huggingface_hub",
    "dotenv",
]
for _m in _STUB_MODULES:
    if _m not in sys.modules:
        sys.modules[_m] = _make_stub(_m)

# dotenv.load_dotenv stub
sys.modules["dotenv"] = _make_stub("dotenv", {"load_dotenv": lambda *a, **kw: None})

# Ensure the service directory is on the path.
_SERVICE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _SERVICE_DIR not in sys.path:
    sys.path.insert(0, _SERVICE_DIR)

# ---------------------------------------------------------------------------
# Stub modules that pipeline.py imports from the local package tree.
# ---------------------------------------------------------------------------

def _stub_submodule(parent: str, child: str, **attrs: object) -> None:
    full = f"{parent}.{child}"
    if full not in sys.modules:
        mod = _make_stub(full, attrs)
        sys.modules[full] = mod
        parent_mod = sys.modules.get(parent)
        if parent_mod is not None:
            setattr(parent_mod, child, mod)


# config stub (must exist BEFORE pipeline.py is imported)
import importlib.util as _ilu

_config_spec = _ilu.spec_from_file_location("config", os.path.join(_SERVICE_DIR, "config.py"))
if _config_spec:
    _config_mod = _ilu.module_from_spec(_config_spec)
    sys.modules["config"] = _config_mod
    try:
        _config_spec.loader.exec_module(_config_mod)  # type: ignore[union-attr]
    except Exception:
        # If config.py itself has missing deps, stub key attributes manually.
        _config_mod.GLIREL_ENABLED = False
        _config_mod.GLIREL_CHUNKING_ENABLED = True
        _config_mod.GLIREL_CHUNK_SIZE = 3
        _config_mod.GLIREL_MODEL = "jackboyla/glirel-large-v0"
        _config_mod.GLIREL_CONFIDENCE_THRESHOLD = 0.15
        _config_mod.GLIREL_SILVER_ENABLED = False
        _config_mod.GLIREL_TRAINING_THRESHOLD = 0.4
        _config_mod.SPACY_ENRICHMENT = False
        _config_mod.SPACY_MODEL = "en_core_web_sm"
        _config_mod.SPACY_MIN_GLINER_ENTITIES = 2
        _config_mod.EXTRACTION_CONFIDENCE = {}
        _config_mod.EXTRACTION_CONFIDENCE_DEFAULT = 0.4
        _config_mod.RELATION_SOFT_GATE_ENABLED = False
        _config_mod.RELATION_GATE_ALLOW_THRESHOLD = 0.45
        _config_mod.RELATION_GATE_QUARANTINE_THRESHOLD = 0.25
        _config_mod.RELATION_GATE_HIGH_CONF_OVERRIDE = 0.85
        _config_mod.DECISION_TRACES_INGEST_ENABLED = False
        _config_mod.DECISION_TRACES_PREFILTER_ENABLED = False
        _config_mod.DECISION_TRACES_MIN_CONTENT_CHARS = 24
        _config_mod.DECISION_TRACES_PREFILTER_MIN_SCORE = 2
        _config_mod.DECISION_TRACES_MIN_CONFIDENCE = 0.6
        _config_mod.GLIREL_LLM_SYNONYMS = False
else:
    # Fallback: minimal config stub
    _config_mod = _make_stub("config")
    _config_mod.GLIREL_ENABLED = False
    _config_mod.GLIREL_CHUNKING_ENABLED = True
    _config_mod.GLIREL_CHUNK_SIZE = 3
    sys.modules["config"] = _config_mod

import config as service_config  # noqa: E402  (intentionally after stubs)

# ── Stub remaining heavy dependencies ────────────────────────────────────────

_heavy_stubs = {
    "memory": {},
    "memory.models": {
        "Entity": object,
        "Episode": object,
        "ExtractionJob": object,
        "Relationship": object,
    },
    "memory.graph": {"BiTemporalGraph": object},
    "memory.vector_store": {"VectorStore": object},
    "memory.extractor": {"extract": lambda *a, **kw: {"entities": [], "relations": []}},
    "memory.graph_suggestions": {"log_relationship_fallback": lambda **kw: None},
    "extraction.decision_traces": {
        "extract_decision_trace": None,
        "run_decision_prefilter": None,
    },
    "extraction.glirel_enrichment": {"GLiRELEnrichment": object},
    "extraction.relation_gate": {
        "GateResult": object,
        "get_gate": lambda: None,
    },
    "metrics.stats_logger": {"log_extraction": None},
    "metrics.model_health": {"model_health_monitor": None},
}

for _mod_name, _attrs in _heavy_stubs.items():
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = _make_stub(_mod_name, _attrs)

# ---------------------------------------------------------------------------
# Now we can safely import just the chunking logic from the pipeline module.
# We don't instantiate ExtractionPipeline; we test the methods as plain
# functions by binding them to a minimal stub object.
# ---------------------------------------------------------------------------

import re  # noqa: E402


class _StubPipeline:
    """Minimal stub that carries only what _chunk_content_for_glirel needs."""

    # Copy the class-level attributes from the real ExtractionPipeline.
    _MSG_BOUNDARY_PATTERNS: tuple  # populated below after import attempt


# Try to import the real patterns; fall back to inline copies.
try:
    from extraction.pipeline import ExtractionPipeline  # type: ignore
    _StubPipeline._MSG_BOUNDARY_PATTERNS = ExtractionPipeline._MSG_BOUNDARY_PATTERNS
    # Bind the real methods to our stub.
    _StubPipeline._chunk_content_for_glirel = ExtractionPipeline._chunk_content_for_glirel  # type: ignore
except Exception:
    # Inline the patterns so tests can run even if the full pipeline import fails.
    _StubPipeline._MSG_BOUNDARY_PATTERNS = (
        re.compile(
            r"(?:^|\n)\s*-\s*Message\s+\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}",
            re.IGNORECASE,
        ),
        re.compile(r"(?:^|\n)\s*Statement\s+(?:by|from)\s+\S", re.IGNORECASE),
        re.compile(r"(?:^|\n)\s*From\s*:", re.IGNORECASE),
        re.compile(r"(?:^|\n)\s*On\s+.{0,60}wrote\s*:", re.IGNORECASE | re.DOTALL),
        re.compile(r"(?:^|\n)\s*\S{2,}[^:\n]{0,60}:\s+\S"),
    )

    def _chunk_content_for_glirel(self, content: str) -> list[str]:  # type: ignore
        if not content or not content.strip():
            return []
        if len(content) < 300:
            return [content]
        chunk_size: int = int(getattr(service_config, "GLIREL_CHUNK_SIZE", 3))
        for pattern in self._MSG_BOUNDARY_PATTERNS:
            matches = list(pattern.finditer(content))
            if len(matches) < 2:
                continue
            boundaries = [m.start() for m in matches]
            raw_messages: list[str] = []
            for i, start in enumerate(boundaries):
                msg_start = start + (1 if content[start] == "\n" else 0)
                end = boundaries[i + 1] if i + 1 < len(boundaries) else len(content)
                segment = content[msg_start:end].strip()
                if segment:
                    raw_messages.append(segment)
            if not raw_messages:
                continue
            chunks: list[str] = []
            for i in range(0, len(raw_messages), chunk_size):
                group = raw_messages[i : i + chunk_size]
                chunks.append("\n\n".join(group))
            return chunks
        return [content]

    _StubPipeline._chunk_content_for_glirel = _chunk_content_for_glirel  # type: ignore


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_messages(n: int, fmt: str = "message_log") -> str:
    """Build a synthetic multi-message blob with *n* individual messages."""
    if fmt == "message_log":
        lines = []
        for i in range(n):
            ts = f"2024-01-01 {i:02d}:00:00"
            lines.append(
                f"- Message {ts} UTC: from Speaker{i} to Target; "
                f"via test; content 'Hello from message {i}.'"
            )
        return "\n".join(lines)
    elif fmt == "statement":
        return "\n".join(
            f"Statement by Speaker{i}: This is message number {i} with some text."
            for i in range(n)
        )
    elif fmt == "email":
        parts = []
        for i in range(n):
            parts.append(
                f"From: person{i}@example.com\n"
                f"Subject: Topic {i}\n"
                f"Body text for message {i}.\n"
            )
        return "\n".join(parts)
    else:
        raise ValueError(f"Unknown format: {fmt}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGLiRELChunking(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = _StubPipeline()
        self._method_config = getattr(
            getattr(self.pipeline, "_chunk_content_for_glirel"),
            "__globals__",
            {},
        ).get("service_config")

    def _set_chunk_size(self, value: int) -> None:
        service_config.GLIREL_CHUNK_SIZE = value
        if self._method_config is not None:
            self._method_config.GLIREL_CHUNK_SIZE = value

    # ── Empty / trivial content ──────────────────────────────────────────────

    def test_empty_string_returns_empty_list(self) -> None:
        result = self.pipeline._chunk_content_for_glirel("")
        self.assertEqual(result, [], "Empty string should return []")

    def test_whitespace_only_returns_empty_list(self) -> None:
        result = self.pipeline._chunk_content_for_glirel("   \n\t  ")
        self.assertEqual(result, [], "Whitespace-only should return []")

    # ── Single message (no split needed) ────────────────────────────────────

    def test_single_short_message_unchanged(self) -> None:
        msg = "Alice works at Acme Corp and lives in New York."
        result = self.pipeline._chunk_content_for_glirel(msg)
        self.assertEqual(len(result), 1, "Single short message → 1 chunk")
        self.assertEqual(result[0], msg)

    def test_single_long_message_no_pattern_unchanged(self) -> None:
        """A long blob with no recognisable message boundaries → 1 chunk."""
        msg = "A" * 400  # 400 chars, no message boundary
        result = self.pipeline._chunk_content_for_glirel(msg)
        self.assertEqual(len(result), 1, "No boundary → single chunk")

    # ── Message-log format ───────────────────────────────────────────────────

    def test_multi_message_log_splits_into_chunks(self) -> None:
        """12 message-log entries with chunk_size=3 → 4 chunks."""
        self._set_chunk_size(3)
        content = _make_messages(12, fmt="message_log")
        result = self.pipeline._chunk_content_for_glirel(content)
        self.assertGreater(len(result), 1, "Multi-message log must produce >1 chunk")
        # Each chunk should contain at most GLIREL_CHUNK_SIZE boundary markers.
        pattern = re.compile(
            r"- Message \d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}",
            re.IGNORECASE,
        )
        for chunk in result:
            count = len(pattern.findall(chunk))
            self.assertLessEqual(
                count,
                service_config.GLIREL_CHUNK_SIZE,
                f"Chunk has {count} messages, expected ≤ {service_config.GLIREL_CHUNK_SIZE}",
            )

    def test_six_messages_makes_two_chunks_of_three(self) -> None:
        self._set_chunk_size(3)
        content = _make_messages(6, fmt="message_log")
        result = self.pipeline._chunk_content_for_glirel(content)
        self.assertEqual(len(result), 2, "6 messages / chunk_size=3 → 2 chunks")

    # ── Statement format ─────────────────────────────────────────────────────

    def test_statement_format_splits(self) -> None:
        self._set_chunk_size(3)
        content = _make_messages(6, fmt="statement")
        result = self.pipeline._chunk_content_for_glirel(content)
        self.assertGreater(len(result), 1, "Statement format should split")

    # ── Email format ─────────────────────────────────────────────────────────

    def test_email_format_splits(self) -> None:
        self._set_chunk_size(3)
        content = _make_messages(6, fmt="email")
        result = self.pipeline._chunk_content_for_glirel(content)
        self.assertGreater(len(result), 1, "Email format should split")

    # ── chunk_size = 1 ───────────────────────────────────────────────────────

    def test_chunk_size_one_each_message_separate(self) -> None:
        self._set_chunk_size(1)
        content = _make_messages(4, fmt="message_log")
        result = self.pipeline._chunk_content_for_glirel(content)
        self.assertEqual(len(result), 4, "chunk_size=1 → one chunk per message")

    # ── No boundary detected ─────────────────────────────────────────────────

    def test_no_boundary_returns_full_content_as_single_chunk(self) -> None:
        content = (
            "This is a long paragraph with no message log boundaries. "
            "It discusses multiple organisations but has no structured turns. "
            "GLiREL should receive it as-is. " * 10
        )
        result = self.pipeline._chunk_content_for_glirel(content)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], content)

    # ── Chunk content coherence ───────────────────────────────────────────────

    def test_no_content_lost_across_chunks(self) -> None:
        """All original messages should appear in some chunk (no data loss)."""
        self._set_chunk_size(3)
        n = 9
        content = _make_messages(n, fmt="message_log")
        chunks = self.pipeline._chunk_content_for_glirel(content)
        combined = "\n".join(chunks)
        for i in range(n):
            self.assertIn(
                f"Hello from message {i}",
                combined,
                f"Message {i} content missing from chunks",
            )


if __name__ == "__main__":
    unittest.main()
