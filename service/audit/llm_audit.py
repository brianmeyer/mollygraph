"""Relationship audit pipeline.

Local-only mode is the default. LLM-backed audit is optional and can be
enabled with provider/model configuration (including local Ollama models).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Iterable

import config
from audit.signals import get_signal_bus
from evolution.audit_feedback import record_audit_feedback_batch
from memory.graph import VALID_REL_TYPES
from memory.graph_suggestions import build_suggestion_digest, run_auto_adoption
from runtime_graph import require_graph_instance
from runtime_vector_store import get_vector_store_instance

log = logging.getLogger(__name__)

_BATCH_SIZE = 25  # Small batches prevent JSON truncation in LLM responses

# ---------------------------------------------------------------------------
# Audit state helpers  (coverage metrics + last_full_sweep)
# ---------------------------------------------------------------------------

def _audit_state_path() -> "Path":
    from pathlib import Path
    state_dir = Path.home() / ".graph-memory"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / "audit_state.json"


def _load_audit_state() -> dict[str, Any]:
    path = _audit_state_path()
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            log.debug("Failed to read audit_state.json", exc_info=True)
    return {}


def _save_audit_state(state: dict[str, Any]) -> None:
    path = _audit_state_path()
    try:
        # Atomic write: temp file + os.replace() prevents corruption on crash mid-write.
        tmp_fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                json.dump(state, fh, indent=2, default=str)
            os.replace(tmp_name, str(path))
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
    except Exception:
        log.debug("Failed to write audit_state.json", exc_info=True)


# ---------------------------------------------------------------------------
# Training signal helpers
# ---------------------------------------------------------------------------

def _audit_signals_path() -> "Path":
    """Return path to the audit training signals JSONL file."""
    from pathlib import Path
    signals_dir = Path.home() / ".graph-memory" / "training"
    signals_dir.mkdir(parents=True, exist_ok=True)
    return signals_dir / "audit_signals.jsonl"


def _write_audit_signal(signal: dict[str, Any]) -> None:
    """Append one training signal record to audit_signals.jsonl."""
    try:
        path = _audit_signals_path()
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(signal, ensure_ascii=True) + "\n")
    except Exception:
        log.debug("Failed writing audit signal", exc_info=True)


SYSTEM_PROMPT = (
    "You are auditing a personal knowledge graph extracted from conversations. "
    "Return valid JSON only."
)


def _relationship_lines(rels: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for idx, rel in enumerate(rels, start=1):
        snippets = rel.get("context_snippets") or []
        if isinstance(snippets, list):
            snippets = [str(s) for s in snippets[:3] if str(s).strip()]
        else:
            snippets = []
        snippet_text = " | ".join(snippets) if snippets else "none"

        lines.append(
            f"{idx}. {rel.get('head', '')} ({rel.get('head_type', '?')}) "
            f"--[{rel.get('rel_type', '?')}]--> "
            f"{rel.get('tail', '')} ({rel.get('tail_type', '?')})\n"
            f"   strength={rel.get('strength', 0)}, mentions={rel.get('mention_count', 0)}, "
            f"audit_status={rel.get('audit_status', None)}\n"
            f"   context={snippet_text}"
        )

    return "\n".join(lines)


def build_audit_prompt(rels: list[dict[str, Any]]) -> str:
    rel_list = _relationship_lines(rels)
    valid_types = ", ".join(sorted(VALID_REL_TYPES))

    return (
        "You are auditing a personal knowledge graph extracted from conversations.\n\n"
        f"Valid relationship types: {valid_types}\n\n"
        "This graph tracks entities Brian knows, uses, and talks about.\n\n"
        "Review each relationship below. For each, return a verdict:\n"
        '- "verify"     -> relationship is correct, mark as verified\n'
        '- "reclassify" -> wrong type, suggest the correct type from the valid list\n'
        '- "quarantine" -> relationship exists but is suspicious or uncertain, flag for review\n'
        '- "delete"     -> relationship is wrong, self-referential, or spam\n\n'
        "IMPORTANT PATTERNS TO CATCH:\n"
        "- Type mismatches (Person->Place with WORKS_AT instead of LOCATED_IN)\n"
        "- Contradictions (Person MENTORS X and MENTORED_BY X simultaneously)\n"
        "- Multiple competing WORKS_AT for same person (keep strongest, quarantine others)\n"
        "- Persistent RELATED_TO edges with 3+ mentions (suggest specific type in notes)\n"
        "- Low-confidence single-mention relationships with weak context\n\n"
        "Relationships to review:\n"
        f"{rel_list}\n\n"
        "Return JSON array:\n"
        "[\n"
        "  {\n"
        '    "index": 1,\n'
        '    "verdict": "verify|reclassify|quarantine|delete",\n'
        '    "suggested_type": "WORKS_AT",\n'
        '    "note": "optional brief reason"\n'
        "  }\n"
        "]\n"
        "Return ONLY the JSON array."
    )


def _extract_json_array(raw: str) -> str:
    import re

    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].startswith("```"):
            text = "\n".join(lines[1:-1]).strip()
        else:
            text = "\n".join(lines[1:]).strip()

    start = text.find("[")
    if start < 0:
        return text

    end = text.rfind("]")
    if end > start:
        candidate = text[start : end + 1]
        # Try parsing as-is first
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
        # Repair: some LLMs drop opening braces on objects inside arrays
        repaired = re.sub(
            r'(?<=[,\[])\s*"index"',
            ' {"index"',
            candidate,
        )
        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            pass

    # Truncated response (no closing ]) or all repairs failed:
    # Extract all complete {...} objects from after the opening [
    remainder = text[start:]
    obj_pattern = re.compile(r'\{[^{}]*\}')
    objects = obj_pattern.findall(remainder)
    if not objects:
        # Try wrapping bare key-value groups in braces
        obj_pattern2 = re.compile(
            r'"index"\s*:\s*\d+\s*,\s*"verdict"\s*:\s*"[^"]*"'
            r'(?:\s*,\s*"suggested_type"\s*:\s*"[^"]*")?'
            r'(?:\s*,\s*"note"\s*:\s*"[^"]*")?',
        )
        matches = obj_pattern2.findall(remainder)
        objects = ["{" + m.rstrip().rstrip(",") + "}" for m in matches]
    if objects:
        # Validate each object individually, keep only parseable ones
        valid = []
        for obj in objects:
            try:
                json.loads(obj)
                valid.append(obj)
            except json.JSONDecodeError:
                continue
        if valid:
            repaired = "[" + ",".join(valid) + "]"
            log.info("Recovered %d/%d objects from truncated/broken JSON", len(valid), len(objects))
            return repaired
    return text


def parse_verdicts(raw: str, batch_len: int) -> list[dict[str, Any]]:
    text = _extract_json_array(raw)
    if not text:
        return []

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        log.warning("Audit model response was not valid JSON: %s", text[:280])
        return []

    if not isinstance(payload, list):
        return []

    verdicts: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue

        index = item.get("index")
        verdict = str(item.get("verdict") or "").strip().lower()
        suggested_type = str(item.get("suggested_type") or "").strip().upper().replace(" ", "_")
        note = str(item.get("note") or "").strip()

        if not isinstance(index, int):
            continue
        if not (1 <= index <= batch_len):
            continue
        if verdict not in {"verify", "reclassify", "quarantine", "delete"}:
            continue

        verdicts.append(
            {
                "index": index,
                "verdict": verdict,
                "suggested_type": suggested_type,
                "note": note,
            }
        )

    return verdicts


async def _invoke_openai_compatible(
    provider: str,
    prompt: str,
    model: str,
    base_url: str,
    api_key: str = "",
    require_api_key: bool = True,
    max_tokens: int = 4096,
) -> tuple[str, int]:
    import httpx

    if require_api_key and not api_key:
        raise RuntimeError(f"{provider} API key is not set")

    # Kimi k2.5 is a thinking model that only accepts temperature=1.0
    temp = 1.0 if provider in ("moonshot", "kimi") else 0.1

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temp,
        "max_tokens": max_tokens,
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    started = time.monotonic()
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    async with httpx.AsyncClient(timeout=180) as client:
        for attempt in range(3):
            try:
                resp = await client.post(endpoint, headers=headers, json=body)
                if resp.status_code >= 400:
                    log.warning("%s audit HTTP %s: %s", provider, resp.status_code, resp.text[:500])
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return str(content), int((time.monotonic() - started) * 1000)
            except Exception as exc:
                if attempt == 2:
                    raise
                await asyncio.sleep(2 * (attempt + 1))
                log.debug("%s audit retry %d: %s", provider, attempt + 1, exc)

    return "", int((time.monotonic() - started) * 1000)


async def _invoke_gemini(prompt: str, model: str) -> tuple[str, int]:
    return await _invoke_openai_compatible(
        provider="gemini",
        prompt=prompt,
        model=model,
        base_url=config.GEMINI_BASE_URL,
        api_key=config.GOOGLE_API_KEY,
        require_api_key=True,
        max_tokens=8192,
    )


async def _invoke_moonshot(prompt: str, model: str) -> tuple[str, int]:
    return await _invoke_openai_compatible(
        provider="moonshot",
        prompt=prompt,
        model=model,
        base_url=config.MOONSHOT_BASE_URL,
        api_key=config.MOONSHOT_API_KEY,
        require_api_key=True,
        max_tokens=4096,
    )


async def _invoke_groq(prompt: str, model: str) -> tuple[str, int]:
    return await _invoke_openai_compatible(
        provider="groq",
        prompt=prompt,
        model=model,
        base_url=config.GROQ_BASE_URL,
        api_key=config.GROQ_API_KEY,
        require_api_key=True,
        max_tokens=4096,
    )


async def _invoke_ollama(prompt: str, model: str) -> tuple[str, int]:
    return await _invoke_openai_compatible(
        provider="ollama",
        prompt=prompt,
        model=model,
        base_url=config.OLLAMA_CHAT_BASE_URL,
        api_key=config.OLLAMA_API_KEY,
        require_api_key=False,
        max_tokens=4096,
    )


async def _invoke_openai(prompt: str, model: str) -> tuple[str, int]:
    return await _invoke_openai_compatible(
        provider="openai",
        prompt=prompt,
        model=model,
        base_url=config.OPENAI_BASE_URL,
        api_key=config.OPENAI_API_KEY,
        require_api_key=True,
        max_tokens=4096,
    )


async def _invoke_openrouter(prompt: str, model: str) -> tuple[str, int]:
    return await _invoke_openai_compatible(
        provider="openrouter",
        prompt=prompt,
        model=model,
        base_url=config.OPENROUTER_BASE_URL,
        api_key=config.OPENROUTER_API_KEY,
        require_api_key=True,
        max_tokens=4096,
    )


async def _invoke_together(prompt: str, model: str) -> tuple[str, int]:
    return await _invoke_openai_compatible(
        provider="together",
        prompt=prompt,
        model=model,
        base_url=config.TOGETHER_BASE_URL,
        api_key=config.TOGETHER_API_KEY,
        require_api_key=True,
        max_tokens=4096,
    )


async def _invoke_fireworks(prompt: str, model: str) -> tuple[str, int]:
    return await _invoke_openai_compatible(
        provider="fireworks",
        prompt=prompt,
        model=model,
        base_url=config.FIREWORKS_BASE_URL,
        api_key=config.FIREWORKS_API_KEY,
        require_api_key=True,
        max_tokens=4096,
    )


async def _invoke_anthropic(prompt: str, model: str) -> tuple[str, int]:
    import httpx

    if not config.ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    body = {
        "model": model,
        "max_tokens": 8192,
        "temperature": 0.1,
        "system": SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    }

    started = time.monotonic()
    async with httpx.AsyncClient(timeout=240) as client:
        for attempt in range(3):
            try:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": config.ANTHROPIC_API_KEY,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json=body,
                )
                if resp.status_code >= 400:
                    log.warning("Anthropic audit HTTP %s: %s", resp.status_code, resp.text[:500])
                resp.raise_for_status()
                data = resp.json()
                blocks = data.get("content", [])
                text_parts: list[str] = []
                if isinstance(blocks, list):
                    for block in blocks:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(str(block.get("text") or ""))
                content = "\n".join(text_parts).strip()
                latency_ms = int((time.monotonic() - started) * 1000)
                return content, latency_ms
            except Exception as exc:
                if attempt == 2:
                    raise
                await asyncio.sleep(3 * (attempt + 1))
                log.debug("Anthropic audit retry %d: %s", attempt + 1, exc)

    return "", int((time.monotonic() - started) * 1000)


async def call_audit_model(prompt: str, schedule: str, model_override: str | None = None) -> dict[str, Any]:
    """Call audit model using the configured provider tier chain.

    Tier order is driven by ``MOLLYGRAPH_AUDIT_PROVIDER_TIERS`` (default:
    deterministic,local,primary,fallback).  The "deterministic" tier is a
    no-op here (handled by ``run_llm_audit`` before this is called).
    Local/primary/fallback map to their respective provider/model env vars.

    Backwards compat: if ``AUDIT_PROVIDER_ORDER`` is set to a non-"none"
    value and ``MOLLYGRAPH_AUDIT_TIER_PRIMARY`` is empty, the legacy value
    becomes the primary-tier provider.
    """
    normalized = schedule.strip().lower()

    def _default_model_for(provider: str) -> str:
        """Best-effort default model when per-tier model is unset."""
        schedule_model = config.AUDIT_MODEL_WEEKLY if normalized == "weekly" else config.AUDIT_MODEL_NIGHTLY
        provider_fallbacks: dict[str, str] = {
            "gemini": "gemini-2.0-flash",
            "moonshot": "kimi-k2.5",
            "kimi": "kimi-k2.5",
            "groq": "llama-3.3-70b-versatile",
            "anthropic": "claude-3-5-sonnet-latest",
            "opus": "claude-opus-4-5",
            "ollama": "llama3.1:8b",
            "local": "llama3.1:8b",
            "openai": "gpt-4o-mini",
            "openrouter": "meta-llama/llama-3.1-70b-instruct",
            "together": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "fireworks": "accounts/fireworks/models/llama-v3p1-70b-instruct",
        }
        return str(schedule_model or provider_fallbacks.get(provider, "llama3.1:8b")).strip()

    async def _run_provider(provider: str, model: str) -> tuple[str, int, str]:
        if provider in {"gemini"}:
            content, latency_ms = await _invoke_gemini(prompt, model)
            return content, latency_ms, "gemini"
        if provider in {"moonshot", "kimi"}:
            content, latency_ms = await _invoke_moonshot(prompt, model)
            return content, latency_ms, "moonshot"
        if provider in {"groq"}:
            content, latency_ms = await _invoke_groq(prompt, model)
            return content, latency_ms, "groq"
        if provider in {"anthropic", "opus"}:
            content, latency_ms = await _invoke_anthropic(prompt, model)
            return content, latency_ms, "anthropic"
        if provider in {"ollama", "local"}:
            content, latency_ms = await _invoke_ollama(prompt, model)
            return content, latency_ms, "ollama"
        if provider in {"openai"}:
            content, latency_ms = await _invoke_openai(prompt, model)
            return content, latency_ms, "openai"
        if provider in {"openrouter"}:
            content, latency_ms = await _invoke_openrouter(prompt, model)
            return content, latency_ms, "openrouter"
        if provider in {"together"}:
            content, latency_ms = await _invoke_together(prompt, model)
            return content, latency_ms, "together"
        if provider in {"fireworks"}:
            content, latency_ms = await _invoke_fireworks(prompt, model)
            return content, latency_ms, "fireworks"
        raise RuntimeError(f"Unsupported audit provider: {provider}")

    def _resolve_provider_from_override(raw: str) -> str:
        lowered = raw.lower()
        if "ollama" in lowered or lowered.startswith("local:"):
            return "ollama"
        if "moonshot" in lowered or "kimi" in lowered:
            return "moonshot"
        if "groq" in lowered:
            return "groq"
        if "claude" in lowered or "anthropic" in lowered or "opus" in lowered:
            return "anthropic"
        if "gemini" in lowered:
            return "gemini"
        if "gpt" in lowered or "openai" in lowered:
            return "openai"
        if "openrouter" in lowered:
            return "openrouter"
        if "together" in lowered:
            return "together"
        if "fireworks" in lowered:
            return "fireworks"
        return ""

    # ── Guard: LLM disabled ───────────────────────────────────────────────
    if not config.AUDIT_LLM_ENABLED and not model_override:
        return {
            "provider": "disabled",
            "model": "",
            "latency_ms": 0,
            "content": "",
            "fallback": "",
            "skipped": True,
        }

    # ── model_override: bypass tier chain entirely ────────────────────────
    if model_override:
        raw_override = model_override.strip()
        provider = ""
        model = raw_override

        # Accept "provider/model" notation
        if "/" in raw_override:
            maybe_provider, maybe_model = raw_override.split("/", 1)
            if maybe_provider.strip().lower() in {
                "gemini", "moonshot", "kimi", "groq", "anthropic", "opus",
                "ollama", "local", "openai", "openrouter", "together", "fireworks",
            }:
                provider = maybe_provider.strip().lower()
                model = maybe_model.strip()

        if not provider:
            provider = _resolve_provider_from_override(raw_override)

        if not provider or provider in {"none", "disabled"}:
            return {
                "provider": "disabled",
                "model": "",
                "latency_ms": 0,
                "content": "",
                "fallback": "",
                "skipped": True,
            }

        content, latency_ms, resolved_provider = await _run_provider(provider, model)
        return {
            "provider": resolved_provider,
            "model": model,
            "latency_ms": latency_ms,
            "content": content,
            "fallback": "",
            "skipped": False,
            "tier": "override",
        }

    # ── Tier chain ────────────────────────────────────────────────────────
    # Backwards compat: if AUDIT_PROVIDER_ORDER is set to a specific provider
    # and MOLLYGRAPH_AUDIT_TIER_PRIMARY is unset, use the legacy value as primary.
    tier_primary_provider = (getattr(config, "AUDIT_TIER_PRIMARY", "") or "").strip().lower()
    if not tier_primary_provider:
        legacy_order = str(getattr(config, "AUDIT_PROVIDER_ORDER", "none")).strip().lower()
        if legacy_order and legacy_order not in {"none", "disabled"}:
            tier_primary_provider = legacy_order.split(",")[0].strip()
            log.debug("Audit tier 'primary' resolved from legacy AUDIT_PROVIDER_ORDER: %s", tier_primary_provider)

    tier_local_provider    = (getattr(config, "AUDIT_TIER_LOCAL",    "") or "ollama").strip().lower()
    tier_fallback_provider = (getattr(config, "AUDIT_TIER_FALLBACK", "") or "").strip().lower()

    # Model per tier (fall back to schedule default if unset)
    tier_local_model    = (getattr(config, "AUDIT_MODEL_LOCAL",    "") or "").strip() or _default_model_for(tier_local_provider)
    tier_primary_model  = (getattr(config, "AUDIT_MODEL_PRIMARY",  "") or "").strip() or (
        _default_model_for(tier_primary_provider) if tier_primary_provider else ""
    )
    tier_fallback_model = (getattr(config, "AUDIT_MODEL_FALLBACK", "") or "").strip() or (
        _default_model_for(tier_fallback_provider) if tier_fallback_provider else ""
    )

    tier_configs: dict[str, tuple[str, str]] = {
        "local":    (tier_local_provider,    tier_local_model),
        "primary":  (tier_primary_provider,  tier_primary_model),
        "fallback": (tier_fallback_provider, tier_fallback_model),
    }

    provider_tiers: list[str] = [
        t.strip().lower()
        for t in getattr(config, "AUDIT_PROVIDER_TIERS", ["deterministic", "local", "primary", "fallback"])
        if t.strip()
    ]

    tried: list[str] = []
    errors: list[str] = []

    for tier in provider_tiers:
        if tier in {"deterministic", ""}:
            # Deterministic checks are always done in run_llm_audit; skip here.
            continue

        entry = tier_configs.get(tier)
        if entry is None:
            log.warning("Unknown audit tier %r — skipping", tier)
            continue

        provider, model = entry
        if not provider:
            log.debug("Audit tier %r: provider not configured — skipping", tier)
            continue
        if provider in {"none", "disabled"}:
            continue

        try:
            content, latency_ms, resolved_provider = await _run_provider(provider, model)
            log.info("Audit tier %r succeeded (provider=%s model=%s)", tier, resolved_provider, model)
            return {
                "provider": resolved_provider,
                "model": model,
                "latency_ms": latency_ms,
                "content": content,
                "fallback": " -> ".join(tried) if tried else "",
                "skipped": False,
                "tier": tier,
            }
        except Exception as exc:
            err_str = f"{tier}/{provider}: {exc}"
            errors.append(err_str)
            tried.append(f"{tier}/{provider}")
            log.warning("Audit tier %r (%s) failed: %s — trying next tier", tier, provider, exc)

    # All LLM tiers exhausted
    if tried:
        raise RuntimeError(f"All audit tiers failed: {' | '.join(errors)}")

    # No LLM tiers were configured (only deterministic) — return skipped
    return {
        "provider": "deterministic",
        "model": "",
        "latency_ms": 0,
        "content": "",
        "fallback": "",
        "skipped": True,
    }


async def apply_verdicts(
    rels: list[dict[str, Any]],
    verdicts: list[dict[str, Any]],
    dry_run: bool,
    max_auto_deletes: int | None = None,
    deletes_so_far: int = 0,
) -> dict[str, int]:
    """Apply LLM audit verdicts to the graph.

    Args:
        rels: Relationship records that were audited.
        verdicts: LLM verdict list (index-aligned with rels).
        dry_run: When True, log only — do not mutate the graph.
        max_auto_deletes: Cap on total delete actions for this run.  None = no cap.
        deletes_so_far: Running delete count already applied in previous batches.
    """
    graph = require_graph_instance()
    auto_fixed = 0
    quarantined = 0
    deleted = 0
    verified = 0

    now_iso = datetime.now(timezone.utc).isoformat()

    for verdict in verdicts:
        idx = int(verdict["index"]) - 1
        if not (0 <= idx < len(rels)):
            continue

        rel = rels[idx]
        decision = verdict["verdict"]
        suggested_type = verdict.get("suggested_type", "")

        # Grab first context snippet for training signals.
        snippets = rel.get("context_snippets") or []
        context = snippets[0] if snippets else ""

        if decision == "verify":
            verified += 1
            if not dry_run:
                graph.set_relationship_audit_status(rel["head"], rel["tail"], rel["rel_type"], "verified")
            _write_audit_signal({
                "timestamp": now_iso,
                "signal": "positive",
                "head": rel["head"],
                "tail": rel["tail"],
                "relation": rel["rel_type"],
                "context": context,
                "source": "verify",
            })
            get_signal_bus().publish("relationship_verified", {
                "head": rel["head"],
                "tail": rel["tail"],
                "rel_type": rel["rel_type"],
                "dry_run": dry_run,
            })
            continue

        if decision == "reclassify":
            if suggested_type in VALID_REL_TYPES:
                auto_fixed += 1
                # Old label is a hard negative; new label is a positive.
                _write_audit_signal({
                    "timestamp": now_iso,
                    "signal": "negative",
                    "head": rel["head"],
                    "tail": rel["tail"],
                    "relation": rel["rel_type"],
                    "context": context,
                    "source": "reclassify",
                    "corrected_to": suggested_type,
                })
                _write_audit_signal({
                    "timestamp": now_iso,
                    "signal": "positive",
                    "head": rel["head"],
                    "tail": rel["tail"],
                    "relation": suggested_type,
                    "context": context,
                    "source": "reclassify",
                })
                if not dry_run:
                    graph.reclassify_relationship(
                        rel["head"],
                        rel["tail"],
                        rel["rel_type"],
                        suggested_type,
                        float(rel.get("strength") or 0.5),
                        int(rel.get("mention_count") or 1),
                        rel.get("context_snippets") if isinstance(rel.get("context_snippets"), list) else [],
                        str(rel.get("first_mentioned") or "") or None,
                    )
                    # Mark the new relationship as verified + stamp last_audited_at
                    graph.set_relationship_audit_status(rel["head"], rel["tail"], suggested_type, "verified")
                get_signal_bus().publish("relationship_reclassified", {
                    "head": rel["head"],
                    "tail": rel["tail"],
                    "old_type": rel["rel_type"],
                    "new_type": suggested_type,
                    "dry_run": dry_run,
                })
            else:
                quarantined += 1
                _write_audit_signal({
                    "timestamp": now_iso,
                    "signal": "negative",
                    "head": rel["head"],
                    "tail": rel["tail"],
                    "relation": rel["rel_type"],
                    "context": context,
                    "source": "quarantine",
                })
                if not dry_run:
                    graph.set_relationship_audit_status(rel["head"], rel["tail"], rel["rel_type"], "quarantined")
                get_signal_bus().publish("relationship_quarantined", {
                    "head": rel["head"],
                    "tail": rel["tail"],
                    "rel_type": rel["rel_type"],
                    "reason": "reclassify_invalid_type",
                    "suggested_type": suggested_type,
                    "dry_run": dry_run,
                })
            continue

        if decision == "quarantine":
            quarantined += 1
            _write_audit_signal({
                "timestamp": now_iso,
                "signal": "negative",
                "head": rel["head"],
                "tail": rel["tail"],
                "relation": rel["rel_type"],
                "context": context,
                "source": "quarantine",
            })
            if not dry_run:
                graph.set_relationship_audit_status(rel["head"], rel["tail"], rel["rel_type"], "quarantined")
            get_signal_bus().publish("relationship_quarantined", {
                "head": rel["head"],
                "tail": rel["tail"],
                "rel_type": rel["rel_type"],
                "reason": "audit_quarantine",
                "dry_run": dry_run,
            })
            continue

        if decision == "delete":
            # Enforce blast-radius cap: skip delete if the cap has been reached.
            if max_auto_deletes is not None and (deletes_so_far + deleted) >= max_auto_deletes:
                log.warning(
                    "audit auto-delete cap reached (%d/%d) — skipping delete of %s -[%s]-> %s",
                    deletes_so_far + deleted,
                    max_auto_deletes,
                    rel["head"],
                    rel["rel_type"],
                    rel["tail"],
                )
                continue
            deleted += 1
            _write_audit_signal({
                "timestamp": now_iso,
                "signal": "negative",
                "head": rel["head"],
                "tail": rel["tail"],
                "relation": rel["rel_type"],
                "context": context,
                "source": "delete",
            })
            if not dry_run:
                graph.delete_specific_relationship(rel["head"], rel["tail"], rel["rel_type"])
            get_signal_bus().publish("relationship_removed", {
                "head": rel["head"],
                "tail": rel["tail"],
                "rel_type": rel["rel_type"],
                "dry_run": dry_run,
            })

    return {
        "auto_fixed": auto_fixed,
        "quarantined": quarantined,
        "deleted": deleted,
        "verified": verified,
    }


def _chunked(rows: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for idx in range(0, len(rows), size):
        yield rows[idx : idx + size]


def _write_maintenance_report(schedule: str, result: dict[str, Any]) -> str:
    config.MAINTENANCE_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    filename = f"audit-{schedule}-{now.strftime('%Y%m%d-%H%M%S-%f')}.md"
    path = config.MAINTENANCE_DIR / filename

    lines = [
        f"# Graph Audit Report ({schedule})",
        "",
        f"- Generated: {now.isoformat()}",
        f"- Model: {result.get('audit_model_model', '')}",
        f"- Provider: {result.get('audit_provider', '')}",
        f"- Relationships scanned: {result.get('relationships_scanned', 0)}",
        f"- Verified: {result.get('verified', 0)}",
        f"- Auto-fixed: {result.get('auto_fixed', 0)}",
        f"- Quarantined: {result.get('quarantined', 0)}",
        f"- Deleted: {result.get('deleted', 0)}",
        f"- Feedback rows written: {result.get('feedback_written', 0)}",
        f"- Feedback positives: {result.get('feedback_positive_labels', 0)}",
        f"- Feedback negatives: {result.get('feedback_negative_labels', 0)}",
        f"- Self-refs deleted: {result.get('self_refs_deleted', 0)}",
        f"- Orphans deleted: {result.get('orphans_deleted', 0)}",
        f"- Strength decay updated: {result.get('strength_decay_updated', 0)}",
        "",
        "## Coverage",
        "",
        f"- Total relationships: {result.get('total_relationships', 0)}",
        f"- Unaudited count: {result.get('unaudited_count', 0)}",
        f"- Flagged count: {result.get('flagged_count', 0)}",
        f"- Coverage: {result.get('coverage_pct', 0.0)}%",
        f"- Last full sweep: {result.get('last_full_sweep', 'never')}",
        "",
        "## Summary",
        "",
        str(result.get("summary", "")),
        "",
    ]

    digest = str(result.get("suggestion_digest") or "")
    if digest:
        lines.extend(["## Suggestions", "", digest, ""])

    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


async def run_llm_audit(
    limit: int = 500,
    dry_run: bool = False,
    schedule: str = "nightly",
    model_override: str | None = None,
) -> dict[str, Any]:
    """Run cleanup + optional LLM audit + suggestion tracking."""
    started = time.monotonic()
    graph = require_graph_instance()

    self_refs_deleted = graph.delete_self_referencing_rels()
    orphans_deleted = graph.delete_orphan_entities_sync(
        vector_store=get_vector_store_instance()
    )
    strength_decay_updated = graph.run_strength_decay_sync()

    rels = graph.get_relationships_for_audit(limit=max(1, int(limit)), schedule=schedule)

    auto_fixed = 0
    quarantined = 0
    deleted = 0
    verified = 0
    relationships_reviewed = 0  # total that received an LLM verdict
    batches = 0
    parse_failures = 0
    total_latency_ms = 0
    chosen_model = ""
    chosen_provider = ""
    fallback_note = ""
    llm_enabled = bool(config.AUDIT_LLM_ENABLED or model_override)
    feedback_written = 0
    feedback_positive_labels = 0
    feedback_negative_labels = 0
    feedback_file = ""

    if rels:
        for batch in _chunked(rels, _BATCH_SIZE):
            batches += 1
            prompt = build_audit_prompt(batch)
            try:
                llm_result = await call_audit_model(prompt, schedule=schedule, model_override=model_override)
                chosen_model = str(llm_result.get("model") or chosen_model)
                chosen_provider = str(llm_result.get("provider") or chosen_provider)
                fallback_note = str(llm_result.get("fallback") or fallback_note)
                total_latency_ms += int(llm_result.get("latency_ms") or 0)
                if bool(llm_result.get("skipped")):
                    continue

                verdicts = parse_verdicts(str(llm_result.get("content") or ""), len(batch))
                if not verdicts:
                    parse_failures += 1
                    continue

                relationships_reviewed += len(verdicts)

                # Determine delete cap: use config value, honour dry_run (no cap in dry runs).
                max_deletes_cap: int | None = None
                if not dry_run:
                    # Percentage-based cap: 5% of reviewed, clamped to [min, max]
                    pct_cap = int(len(rels) * config.AUDIT_AUTO_DELETE_PCT) if rels else config.AUDIT_AUTO_DELETE_MIN
                    max_deletes_cap = max(config.AUDIT_AUTO_DELETE_MIN, min(pct_cap, config.AUDIT_AUTO_DELETE_MAX))

                # Check if we have already hit the cap before applying this batch.
                if max_deletes_cap is not None and deleted >= max_deletes_cap:
                    log.warning(
                        "audit auto-delete cap (%d) already reached — skipping delete verdicts for this batch",
                        max_deletes_cap,
                    )
                    # Still apply non-delete verdicts (verify, reclassify, quarantine).
                    verdicts_no_delete = [v for v in verdicts if v.get("verdict") != "delete"]
                    outcome = await apply_verdicts(
                        batch, verdicts_no_delete, dry_run=dry_run,
                        max_auto_deletes=max_deletes_cap, deletes_so_far=deleted,
                    )
                else:
                    outcome = await apply_verdicts(
                        batch, verdicts, dry_run=dry_run,
                        max_auto_deletes=max_deletes_cap, deletes_so_far=deleted,
                    )
                auto_fixed += outcome["auto_fixed"]
                quarantined += outcome["quarantined"]
                deleted += outcome["deleted"]
                verified += outcome["verified"]

                feedback = record_audit_feedback_batch(
                    batch,
                    verdicts,
                    schedule=schedule,
                    provider=chosen_provider,
                    model=chosen_model,
                    dry_run=dry_run,
                )
                feedback_written += int(feedback.get("written", 0) or 0)
                feedback_positive_labels += int(feedback.get("positive_labels", 0) or 0)
                feedback_negative_labels += int(feedback.get("negative_labels", 0) or 0)
                feedback_file = str(feedback.get("file") or feedback_file)
            except Exception:
                log.error("Audit batch failed", exc_info=True)
                parse_failures += 1

    suggestion_digest = build_suggestion_digest()
    auto_adoption_result = run_auto_adoption()

    # ── Coverage metrics ──────────────────────────────────────────────────
    try:
        coverage = graph.get_audit_coverage_metrics()
    except Exception:
        log.debug("Failed to get audit coverage metrics", exc_info=True)
        coverage = {"total_relationships": 0, "unaudited_count": 0, "flagged_count": 0, "coverage_pct": 0.0}

    # ── Audit state (last_full_sweep) ─────────────────────────────────────
    audit_state = _load_audit_state()
    last_full_sweep = audit_state.get("last_full_sweep", "")
    now_iso = datetime.now(timezone.utc).isoformat()
    if schedule.strip().lower() == "weekly" and not dry_run:
        audit_state["last_full_sweep"] = now_iso
        _save_audit_state(audit_state)
        last_full_sweep = now_iso

    summary = (
        f"Reviewed {len(rels)} relationships: "
        f"{auto_fixed} auto-fixed, {quarantined} quarantined, {verified} verified"
    )
    if not llm_enabled:
        summary = (
            f"Reviewed {len(rels)} relationships in local-only mode: "
            "LLM audit is disabled (set AUDIT_LLM_ENABLED=true to enable provider calls)."
        )

    status = "pass" if parse_failures == 0 else "warn"
    duration_seconds = round(time.monotonic() - started, 3)

    result = {
        "status": status,
        "summary": summary,
        "relationships_scanned": len(rels),
        "relationships_reviewed": relationships_reviewed,
        "relationships_approved": verified,
        "relationships_flagged": quarantined,
        "relationships_reclassified": auto_fixed,
        "auto_fixed": auto_fixed,
        "quarantined": quarantined,
        "deleted": deleted,
        "verified": verified,
        "self_refs_deleted": self_refs_deleted,
        "orphans_deleted": orphans_deleted,
        "strength_decay_updated": strength_decay_updated,
        "audit_model_model": chosen_model or (model_override or (config.AUDIT_MODEL_NIGHTLY if llm_enabled else "")),
        "audit_provider": chosen_provider or ("disabled" if not llm_enabled else "unknown"),
        "audit_model_latency_ms": total_latency_ms,
        "batches": batches,
        "parse_failures": parse_failures,
        "dry_run": bool(dry_run),
        "schedule": schedule,
        "llm_enabled": llm_enabled,
        "fallback": fallback_note,
        "suggestion_digest": suggestion_digest,
        "auto_adoption_result": auto_adoption_result,
        "feedback_written": feedback_written,
        "feedback_positive_labels": feedback_positive_labels,
        "feedback_negative_labels": feedback_negative_labels,
        "feedback_file": feedback_file,
        "duration_seconds": duration_seconds,
        # ── Coverage metrics ─────────────────────────────────────────────
        "total_relationships": coverage["total_relationships"],
        "unaudited_count": coverage["unaudited_count"],
        "coverage_pct": coverage["coverage_pct"],
        "flagged_count": coverage["flagged_count"],
        "last_full_sweep": last_full_sweep,
    }

    report_path = _write_maintenance_report(schedule, result)
    result["report_path"] = report_path
    return result


__all__ = [
    "apply_verdicts",
    "build_audit_prompt",
    "call_audit_model",
    "parse_verdicts",
    "run_llm_audit",
]
