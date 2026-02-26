"""Decision trace auto-detection helpers for ingestion (Phase 2)."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import config
from audit import llm_audit

log = logging.getLogger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "y", "on"}

_DECISION_STRONG_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:i|we)\s+(?:decided|decide|chose|choose|opted|opt)\b", re.IGNORECASE),
    re.compile(r"\blet['’]?s\s+(?:go with|do it|ship it|proceed|move forward)\b", re.IGNORECASE),
    re.compile(r"\b(?:approved|green[- ]?lit|signed off)\b", re.IGNORECASE),
    re.compile(r"\b(?:final decision|decision made|we are going with)\b", re.IGNORECASE),
    re.compile(r"\b(?:instead of|rather than|vs\.?)\b", re.IGNORECASE),
)

_DECISION_WEAK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:should use|should do|will use|we'll use)\b", re.IGNORECASE),
    re.compile(r"\b(?:plan is|moving to|switching to|migrating to)\b", re.IGNORECASE),
    re.compile(r"\b(?:approved by|agreed to|settled on)\b", re.IGNORECASE),
    re.compile(r"\bdecision\b", re.IGNORECASE),
)

_PROMO_NOISE_MARKERS: tuple[str, ...] = (
    "unsubscribe",
    "limited time",
    "buy now",
    "promo code",
    "% off",
    "sponsored",
    "deal alert",
    "click here",
    "free shipping",
)

_DECISION_SYSTEM_PROMPT = (
    "You extract decision traces from ingestion text. Return strict JSON only."
)


@dataclass(frozen=True)
class DecisionPrefilterResult:
    passed: bool
    score: int
    reason: str


@dataclass(frozen=True)
class DecisionPayload:
    decision: str
    reasoning: str
    alternatives: list[str]
    inputs: list[str]
    outcome: str
    decided_by: str
    confidence: float


@dataclass(frozen=True)
class DecisionExtractionResult:
    is_decision: bool
    payload: DecisionPayload | None
    provider: str
    model: str
    tier: str
    latency_ms: int
    error: str = ""


def run_decision_prefilter(
    *,
    content: str,
    source: str,
    raw_source: str | None = None,
) -> DecisionPrefilterResult:
    """Cheap deterministic gate for decision-like text.

    This is intentionally lexical/heuristic so it's fast enough to run on every
    ingest job before any LLM call.
    """
    text = str(content or "").strip()
    if not text:
        return DecisionPrefilterResult(passed=False, score=0, reason="empty_content")

    if is_low_signal_source(source=source, raw_source=raw_source):
        return DecisionPrefilterResult(passed=False, score=0, reason="blocked_source")

    min_chars = int(getattr(config, "DECISION_TRACES_MIN_CONTENT_CHARS", 24))
    if len(text) < min_chars:
        return DecisionPrefilterResult(passed=False, score=0, reason="too_short")

    lowered = text.lower()
    noise_hits = sum(1 for marker in _PROMO_NOISE_MARKERS if marker in lowered)
    if noise_hits >= 2:
        return DecisionPrefilterResult(passed=False, score=0, reason="promo_noise")

    score = 0
    for pattern in _DECISION_STRONG_PATTERNS:
        if pattern.search(text):
            score += 2
    for pattern in _DECISION_WEAK_PATTERNS:
        if pattern.search(text):
            score += 1

    min_score = int(getattr(config, "DECISION_TRACES_PREFILTER_MIN_SCORE", 2))
    if not bool(getattr(config, "DECISION_TRACES_PREFILTER_ENABLED", True)):
        return DecisionPrefilterResult(passed=True, score=max(score, min_score), reason="prefilter_disabled")

    if score < min_score:
        return DecisionPrefilterResult(passed=False, score=score, reason="insufficient_signal")

    return DecisionPrefilterResult(passed=True, score=score, reason="matched")


def is_low_signal_source(*, source: str, raw_source: str | None = None) -> bool:
    blocked_tokens = [
        token for token in getattr(config, "DECISION_TRACES_SOURCE_BLOCKLIST", []) if str(token).strip()
    ]
    if not blocked_tokens:
        return False

    source_norm = str(source or "").strip().lower()
    raw_norm = str(raw_source or "").strip().lower()
    haystacks = [source_norm, raw_norm]

    for blocked in blocked_tokens:
        marker = str(blocked).strip().lower()
        if marker and any(marker in hay for hay in haystacks):
            return True
    return False


def _build_decision_prompt(*, content: str, source: str, speaker: str | None) -> str:
    speaker_text = speaker or ""
    return (
        f"{_DECISION_SYSTEM_PROMPT}\n\n"
        "Classify whether the text includes a concrete decision that was made.\n"
        "If yes, extract a structured payload.\n\n"
        "Rules:\n"
        "- Return JSON object only (no markdown).\n"
        "- If no decision is present, set is_decision=false and keep fields empty.\n"
        "- confidence must be a float between 0 and 1.\n"
        "- alternatives and inputs must be arrays of short strings.\n\n"
        "JSON schema:\n"
        "{\n"
        '  "is_decision": boolean,\n'
        '  "decision": string,\n'
        '  "reasoning": string,\n'
        '  "alternatives": string[],\n'
        '  "inputs": string[],\n'
        '  "outcome": string,\n'
        '  "decided_by": string,\n'
        '  "confidence": number\n'
        "}\n\n"
        f"source: {source}\n"
        f"speaker: {speaker_text}\n"
        "text:\n"
        f"{content}"
    )


def _extract_json_object(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""

    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].startswith("```"):
            text = "\n".join(lines[1:-1]).strip()
        else:
            text = "\n".join(lines[1:]).strip()

    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        return text[start : end + 1]
    return text


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in _TRUE_VALUES


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp_confidence(value: Any) -> float:
    return max(0.0, min(1.0, _safe_float(value, default=0.0)))


def _clean_list(value: Any, *, max_items: int) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in value:
        item = str(raw or "").strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(item[:240])
        if len(cleaned) >= max_items:
            break
    return cleaned


def _parse_decision_payload(
    *,
    raw: str,
    speaker: str | None,
) -> tuple[bool, DecisionPayload | None, str]:
    candidate = _extract_json_object(raw)
    if not candidate:
        return False, None, "empty_response"

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return False, None, "invalid_json"

    if not isinstance(data, dict):
        return False, None, "invalid_shape"

    is_decision = _normalize_bool(data.get("is_decision"))
    if not is_decision:
        return False, None, ""

    decision = str(data.get("decision") or "").strip()
    if not decision:
        return False, None, "missing_decision"

    alternatives = _clean_list(
        data.get("alternatives"),
        max_items=max(1, int(getattr(config, "DECISION_TRACES_MAX_ALTERNATIVES", 6))),
    )
    inputs = _clean_list(
        data.get("inputs"),
        max_items=max(1, int(getattr(config, "DECISION_TRACES_MAX_INPUTS", 8))),
    )
    decided_by = str(data.get("decided_by") or "").strip() or str(speaker or "").strip() or "Unknown"

    payload = DecisionPayload(
        decision=decision[:400],
        reasoning=str(data.get("reasoning") or "").strip()[:800],
        alternatives=alternatives,
        inputs=inputs,
        outcome=str(data.get("outcome") or "").strip()[:400],
        decided_by=decided_by[:120],
        confidence=_clamp_confidence(data.get("confidence")),
    )
    return True, payload, ""


async def _invoke_provider(*, provider: str, model: str, prompt: str) -> tuple[str, int, str]:
    normalized = str(provider or "").strip().lower()
    if normalized in {"gemini"}:
        content, latency_ms = await llm_audit._invoke_gemini(prompt, model)
        return content, latency_ms, "gemini"
    if normalized in {"moonshot", "kimi"}:
        content, latency_ms = await llm_audit._invoke_moonshot(prompt, model)
        return content, latency_ms, "moonshot"
    if normalized in {"groq"}:
        content, latency_ms = await llm_audit._invoke_groq(prompt, model)
        return content, latency_ms, "groq"
    if normalized in {"anthropic", "opus"}:
        content, latency_ms = await llm_audit._invoke_anthropic(prompt, model)
        return content, latency_ms, "anthropic"
    if normalized in {"ollama", "local"}:
        content, latency_ms = await llm_audit._invoke_ollama(prompt, model)
        return content, latency_ms, "ollama"
    if normalized in {"openai"}:
        content, latency_ms = await llm_audit._invoke_openai(prompt, model)
        return content, latency_ms, "openai"
    if normalized in {"openrouter"}:
        content, latency_ms = await llm_audit._invoke_openrouter(prompt, model)
        return content, latency_ms, "openrouter"
    if normalized in {"together"}:
        content, latency_ms = await llm_audit._invoke_together(prompt, model)
        return content, latency_ms, "together"
    if normalized in {"fireworks"}:
        content, latency_ms = await llm_audit._invoke_fireworks(prompt, model)
        return content, latency_ms, "fireworks"
    raise RuntimeError(f"Unsupported decision-trace provider: {provider}")


def _provider_chain() -> list[tuple[str, str, str]]:
    primary_provider = str(getattr(config, "DECISION_TRACES_TIER_PRIMARY", "")).strip().lower()
    primary_model = str(getattr(config, "DECISION_TRACES_MODEL_PRIMARY", "")).strip()
    fallback_provider = str(getattr(config, "DECISION_TRACES_TIER_FALLBACK", "")).strip().lower()
    fallback_model = str(getattr(config, "DECISION_TRACES_MODEL_FALLBACK", "")).strip()

    chain: list[tuple[str, str, str]] = []
    if primary_provider and primary_provider not in {"none", "disabled"}:
        chain.append(("primary", primary_provider, primary_model))
    if (
        fallback_provider
        and fallback_provider not in {"none", "disabled"}
        and (fallback_provider, fallback_model) != (primary_provider, primary_model)
    ):
        chain.append(("fallback", fallback_provider, fallback_model))
    return chain


async def extract_decision_trace(
    *,
    content: str,
    source: str,
    speaker: str | None = None,
) -> DecisionExtractionResult:
    """Run primary/fallback LLM extraction for decision traces."""
    chain = _provider_chain()
    if not chain:
        return DecisionExtractionResult(
            is_decision=False,
            payload=None,
            provider="",
            model="",
            tier="",
            latency_ms=0,
            error="no_provider_configured",
        )

    prompt = _build_decision_prompt(content=content, source=source, speaker=speaker)
    errors: list[str] = []

    for tier, provider, model in chain:
        try:
            raw, latency_ms, resolved_provider = await _invoke_provider(
                provider=provider,
                model=model,
                prompt=prompt,
            )
        except Exception as exc:
            errors.append(f"{tier}/{provider}: {exc}")
            continue

        is_decision, payload, parse_error = _parse_decision_payload(raw=raw, speaker=speaker)
        if parse_error:
            errors.append(f"{tier}/{provider}: {parse_error}")
            continue

        return DecisionExtractionResult(
            is_decision=is_decision,
            payload=payload,
            provider=resolved_provider,
            model=model,
            tier=tier,
            latency_ms=latency_ms,
            error="",
        )

    error_text = " | ".join(errors) if errors else "llm_unavailable"
    log.debug("Decision trace extraction failed: %s", error_text)
    return DecisionExtractionResult(
        is_decision=False,
        payload=None,
        provider="",
        model="",
        tier="",
        latency_ms=0,
        error=error_text,
    )


__all__ = [
    "DecisionExtractionResult",
    "DecisionPayload",
    "DecisionPrefilterResult",
    "extract_decision_trace",
    "is_low_signal_source",
    "run_decision_prefilter",
]
