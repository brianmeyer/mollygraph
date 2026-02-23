"""Relationship audit pipeline.

Local-only mode is the default. LLM-backed audit is optional and can be
enabled with provider/model configuration (including local Ollama models).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Iterable

import config
from evolution.audit_feedback import record_audit_feedback_batch
from memory.bitemporal_graph import VALID_REL_TYPES
from memory.graph_suggestions import build_suggestion_digest, run_auto_adoption
from runtime_graph import require_graph_instance

log = logging.getLogger(__name__)

_BATCH_SIZE = 500


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
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].startswith("```"):
            text = "\n".join(lines[1:-1]).strip()
        else:
            text = "\n".join(lines[1:]).strip()

    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        return text[start : end + 1]
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

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
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
    """Call audit model using configured provider chain."""
    normalized = schedule.strip().lower()
    order = [p.strip().lower() for p in str(config.AUDIT_PROVIDER_ORDER).split(",") if p.strip()]
    if not order:
        order = ["none"]

    def _default_model_for(provider: str) -> str:
        default_model = config.AUDIT_MODEL_WEEKLY if normalized == "weekly" else config.AUDIT_MODEL_NIGHTLY
        gemini_model = (config.AUDIT_MODEL_WEEKLY if normalized == "weekly" else config.AUDIT_MODEL_PRIMARY) or default_model
        provider_to_model = {
            "gemini": gemini_model,
            "moonshot": config.AUDIT_MODEL_SECONDARY or "kimi-k2.5",
            "kimi": config.AUDIT_MODEL_SECONDARY or "kimi-k2.5",
            "groq": config.AUDIT_MODEL_TERTIARY or "gpt-oss-120b",
            "anthropic": config.AUDIT_MODEL_WEEKLY or "claude-3-5-sonnet-latest",
            "opus": config.AUDIT_MODEL_WEEKLY or "claude-3-opus-latest",
            "ollama": config.AUDIT_MODEL_LOCAL or default_model,
            "local": config.AUDIT_MODEL_LOCAL or default_model,
        }
        return str(provider_to_model.get(provider) or default_model).strip()

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
        raise RuntimeError(f"Unsupported audit provider: {provider}")

    if not config.AUDIT_LLM_ENABLED and not model_override:
        return {
            "provider": "disabled",
            "model": "",
            "latency_ms": 0,
            "content": "",
            "fallback": "",
            "skipped": True,
        }

    if model_override:
        raw_override = model_override.strip()
        lowered = raw_override.lower()
        provider = ""
        model = raw_override
        if "/" in raw_override:
            maybe_provider, maybe_model = raw_override.split("/", 1)
            if maybe_provider.strip().lower() in {"gemini", "moonshot", "kimi", "groq", "anthropic", "opus", "ollama", "local"}:
                provider = maybe_provider.strip().lower()
                model = maybe_model.strip()

        if not provider:
            if "ollama" in lowered or lowered.startswith("local:"):
                provider = "ollama"
            elif "moonshot" in lowered or "kimi" in lowered:
                provider = "moonshot"
            elif "groq" in lowered:
                provider = "groq"
            elif "claude" in lowered or "anthropic" in lowered or "opus" in lowered:
                provider = "anthropic"
            elif "gemini" in lowered:
                provider = "gemini"
            else:
                provider = next((p for p in order if p not in {"none", "disabled"}), "ollama")

        if provider in {"none", "disabled"}:
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
        }

    errors: list[str] = []
    for idx, provider in enumerate(order):
        if provider in {"none", "disabled"}:
            return {
                "provider": "disabled",
                "model": "",
                "latency_ms": 0,
                "content": "",
                "fallback": " -> ".join(order[:idx]) if idx > 0 else "",
                "skipped": True,
            }

        model = _default_model_for(provider)
        try:
            content, latency_ms, resolved_provider = await _run_provider(provider, model)
            return {
                "provider": resolved_provider,
                "model": model,
                "latency_ms": latency_ms,
                "content": content,
                "fallback": " -> ".join(order[:idx]) if idx > 0 else "",
                "skipped": False,
            }
        except Exception as exc:
            errors.append(f"{provider}:{exc}")
            continue

    raise RuntimeError(f"All audit providers failed: {' | '.join(errors)}")


async def apply_verdicts(
    rels: list[dict[str, Any]],
    verdicts: list[dict[str, Any]],
    dry_run: bool,
) -> dict[str, int]:
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
            continue

        if decision == "delete":
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
    orphans_deleted = graph.delete_orphan_entities_sync()
    strength_decay_updated = graph.run_strength_decay_sync()

    rels = graph.get_relationships_for_audit(limit=max(1, int(limit)))

    auto_fixed = 0
    quarantined = 0
    deleted = 0
    verified = 0
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

                outcome = await apply_verdicts(batch, verdicts, dry_run=dry_run)
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
        "auto_fixed": auto_fixed,
        "quarantined": quarantined,
        "deleted": deleted,
        "verified": verified,
        "self_refs_deleted": self_refs_deleted,
        "orphans_deleted": orphans_deleted,
        "strength_decay_updated": strength_decay_updated,
        "audit_model_model": chosen_model or (model_override or (config.AUDIT_MODEL_LOCAL if llm_enabled else "")),
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
