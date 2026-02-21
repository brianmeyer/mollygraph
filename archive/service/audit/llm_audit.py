"""LLM-powered relationship audit pipeline.

Nightly schedule (Mon-Sat):
- Kimi (Moonshot) for quality audit before LoRA training
- Falls back to Gemini if Kimi unavailable

Weekly schedule (Sunday):
- Opus (Anthropic) for deep audit before full training
- Falls back to Gemini if Anthropic key is absent.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Iterable

import config
from memory.graph import (
    VALID_REL_TYPES,
    delete_orphan_entities_sync,
    delete_self_referencing_rels,
    delete_specific_relationship,
    get_relationships_for_audit,
    reclassify_relationship,
    run_strength_decay_sync,
    set_relationship_audit_status,
)
from memory.graph_suggestions import build_suggestion_digest, run_auto_adoption

log = logging.getLogger(__name__)

_BATCH_SIZE = 500


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


async def _invoke_gemini(prompt: str, model: str) -> tuple[str, int]:
    import httpx

    if not config.GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY is not set")

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 8192,
    }

    started = time.monotonic()
    async with httpx.AsyncClient(timeout=180) as client:
        for attempt in range(3):
            try:
                resp = await client.post(
                    f"{config.GEMINI_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {config.GOOGLE_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                )
                if resp.status_code >= 400:
                    log.warning("Gemini audit HTTP %s: %s", resp.status_code, resp.text[:500])
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                latency_ms = int((time.monotonic() - started) * 1000)
                return str(content), latency_ms
            except Exception as exc:
                if attempt == 2:
                    raise
                await asyncio.sleep(2 * (attempt + 1))
                log.debug("Gemini audit retry %d: %s", attempt + 1, exc)

    return "", int((time.monotonic() - started) * 1000)


async def _invoke_kimi(prompt: str, model: str) -> tuple[str, int]:
    """Invoke Kimi via OpenClaw Gateway or Moonshot API."""
    import httpx

    # Try OpenClaw Gateway first (preferred for integrated deployment)
    gateway_url = os.environ.get("OPENCLAW_GATEWAY_URL", "ws://127.0.0.1:18789")
    gateway_token = os.environ.get("OPENCLAW_GATEWAY_TOKEN", "")

    # Convert ws:// to http:// for API calls
    http_url = gateway_url.replace("ws://", "http://").replace("wss://", "https://")

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 8192,
    }

    started = time.monotonic()
    async with httpx.AsyncClient(timeout=180) as client:
        for attempt in range(3):
            try:
                # Try OpenClaw Gateway agent endpoint
                headers = {"Content-Type": "application/json"}
                if gateway_token:
                    headers["Authorization"] = f"Bearer {gateway_token}"

                resp = await client.post(
                    f"{http_url}/agent/turn",
                    headers=headers,
                    json=body,
                )
                if resp.status_code >= 400:
                    log.warning("Kimi audit HTTP %s: %s", resp.status_code, resp.text[:500])
                resp.raise_for_status()
                data = resp.json()
                content = data.get("content") or data.get("message", {}).get("content", "")
                latency_ms = int((time.monotonic() - started) * 1000)
                return str(content), latency_ms
            except Exception as exc:
                if attempt == 2:
                    # Fallback to Gemini if Kimi unavailable
                    log.warning("Kimi audit failed after 3 attempts, falling back to Gemini: %s", exc)
                    raise
                await asyncio.sleep(2 * (attempt + 1))
                log.debug("Kimi audit retry %d: %s", attempt + 1, exc)

    return "", int((time.monotonic() - started) * 1000)


async def _invoke_opus(prompt: str, model: str) -> tuple[str, int]:
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
                    log.warning("Opus audit HTTP %s: %s", resp.status_code, resp.text[:500])
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
                log.debug("Opus audit retry %d: %s", attempt + 1, exc)

    return "", int((time.monotonic() - started) * 1000)


async def call_audit_model(prompt: str, schedule: str, model_override: str | None = None) -> dict[str, Any]:
    """Call nightly/weekly audit model with fallback behavior."""
    normalized = schedule.strip().lower()

    if normalized == "weekly":
        # Weekly: Opus deep audit before full training
        target_model = model_override or config.AUDIT_MODEL_WEEKLY
        if config.ANTHROPIC_API_KEY:
            content, latency_ms = await _invoke_opus(prompt, target_model)
            return {
                "provider": "anthropic",
                "model": target_model,
                "latency_ms": latency_ms,
                "content": content,
                "fallback": "",
            }
        # Fallback to Gemini if Opus unavailable
        fallback_model = config.AUDIT_MODEL_NIGHTLY.replace("kimi", "gemini")
        content, latency_ms = await _invoke_gemini(prompt, fallback_model)
        return {
            "provider": "gemini",
            "model": fallback_model,
            "latency_ms": latency_ms,
            "content": content,
            "fallback": "weekly_opus_unavailable_fallback_to_gemini",
        }

    # Nightly: Kimi audit before LoRA training
    target_model = model_override or config.AUDIT_MODEL_NIGHTLY
    if "kimi" in target_model.lower():
        try:
            content, latency_ms = await _invoke_kimi(prompt, target_model)
            return {
                "provider": "moonshot",
                "model": target_model,
                "latency_ms": latency_ms,
                "content": content,
                "fallback": "",
            }
        except Exception:
            # Fallback to Gemini if Kimi unavailable
            log.warning("Kimi unavailable, falling back to Gemini for nightly audit")
            target_model = "gemini-2.5-flash"

        # Weekly path requested but no Anthropic key; fallback to nightly Gemini.
        fallback_model = config.AUDIT_MODEL_NIGHTLY
        content, latency_ms = await _invoke_gemini(prompt, fallback_model)
        return {
            "provider": "gemini",
            "model": fallback_model,
            "latency_ms": latency_ms,
            "content": content,
            "fallback": "weekly_opus_unavailable_fallback_to_gemini",
        }

    target_model = model_override or config.AUDIT_MODEL_NIGHTLY
    content, latency_ms = await _invoke_gemini(prompt, target_model)
    return {
        "provider": "gemini",
        "model": target_model,
        "latency_ms": latency_ms,
        "content": content,
        "fallback": "",
    }


async def apply_verdicts(
    rels: list[dict[str, Any]],
    verdicts: list[dict[str, Any]],
    dry_run: bool,
) -> dict[str, int]:
    auto_fixed = 0
    quarantined = 0
    deleted = 0
    verified = 0

    for verdict in verdicts:
        idx = int(verdict["index"]) - 1
        if not (0 <= idx < len(rels)):
            continue

        rel = rels[idx]
        decision = verdict["verdict"]
        suggested_type = verdict.get("suggested_type", "")

        if decision == "verify":
            verified += 1
            if not dry_run:
                set_relationship_audit_status(rel["head"], rel["tail"], rel["rel_type"], "verified")
            continue

        if decision == "reclassify":
            if suggested_type in VALID_REL_TYPES:
                auto_fixed += 1
                if not dry_run:
                    reclassify_relationship(
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
                if not dry_run:
                    set_relationship_audit_status(rel["head"], rel["tail"], rel["rel_type"], "quarantined")
            continue

        if decision == "quarantine":
            quarantined += 1
            if not dry_run:
                set_relationship_audit_status(rel["head"], rel["tail"], rel["rel_type"], "quarantined")
            continue

        if decision == "delete":
            deleted += 1
            if not dry_run:
                delete_specific_relationship(rel["head"], rel["tail"], rel["rel_type"])

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
    filename = f"audit-{schedule}-{now.strftime('%Y%m%d-%H%M%S')}.md"
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
    """Run cleanup + LLM audit + suggestion tracking."""
    started = time.monotonic()

    self_refs_deleted = delete_self_referencing_rels()
    orphans_deleted = delete_orphan_entities_sync()
    strength_decay_updated = run_strength_decay_sync()

    rels = get_relationships_for_audit(limit=max(1, int(limit)))

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

                verdicts = parse_verdicts(str(llm_result.get("content") or ""), len(batch))
                if not verdicts:
                    parse_failures += 1
                    continue

                outcome = await apply_verdicts(batch, verdicts, dry_run=dry_run)
                auto_fixed += outcome["auto_fixed"]
                quarantined += outcome["quarantined"]
                deleted += outcome["deleted"]
                verified += outcome["verified"]
            except Exception:
                log.error("Audit batch failed", exc_info=True)
                parse_failures += 1

    suggestion_digest = build_suggestion_digest()
    auto_adoption_result = run_auto_adoption()

    summary = (
        f"Reviewed {len(rels)} relationships: "
        f"{auto_fixed} auto-fixed, {quarantined} quarantined, {verified} verified"
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
        "audit_model_model": chosen_model or (model_override or config.AUDIT_MODEL_NIGHTLY),
        "audit_provider": chosen_provider or "gemini",
        "audit_model_latency_ms": total_latency_ms,
        "batches": batches,
        "parse_failures": parse_failures,
        "dry_run": bool(dry_run),
        "schedule": schedule,
        "fallback": fallback_note,
        "suggestion_digest": suggestion_digest,
        "auto_adoption_result": auto_adoption_result,
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
