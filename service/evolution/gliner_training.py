"""Standalone GLiNER training pipeline for graph-memory.

This is adapted from Molly's ``evolution/gliner_training.py`` with Molly-specific
engine/infra dependencies removed in favor of a simple state file.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import config

log = logging.getLogger(__name__)

GLINER_BASE_MODEL = config.GLINER_BASE_MODEL
GLINER_BENCHMARK_SEED = config.GLINER_BENCHMARK_SEED
GLINER_BENCHMARK_EVAL_RATIO = config.GLINER_BENCHMARK_EVAL_RATIO
GLINER_BENCHMARK_THRESHOLD = config.GLINER_BENCHMARK_THRESHOLD
GLINER_FINETUNE_COOLDOWN_DAYS = config.GLINER_FINETUNE_COOLDOWN_DAYS
GLINER_TRAINING_SCAN_LIMIT = config.GLINER_TRAINING_SCAN_LIMIT


class GLiNERTrainingService:
    _GLINER_MAX_RUNS = 3
    _GLINER_MAX_BACKUPS = 2

    def __init__(self, state_file: Path | None = None):
        self._state_file = state_file or config.STATE_FILE
        self.state: dict[str, Any] = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        if not self._state_file.exists():
            return {}
        try:
            payload = json.loads(self._state_file.read_text(encoding="utf-8"))
        except Exception:
            log.warning("Failed to parse state file %s", self._state_file, exc_info=True)
            return {}
        return payload if isinstance(payload, dict) else {}

    def save_state(self) -> None:
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        tmp.replace(self._state_file)

    def refresh_state(self) -> None:
        self.state = self._load_state()

    def gliner_training_dir(self) -> Path:
        return config.TRAINING_DIR

    def gliner_models_dir(self) -> Path:
        return config.MODELS_DIR

    def gliner_candidate_model_dir(self) -> Path:
        return self.gliner_models_dir() / "gliner_candidate"

    def gliner_active_model_dir(self) -> Path:
        return self.gliner_models_dir() / "gliner_active"

    def gliner_training_config_path(self) -> Path:
        return config.GRAPH_MEMORY_DIR / "gliner_finetune_config.json"

    async def run_gliner_nightly_cycle(self, force: bool = False) -> dict[str, Any]:
        self.refresh_state()

        accumulation = await asyncio.to_thread(
            self.accumulate_gliner_training_data,
            GLINER_TRAINING_SCAN_LIMIT,
        )
        total_examples = int(accumulation.get("total_examples", 0))
        required = int(config.GLINER_FINETUNE_MIN_EXAMPLES)

        progress_line = f"GLiNER training data: {total_examples}/{required} examples accumulated"
        self.state["gliner_training_examples"] = total_examples
        self.state["gliner_last_result"] = progress_line
        self.state["gliner_last_cycle_status"] = "accumulated"
        self.save_state()

        if total_examples < required:
            return {
                "status": "insufficient_examples",
                "count": total_examples,
                "required": required,
                "accumulation": accumulation,
                "message": progress_line,
            }

        now_utc = datetime.now(timezone.utc)
        last_run_iso = str(self.state.get("gliner_last_finetune_at", "")).strip()
        last_run = _parse_datetime(last_run_iso)

        if not force and last_run and (now_utc - last_run) < timedelta(days=GLINER_FINETUNE_COOLDOWN_DAYS):
            elapsed = now_utc - last_run
            remaining = timedelta(days=GLINER_FINETUNE_COOLDOWN_DAYS) - elapsed
            hours_remaining = max(0, int(remaining.total_seconds() // 3600))
            cooldown_line = (
                f"GLiNER fine-tune skipped: last run {last_run.date().isoformat()} "
                f"({hours_remaining}h cooldown remaining)."
            )
            self.state["gliner_last_result"] = cooldown_line
            self.state["gliner_last_cycle_status"] = "cooldown_active"
            self.save_state()
            return {
                "status": "cooldown_active",
                "count": total_examples,
                "required": required,
                "last_run": last_run.isoformat(),
                "accumulation": accumulation,
                "message": cooldown_line,
                "force": bool(force),
            }

        pipeline = await self.run_gliner_finetune_pipeline()
        return {
            "status": "finetune_triggered",
            "accumulation": accumulation,
            "pipeline": pipeline,
            "force": bool(force),
        }

    async def run_gliner_finetune_pipeline(self) -> dict[str, Any]:
        rows = await asyncio.to_thread(self.load_accumulated_gliner_examples)
        total_rows = len(rows)
        required = int(config.GLINER_FINETUNE_MIN_EXAMPLES)

        if total_rows < required:
            msg = f"GLiNER training data: {total_rows}/{required} examples accumulated"
            self.state["gliner_training_examples"] = total_rows
            self.state["gliner_last_result"] = msg
            self.state["gliner_last_cycle_status"] = "insufficient_examples"
            self.save_state()
            return {
                "status": "insufficient_examples",
                "count": total_rows,
                "required": required,
                "message": msg,
            }

        train_rows, eval_rows = self.split_holdout_rows(
            rows,
            eval_ratio=GLINER_BENCHMARK_EVAL_RATIO,
            seed=GLINER_BENCHMARK_SEED,
        )
        if not train_rows or not eval_rows:
            self.state["gliner_last_cycle_status"] = "split_failed"
            self.state["gliner_last_result"] = "GLiNER fine-tune skipped: invalid train/eval split."
            self.save_state()
            return {
                "status": "split_failed",
                "count": total_rows,
                "train_count": len(train_rows),
                "eval_count": len(eval_rows),
            }

        strategy = await asyncio.to_thread(self.select_gliner_training_strategy, total_rows)
        mode = str(strategy.get("mode") or "lora").strip().lower()
        if mode not in {"lora", "full"}:
            mode = "lora"

        # Pre-training audit with Kimi (gates the training)
        pretrain_audit = await self._run_pretraining_audit(mode)
        if not pretrain_audit.get("passed", True):
            self.state["gliner_last_cycle_status"] = "pretrain_audit_failed"
            self.state["gliner_last_result"] = f"GLiNER {mode} blocked: Kimi pre-training audit failed - {pretrain_audit.get('reason', 'quality checks not met')}"
            self.save_state()
            return {
                "status": "pretrain_audit_failed",
                "count": total_rows,
                "pretrain_audit": pretrain_audit,
                "message": self.state["gliner_last_result"],
            }

        self.state["gliner_last_finetune_at"] = datetime.now(timezone.utc).isoformat()
        self.state["gliner_training_examples"] = total_rows
        self.state["gliner_last_cycle_status"] = "finetune_started"
        self.state["gliner_last_training_strategy"] = mode
        self.save_state()

        fine_tune = await asyncio.to_thread(self.fine_tune_gliner_candidate, train_rows, mode)
        if not fine_tune.get("ok", False):
            self.state["gliner_last_cycle_status"] = "finetune_failed"
            self.state["gliner_last_result"] = f"GLiNER {mode} fine-tune failed before benchmarking."
            self.save_state()
            return {
                "status": "finetune_failed",
                "count": total_rows,
                "train_count": len(train_rows),
                "eval_count": len(eval_rows),
                "training_strategy": strategy,
                "fine_tune": fine_tune,
            }

        candidate_model_ref = str(fine_tune.get("candidate_model") or "").strip()
        benchmark = await asyncio.to_thread(
            self.benchmark_finetune_candidate,
            eval_rows,
            candidate_model_ref,
            len(train_rows),
        )

        benchmark_ok = bool(benchmark.get("ok", False))
        improvement = float(benchmark.get("improvement", 0.0) or 0.0)
        threshold = float(config.GLINER_FINETUNE_BENCHMARK_THRESHOLD)
        should_deploy = benchmark_ok and improvement >= threshold

        payload = {
            "count": total_rows,
            "train_count": len(train_rows),
            "eval_count": len(eval_rows),
            "training_strategy": strategy,
            "fine_tune": fine_tune,
            "benchmark": benchmark,
            "threshold": threshold,
            "should_deploy": should_deploy,
        }

        status = "below_threshold"
        if should_deploy:
            deploy = await asyncio.to_thread(
                self.deploy_gliner_candidate_model,
                Path(candidate_model_ref),
                benchmark,
                fine_tune,
            )
            payload["deploy"] = deploy
            if deploy.get("ok"):
                status = "deployed"
                self.state["gliner_last_result"] = (
                    f"GLiNER {mode} +{improvement:.4f} F1 on {len(eval_rows)} eval rows; deployed."
                )
                self.state["gliner_last_cycle_status"] = "deployed"
                self.save_state()
            else:
                status = "deploy_failed"
                self.state["gliner_last_result"] = (
                    f"GLiNER {mode} benchmark passed (+{improvement:.4f}) but deploy failed."
                )
                self.state["gliner_last_cycle_status"] = "deploy_failed"
                self.save_state()
        else:
            await asyncio.to_thread(
                self.discard_gliner_candidate_model,
                Path(candidate_model_ref) if candidate_model_ref else None,
            )
            self.state["gliner_last_cycle_status"] = "below_threshold"
            self.state["gliner_last_result"] = (
                f"GLiNER {mode} candidate rejected (+{improvement:.4f} < {threshold:.4f})."
            )
            self.save_state()

        await asyncio.to_thread(
            self.record_gliner_benchmark,
            mode,
            benchmark,
            status,
            total_rows,
        )

        return {
            "status": status,
            "count": total_rows,
            "train_count": len(train_rows),
            "eval_count": len(eval_rows),
            "training_strategy": strategy,
            "benchmark": benchmark,
            "fine_tune": fine_tune,
            **({"deploy": payload.get("deploy")} if "deploy" in payload else {}),
        }

    def accumulate_gliner_training_data(self, limit: int = GLINER_TRAINING_SCAN_LIMIT) -> dict[str, Any]:
        from memory.graph import get_driver

        training_dir = self.gliner_training_dir()
        training_dir.mkdir(parents=True, exist_ok=True)

        seen_episode_ids = self.load_existing_gliner_episode_ids(training_dir)
        opus_analysis_text = self.latest_maintenance_analysis_text()
        cursor = str(self.state.get("gliner_training_cursor", "")).strip()

        where_cursor = (
            "AND datetime(coalesce(ep.created_at, toString(ep.ingested_at), toString(ep.occurred_at))) > datetime($cursor)"
            if cursor
            else ""
        )
        params: dict[str, Any] = {"limit": int(limit)}
        if cursor:
            params["cursor"] = cursor

        driver = get_driver()
        with driver.session() as session:
            rows = [
                dict(record)
                for record in session.run(
                    f"""
                    MATCH (ep:Episode)
                    WHERE ep.content_preview IS NOT NULL
                      AND trim(ep.content_preview) <> ''
                      AND ep.entities_extracted IS NOT NULL
                      AND size(ep.entities_extracted) > 0
                      {where_cursor}
                    RETURN ep.id AS episode_id,
                           coalesce(ep.created_at, toString(ep.ingested_at), toString(ep.occurred_at)) AS created_at,
                           ep.content_preview AS source_text,
                           ep.entities_extracted AS entity_names
                    ORDER BY coalesce(ep.created_at, toString(ep.ingested_at), toString(ep.occurred_at)) ASC
                    LIMIT $limit
                    """,
                    **params,
                )
            ]

            new_examples: list[dict[str, Any]] = []
            latest_seen_created_at = cursor

            for row in rows:
                episode_id = str(row.get("episode_id") or "").strip()
                if not episode_id:
                    continue

                if episode_id in seen_episode_ids:
                    if row.get("created_at"):
                        latest_seen_created_at = str(row["created_at"])
                    continue

                example = self.build_training_example_from_episode(
                    session=session,
                    episode=row,
                    opus_analysis_text=opus_analysis_text,
                )
                if example:
                    new_examples.append(example)
                    seen_episode_ids.add(episode_id)

                if row.get("created_at"):
                    latest_seen_created_at = str(row["created_at"])

        batch_path: str | None = None
        if new_examples:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            path = training_dir / f"examples-{ts}.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for example in new_examples:
                    f.write(json.dumps(example, ensure_ascii=True) + "\n")
            batch_path = str(path)
            log.info("GLiNER accumulation: wrote %d examples to %s", len(new_examples), path)
        else:
            log.info("GLiNER accumulation: no new examples this run")

        if latest_seen_created_at:
            self.state["gliner_training_cursor"] = latest_seen_created_at

        total_examples = self.count_accumulated_gliner_examples(training_dir)
        self.state["gliner_training_examples"] = total_examples
        self.save_state()

        return {
            "new_examples": len(new_examples),
            "total_examples": total_examples,
            "batch_path": batch_path,
            "cursor": str(self.state.get("gliner_training_cursor", "")),
        }

    def load_existing_gliner_episode_ids(self, training_dir: Path) -> set[str]:
        seen: set[str] = set()
        for path in sorted(training_dir.glob("*.jsonl")):
            try:
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(payload, dict):
                            continue
                        episode_id = str(payload.get("episode_id") or "").strip()
                        if episode_id:
                            seen.add(episode_id)
            except Exception:
                log.debug("Failed scanning training file %s", path, exc_info=True)
        return seen

    def latest_maintenance_analysis_text(self) -> str:
        maintenance_dir = config.MAINTENANCE_DIR
        if not maintenance_dir.exists():
            return ""

        candidates = sorted(maintenance_dir.glob("*.md"))
        if not candidates:
            return ""

        latest_path = candidates[-1]
        try:
            text = latest_path.read_text(encoding="utf-8")
        except Exception:
            return ""

        marker = "\n## Analysis\n"
        if marker in text:
            text = text.split(marker, 1)[1]

        return self.normalize_entity_text(text)

    def build_training_example_from_episode(
        self,
        session: Any,
        episode: dict[str, Any],
        opus_analysis_text: str,
    ) -> dict[str, Any] | None:
        source_text = str(episode.get("source_text") or "").strip()
        if not source_text:
            return None

        entity_names_raw = episode.get("entity_names") or []
        if not isinstance(entity_names_raw, list):
            return None

        entity_names = sorted({str(name).strip() for name in entity_names_raw if str(name).strip()})
        if not entity_names:
            return None

        entity_rows = [
            dict(record)
            for record in session.run(
                """
                MATCH (e:Entity)
                WHERE e.name IN $names
                RETURN e.name AS name,
                       e.entity_type AS entity_type,
                       coalesce(e.mention_count, 0) AS mention_count,
                       EXISTS { MATCH (e)--(:Entity) } AS has_relationship
                """,
                names=entity_names,
            )
        ]
        if not entity_rows:
            return None

        selected_entities: list[dict[str, Any]] = []
        signal_counts = {
            "opus_confirmed": 0,
            "multi_mentions": 0,
            "relationship_backed": 0,
        }

        for entity in entity_rows:
            name = str(entity.get("name") or "").strip()
            label = str(entity.get("entity_type") or "Concept").strip() or "Concept"
            mention_count = int(entity.get("mention_count") or 0)
            has_relationship = bool(entity.get("has_relationship"))
            normalized_name = self.normalize_entity_text(name)

            if not normalized_name:
                continue

            opus_confirmed = bool(opus_analysis_text) and normalized_name in opus_analysis_text
            multi_mentions = mention_count >= 2
            relationship_backed = has_relationship

            if not (opus_confirmed or multi_mentions or relationship_backed):
                continue

            if opus_confirmed:
                signal_counts["opus_confirmed"] += 1
            if multi_mentions:
                signal_counts["multi_mentions"] += 1
            if relationship_backed:
                signal_counts["relationship_backed"] += 1

            selected_entities.append({"text": name, "label": label})

        if not selected_entities:
            return None

        selected_names = sorted({str(ent["text"]).strip() for ent in selected_entities})
        relation_rows = [
            dict(record)
            for record in session.run(
                """
                MATCH (h:Entity)-[r]->(t:Entity)
                WHERE h.name IN $names AND t.name IN $names
                  AND (r.audit_status IS NULL OR r.audit_status <> 'quarantined')
                RETURN h.name AS head,
                       t.name AS tail,
                       type(r) AS label,
                       coalesce(r.mention_count, 0) AS mention_count
                """,
                names=selected_names,
            )
        ]

        relations: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str, str]] = set()
        for rel in relation_rows:
            head = str(rel.get("head") or "").strip()
            tail = str(rel.get("tail") or "").strip()
            label = str(rel.get("label") or "").strip()
            if not head or not tail or not label:
                continue
            key = (head, label, tail)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            relations.append({"head": head, "tail": tail, "label": label})

        return {
            "episode_id": str(episode.get("episode_id") or ""),
            "created_at": str(episode.get("created_at") or ""),
            "source_text": source_text,
            "extracted_entities": selected_entities,
            "extracted_relations": relations,
            "quality_signals": signal_counts,
        }

    def count_accumulated_gliner_examples(self, training_dir: Path | None = None) -> int:
        target_dir = training_dir or self.gliner_training_dir()
        if not target_dir.exists():
            return 0

        total = 0
        for path in target_dir.glob("*.jsonl"):
            try:
                with path.open("r", encoding="utf-8") as f:
                    total += sum(1 for line in f if line.strip())
            except Exception:
                log.debug("Failed counting examples in %s", path, exc_info=True)

        return total

    def load_accumulated_gliner_examples(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        training_dir = self.gliner_training_dir()
        if not training_dir.exists():
            return rows

        for path in sorted(training_dir.glob("*.jsonl")):
            try:
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(payload, dict):
                            continue
                        row = self.to_benchmark_row(payload)
                        if row:
                            rows.append(row)
            except Exception:
                log.debug("Failed loading examples from %s", path, exc_info=True)

        return rows

    @staticmethod
    def to_benchmark_row(payload: dict[str, Any]) -> dict[str, Any] | None:
        text = str(payload.get("source_text") or payload.get("text") or "").strip()
        entities = payload.get("extracted_entities", payload.get("entities", []))
        relations = payload.get("extracted_relations", payload.get("relations", []))

        if not text or not isinstance(entities, list) or not entities:
            return None

        return {
            "episode_id": str(payload.get("episode_id") or ""),
            "created_at": str(payload.get("created_at") or ""),
            "text": text,
            "entities": entities,
            "relations": relations if isinstance(relations, list) else [],
        }

    async def _run_pretraining_audit(self, mode: str) -> dict[str, Any]:
        """Run Kimi pre-training audit to gate GLiNER training.
        
        Kimi reviews training data quality before allowing LoRA or full training.
        """
        try:
            from audit.llm_audit import call_audit_model, build_audit_prompt
            from memory.graph import get_relationships_for_audit
            
            # Sample recent relationships for quality check
            rels = get_relationships_for_audit(limit=100)
            if not rels:
                return {"passed": True, "reason": "no_relationships_to_audit"}
            
            prompt = build_audit_prompt(rels)
            
            # Call Kimi for pre-training audit
            audit_model = getattr(config, "AUDIT_MODEL_PRETRAIN", "kimi-k2.5")
            llm_result = await call_audit_model(prompt, schedule="nightly", model_override=audit_model)
            
            content = str(llm_result.get("content") or "")
            provider = str(llm_result.get("provider") or "unknown")
            fallback = str(llm_result.get("fallback") or "")
            
            # Parse verdicts
            from audit.llm_audit import parse_verdicts
            verdicts = parse_verdicts(content, len(rels))
            
            if not verdicts:
                return {"passed": False, "reason": "audit_parse_failed", "provider": provider}
            
            # Calculate quality metrics
            total = len(verdicts)
            verify_count = sum(1 for v in verdicts if v.get("verdict") == "verify")
            delete_count = sum(1 for v in verdicts if v.get("verdict") == "delete")
            quarantine_count = sum(1 for v in verdicts if v.get("verdict") == "quarantine")
            
            verify_ratio = verify_count / total if total > 0 else 0
            delete_ratio = delete_count / total if total > 0 else 0
            
            # Quality gates
            passed = verify_ratio >= 0.6 and delete_ratio <= 0.15
            
            return {
                "passed": passed,
                "reason": f"verify_ratio={verify_ratio:.2f}, delete_ratio={delete_ratio:.2f}" if not passed else "quality_acceptable",
                "provider": provider,
                "fallback": fallback,
                "total_audited": total,
                "verify_count": verify_count,
                "delete_count": delete_count,
                "quarantine_count": quarantine_count,
            }
        except Exception as exc:
            log.warning("Pre-training audit failed, allowing training to proceed: %s", exc)
            return {"passed": True, "reason": f"audit_error_allowing_train: {exc}"}

    def to_gliner_training_record(self, row: dict[str, Any]) -> dict[str, Any] | None:
        text = str(row.get("text") or "").strip()
        if not text:
            return None

        text_norm = self.normalize_entity_text(text)

        entities_map: dict[str, list[str]] = {}
        entity_names: set[str] = set()

        for entity in row.get("entities") or []:
            if isinstance(entity, str):
                name = entity.strip()
                label = "Concept"
            elif isinstance(entity, dict):
                name = str(entity.get("text") or entity.get("name") or "").strip()
                label = str(entity.get("label") or entity.get("type") or "Concept").strip() or "Concept"
            else:
                continue

            if not name:
                continue
            if self.normalize_entity_text(name) not in text_norm:
                continue

            entities_map.setdefault(label, [])
            if name not in entities_map[label]:
                entities_map[label].append(name)
                entity_names.add(name)

        relations_out: list[dict[str, Any]] = []
        for rel in row.get("relations") or []:
            if not isinstance(rel, dict):
                continue
            head = str(rel.get("head") or "").strip()
            tail = str(rel.get("tail") or "").strip()
            label = str(rel.get("label") or "").strip().lower().replace("_", " ")

            if not head or not tail or not label:
                continue
            if head not in entity_names or tail not in entity_names:
                continue
            if self.normalize_entity_text(head) not in text_norm:
                continue
            if self.normalize_entity_text(tail) not in text_norm:
                continue

            relations_out.append({label: {"head": head, "tail": tail}})

        output: dict[str, Any] = {}
        if entities_map:
            output["entities"] = entities_map
        if relations_out:
            output["relations"] = relations_out
        if not output:
            return None

        return {"input": text, "output": output}

    def active_gliner_model_ref(self) -> str:
        active_dir = self.gliner_active_model_dir()
        if active_dir.exists():
            return str(active_dir)

        active_state_ref = str(self.state.get("gliner_active_model_ref", "")).strip()
        if active_state_ref and Path(active_state_ref).exists():
            return active_state_ref

        return GLINER_BASE_MODEL

    def select_gliner_training_strategy(self, total_examples: int) -> dict[str, Any]:
        full_min_examples = max(1, int(config.GLINER_FULL_FINETUNE_MIN_EXAMPLES))
        plateau_window = max(1, int(config.GLINER_LORA_PLATEAU_WINDOW))
        plateau_epsilon = max(0.0, float(config.GLINER_LORA_PLATEAU_EPSILON))

        if total_examples < full_min_examples:
            return {
                "mode": "lora",
                "reason": "insufficient_examples_for_full_finetune",
                "full_min_examples": full_min_examples,
                "total_examples": total_examples,
            }

        history = self.state.get("gliner_benchmark_history")
        if not isinstance(history, list):
            history = []

        recent_lora = [
            row
            for row in history
            if isinstance(row, dict)
            and str(row.get("strategy", "")).lower() == "lora"
            and bool(row.get("benchmark_ok"))
        ][-plateau_window:]

        if len(recent_lora) < plateau_window:
            return {
                "mode": "lora",
                "reason": "not_enough_lora_history",
                "required_runs": plateau_window,
                "available_runs": len(recent_lora),
                "total_examples": total_examples,
            }

        improvements = [float(row.get("improvement", 0.0) or 0.0) for row in recent_lora]
        max_gain = max(improvements) if improvements else 0.0
        plateaued = max_gain <= plateau_epsilon

        if plateaued:
            return {
                "mode": "full",
                "reason": "lora_plateau_detected",
                "plateau_window": plateau_window,
                "plateau_epsilon": plateau_epsilon,
                "recent_improvements": improvements,
                "total_examples": total_examples,
            }

        return {
            "mode": "lora",
            "reason": "lora_still_improving",
            "plateau_window": plateau_window,
            "plateau_epsilon": plateau_epsilon,
            "recent_improvements": improvements,
            "total_examples": total_examples,
        }

    def record_gliner_benchmark(
        self,
        strategy: str,
        benchmark: dict[str, Any],
        status: str,
        total_examples: int,
    ) -> None:
        history = self.state.get("gliner_benchmark_history")
        if not isinstance(history, list):
            history = []

        history.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "strategy": strategy,
                "status": status,
                "total_examples": int(total_examples),
                "benchmark_ok": bool(benchmark.get("ok", False)),
                "improvement": float(benchmark.get("improvement", 0.0) or 0.0),
                "base_score": float(benchmark.get("base_score", 0.0) or 0.0),
                "candidate_score": float(benchmark.get("candidate_score", 0.0) or 0.0),
                "eval_count": int(benchmark.get("split", {}).get("eval_count", 0) or 0),
            }
        )

        self.state["gliner_benchmark_history"] = history[-20:]
        self.save_state()

    def fine_tune_gliner_candidate(self, train_rows: list[dict[str, Any]], mode: str = "lora") -> dict[str, Any]:
        train_records = [record for record in (self.to_gliner_training_record(r) for r in train_rows) if record]
        if not train_records:
            return {"ok": False, "error": "no_valid_train_records"}

        try:
            from gliner2.training.trainer import train_gliner2
        except Exception as exc:
            return {"ok": False, "error": f"training_import_failed: {exc}"}

        models_dir = self.gliner_models_dir()
        candidate_dir = self.gliner_candidate_model_dir()
        runs_dir = models_dir / "gliner_runs"
        splits_dir = self.gliner_training_dir() / "splits"

        models_dir.mkdir(parents=True, exist_ok=True)
        runs_dir.mkdir(parents=True, exist_ok=True)
        splits_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        train_split_path = splits_dir / f"train-{ts}.jsonl"
        with train_split_path.open("w", encoding="utf-8") as f:
            for record in train_records:
                f.write(json.dumps(record, ensure_ascii=True) + "\n")

        output_dir = runs_dir / ts
        output_dir.mkdir(parents=True, exist_ok=True)

        batch_size = min(4, max(1, len(train_records)))
        normalized_mode = "full" if str(mode).strip().lower() == "full" else "lora"
        use_lora = normalized_mode == "lora"
        num_epochs = 1 if use_lora else 2

        try:
            result = train_gliner2(
                model_path=self.active_gliner_model_ref(),
                train_data=train_records,
                output_dir=str(output_dir),
                num_epochs=num_epochs,
                batch_size=batch_size,
                eval_strategy="no",
                fp16=False,
                bf16=False,
                num_workers=0,
                logging_steps=max(1, min(25, len(train_records))),
                use_lora=use_lora,
                save_adapter_only=False,
                seed=GLINER_BENCHMARK_SEED,
            )
        except Exception as exc:
            log.error("GLiNER fine-tune failed", exc_info=True)
            return {"ok": False, "error": str(exc)}

        final_dir = output_dir / "final"
        if not final_dir.exists():
            best_dir = output_dir / "best"
            if best_dir.exists():
                final_dir = best_dir
            else:
                return {
                    "ok": False,
                    "error": "trained_model_not_found",
                    "output_dir": str(output_dir),
                }

        self.discard_gliner_candidate_model(candidate_dir)
        shutil.copytree(final_dir, candidate_dir, dirs_exist_ok=False)

        metadata = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "base_model": self.active_gliner_model_ref(),
            "train_examples": len(train_records),
            "batch_size": batch_size,
            "mode": normalized_mode,
            "num_epochs": num_epochs,
            "output_dir": str(output_dir),
            "result": result,
        }
        metadata_path = candidate_dir / "fine_tune_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

        self.prune_gliner_dirs()

        return {
            "ok": True,
            "candidate_model": str(candidate_dir),
            "train_split_path": str(train_split_path),
            "output_dir": str(output_dir),
            "metadata_path": str(metadata_path),
            "mode": normalized_mode,
            "result": result,
        }

    def prune_gliner_dirs(self) -> dict[str, Any]:
        pruned: dict[str, list[str]] = {"runs": [], "backups": []}

        for subdir, limit in [
            (self.gliner_models_dir() / "gliner_runs", self._GLINER_MAX_RUNS),
            (self.gliner_models_dir() / "gliner_backups", self._GLINER_MAX_BACKUPS),
        ]:
            if not subdir.exists():
                continue

            dirs = sorted([d for d in subdir.iterdir() if d.is_dir()], key=lambda p: p.name, reverse=True)
            for old_dir in dirs[limit:]:
                try:
                    shutil.rmtree(old_dir, ignore_errors=True)
                    pruned[subdir.name].append(old_dir.name)
                    log.info("Pruned GLiNER dir: %s", old_dir)
                except Exception:
                    log.warning("Failed pruning %s", old_dir, exc_info=True)

        return pruned

    @staticmethod
    def discard_gliner_candidate_model(candidate_path: Path | None) -> None:
        if candidate_path and candidate_path.exists():
            shutil.rmtree(candidate_path, ignore_errors=True)

    def deploy_gliner_candidate_model(
        self,
        candidate_path: Path,
        benchmark: dict[str, Any],
        fine_tune: dict[str, Any],
    ) -> dict[str, Any]:
        if not candidate_path.exists():
            return {"ok": False, "error": "candidate_path_missing"}

        active_dir = self.gliner_active_model_dir()
        active_dir.parent.mkdir(parents=True, exist_ok=True)
        previous_active_ref = self.active_gliner_model_ref()

        backup_dir: Path | None = None
        if active_dir.exists():
            backup_root = self.gliner_models_dir() / "gliner_backups"
            backup_root.mkdir(parents=True, exist_ok=True)
            backup_dir = backup_root / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            shutil.copytree(active_dir, backup_dir, dirs_exist_ok=False)

        if active_dir.exists():
            shutil.rmtree(active_dir, ignore_errors=True)
        shutil.copytree(candidate_path, active_dir, dirs_exist_ok=False)

        deployed_at = datetime.now(timezone.utc).isoformat()
        self.state["gliner_active_model_ref"] = str(active_dir)
        self.state["gliner_last_deployed_at"] = deployed_at
        self.save_state()

        config_payload = {
            "updated_at": deployed_at,
            "active_model_ref": str(active_dir),
            "previous_model_ref": previous_active_ref,
            "backup_model_ref": str(backup_dir) if backup_dir else None,
            "benchmark": benchmark,
            "fine_tune": {
                "mode": fine_tune.get("mode", "lora"),
                "train_split_path": fine_tune.get("train_split_path"),
                "output_dir": fine_tune.get("output_dir"),
                "metadata_path": fine_tune.get("metadata_path"),
            },
        }

        config_path = self.gliner_training_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config_payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

        pruned = self.prune_gliner_dirs()

        return {
            "ok": True,
            "active_model": str(active_dir),
            "backup_model": str(backup_dir) if backup_dir else None,
            "training_config": str(config_path),
            "pruned": pruned,
        }

    @staticmethod
    def split_holdout_rows(
        rows: list[dict[str, Any]],
        eval_ratio: float = GLINER_BENCHMARK_EVAL_RATIO,
        seed: int = GLINER_BENCHMARK_SEED,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not rows:
            return [], []

        indices = list(range(len(rows)))
        random.Random(seed).shuffle(indices)

        if len(rows) == 1:
            eval_count = 1
        else:
            eval_count = max(1, int(round(len(rows) * eval_ratio)))
            eval_count = min(eval_count, len(rows) - 1)

        eval_idx = set(indices[:eval_count])
        train_rows = [rows[i] for i in range(len(rows)) if i not in eval_idx]
        eval_rows = [rows[i] for i in range(len(rows)) if i in eval_idx]

        return train_rows, eval_rows

    @staticmethod
    def normalize_entity_text(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip().lower()
        return re.sub(r"\s+", " ", text)

    def extract_expected_entity_set(self, row: dict[str, Any]) -> set[str]:
        entities = row.get("entities") or []
        if not isinstance(entities, list):
            return set()

        normalized: set[str] = set()
        for entity in entities:
            if isinstance(entity, str):
                name = self.normalize_entity_text(entity)
            elif isinstance(entity, dict):
                name = self.normalize_entity_text(
                    entity.get("text") or entity.get("name") or entity.get("entity")
                )
            else:
                name = ""

            if name:
                normalized.add(name)

        return normalized

    @staticmethod
    def extract_predicted_entity_set(result: Any) -> set[str]:
        if not isinstance(result, dict):
            return set()

        entity_dict = result.get("entities", result)
        if not isinstance(entity_dict, dict):
            return set()

        predicted: set[str] = set()
        for items in entity_dict.values():
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, str):
                    name = GLiNERTrainingService.normalize_entity_text(item)
                elif isinstance(item, dict):
                    name = GLiNERTrainingService.normalize_entity_text(item.get("text"))
                else:
                    name = ""
                if name:
                    predicted.add(name)

        return predicted

    @staticmethod
    def compute_prf_metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
        precision = (tp / (tp + fp)) if (tp + fp) else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    def load_gliner_entity_model(self, model_ref: str) -> tuple[Any, Any]:
        from gliner2 import GLiNER2
        from memory.extractor import ENTITY_SCHEMA

        model = GLiNER2.from_pretrained(model_ref)
        schema = model.create_schema().entities(ENTITY_SCHEMA)
        return model, schema

    def evaluate_model_on_rows(
        self,
        model_ref: str,
        rows: list[dict[str, Any]],
        threshold: float = GLINER_BENCHMARK_THRESHOLD,
    ) -> dict[str, Any]:
        if not rows:
            return {
                "ok": False,
                "error": "empty_eval_set",
                "metrics": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "counts": {"tp": 0, "fp": 0, "fn": 0},
                "rows_total": 0,
                "rows_evaluated": 0,
                "rows_failed": 0,
                "latency_ms_avg": 0.0,
            }

        try:
            model, schema = self.load_gliner_entity_model(model_ref)
        except Exception as exc:
            return {
                "ok": False,
                "error": f"model_load_failed: {exc}",
                "metrics": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "counts": {"tp": 0, "fp": 0, "fn": 0},
                "rows_total": len(rows),
                "rows_evaluated": 0,
                "rows_failed": len(rows),
                "latency_ms_avg": 0.0,
            }

        tp = 0
        fp = 0
        fn = 0
        rows_evaluated = 0
        rows_failed = 0
        latency_sum_ms = 0.0
        failure_samples: list[dict[str, Any]] = []

        for idx, row in enumerate(rows):
            text = str(row.get("text") or "").strip()
            if not text:
                rows_failed += 1
                if len(failure_samples) < 5:
                    failure_samples.append({"row_index": idx, "error": "missing_text"})
                continue

            expected = self.extract_expected_entity_set(row)
            try:
                t0 = time.monotonic()
                result = model.extract(text, schema, threshold=threshold, include_confidence=True)
                latency_sum_ms += (time.monotonic() - t0) * 1000.0
            except Exception as exc:
                rows_failed += 1
                if len(failure_samples) < 5:
                    failure_samples.append({"row_index": idx, "error": str(exc)[:300]})
                continue

            predicted = self.extract_predicted_entity_set(result)
            tp += len(predicted & expected)
            fp += len(predicted - expected)
            fn += len(expected - predicted)
            rows_evaluated += 1

        if rows_evaluated == 0:
            return {
                "ok": False,
                "error": "all_inference_failed",
                "metrics": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "counts": {"tp": tp, "fp": fp, "fn": fn},
                "rows_total": len(rows),
                "rows_evaluated": 0,
                "rows_failed": rows_failed,
                "latency_ms_avg": 0.0,
                "failure_samples": failure_samples,
            }

        metrics = self.compute_prf_metrics(tp=tp, fp=fp, fn=fn)
        return {
            "ok": rows_failed == 0,
            "error": "" if rows_failed == 0 else "partial_inference_failures",
            "metrics": metrics,
            "counts": {"tp": tp, "fp": fp, "fn": fn},
            "rows_total": len(rows),
            "rows_evaluated": rows_evaluated,
            "rows_failed": rows_failed,
            "latency_ms_avg": round(latency_sum_ms / rows_evaluated, 2),
            "failure_samples": failure_samples,
        }

    def benchmark_finetune_candidate(
        self,
        rows: list[dict[str, Any]],
        candidate_model_ref: str | None = None,
        train_count: int | None = None,
    ) -> dict[str, Any]:
        eval_rows = rows
        resolved_train_count = int(train_count or 0)

        if train_count is None:
            split_train, split_eval = self.split_holdout_rows(rows)
            eval_rows = split_eval
            resolved_train_count = len(split_train)

        if not eval_rows:
            return {
                "ok": False,
                "base_score": 0.0,
                "candidate_score": 0.0,
                "improvement": 0.0,
                "split": {
                    "seed": GLINER_BENCHMARK_SEED,
                    "train_count": resolved_train_count,
                    "eval_count": 0,
                },
                "failure": {"reason": "no_eval_rows"},
            }

        base_model_ref = self.active_gliner_model_ref()
        candidate_model_ref = str(candidate_model_ref or "").strip()
        if not candidate_model_ref:
            candidate_path = self.gliner_candidate_model_dir()
            if candidate_path.exists():
                candidate_model_ref = str(candidate_path)

        if not candidate_model_ref:
            return {
                "ok": False,
                "base_score": 0.0,
                "candidate_score": 0.0,
                "improvement": 0.0,
                "split": {
                    "seed": GLINER_BENCHMARK_SEED,
                    "train_count": resolved_train_count,
                    "eval_count": len(eval_rows),
                },
                "failure": {"reason": "candidate_model_missing"},
            }

        base_eval = self.evaluate_model_on_rows(base_model_ref, eval_rows)
        candidate_eval = self.evaluate_model_on_rows(candidate_model_ref, eval_rows)

        base_score = float(base_eval.get("metrics", {}).get("f1", 0.0) or 0.0)
        candidate_score = float(candidate_eval.get("metrics", {}).get("f1", 0.0) or 0.0)
        benchmark_ok = bool(base_eval.get("ok")) and bool(candidate_eval.get("ok"))
        improvement = (candidate_score - base_score) if benchmark_ok else 0.0

        failure_details = []
        if not base_eval.get("ok"):
            failure_details.append({"model": "base", "error": base_eval.get("error", "unknown")})
        if not candidate_eval.get("ok"):
            failure_details.append(
                {"model": "candidate", "error": candidate_eval.get("error", "unknown")}
            )

        return {
            "ok": benchmark_ok,
            "split": {
                "seed": GLINER_BENCHMARK_SEED,
                "train_count": resolved_train_count,
                "eval_count": len(eval_rows),
            },
            "base_model": base_model_ref,
            "candidate_model": candidate_model_ref or None,
            "base": base_eval,
            "candidate": candidate_eval,
            "base_score": round(base_score, 4),
            "candidate_score": round(candidate_score, 4),
            "improvement": round(improvement, 4),
            "failure": None if benchmark_ok else {"reason": "model_evaluation_failed", "details": failure_details},
        }

    def status(self) -> dict[str, Any]:
        self.refresh_state()

        total_examples = int(self.state.get("gliner_training_examples", 0) or 0)
        required = int(config.GLINER_FINETUNE_MIN_EXAMPLES)

        last_finetune_iso = str(self.state.get("gliner_last_finetune_at", "") or "")
        last_finetune = _parse_datetime(last_finetune_iso)

        cooldown_remaining_hours = 0
        if last_finetune:
            now = datetime.now(timezone.utc)
            elapsed = now - last_finetune
            cooldown = timedelta(days=GLINER_FINETUNE_COOLDOWN_DAYS)
            if elapsed < cooldown:
                cooldown_remaining_hours = int((cooldown - elapsed).total_seconds() // 3600)

        return {
            "examples_accumulated": total_examples,
            "required": required,
            "last_finetune_at": last_finetune_iso,
            "last_strategy": str(self.state.get("gliner_last_training_strategy", "lora") or "lora"),
            "last_result": str(self.state.get("gliner_last_result", "") or ""),
            "last_cycle_status": str(self.state.get("gliner_last_cycle_status", "") or ""),
            "cooldown_remaining_hours": max(0, cooldown_remaining_hours),
            "active_model": self.active_gliner_model_ref(),
            "base_model": GLINER_BASE_MODEL,
        }


def _parse_datetime(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


_SERVICE: GLiNERTrainingService | None = None


def _get_service() -> GLiNERTrainingService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = GLiNERTrainingService()
    return _SERVICE


async def run_gliner_finetune_pipeline(force: bool = False) -> dict[str, Any]:
    """Run nightly GLiNER cycle: accumulate + (optional) fine-tune pipeline."""
    service = _get_service()
    return await service.run_gliner_nightly_cycle(force=force)


async def run_gliner_accumulation(limit: int = GLINER_TRAINING_SCAN_LIMIT) -> dict[str, Any]:
    """Accumulate training data without triggering fine-tune."""
    service = _get_service()
    return await asyncio.to_thread(service.accumulate_gliner_training_data, limit)


def get_gliner_stats() -> dict[str, Any]:
    """Return GLiNER status snapshot from state.json and model dirs."""
    service = _get_service()
    return service.status()


__all__ = [
    "GLINER_BASE_MODEL",
    "GLINER_BENCHMARK_EVAL_RATIO",
    "GLINER_BENCHMARK_SEED",
    "GLINER_BENCHMARK_THRESHOLD",
    "GLINER_FINETUNE_COOLDOWN_DAYS",
    "GLINER_TRAINING_SCAN_LIMIT",
    "GLiNERTrainingService",
    "get_gliner_stats",
    "run_gliner_accumulation",
    "run_gliner_finetune_pipeline",
]
