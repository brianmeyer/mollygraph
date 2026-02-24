# MollyGraph QA / Reliability Audit

**Date:** 2026-02-24  
**Auditor Role:** QA Engineer + SRE  
**Repo:** `/Users/brianmeyer/mollygraph/service`  
**Methodology:** Full static review of all non-venv Python source files + cross-referencing known production incidents

---

## Severity Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 5     |
| HIGH     | 11    |
| MEDIUM   | 9     |
| LOW      | 6     |

---

## CRITICAL

---

### C-1 — Queue jobs permanently stuck as `processing` after ungraceful shutdown

**File:** `extraction/queue.py:claim_next()`, `QueueWorker._process_job()`  
**Scenario:** Worker process is killed with `SIGKILL` (or OOMs) while a job has been claimed and marked `processing`. On restart the job stays `processing` forever — `claim_next()` only fetches `status = 'pending'` rows.  
**Impact:** Any content ingested at the time of a crash is silently dropped. Queue grows stale. No dead-letter queue. No visibility into how long jobs have been "processing". Operators have no way to know the job needs to be replayed without direct DB inspection.  
**Blast radius:** All jobs in-flight at crash time. With `max_concurrent=3` that's up to 3 lost jobs per crash.  
**Mitigation:**
1. Add a `processing_timeout_seconds` field (e.g., 5 min). On `claim_next()`, also reset jobs where `status = 'processing' AND started_at < now - timeout` back to `pending`.
2. Add a periodic watchdog query (or run it inside the worker's `start()` loop) to detect and reset timed-out processing jobs.
3. Add a `retry_count` column and move jobs exceeding N retries to `status = 'dead'` (dead-letter queue).
4. Expose `stuck_jobs` count in `/health`.

---

### C-2 — No port-conflict detection at startup; new process silently fails

**File:** `main.py` (startup), `config.py:PORT`  
**Scenario:** (Confirmed production incident.) An old process holds port 7422. Uvicorn's bind fails with `[Errno 48] Address already in use`. If run as a background daemon/launchd agent, the new process exits immediately with no user-visible error. Queries continue hitting the old (stale) process.  
**Impact:** Operators believe the service started successfully. All ingestion goes to the stale process. After the stale process is killed, the new one is already gone — service is fully down.  
**Mitigation:**
1. Write a PID file at startup (`config.GRAPH_MEMORY_DIR / "mollygraph.pid"`). On startup, check if the PID file exists and if that PID is alive; warn and attempt to clean up or refuse to start.
2. Add a pre-bind socket check: `socket.connect_ex(('127.0.0.1', PORT)) == 0` → log `CRITICAL: port already in use, refusing to start` and exit with code 1.
3. Add a health-check wrapper script that validates the PID file and does a `/health` request after startup before reporting success.

---

### C-3 — Partial entity/episode writes with no automatic rollback or recovery

**File:** `extraction/pipeline.py:process_job()`, lines ~260–510  
**Scenario:** Inside `process_job()`, writes happen in this sequence:
1. `upsert_entity()` for each entity → Neo4j ✓  
2. `vector_store.add_entity()` for each entity → Zvec ✓  
3. `create_episode()` → Neo4j ✓  
4. `upsert_relationship()` for each relationship → Neo4j ✓  

If Neo4j goes down between steps 1 and 4, or Zvec fails silently at step 2, the graph and vector store diverge. Entities exist in Neo4j without vectors. Episode may not be created. Relationships may be missing.  
**Impact:** Graph/vector divergence causes search quality degradation that's hard to diagnose. Partially-written episodes feed into training data with missing relationship context.  
**Existing mitigation:** `_mark_episode_incomplete()` stamps `ep.incomplete = true` and logs a WARNING — but only if `episode_id_written is not None` (i.e., episode was created before the failure). If the failure happens during entity writes (before episode creation), nothing is marked. Additionally, `_mark_episode_incomplete` itself can fail silently if Neo4j is still down.  
**Mitigation:**
1. Mark the episode at the START of the job (or mark the job payload itself) before any writes begin.
2. Add a startup reconciliation pass: query `MATCH (ep:Episode {incomplete: true})` and optionally re-queue those episode IDs for reprocessing.
3. Expose `incomplete_episodes` count in `/health` and `/stats`.

---

### C-4 — Zvec degraded mode: silent data loss, invisible to queries

**File:** `memory/vector_store.py:ZvecBackend._init_collection()`, lines ~260–295  
**Scenario:** If `zvec.open()` raises `RuntimeError` (e.g., corrupted collection, LOCK race), `self.collection = None`. All subsequent `add_entity()` calls log at DEBUG level and return silently. All `similarity_search()` calls return `[]`. The service starts and appears healthy — `/health` says "healthy" — but vector search returns nothing.  
**Impact:** The query endpoint falls through to graph-only search (degraded quality) with no indication that vectors are unavailable. Training data from degraded period uses only graph signals. The `/health` endpoint does NOT report vector degradation.  
**Mitigation:**
1. `get_stats()` already returns `{"degraded": True}` in the `collection is None` case — but `/health` doesn't surface this. Add a `vector_degraded` field to `/health` that checks `vector_store.get_stats().get("degraded", False)`.
2. Log at `ERROR` (not `DEBUG`) when `add_entity()` is a no-op due to degradation.
3. Consider raising on degraded write so the queue job fails with an explicit error rather than silently succeeding with incomplete data.

---

### C-5 — Audit auto-delete has no blast radius cap

**File:** `audit/llm_audit.py:apply_verdicts()`, `config.py:AUDIT_AUTO_DELETE`  
**Scenario:** When `MOLLYGRAPH_AUDIT_AUTO_DELETE=true`, every `verdict == "delete"` in the LLM response results in a real relationship deletion (`graph.delete_specific_relationship()`). A hallucinating LLM responding to a 500-item batch could return 500 `"delete"` verdicts. There is no per-cycle cap on deletions.  
**Impact:** An LLM hallucination or a misconfigured model could wipe the entire relationship graph in a single nightly run. There is no confirmation step, no "preview mode" that requires sign-off, and no automatic backup before the delete sweep.  
**Blast radius:** Up to `_BATCH_SIZE = 500` relationships deleted per audit invocation (unlimited invocations per cycle).  
**Mitigation:**
1. Add `AUDIT_MAX_AUTO_DELETE_PER_CYCLE` config (default 20). If the delete count would exceed this, abort and log `CRITICAL` instead.
2. Require `dry_run=True` by default for `delete` verdicts; only promote to real delete after operator review, or use a two-stage approach: first run marks for deletion (quarantine), a separate pass reviews and confirms.
3. Take a Neo4j dump (or at minimum a Cypher export of the affected relationships) before executing any deletes.

---

## HIGH

---

### H-1 — Neo4j down at startup → unrecoverable crash with no retry

**File:** `main.py:lifespan()`, `memory/graph/core.py:BiTemporalGraph.__init__()`  
**Scenario:** If Neo4j is down when `BiTemporalGraph.__init__()` is called, `_ensure_indexes()` → `session.run()` raises. The exception propagates up through `lifespan()` and crashes FastAPI startup. The process exits entirely.  
**Impact:** Transient Neo4j restarts (e.g., Docker container health check flap) cause the service to refuse to start. Requires manual restart. If the service is daemonized, it may not be automatically restarted.  
**Mitigation:**
1. Add retry loop in `lifespan()` for Neo4j connection: up to 5 attempts with exponential backoff (1, 2, 4, 8, 16s).
2. Separate driver creation from index creation: allow the service to start in a `neo4j_unavailable` state, report that in `/health`, and retry connectivity in the background.

---

### H-2 — Queue cleanup uses f-string interpolation (minor injection risk in internal API)

**File:** `extraction/queue.py:cleanup_old()`, line:
```python
AND completed_at < datetime('now', '-{} days')".format(days)
```
**Scenario:** `days` is a Python int (safe here), but the pattern is dangerous — if the call site ever passes a user-supplied string, this is SQL injection.  
**Impact:** Currently low risk (internal call), but a code smell that could become critical if the cleanup endpoint is ever exposed via API without sanitization.  
**Mitigation:** Use parameterized queries: `datetime('now', '-' || ? || ' days')` with `(days,)`.

---

### H-3 — Worker auto-restart limited to exactly 1 attempt, then permanent degraded state

**File:** `main.py:/health`, `_worker_restart_count`  
**Scenario:** Worker crashes, is restarted once. If it crashes again (e.g., persistent Neo4j failure), `_worker_restart_count >= 1` prevents further restarts. Service reports `"degraded"` indefinitely until a human intervenes.  
**Impact:** If the underlying cause of the crash is transient (e.g., brief Neo4j timeout), the service stays broken unnecessarily. No alert is sent — the degraded state is only visible on `/health` polling.  
**Mitigation:**
1. Implement exponential-backoff restart policy (max 5 attempts). Reset counter after a clean run of N minutes.
2. Log `CRITICAL` when `worker_restart_count` exceeds threshold (triggers monitoring alert if log ingestion is set up).
3. Expose `worker_restart_count` as a Prometheus metric or webhook alert.

---

### H-4 — Model health monitor state is in-memory only; survives neither restart nor rollback

**File:** `metrics/model_health.py:ModelHealthMonitor`  
**Scenario:** `baseline_fallback_rate` and all rolling windows are in-memory. On process restart (including the crash→restart cycles described in C-1, H-3), they reset to `None`. Until the next training run calls `set_baseline()`, `check_health()` returns `"monitoring"` (no baseline) and rollback never triggers.  
**Impact:** Post-restart quality degradation (e.g., bad model was deployed, then service restarted) goes undetected. The guard that should protect against bad LoRA deployments is silently disabled.  
**Mitigation:**
1. Persist baseline to disk (`GRAPH_MEMORY_DIR / "model_health_baseline.json"`) after `set_baseline()`.
2. On startup, load baseline from disk if present (with a staleness check — ignore if >7 days old).

---

### H-5 — Rollback silently aborts if backup model path doesn't exist

**File:** `metrics/model_health.py:_trigger_rollback()`, lines ~190–240  
**Scenario:** Rollback reads `backup_model_ref` from `gliner_finetune_config.json`. If that file doesn't exist, if `backup_model_ref` key is missing, or if the path doesn't exist on disk, the rollback logs `ERROR` and returns — the bad model remains active.  
**Impact:** The entire rollback safety mechanism fails silently. Quality degradation continues unmitigated. There is no alert beyond a log line.  
**Mitigation:**
1. At training deploy time, always verify `backup_model_ref` exists before stamping it.
2. If `_trigger_rollback()` fails, raise an exception so the failure propagates to monitoring (log at `CRITICAL`, send a structured alert).
3. Maintain the last 2 backups (already configured via `_GLINER_MAX_BACKUPS = 2`) but verify they exist at the time rollback is needed, not just at deploy time.

---

### H-6 — GLiNER model OOM: no memory limit, no graceful degradation

**File:** `memory/extractor.py` (not shown but imported), `extraction/pipeline.py`  
**Scenario:** (Confirmed production issue with Ollama eating 20GB.) GLiNER2 large model loads ~2-4GB of model weights at startup. Under memory pressure (Ollama KV cache, macOS unified memory), the `torch` inference call inside `asyncio.to_thread()` can OOM-kill the process with SIGKILL.  
**Impact:** Process dies mid-job, leaving stuck queue entries (C-1) and partial writes (C-3).  
**Mitigation:**
1. Add `MOLLYGRAPH_GLINER_MAX_MEMORY_MB` env var; before inference, check available memory with `psutil.virtual_memory().available` and skip extraction with a fallback result if below threshold.
2. Add Ollama keep-alive=0 config to prevent KV cache accumulation: `OLLAMA_KEEP_ALIVE=0` in the Ollama server's environment, or pass `keep_alive: 0` in API calls.
3. Consider lazy model loading: don't load GLiNER until first extraction request, rather than at startup.

---

### H-7 — Training pipeline: `state.json` parsed without holding lock; stale read race

**File:** `evolution/gliner_training.py:GLiNERTrainingService`, `_load_state()`, `refresh_state()`, `save_state()`  
**Scenario:** `save_state()` acquires `self._state_lock` and uses atomic rename (correct). However, `refresh_state()` calls `_load_state()` which reads from disk *without holding the lock*. If `refresh_state()` and `save_state()` race (e.g., from concurrent API calls to `/train` and `/maintenance`), `refresh_state()` may read a partial or stale file.  
**Impact:** Training cursor position or cooldown timestamps can be lost or misread, causing reprocessing of already-seen episodes or skipping training when it should run.  
**Mitigation:** Make `refresh_state()` acquire `self._state_lock` before calling `_load_state()`.

---

### H-8 — Training data: no input validation before writing to JSONL files

**File:** `evolution/gliner_training.py:accumulate_gliner_training_data()`, `build_training_example_from_episode()`  
**Scenario:** Training examples are built from episode content previews stored in Neo4j. There is no sanitization of the `source_text` field beyond length truncation. Malformed Neo4j data (e.g., mixed string/DateTime timestamp that breaks the Cypher query — confirmed production issue) can cause bad examples to be written or query failures to silently skip episodes.  
**Impact:** Corrupted or adversarial content in Neo4j can enter training data and degrade or poison the GLiNER2 model.  
**Mitigation:**
1. Validate training examples against a schema before writing (entity labels must be in `_ALLOWED_ENTITY_TYPES`, relation labels in `_ALLOWED_REL_TYPES`).
2. Check `source_text` length, encoding, and basic sanity (non-empty, not just whitespace) before including.
3. The existing 24-hour audit delay is a good safeguard — keep it.

---

### H-9 — Neo4j indexes created without UNIQUENESS constraint on Entity.name

**File:** `memory/graph/core.py:_ensure_indexes()`  
**Scenario:** `CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)` creates a non-unique index. Two concurrent `upsert_entity()` calls for the same entity name can race and create duplicate Entity nodes, because `MERGE` in entity upsert uses `name` + `entity_type` but without a unique constraint, Neo4j cannot enforce this at the DB level.  
**Impact:** Duplicate entity nodes accumulate. Graph queries that `MATCH (e:Entity {name: $name})` return multiple results; Cypher's `.single()` raises `ResultNotSingleError`. Relationships may be created between different copies of the "same" entity.  
**Mitigation:**
1. Add `CREATE CONSTRAINT entity_name_type_unique IF NOT EXISTS FOR (e:Entity) REQUIRE (e.name, e.entity_type) IS UNIQUE`.
2. Add a migration step that deduplicates existing Entity nodes before applying the constraint.

---

### H-10 — Zvec collection schema hardcodes embedding dimension=768

**File:** `memory/vector_store.py:ZvecBackend._init_collection()`, line:
```python
zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32, dimension=768, ...)
```
**Scenario:** The collection schema is created once. If the embedding provider is later switched to a model with a different output dimension (e.g., `nomic-embed-text` = 768, `all-MiniLM-L6-v2` = 384, Jina v5-nano = 512), `add_entity()` will fail silently or corrupt the index because the vector dimension doesn't match.  
**Impact:** Silent vector corruption or insertion failures when the embedding model changes, leading to degraded search with no error surfaced.  
**Mitigation:**
1. Store the embedding dimension in the collection metadata (or a sidecar file like `zvec_collection/schema.json`).
2. On startup, compare the configured embedding model's dimension to the stored dimension. If mismatched, refuse to start and prompt for explicit reindex.
3. The startup auto-reindex (in `main.py:lifespan()`) already handles empty Zvec — extend it to also check dimension mismatch.

---

### H-11 — audit_state.json written non-atomically; crash during write corrupts state

**File:** `audit/llm_audit.py:_save_audit_state()`, lines:
```python
with path.open("w", encoding="utf-8") as f:
    json.dump(state, f, indent=2, default=str)
```
**Scenario:** If the process is killed during the `json.dump()` write (e.g., OOM, SIGKILL), the file contains partial JSON. On next startup, `_load_audit_state()` fails to parse and returns `{}`, losing coverage metrics and `last_full_sweep` timestamp.  
**Impact:** The audit restarts from scratch, re-auditing already-reviewed relationships. Coverage metrics reset to zero.  
**Mitigation:** Use atomic write pattern (temp file + rename), same as `GLiNERTrainingService.save_state()` already does correctly.

---

## MEDIUM

---

### M-1 — `/health` endpoint does not verify Neo4j connectivity

**File:** `main.py:/health`  
**Scenario:** `/health` checks worker task liveness and vector stats but never runs a Neo4j query. If Neo4j is down, `/health` returns `"healthy"` as long as the worker task is running (the worker will fail on the next job claim, not on health check).  
**Impact:** Load balancers and monitoring systems see a healthy service while the database is down. Ingestion jobs queue up and fail silently.  
**Mitigation:** Add a lightweight Neo4j probe in `/health`: `session.run("RETURN 1")` with a 2s timeout. Cache result for 10s to avoid hot-path overhead.

---

### M-2 — Maintenance lock: `fcntl.flock` is process-level, not thread-level

**File:** `maintenance/lock.py:maintenance_lock()`  
**Scenario:** `fcntl.flock()` is held by the file descriptor, but in a Python async process all handlers share the same OS process. If two concurrent HTTP requests trigger `/maintenance` simultaneously (or if `/maintenance` is triggered while a previous one is still running), the second one sees `LOCK_NB` fail and raises `RuntimeError("Maintenance lock already held")` — which is correct. However, the lock is also checked in the queue worker via `is_maintenance_locked()`, which reads the lock file (no flock). The check is advisory — the queue could start a job between the `is_maintenance_locked()` check and the actual lock acquisition, creating a race window.  
**Impact:** In theory, a queue job could start processing (writing to Neo4j) while maintenance is actively mutating the graph (merge, delete, reclassify). This could produce inconsistent write ordering.  
**Mitigation:** The current approach is mostly safe given the async nature. To harden: add a short sleep after `is_maintenance_locked()` returns True and re-check before yielding, ensuring the queue worker waits out transient lock acquisition races.

---

### M-3 — Suggestion JSONL files grow unboundedly; no rotation/cleanup

**File:** `memory/graph_suggestions.py:_append_jsonl()`  
**Scenario:** Every rejected or fallback relationship is appended to `~/.graph-memory/suggestions/YYYY-MM-DD.jsonl`. On a busy day (many ingest calls), these files can grow to hundreds of MB. There is no cleanup path for files older than the build_suggestion_digest() window.  
**Impact:** Disk exhaustion over time. On a 256GB Mac mini with other workloads, this could contribute to disk full events.  
**Mitigation:** Add a cleanup job (in `run_maintenance_cycle()`) that deletes suggestion files older than 30 days. Or cap file size and rotate.

---

### M-4 — schema_drift.json written non-atomically

**File:** `main.py:/metrics/schema-drift`, line:
```python
drift_file.write_text(json.dumps(history, indent=2))
```
**Scenario:** Same issue as H-11. If the process dies mid-write, `drift_file` contains partial JSON. On next call to `/metrics/schema-drift`, `json.loads()` fails and `history = []`, losing all historical drift data.  
**Impact:** Schema drift detection loses history; alarms may not fire when they should.  
**Mitigation:** Use atomic write (temp + rename).

---

### M-5 — SQLite queue WAL never checkpointed; WAL can grow unboundedly

**File:** `extraction/queue.py:cleanup_old()`  
**Scenario:** `PRAGMA journal_mode=WAL` is set but `PRAGMA wal_checkpoint(FULL)` is never called. After `VACUUM` in `cleanup_old()`, the WAL file may not shrink. Under heavy ingestion, the WAL file (`extraction_queue.db-wal`) can grow to multiple GB.  
**Impact:** Disk pressure. SQLite reads become slower as WAL grows larger (every read must merge WAL into the base DB).  
**Mitigation:** Call `PRAGMA wal_checkpoint(TRUNCATE)` after `VACUUM` in `cleanup_old()`. Also set `PRAGMA wal_autocheckpoint=1000`.

---

### M-6 — Training: fixed benchmark seed means eval split is never independent

**File:** `evolution/gliner_training.py:GLINER_BENCHMARK_SEED = 1337`, `split_holdout_rows()`  
**Scenario:** `random.seed(GLINER_BENCHMARK_SEED)` is used every training run. The same episodes end up in the eval set every time. Over many training cycles, the model can overfit to this fixed eval set without it being detected.  
**Impact:** Benchmark scores look good but don't generalize to production data. The deploy/reject decision becomes less reliable over time.  
**Mitigation:** Rotate the seed each training run (e.g., `seed = int(datetime.now().timestamp()) % 100000`) or use stratified sampling with a new seed each run. Log the seed used in the run audit trail.

---

### M-7 — Training `accumulate` doesn't validate Cypher timestamp types; mixed string/DateTime crash

**File:** `evolution/gliner_training.py:accumulate_gliner_training_data()`, Cypher query with `datetime(coalesce(ep.created_at, toString(ep.ingested_at), toString(ep.occurred_at)))`  
**Scenario:** (Confirmed production issue.) Some Episode nodes have `created_at` as a Neo4j DateTime object, others have it as a string. The `coalesce()` + `datetime()` call works when the field is consistently one type. However, if `ep.ingested_at` or `ep.occurred_at` is a Neo4j DateTime, `toString()` produces a non-ISO-8601 format that `datetime()` rejects, causing a Cypher error that skips the entire batch.  
**Impact:** Training accumulation silently skips episodes, causing cursor to not advance. Eventually training never gets enough examples.  
**Mitigation:**
1. Normalize all Episode temporal properties to ISO-8601 strings at write time in `create_episode()`.
2. Run a one-time migration: `MATCH (ep:Episode) WHERE ep.created_at IS NOT NULL SET ep.created_at = toString(ep.created_at)`.
3. Add a `backfill_temporal_properties_sync()` run during maintenance that covers Episodes (currently it only covers Entity and generic relationship nodes — check `memory/graph/maintenance.py`).

---

### M-8 — Rollback does not invalidate the in-memory GLiNER model cache

**File:** `metrics/model_health.py:_trigger_rollback()`, `memory/extractor.py` (hot-reload via mtime)  
**Scenario:** After rollback swaps the active model directory, the in-memory GLiNER model (`_get_model()` mtime check) should detect the change on the next extraction. However, if the rollback's `temp_dir.rename(active_path)` doesn't change the mtime of the *directory itself* (only its contents), the mtime check may not trigger.  
**Impact:** The bad model remains in memory until the next manual restart or until a new extraction happens and the mtime check happens to fire.  
**Mitigation:** After rollback completes, explicitly call `memory_extractor.invalidate_model_cache()` (already available as `_refresh_extractor_runtime()` in `main.py`). Or touch a sentinel file in the model directory to guarantee mtime change.

---

### M-9 — No graceful drain of in-flight jobs on SIGTERM

**File:** `main.py:lifespan()` finally block  
**Scenario:** On SIGTERM (normal shutdown), the lifespan `finally` block calls `queue_worker.stop()` (sets `running = False`) and `_worker_task.cancel()`. Any jobs in `_tasks` (up to `max_concurrent=3`) are cancelled mid-processing via `asyncio.CancelledError`. Neo4j writes in progress are abandoned without `_mark_episode_incomplete()`.  
**Impact:** Same as C-3 — partial writes, stuck jobs — but triggered by every normal restart, not just crashes.  
**Mitigation:** In the `finally` block, wait for `self._tasks` to drain before cancelling the worker task:
```python
if queue_worker is not None:
    queue_worker.stop()
    # Give in-flight tasks a chance to complete
    pending = [t for t in queue_worker._tasks if not t.done()]
    if pending:
        await asyncio.wait(pending, timeout=30.0)
```

---

## LOW

---

### L-1 — Default API key is `dev-key-change-in-production`

**File:** `config.py:API_KEY`  
**Scenario:** The default API key is hardcoded as a well-known string. If `.env` is not configured in production, the service accepts this key from anyone.  
**Impact:** Unauthorized data ingestion, graph manipulation, or audit triggering from the local network.  
**Mitigation:** Add a startup check: if `API_KEY == "dev-key-change-in-production"` and `not TEST_MODE`, log `WARNING: using default dev API key — set MOLLYGRAPH_API_KEY in production`.

---

### L-2 — No structured alerting for CRITICAL conditions

**File:** Multiple  
**Scenario:** Model rollback (`model_health.py`), worker death (`main.py:/health`), audit blast (`llm_audit.py`), Neo4j down at startup — all produce log lines at `ERROR`/`CRITICAL` but nothing external (webhook, email, desktop notification). MollyGraph runs headlessly as a background service.  
**Impact:** Production incidents go unnoticed until a user notices degraded query quality.  
**Mitigation:** Add a simple alerting hook: `config.ALERT_WEBHOOK_URL` (optional). On `CRITICAL` events, POST a JSON payload to the webhook. OpenClaw could receive these via a local HTTP endpoint.

---

### L-3 — Backup model count capped at 2 but only 1 is ever reliably available

**File:** `evolution/gliner_training.py:_GLINER_MAX_BACKUPS = 2`, `deploy_gliner_candidate_model()`  
**Scenario:** Deploy renames current active → backup and installs candidate as active. With `_GLINER_MAX_BACKUPS = 2`, the older backup may be pruned. If rollback triggers, there's only 1 backup. If that backup is itself corrupt, rollback fails (see H-5).  
**Impact:** One bad training run can leave the system with no valid model if both the candidate and backup are corrupt.  
**Mitigation:** Keep 3 backups minimum. Verify each backup's integrity (file size > 0, key files present) before pruning older ones.

---

### L-4 — `cleanup_old()` runs VACUUM inline, which blocks SQLite writers

**File:** `extraction/queue.py:cleanup_old()`  
**Scenario:** `VACUUM` in SQLite is a blocking full-table-rewrite operation. Running it on the queue DB while the worker is actively claiming/completing jobs blocks all SQLite access for the duration.  
**Impact:** Queue processing pauses during vacuum (potentially several seconds). Under load this creates a visible latency spike.  
**Mitigation:** Run `VACUUM` in a separate connection with `PRAGMA auto_vacuum=INCREMENTAL` so cleanup happens incrementally, or run `cleanup_old()` during maintenance window when the queue is paused.

---

### L-5 — GLiNER training `_finetune_running` is a class variable (process-scoped, not persistent)

**File:** `evolution/gliner_training.py:GLiNERTrainingService._finetune_running`  
**Scenario:** `_finetune_running = False` is a class-level boolean. It prevents concurrent fine-tune runs within the same process. However, if two separate HTTP requests both hit `/train` before the first one sets the flag (e.g., due to async scheduling), both could see `_finetune_running = False` and both start training simultaneously.  
**Impact:** Two simultaneous training runs both write to `gliner_candidate/`, creating a race condition in model file writes.  
**Mitigation:** Replace the class boolean with a `asyncio.Lock` (or `threading.Lock`) acquired at the start of `run_gliner_finetune_pipeline()`. Current implementation has a TOCTOU gap between the `if _finetune_running` check and `GLiNERTrainingService._finetune_running = True`.

---

### L-6 — Ingestion counters reset on day rollover; no persistence across restarts

**File:** `extraction/pipeline.py:ExtractionPipeline._counter_date`, `_ingestion_counters`  
**Scenario:** Daily ingestion counters (`jobs_processed`, entity/relationship counts, fallback rates) are class-level in-memory only. They reset every midnight AND on every process restart. The `/metrics/dashboard` view can show misleading partial-day counts after a restart.  
**Impact:** Low — purely a metrics accuracy issue. Does not affect correctness.  
**Mitigation:** Persist counter snapshots to `LOGS_DIR / "daily_counters.json"` and load on startup if the file's date matches today's date.

---

## Backup / Recovery Assessment

| Asset | Current Backup | Recovery Path |
|-------|---------------|---------------|
| Neo4j graph data | **None** (no automated dump) | Manual `neo4j-admin dump` required. No restore procedure documented. |
| Zvec collection | **None** | Startup auto-reindex from Neo4j (entities only; episode vectors are lost). Quality degrades until reindex completes. |
| SQLite queue DB | **None** (WAL provides crash safety, not backup) | Lost jobs; manually re-ingest content. |
| GLiNER active model | 1-2 backups in `models/gliner_backup_*` | Rollback via `model_health_monitor._trigger_rollback()` — but see H-5 for failure modes. |
| Training examples (JSONL) | **None** | Lost on disk failure; would require re-accumulation from Neo4j episodes. |
| state.json | **None** | Lost; training cursor resets, causing full re-scan of episodes. |
| audit_state.json | **None** | Lost; audit coverage resets. |

**Recommendation:** Add a nightly backup job:
1. `neo4j-admin dump --database=neo4j --to=~/.graph-memory/backups/neo4j-$(date +%Y%m%d).dump`
2. `cp -r ~/.graph-memory/zvec_collection ~/.graph-memory/backups/zvec-$(date +%Y%m%d)/` (or skip if Zvec is rebuildable from Neo4j)
3. Keep 7 daily backups, rotating old ones.

---

## Monitoring Gaps

| Failure Mode | Currently Detected? | How to Add |
|---|---|---|
| Neo4j down mid-operation | NO — only detected when a job fails | Add Neo4j liveness probe to `/health` |
| Zvec degraded (collection=None) | NO — `/health` doesn't check | Add `vector_degraded` field to `/health` |
| Queue jobs stuck in `processing` | NO — no watchdog | Add `stuck_jobs_count` to `/health` |
| Port conflict on startup | NO | PID file check at startup |
| OOM kill (SIGKILL) | NO — process disappears | External process supervisor (launchd healthcheck) |
| Training data corruption | Partial (24h delay + stale cleanup) | Schema validation before JSONL write |
| Disk full | NO | Add `disk_free_mb` to `/health` using `shutil.disk_usage()` |
| GLiNER model mtime stale after rollback | NO | Call `invalidate_model_cache()` after rollback |

---

## Quick-Win Priority List (Recommended Order)

1. **[C-1]** Add stuck-job watchdog + dead-letter queue in `ExtractionQueue` (~50 lines)
2. **[C-4]** Surface Zvec degradation in `/health` (~5 lines)
3. **[M-1]** Add Neo4j liveness probe to `/health` (~10 lines)
4. **[C-3]** Mark episode incomplete at job START, before any writes (~5 lines)
5. **[H-11]** Atomic write for `audit_state.json` (~10 lines)
6. **[M-4]** Atomic write for `schema_drift.json` (~5 lines)
7. **[C-5]** Add `AUDIT_MAX_AUTO_DELETE_PER_CYCLE` cap (~15 lines)
8. **[H-4]** Persist model health baseline to disk (~20 lines)
9. **[M-9]** Graceful drain of in-flight jobs on SIGTERM (~15 lines)
10. **[C-2]** PID file + port conflict check at startup (~30 lines)

---

*Audit completed 2026-02-24. All file:line references based on codebase snapshot at that date.*
