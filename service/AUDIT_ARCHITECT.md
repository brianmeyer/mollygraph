# MollyGraph Architectural Audit
**Date:** 2026-02-24  
**Auditor role:** Senior Systems Architect  
**Codebase:** `/Users/brianmeyer/mollygraph/service`  
**Commit context:** Post-GLiREL enrichment, parallel retrieval, tiered fallbacks, auto-adoption, delete/prune endpoints, nightly audit, LoRA training loop

---

## Executive Summary

The codebase is architecturally sound for a personal-scale knowledge graph service. The async/thread boundary handling is mostly correct, the persistence layer is appropriate (SQLite WAL + Neo4j), and the fallback chains are thoughtfully tiered. However, there are **three concurrency bugs that can cause silent data corruption or crashes in production**, one dead-code path in the training loop, and several scalability cliffs that will hit around 10K–50K entities.

---

## CRITICAL Findings

### C-1 — `SqliteVecBackend` single connection shared across threads without mutex
**File:** `memory/vector_store.py` line ~92  
**Severity:** CRITICAL  

`SqliteVecBackend.__init__` opens one `sqlite3.Connection` and stores it as `self.db`. Every `asyncio.to_thread(vector_store.similarity_search, ...)` runs in a different OS thread from the thread pool. `sqlite3.connect(check_same_thread=False)` suppresses the default safety check, but SQLite connections are **not thread-safe for concurrent writes** — serialization must be enforced by the caller.

The query path (`main.py:1130`) runs `_graph_branch()` and `_vector_branch()` concurrently via `asyncio.gather`, where `_vector_branch` calls `asyncio.to_thread(vector_store.similarity_search, ...)`. If an ingestion job is simultaneously running `asyncio.to_thread(vector_store.add_entity, ...)` from the queue worker, two OS threads can be executing SQLite calls on `self.db` simultaneously.

**Impact:** Crashes (`DatabaseError: database disk image is malformed`), silent corruption of `dense_vectors` or `entity_meta` tables. Hard to reproduce locally but guaranteed to surface under load.

**Fix:** Add a `threading.Lock` to `SqliteVecBackend` and acquire it around every public method that touches `self.db`:

```python
class SqliteVecBackend(VectorStoreBackend):
    def __init__(self, ...):
        ...
        self._lock = threading.Lock()
    
    def add_entity(self, ...):
        with self._lock:
            ...  # all self.db calls
    
    def similarity_search(self, ...):
        with self._lock:
            ...
```

Note: `ZvecBackend` should be audited for the same issue — Zvec's thread-safety guarantees are not documented in this codebase.

---

### C-2 — `require_graph_instance()` leaks un-registered Neo4j driver instances
**File:** `runtime_graph.py` lines 28–38  
**Severity:** CRITICAL  

```python
def require_graph_instance() -> BiTemporalGraph:
    global _GRAPH_INSTANCE
    if _GRAPH_INSTANCE is not None:        # read without lock
        return _GRAPH_INSTANCE

    with _GRAPH_LOCK:
        if _GRAPH_INSTANCE is None:        # correct double-check
            _GRAPH_INSTANCE = BiTemporalGraph(...)
        return _GRAPH_INSTANCE
```

The pattern is mostly correct, but there is a critical gap: when `require_graph_instance()` creates and stores `_GRAPH_INSTANCE`, it does NOT call `set_graph_instance()`. This means the new instance bypasses the close-old-instance logic in `set_graph_instance()`.

More critically: if `set_graph_instance(None)` is called (e.g., during shutdown cleanup), then `get_graph_instance()` returns `None`, but `require_graph_instance()` creates a **new** `BiTemporalGraph` instance and stashes it in `_GRAPH_INSTANCE`. This newly created instance is never registered through `set_graph_instance`, so if `set_graph_instance(new_graph)` is later called again, the instance created by `require_graph_instance()` is **never closed** — leaking the Neo4j connection pool.

**Impact:** Neo4j connection pool exhaustion over time (default pool size 100); `ServiceUnavailable` errors.

**Fix:** Replace the body of `require_graph_instance()`:

```python
def require_graph_instance() -> BiTemporalGraph:
    if _GRAPH_INSTANCE is not None:
        return _GRAPH_INSTANCE
    with _GRAPH_LOCK:
        if _GRAPH_INSTANCE is None:
            new_instance = BiTemporalGraph(config.NEO4J_URI, config.NEO4J_USER, config.NEO4J_PASSWORD)
            # Use set_graph_instance to go through the proper registration path
            _GRAPH_INSTANCE = new_instance  # still need direct set to avoid re-entry
        return _GRAPH_INSTANCE
```

Better yet: move `set_graph_instance` to be the **only** place that writes `_GRAPH_INSTANCE` and have `require_graph_instance` call `set_graph_instance` if uninitialized.

---

### C-3 — `maintenance_lock` is ineffective against concurrent API-triggered deletes
**File:** `maintenance/lock.py`, `main.py` delete/prune endpoints  
**Severity:** CRITICAL (data integrity)

The maintenance lock prevents the **queue worker** from claiming new jobs while `run_maintenance_cycle()` is active (via `is_maintenance_locked()` check in `QueueWorker.start()`). However, the delete endpoints in `main.py` (`/entity/{name}`, `/prune`) call `graph.delete_entity()` and `vector_store.remove_entity()` **without checking the lock**:

```python
# main.py ~line 824 — no lock check
deleted = await asyncio.to_thread(graph.delete_entity, name)
```

Meanwhile, `run_maintenance_cycle()` may be iterating over entity sets to audit, re-index, or prune. A concurrent delete of an entity mid-audit can produce:
- `KeyError` / `NodeNotFound` in the maintenance pipeline (absorbed silently)
- Audit state file recording a verdict for a now-deleted entity, causing the next cycle to error on a no-op apply

**Impact:** Audit state corruption, orphaned vector entries (vector deleted but graph not, or vice versa in a race).

**Fix:** Add a lock check to delete and prune endpoints:

```python
# In delete_entity_endpoint and prune_entities_endpoint:
if is_maintenance_locked():
    raise HTTPException(503, "Maintenance in progress — retry in a few minutes")
```

Or better: implement a proper `asyncio.Event` or `asyncio.Lock` at the service level so delete/prune can coordinate with maintenance without blocking API availability long-term.

---

## HIGH Findings

### H-1 — Dead code: `_ALLOWED_REL_TYPES` is never populated in `gliner_training.py`
**File:** `evolution/gliner_training.py` lines ~35–48  
**Severity:** HIGH (training data quality)

```python
# Module level
_ALLOWED_REL_TYPES: frozenset[str] = frozenset()  # always empty

def _get_allowed_rel_types() -> frozenset[str]:
    ...  # computes types from schema
```

`_get_allowed_rel_types()` is defined but **never called**. `_ALLOWED_REL_TYPES` remains an empty frozenset. Any code that gates on `_ALLOWED_REL_TYPES` (e.g., filtering training examples) silently accepts all relation types, defeating the schema-aligned training goal.

**Fix:** Call `_get_allowed_rel_types()` at module load time or lazily on first use:

```python
def _ensure_rel_types_loaded():
    global _ALLOWED_REL_TYPES
    if not _ALLOWED_REL_TYPES:
        _ALLOWED_REL_TYPES = _get_allowed_rel_types()
```

Call this at the top of any function that uses `_ALLOWED_REL_TYPES`.

---

### H-2 — SQL injection (soft) in `ExtractionQueue.cleanup_old()`
**File:** `extraction/queue.py` lines ~162–172  
**Severity:** HIGH (defensive coding required for any public API)

```python
def cleanup_old(self, days: int = 7):
    with self._get_conn() as conn:
        conn.execute("""
            DELETE FROM jobs 
            WHERE status IN ('completed', 'failed')
            AND completed_at < datetime('now', '-{} days')
        """.format(days))
```

`days` is typed as `int` and validated by Python's type system only at the call site. If `cleanup_old` is ever called via an API endpoint that parses user input, a crafted value (`days = "7); DROP TABLE jobs; --"`) bypasses the int cast only if the endpoint omits validation. Not currently exploitable but violates the parameterized-query contract SQLite provides.

**Fix:**
```python
conn.execute(
    "DELETE FROM jobs WHERE status IN ('completed', 'failed') "
    "AND completed_at < datetime('now', ? || ' days')",
    (f"-{int(days)}",)
)
```

---

### H-3 — `ExtractionQueue._get_conn()` creates a new connection per call; `VACUUM` inside transaction
**File:** `extraction/queue.py` lines ~65–75, ~170  
**Severity:** HIGH (performance + correctness)

1. A new `sqlite3.connect()` is opened and closed on every single queue operation (submit, claim, complete, etc.). At high ingestion rates this adds ~2–5ms of OS overhead per call and can exhaust file descriptors.

2. `cleanup_old()` calls `conn.execute("VACUUM")` inside the `_get_conn()` context. VACUUM cannot run inside a transaction and will throw `OperationalError: cannot VACUUM from within a transaction` in WAL mode when called this way.

**Fix for (1):** Use a connection pool or a single persistent connection protected by `threading.Lock()`.

**Fix for (2):**
```python
def cleanup_old(self, days: int = 7):
    with self._get_conn() as conn:
        conn.execute(...)
        conn.commit()
    # VACUUM outside the context manager (outside any transaction)
    with self._get_conn() as conn:
        conn.isolation_level = None  # autocommit
        conn.execute("VACUUM")
```

---

### H-4 — `BiTemporalGraph._ensure_indexes()` only creates audit indexes for 7 hardcoded rel types
**File:** `memory/graph/core.py` lines ~30–40  
**Severity:** HIGH (query performance)

```python
for rel_type in ["WORKS_AT", "KNOWS", "USES", "MENTIONS", "DISCUSSED_WITH", "IS_A", "PART_OF"]:
    session.run(f"CREATE INDEX rel_audit_status_{rel_type} ...")
```

`VALID_REL_TYPES` in `memory/graph/constants.py` contains ~40 relationship types. Any rel type not in this hardcoded list will execute full relationship scans when filtering by `audit_status`. At 100K relationships, missing indexes cause the nightly audit scan to degrade from O(log n) to O(n).

**Fix:** Drive index creation from `VALID_REL_TYPES`:

```python
from .constants import VALID_REL_TYPES

for rel_type in VALID_REL_TYPES:
    session.run(
        f"CREATE INDEX rel_audit_status_{rel_type} IF NOT EXISTS "
        f"FOR ()-[r:{rel_type}]-() ON (r.audit_status)"
    )
```

---

### H-5 — `_save_audit_state` / `_load_audit_state` have no concurrency protection
**File:** `audit/llm_audit.py` (state file I/O functions)  
**Severity:** HIGH (audit state corruption)

The audit state JSON file is read and written by `run_llm_audit()` with no lock. The `/audit` endpoint is a regular async FastAPI handler — two concurrent POST requests to `/audit` will simultaneously read stale state, process overlapping entity sets, and write conflicting results. The last write wins, potentially rolling back audit progress.

**Fix:** Add a module-level `asyncio.Lock`:

```python
_AUDIT_LOCK = asyncio.Lock()

async def run_llm_audit(...):
    async with _AUDIT_LOCK:
        ...
```

Or at minimum add a file lock using `fcntl.flock` mirroring `maintenance_lock.py`.

---

### H-6 — `graph_suggestions.py` auto-adoption modifies `VALID_REL_TYPES` and extractor schema without invalidating `_REGISTRY_CACHE`
**File:** `memory/graph_suggestions.py` (auto-adoption path)  
**Severity:** HIGH (stale schema in active extractors)

When the auto-adoption pipeline promotes a new relation type, it presumably writes to the extractor schema file. However, `extractor_schema_registry.py` maintains `_REGISTRY_CACHE` that is only invalidated on explicit calls to `set_active_extractor_schema()` or `upload_custom_extractor_schema()`. If `graph_suggestions.py` writes directly to the schema file without going through these functions, the in-memory cache serves stale schema until restart.

**Fix:** Auto-adoption must call `set_active_extractor_schema()` or invalidate `_REGISTRY_CACHE` explicitly:

```python
# In graph_suggestions.py adopt_suggestions():
from extractor_schema_registry import upload_custom_extractor_schema, initialize_extractor_schema_registry
# After schema file update:
global _REGISTRY_CACHE
with _REGISTRY_LOCK:
    _REGISTRY_CACHE = None  # force reload on next access
```

---

## MEDIUM Findings

### M-1 — Parallel `asyncio.gather` in query bypasses backpressure
**File:** `main.py` line ~1130  
**Severity:** MEDIUM

```python
(vector_results, embedding_ms, vector_search_ms) = await asyncio.gather(
    _vector_branch(),
    ...
)
```

`_graph_branch()` and `_vector_branch()` both call `asyncio.to_thread(...)`, dispatching CPU/IO work to the thread pool. Under concurrent query load (N simultaneous `/query` requests), this creates `2*N` thread pool tasks simultaneously, potentially starving the queue worker's `asyncio.to_thread` calls and introducing latency spikes.

**Fix:** Consider a `asyncio.Semaphore` on the query path, or move reranking/embedding to a dedicated thread pool with bounded size.

---

### M-2 — `QueueWorker._process_job` silently discards task exceptions
**File:** `extraction/queue.py` lines ~237–260  
**Severity:** MEDIUM

```python
task = asyncio.create_task(self._process_job(job))
self._tasks.add(task)
task.add_done_callback(self._tasks.discard)
```

If `_process_job` raises an unhandled exception (outside the inner try/except), Python logs a warning about "Task exception was never retrieved" but the exception is lost. The outer `try/except` in `_process_job` catches most cases, but `asyncio.to_thread(self.queue.complete, ...)` can itself fail (e.g., disk full), which is not caught.

**Fix:**
```python
def _on_task_done(self, task: asyncio.Task) -> None:
    self._tasks.discard(task)
    if not task.cancelled() and task.exception() is not None:
        log.error("Queue worker task failed unexpectedly", exc_info=task.exception())

task.add_done_callback(self._on_task_done)
```

---

### M-3 — `GLiNERTrainingService.state` dict mutated without lock in multiple places
**File:** `evolution/gliner_training.py`  
**Severity:** MEDIUM (training state corruption)

`self._state_lock` guards `save_state()` (the file write), but `self.state["last_lora_run"] = ...`, `self.state.get(...)` etc. are scattered throughout methods without locking. Since `run_gliner_finetune_pipeline()` in `main.py` calls into `GLiNERTrainingService` via `asyncio.to_thread`, and the training service might be instantiated as a singleton, two concurrent `/train` API calls (or a manual call + nightly cycle) can corrupt `self.state`.

**Fix:** Either acquire `self._state_lock` around all `self.state` mutations (not just saves), or make `state` a copy-on-write structure.

---

### M-4 — `VectorStore` backend switching leaves orphaned data
**File:** `memory/vector_store.py`, `main.py` embedding config endpoint  
**Severity:** MEDIUM (data integrity)

If `MOLLYGRAPH_VECTOR_BACKEND` is changed from `zvec` to `sqlitevec` at runtime (via the embedding config API), a new `VectorStore` is created pointing to a different backend database. Entities written before the switch remain in the old backend but the new `VectorStore` has no data. Queries return empty results for previously ingested content. There's no migration path.

**Fix:** Document that backend switching requires a full re-index. Enforce this at the API level — reject backend switch requests if the current backend has data, unless `force_reindex=true` is passed.

---

### M-5 — `glirel_enrichment.py` loads GLiREL model into memory per-worker concurrently
**File:** `extraction/glirel_enrichment.py`  
**Severity:** MEDIUM (memory spike at startup)

GLiREL model loading is gated by a module-level singleton (`_glirel_model`) and a `threading.Lock`. However, if `GLIREL_ENABLED=true` and `max_concurrent=3` in `QueueWorker`, three jobs can simultaneously trigger `_load_glirel_model()`. The first acquires the lock and loads; the others wait. This is correct for correctness but causes a ~2–4 GB RSS spike at startup when all 3 workers initialize simultaneously under load.

**Fix:** Eagerly load the GLiREL model at service startup in `lifespan()`, not lazily on first job. This moves the spike to startup time and prevents per-job latency hits.

---

### M-6 — `audit/signals.py` JSONL log file grows unbounded
**File:** `audit/signals.py`  
**Severity:** MEDIUM (disk)

Signal events are appended to a JSONL file with no rotation or size cap. At 100 ingestions/day × 365 days, this file grows to tens of MB and stat() operations on it become noticeable. At 1M ingestions, the file is GB-scale.

**Fix:** Implement log rotation (Python's `logging.handlers.RotatingFileHandler` pattern, or explicit size/age check on write). Cap at 50 MB or 30 days.

---

### M-7 — `cleanup_old` in ExtractionQueue never called automatically
**File:** `extraction/queue.py`  
**Severity:** MEDIUM (disk growth)

`ExtractionQueue.cleanup_old()` is defined but there's no scheduled call in `lifespan()` or maintenance cycle. Completed/failed jobs accumulate indefinitely in `extraction_queue.db`.

**Fix:** Add a daily cleanup call in `run_maintenance_cycle()`:

```python
queue.cleanup_old(days=7)
```

---

## LOW Findings

### L-1 — `config.py` loads `.env` at import time with `override=False`
**File:** `config.py` line 7  
**Severity:** LOW

`load_dotenv(..., override=False)` is correct for not stomping environment variables set by Docker/systemd. However, the `.env` file path is hardcoded to `Path(__file__).parent / ".env"`, meaning `.env` must live in the service directory. This is fine for dev but creates a footgun if `.env` is committed to git (it currently contains `NEO4J_PASSWORD=mollygraph` as default).

**Recommendation:** Add `.env` to `.gitignore`; document that production deployments must set env vars via their deployment mechanism.

---

### L-2 — `BiTemporalGraph.__init__` calls `_ensure_indexes()` synchronously on the neo4j session
**File:** `memory/graph/core.py` lines ~13–16  
**Severity:** LOW

`_ensure_indexes()` runs ~14 `CREATE INDEX IF NOT EXISTS` DDL statements synchronously during `__init__`. If called concurrently by two coroutines (e.g., two simultaneous requests hitting `require_graph_instance()` before initialization completes), both will attempt DDL on Neo4j. Neo4j handles this gracefully (idempotent), but both threads will wait for the same schema lock, adding 100–500ms to startup.

**Recommendation:** Move index creation to a one-time startup task in `lifespan()` after `set_graph_instance()` is called.

---

### L-3 — `maintenance/lock.py` `_cleanup_stale_lock()` has TOCTOU between `exists()` and `read_text()`
**File:** `maintenance/lock.py` lines ~45–70  
**Severity:** LOW

```python
if not _LOCK_PATH.exists():
    return False
content = _LOCK_PATH.read_text().strip()
```

If the lock file is deleted between `exists()` and `read_text()` (e.g., maintenance cycle just finished), `read_text()` raises `FileNotFoundError` which is not caught. In practice this is a 1-in-10,000 race, but the exception would crash the queue worker's poll loop.

**Fix:**
```python
try:
    content = _LOCK_PATH.read_text().strip()
except FileNotFoundError:
    return False
```

---

### L-4 — `gliner_training.py` training JSONL files have no size cap
**File:** `evolution/gliner_training.py`, config `TRAINING_DIR`  
**Severity:** LOW

Training examples accumulate in `~/.graph-memory/training/gliner/*.jsonl`. `GLINER_TRAINING_SCAN_LIMIT` caps how many are loaded for training (default 4000) but doesn't delete old files. Over months of daily ingestion, this directory can accumulate hundreds of MB.

**Fix:** After each training run, archive or rotate JSONL files older than N days (configurable). A simple approach: delete files beyond the last `GLINER_TRAINING_SCAN_LIMIT` examples by mtime.

---

### L-5 — `extractor_schema_registry._REGISTRY_CACHE` read outside lock in `get_effective_extractor_schema`
**File:** `extractor_schema_registry.py` line ~220+  
**Severity:** LOW

`get_effective_extractor_schema()` calls `initialize_extractor_schema_registry()` which acquires `_REGISTRY_LOCK`, then calls `_effective_schema_for_registry(registry)` with a `copy.deepcopy` result. This is safe. However, `_effective_schema_for_registry` is also called directly from `set_active_extractor_schema` **inside** the lock — which is also fine. No actual bug here, but the pattern of calling `_effective_schema_for_registry()` from both inside and outside the lock makes future bugs likely.

**Recommendation:** Consistently ensure `_effective_schema_for_registry` is only called with the lock held, or document clearly that it's safe to call without it (it doesn't access shared mutable state).

---

## Scalability Analysis

### At 10K entities (current scale)
Everything works. Neo4j query times are fast (< 10ms), vector search in SQLite-vec is fast (< 5ms), audit batch of 500 entities completes in minutes.

### At 100K entities  
**Will break:**
- `BiTemporalGraph._ensure_indexes()` missing 33+ rel type indexes (finding H-4) causes audit scans to degrade to full graph scans: ~30 seconds per audit instead of < 1 second.
- `SqliteVecBackend` single-connection without mutex (finding C-1) will cause intermittent crashes under concurrent query/ingest load.
- `audit/signals.py` JSONL file reaches ~50MB, causing measurable stat() overhead on every write.
- `run_maintenance_cycle` fetches all entities for audit in memory. Need to stream/paginate from Neo4j instead of loading all at once.

### At 1M entities
**Will break:**
- `VectorStore.list_all_entity_ids()` returns a list of 1M strings — used during re-index. This is an O(n) memory allocation (~80MB for 1M 64-char IDs). Need pagination.
- Training scan limit `GLINER_TRAINING_SCAN_LIMIT=4000` is reasonable, but `load_audit_feedback_entries()` in `evolution/audit_feedback.py` may load all feedback into memory.
- Neo4j full-graph traversals in `get_graph_stats()` or similar maintenance calls will timeout at default 30s Neo4j transaction timeout.
- `extraction_queue.db` with 1M historical jobs (no auto-cleanup — see M-7) becomes a multi-GB SQLite file. WAL mode degrades when WAL file is large.

### At 10M entities  
Fundamental architectural shift required: Neo4j single node bottleneck, SQLite-vec is not designed for this scale (ZVec is better but still single-process). Would need Neo4j Enterprise + sharded vector index (Milvus/Qdrant/Weaviate).

---

## Module Boundary Assessment

### Well-bounded (no changes recommended)
- `extraction/queue.py` — clean separation of persistence from processing
- `maintenance/lock.py` — correct single-responsibility file lock
- `embedding_registry.py` — good tier chain abstraction
- `extractor_schema_registry.py` — clean registry with proper copy-on-write semantics

### Should be split
- **`main.py`** is a god-module: API routing, job submission logic, query orchestration, re-index logic, delete/prune logic, and startup sequencing all live in one 1800+ line file. Recommend extracting:
  - `api/query.py` — query endpoint + parallel retrieval logic
  - `api/admin.py` — embeddings config, schema config, training trigger
  - `api/ingest.py` — ingest endpoint + validation
  - `startup.py` — lifespan initialization

- **`memory/graph/relationships.py`** calls into `memory/graph_suggestions.py` (logging rejections). This creates a dependency from the core graph layer back to the suggestions/adoption layer — a layer violation. The suggestion logging should be done in the pipeline layer, not in the relationship upsert.

### Should be merged
- `runtime_graph.py`, `runtime_pipeline.py`, `runtime_vector_store.py` are three nearly identical singletons with identical patterns. These could be a single `runtime_registry.py` with a generic `ServiceRegistry` class holding all three. Current pattern has copy-paste divergence risk.

---

## Data Flow Integrity Gaps

### Ingestion → Storage: Partial commit risk
**File:** `extraction/pipeline.py`  
After GLiNER + GLiREL extraction, entities are written to Neo4j and vector store in two separate calls. If Neo4j write succeeds but vector store write fails (e.g., SQLite-vec disk full), the graph has the entity but the vector index doesn't. Queries will find it via graph traversal but not via semantic search. There's no compensating transaction or retry.

**Fix:** Implement a two-phase approach: write to vector store first (idempotent `INSERT OR REPLACE`), then write to Neo4j. On failure, log the entity ID for re-indexing. Or add a reconciliation step in maintenance that finds graph entities missing from vector store.

### Audit → Apply: Partial batch commit
**File:** `audit/llm_audit.py`  
Audit verdicts are applied per-batch to Neo4j. If `run_llm_audit()` processes batches 1–3 and the LLM call fails on batch 4, batches 1–3 are committed and the audit state file is updated to reflect progress. Batch 4 and beyond are left un-audited. On next run, the state file's progress pointer skips already-processed entities — correct. However, if the state file itself fails to write after batch 3 (disk full), the next run re-audits batches 1–3, applying verdicts twice (idempotent for most operations, but `delete` twice is a no-op while `reclassify` twice could flip type back).

**Fix:** Write audit state atomically using temp-file + rename (already done in `save_state()` for GLiNER — apply the same pattern to audit state).

### Delete endpoints: Vector/Graph inconsistency
**File:** `main.py` `delete_entity_endpoint()` lines ~824–833  

```python
deleted = await asyncio.to_thread(graph.delete_entity, name)
...
vector_removed = bool(await asyncio.to_thread(vector_store.remove_entity, entity_id))
```

If Neo4j delete succeeds and vector store delete fails (or vice versa), the entity is in a half-deleted state. The endpoint returns 200 (or 404) but the stores are inconsistent. This is made worse by C-3 (maintenance running concurrently).

**Fix:** Move delete to a transaction wrapper that retries vector store deletion and logs failures for reconciliation. Add a maintenance task that detects graph/vector inconsistencies.

---

## Configuration Architecture

### What works well
- `dotenv` with `override=False` correctly defers to system env
- All config values are module-level constants, not mutable after startup
- Tier chain config (`EMBEDDING_TIER_ORDER`, `AUDIT_PROVIDER_TIERS`) provides clean fallback configuration

### Remaining stomping risk
1. `config.py` is imported before `.env` is loaded in some edge cases if a module imports `config` at the top level during test discovery. The `load_dotenv()` at line 7 of `config.py` fires on first import — this is fine for production but can cause test environment pollution if the `.env` file exists in the service directory during test runs.

2. `EXTRACTOR_BACKEND` is always forced to `"gliner2"` regardless of env var (`config.py` lines ~130–138). The env var is read but immediately overridden. If someone sets `MOLLYGRAPH_EXTRACTOR_BACKEND=future_backend` expecting it to work, it silently falls back. The error log is emitted, but `EXTRACTOR_BACKEND` is correct. This is intentional but surprising.

3. `VALID_REL_TYPES` in `memory/graph/constants.py` is a compile-time constant that must be manually kept in sync with `extractor_schema_registry.py`'s `DEFAULT_RELATION_SCHEMA`. When a new relation type is added to one, it must be added to the other. No validation enforces this alignment.

**Fix:** Derive `VALID_REL_TYPES` dynamically from `DEFAULT_RELATION_SCHEMA` at startup, rather than maintaining a separate list.

---

## Summary Table

| ID | Severity | File | Description |
|----|----------|------|-------------|
| C-1 | CRITICAL | `memory/vector_store.py:92` | SQLiteVecBackend single connection, no mutex — concurrent thread corruption |
| C-2 | CRITICAL | `runtime_graph.py:28` | `require_graph_instance()` bypasses registration, leaks Neo4j connections |
| C-3 | CRITICAL | `main.py:824`, `maintenance/lock.py` | Delete/prune endpoints don't check maintenance lock — data integrity race |
| H-1 | HIGH | `evolution/gliner_training.py:35` | `_ALLOWED_REL_TYPES` never populated — dead code, training not schema-filtered |
| H-2 | HIGH | `extraction/queue.py:162` | `cleanup_old` uses string formatting for SQL — parameterize it |
| H-3 | HIGH | `extraction/queue.py:65,170` | New sqlite3 connection per call + VACUUM inside transaction |
| H-4 | HIGH | `memory/graph/core.py:30` | Only 7/40+ rel types get audit_status indexes |
| H-5 | HIGH | `audit/llm_audit.py` | Audit state file read/write unprotected — concurrent audit requests corrupt state |
| H-6 | HIGH | `memory/graph_suggestions.py` | Auto-adoption doesn't invalidate `_REGISTRY_CACHE` — stale schema served |
| M-1 | MEDIUM | `main.py:1130` | Parallel gather in query unbounded — threadpool saturation under load |
| M-2 | MEDIUM | `extraction/queue.py:250` | Unhandled task exceptions silently discarded |
| M-3 | MEDIUM | `evolution/gliner_training.py` | `state` dict mutated without lock |
| M-4 | MEDIUM | `memory/vector_store.py` | Backend switching leaves orphaned data, no migration path |
| M-5 | MEDIUM | `extraction/glirel_enrichment.py` | GLiREL model lazy-loaded per worker — 2–4GB RSS spike under load |
| M-6 | MEDIUM | `audit/signals.py` | Signal JSONL grows unbounded |
| M-7 | MEDIUM | `extraction/queue.py` | `cleanup_old()` never automatically called |
| L-1 | LOW | `config.py:7` | `.env` committed path risk |
| L-2 | LOW | `memory/graph/core.py:13` | Index creation in constructor, not startup lifespan |
| L-3 | LOW | `maintenance/lock.py:47` | TOCTOU in `_cleanup_stale_lock` — uncaught `FileNotFoundError` |
| L-4 | LOW | `evolution/gliner_training.py` | Training JSONL files accumulate without rotation |
| L-5 | LOW | `extractor_schema_registry.py` | `_REGISTRY_CACHE` read/call pattern inconsistent |

---

*Audit written by senior systems architect persona. All file:line references based on code read 2026-02-24.*
