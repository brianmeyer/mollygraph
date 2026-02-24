# MollyGraph Service — Developer Code Audit

**Date:** 2026-02-24  
**Reviewer:** Senior Developer Persona (automated)  
**Scope:** `extraction/pipeline.py`, `extraction/glirel_enrichment.py`, `extraction/glirel_synonyms.py`, `memory/graph_suggestions.py`, `audit/llm_audit.py`, `main.py`, `memory/vector_store.py`, `evolution/gliner_training.py`

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 1 |
| HIGH     | 3 |
| MEDIUM   | 6 |
| LOW      | 5 |

---

## CRITICAL

---

### C-1: Cypher Injection via `rel_type` f-string interpolation

**File:** `main.py:871`  
**Severity:** CRITICAL  
**Category:** Security / Injection

**Code:**
```python
rec = _session.run(
    f"""
    MATCH (h:Entity {{name: $source}})-[r:`{rel_type}`]-(t:Entity {{name: $target}})
    DELETE r
    RETURN count(r) AS deleted
    """,
    source=source,
    target=target,
).single()
```

`rel_type` comes directly from the API request body (`req.rel_type`), is only `.strip()`-ped, and is interpolated into the Cypher query using an f-string. The backtick-quoting of Neo4j relationship type identifiers is **not injection-safe** — a value containing a backtick (e.g. `` `]+foo[ `` or `` ` DETACH DELETE (:Entity) //`` ) can break out of the identifier scope and inject arbitrary Cypher. Neo4j's Cypher parser processes the entire query string before executing, so there is no parameterisation boundary protecting the relationship type label.

**`DeleteRelationshipRequest` model (main.py:205):**
```python
class DeleteRelationshipRequest(BaseModel):
    source: str
    target: str
    rel_type: str | None = None   # ← no pattern validation
```

**Fix — validate against the allowlist before interpolation:**
```python
from memory.graph import VALID_REL_TYPES

rel_type = req.rel_type.strip() if req.rel_type else None

if rel_type and rel_type.upper().replace(" ", "_") not in VALID_REL_TYPES:
    raise HTTPException(
        status_code=422,
        detail=f"Unknown rel_type '{rel_type}'. Must be one of: {sorted(VALID_REL_TYPES)}",
    )

# Now safe to interpolate — value is in the allowlist
```

Alternatively, add a Pydantic validator on `DeleteRelationshipRequest`:
```python
from pydantic import field_validator

class DeleteRelationshipRequest(BaseModel):
    source: str
    target: str
    rel_type: str | None = None

    @field_validator("rel_type")
    @classmethod
    def validate_rel_type(cls, v: str | None) -> str | None:
        if v is None:
            return None
        normalized = v.strip().upper().replace(" ", "_")
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", normalized):
            raise ValueError(f"rel_type must be UPPER_UNDERSCORE identifiers only, got: {v!r}")
        return normalized
```

---

## HIGH

---

### H-1: `SqliteVecBackend` connection used across threads (thread-safety violation)

**File:** `memory/vector_store.py:76-84` (connection creation) and callers  
**Severity:** HIGH  
**Category:** Resource Management / Thread Safety

**Code:**
```python
class SqliteVecBackend(VectorStoreBackend):
    def __init__(self, db_path: str | Path | None = None):
        ...
        self.db = sqlite3.connect(str(self.db_path))   # ← created in __init__ thread
        ...

    def add_entity(self, ...):
        self.db.execute(...)   # ← called from worker threads
```

`sqlite3.connect()` returns a connection tied to the calling thread by default. The connection is created in `__init__` (on the main/event-loop thread). `refresh_stale_embeddings` (pipeline.py:889) calls `self.vector_store.add_entity(...)` inside `await asyncio.to_thread(...)`, which runs on a `ThreadPoolExecutor` worker thread. The `_reindex_embeddings_sync` path (called via `asyncio.to_thread` in main.py) also calls `add_entity`. These cross-thread uses will raise `ProgrammingError: SQLite objects created in a thread can only be used in that thread`.

**Fix — use `check_same_thread=False` with a threading lock, or create per-thread connections:**
```python
import threading

class SqliteVecBackend(VectorStoreBackend):
    def __init__(self, db_path: ...):
        ...
        # thread_local for per-thread connections avoids locking overhead
        self._thread_local = threading.local()
        self._db_path_str = str(self.db_path)
        self._init_tables()   # still need a connection here for initial setup

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._thread_local, "db"):
            self._thread_local.db = sqlite3.connect(self._db_path_str, check_same_thread=False)
            sqlite_vec.load(self._thread_local.db)
        return self._thread_local.db

    def add_entity(self, ...):
        conn = self._get_conn()
        conn.execute(...)
```

Or the simpler fix — pass `check_same_thread=False` and add a `threading.Lock()` guard around all `db.execute` calls.

---

### H-2: `embedding` variable unbound if `_text_embedding` raises — job fails instead of skipping

**File:** `extraction/pipeline.py:302-320`  
**Severity:** HIGH  
**Category:** Bug / Error Handling

**Code:**
```python
# Keep vector index in sync with graph entities.
_embed_start = time.perf_counter()
try:
    embedding = self._text_embedding(...)   # ← if this raises, embedding is unbound
finally:
    embedding_time_ms += (time.perf_counter() - _embed_start) * 1000
try:                                         # ← SEPARATE try — no except for the first
    _store_start = time.perf_counter()
    try:
        self.vector_store.add_entity(
            ...
            dense_embedding=embedding,       # ← UnboundLocalError if embedding unset
            ...
        )
    finally:
        vector_store_time_ms += (time.perf_counter() - _store_start) * 1000
except Exception:
    log.debug("Vector index upsert failed for %s", entity.name, exc_info=True)
```

If `_text_embedding` raises (the hash fallback prevents this in practice, but a race or threading bug in `_get_embedding_model` could cause it), the `finally` runs and then the exception propagates up through the `for entity in entities:` loop to the outer `except Exception as exc:` at the bottom of `process_job`, causing the **entire job to fail**. The intent is clearly for vector-index failures to be silent/non-fatal (`log.debug` in the second `except`), but the try-boundary split makes this impossible.

**Fix — consolidate both operations into one try/except:**
```python
_embed_start = time.perf_counter()
try:
    embedding = self._text_embedding(
        f"{entity.name} {entity.entity_type} {job.content[:200]}"
    )
    embedding_time_ms += (time.perf_counter() - _embed_start) * 1000
    _store_start = time.perf_counter()
    try:
        self.vector_store.add_entity(
            entity_id=entity_id,
            name=entity.name,
            entity_type=entity.entity_type,
            dense_embedding=embedding,
            content=job.content[:500],
            confidence=entity.confidence,
        )
    finally:
        vector_store_time_ms += (time.perf_counter() - _store_start) * 1000
except Exception:
    embedding_time_ms += (time.perf_counter() - _embed_start) * 1000   # still account for time
    log.debug("Vector index upsert failed for %s", entity.name, exc_info=True)
```

---

### H-3: `_text_embedding` recursion has no depth guard — potential stack overflow

**File:** `extraction/pipeline.py:1077, 1101, 1114`  
**Severity:** HIGH  
**Category:** Bug / Infinite Recursion Risk

**Code (repeated three times):**
```python
# ollama tier failure:
cls._embedding_model = None
cls._embedding_failed_tiers.add("ollama")
return cls._text_embedding(text, dim)   # ← recursive call

# cloud tier failure:
cls._embedding_model = None
cls._embedding_failed_tiers.add("cloud")
return cls._text_embedding(text, dim)   # ← recursive call

# sentence-transformers tier failure:
cls._embedding_model = None
cls._embedding_failed_tiers.add(cls._embedding_active_tier or "sentence-transformers")
return cls._text_embedding(text, dim)   # ← recursive call
```

Each failing tier nulls out `_embedding_model` and recurses. The recursion terminates only when `_get_embedding_model()` returns the `"hash"` sentinel. This is correct in steady state. However, there are two latent risk scenarios:

1. **Race condition**: If two threads call `_text_embedding` concurrently (the `asyncio.to_thread` pattern makes this possible), both might see `_embedding_model is None` simultaneously, both load the same tier, both fail and add to `_embedding_failed_tiers`. This is safe but wasteful.

2. **More dangerous**: If `cls._embedding_active_tier` is `None` when `_text_embedding` is entered (e.g. due to `invalidate_embedding_cache()` being called between `_get_embedding_model` and the encode), then `cls._embedding_failed_tiers.add(cls._embedding_active_tier or "sentence-transformers")` adds `"sentence-transformers"` even if that wasn't the active tier. On the next recursion, `_get_embedding_model` walks the tier chain, all tiers fail, and the hash fallback is loaded — but if at any point the chain is misconfigured to omit "hash" (e.g., `EMBEDDING_TIER_ORDER = ["sentence-transformers"]` in a test config), the function recurses indefinitely until `RecursionError`.

**Fix — add a depth sentinel:**
```python
@classmethod
def _text_embedding(cls, text: str, dim: int = 768, _depth: int = 0) -> list[float]:
    if _depth > 4:
        # All tiers exhausted; use hash directly without further recursion.
        log.warning("_text_embedding: all tiers exhausted, using hash directly")
        return cls._hash_embed(text, dim)
    ...
    # In each tier failure branch:
    return cls._text_embedding(text, dim, _depth=_depth + 1)
```

---

## MEDIUM

---

### M-1: `_finetune_running` bool guard is not safe for concurrent async callers

**File:** `evolution/gliner_training.py:193-203`  
**Severity:** MEDIUM  
**Category:** Bug / Concurrency

**Code:**
```python
_finetune_running = False  # class-level concurrency guard

async def run_gliner_finetune_pipeline(self) -> dict[str, Any]:
    if GLiNERTrainingService._finetune_running:
        log.warning("GLiNER fine-tune already running, skipping concurrent trigger")
        return {"status": "already_running", ...}
    GLiNERTrainingService._finetune_running = True
    try:
        return await self._run_finetune_pipeline_inner()
    finally:
        GLiNERTrainingService._finetune_running = False
```

There is no `await` between the `if` check and the `_finetune_running = True` assignment, so within a single async event loop this is safe (Python's GIL and asyncio's cooperative multitasking guarantee no preemption between those two statements). However:

1. If this method is ever called via `asyncio.to_thread` from two different threads simultaneously, the check-and-set is a TOCTOU race.
2. The guard is a class-level bool, not an `asyncio.Lock` — if `_run_finetune_pipeline_inner` awaits something and the event loop processes another call to `run_gliner_finetune_pipeline`, the check will correctly block, but there's no queuing — the second caller just gets "already_running" and silently does nothing.
3. If the service restarts mid-finetune, `_finetune_running` stays `False` (it's in-process only), which is actually correct.

**Fix — use `asyncio.Lock`:**
```python
_finetune_lock: asyncio.Lock | None = None  # lazy-initialized

@classmethod
def _get_finetune_lock(cls) -> asyncio.Lock:
    if cls._finetune_lock is None:
        cls._finetune_lock = asyncio.Lock()
    return cls._finetune_lock

async def run_gliner_finetune_pipeline(self) -> dict[str, Any]:
    lock = self._get_finetune_lock()
    if lock.locked():
        log.warning("GLiNER fine-tune already running, skipping concurrent trigger")
        return {"status": "already_running", "message": "Fine-tune pipeline already in progress"}
    async with lock:
        return await self._run_finetune_pipeline_inner()
```

---

### M-2: Module-level side effect — `_apply_adopted_schema_on_load()` runs at import time

**File:** `memory/graph_suggestions.py:last ~5 lines`  
**Severity:** MEDIUM  
**Category:** Python Anti-pattern / Testing / Startup Coupling

**Code:**
```python
# ... end of graph_suggestions.py ...
_apply_adopted_schema_on_load()
```

This runs at module import time, which:
1. Reads `~/.graph-memory/adopted_schema.json` from disk on every import.
2. Calls `_adopt_rel_type(rel_type)` which imports from `memory.graph` and `memory.extractor` — side effects that mutate module-level globals (`VALID_REL_TYPES`, `RELATION_SCHEMA`).
3. Any test that does `import memory.graph_suggestions` triggers these IO and mutation side effects, making tests non-hermetic.
4. If the graph connection is not yet ready when the module is imported, the `_adopt_rel_type` call silently fails and adopted schema is lost.

**Fix — call at startup, not import:**
```python
# Remove the bare call at module bottom.
# In main.py lifespan:
@asynccontextmanager
async def lifespan(app: FastAPI):
    ...
    from memory.graph_suggestions import _apply_adopted_schema_on_load
    _apply_adopted_schema_on_load()
    ...
    yield
    ...
```

---

### M-3: `_save_adoption_history` is not atomic — corrupt file on crash

**File:** `memory/graph_suggestions.py` (~line 280-288)  
**Severity:** MEDIUM  
**Category:** Resource Management / Data Integrity

**Code:**
```python
def _save_adoption_history(history: dict[str, dict[str, Any]]) -> None:
    SUGGESTIONS_DIR.mkdir(parents=True, exist_ok=True)
    ...
    payload = {"version": 1, "items": serializable_items}
    _ADOPTION_HISTORY_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
```

`Path.write_text` is not atomic — if the process crashes mid-write, `adoption_history.json` will be partially written and unreadable. Subsequent loads will return `{}` (the `except Exception` swallows parse errors), silently losing all tracking history.

Compare this with `save_state` in `gliner_training.py` which correctly uses atomic rename via `tempfile.mkstemp` + `Path.replace`.

**Fix — use atomic write pattern:**
```python
def _save_adoption_history(history: dict[str, dict[str, Any]]) -> None:
    SUGGESTIONS_DIR.mkdir(parents=True, exist_ok=True)
    ...
    payload = {"version": 1, "items": serializable_items}
    text = json.dumps(payload, indent=2, ensure_ascii=True) + "\n"

    import tempfile, os
    fd, tmp_path = tempfile.mkstemp(dir=SUGGESTIONS_DIR, prefix=".adoption_tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        Path(tmp_path).replace(_ADOPTION_HISTORY_PATH)
    except Exception:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass
        raise
```

---

### M-4: `_ALLOWED_REL_TYPES` global rewrite called on every invocation when schema fails

**File:** `evolution/gliner_training.py` (in `to_gliner_training_record`)  
**Severity:** MEDIUM  
**Category:** Bug / Performance

**Code:**
```python
# Lazily computed allowed relation types (populated on first use)
_ALLOWED_REL_TYPES: frozenset[str] = frozenset()

def to_gliner_training_record(self, row: dict[str, Any]) -> dict[str, Any] | None:
    ...
    global _ALLOWED_REL_TYPES
    if not _ALLOWED_REL_TYPES:
        _ALLOWED_REL_TYPES = _get_allowed_rel_types()
    # Empty frozenset means "allow all"
    _check_rel = bool(_ALLOWED_REL_TYPES)
```

If `_get_allowed_rel_types()` fails (its `except Exception: pass; return frozenset()` is silent) and returns an empty frozenset, `_ALLOWED_REL_TYPES` is assigned `frozenset()`. On the next call, `not _ALLOWED_REL_TYPES` is `True` again, so `_get_allowed_rel_types()` is called again — and fails again. This creates a hot import-and-fail loop for every training record converted when the schema registry is unavailable.

Additionally, using `global` for a class's module-level constant is an anti-pattern. The `global` keyword here modifies module state from an instance method.

**Fix — use a sentinel to distinguish "not loaded" from "empty":**
```python
_ALLOWED_REL_TYPES: frozenset[str] | None = None  # None = not yet loaded; empty = allow all

def to_gliner_training_record(self, row: dict[str, Any]) -> dict[str, Any] | None:
    global _ALLOWED_REL_TYPES
    if _ALLOWED_REL_TYPES is None:
        result = _get_allowed_rel_types()
        _ALLOWED_REL_TYPES = result  # cache even if empty (empty = allow all)
    _check_rel = bool(_ALLOWED_REL_TYPES)
    ...
```

---

### M-5: `_normalize_entity_type` returns `""` for valid GLiNER types — silently drops type constraints

**File:** `extraction/glirel_enrichment.py` (in `_normalize_entity_type`)  
**Severity:** MEDIUM  
**Category:** Bug / Logic Error

**Code:**
```python
_ENTITY_TYPE_ALIASES: dict[str, str] = {
    "person": "person",
    "organization": "organization",
    "org": "organization",
    ...
    # "technology", "project", "concept", "event" are NOT in this dict
}

@classmethod
def _normalize_entity_type(cls, value: str) -> str:
    normalized = cls._normalize(value).replace(" ", "_")
    if not normalized:
        return ""
    return _ENTITY_TYPE_ALIASES.get(normalized, _ENTITY_TYPE_ALIASES.get(normalized.replace("_", " "), ""))
```

GLiNER entity types include `"Technology"`, `"Project"`, `"Concept"`, `"Event"` — none of which are in `_ENTITY_TYPE_ALIASES`. So `_normalize_entity_type("Technology")` returns `""`. In `_build_entity_type_index`:

```python
normalized_type = cls._normalize_entity_type(raw_type)
if not normalized_type:
    continue   # ← "Technology" entities contribute NOTHING to the type index
```

This means entity type constraints in `_RELATION_TYPE_CONSTRAINTS` can never fire for any entity with type "Technology", "Project", etc. The affected constraints (`"works at"`, `"founded"`, `"parent of"`, etc.) require `"person"` types. An entity labelled `"Person"` does correctly resolve via the alias `"person" → "person"`.

The impact is subtle: constraint filtering is only partially effective. Relations like `Technology --[founded]--> Organization` would pass the "founded" type constraint (which requires `head=person`) because the Tech entity's types set is empty → `not head_types` → `return True`.

**Fix — expand `_ENTITY_TYPE_ALIASES` to cover all MollyGraph entity types:**
```python
_ENTITY_TYPE_ALIASES: dict[str, str] = {
    "person":        "person",
    "organization":  "organization",
    "org":           "organization",
    "company":       "organization",
    "institution":   "organization",
    "location":      "location",
    "place":         "location",
    "gpe":           "location",
    "loc":           "location",
    # Add missing types:
    "technology":    "technology",
    "project":       "project",
    "concept":       "concept",
    "event":         "event",
}
```

---

### M-6: `_llm_enrich_synonyms` spawns ThreadPoolExecutor on every call — no caching

**File:** `extraction/glirel_synonyms.py` (~line 230-250)  
**Severity:** MEDIUM  
**Category:** Performance / Resource Management

**Code:**
```python
def _llm_enrich_synonyms(label: str) -> list[str]:
    ...
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        result: dict = pool.submit(asyncio.run, _call()).result(timeout=30)
```

Every call to `add_synonym_group` with auto-generation triggers `_llm_enrich_synonyms`, which:
1. Creates a new `ThreadPoolExecutor`.
2. Spawns a new OS thread.
3. Creates a new `asyncio` event loop via `asyncio.run`.
4. Makes an outbound HTTP request to the audit LLM.

All this per new relation type, potentially during the adoption cycle. If many relations are adopted at once (e.g., a batch run with 10+ new types), this creates 10+ threads and 10+ event loops simultaneously.

The result is also never cached — if `add_synonym_group("works at")` is called twice, two LLM calls are made.

**Fix — call the LLM once and cache; use a module-level executor:**
```python
_llm_synonym_cache: dict[str, list[str]] = {}

def _llm_enrich_synonyms(label: str) -> list[str]:
    if label in _llm_synonym_cache:
        return _llm_synonym_cache[label]
    ...
    result = ...
    _llm_synonym_cache[label] = result
    return result
```

---

## LOW

---

### L-1: `ZvecBackend.list_all_entity_ids` returns `None` but signature says `Optional[List[str]]`

**File:** `memory/vector_store.py` (`ZvecBackend.list_all_entity_ids`)  
**Severity:** LOW  
**Category:** Type Safety / Documentation

**Code:**
```python
def list_all_entity_ids(self) -> Optional[List[str]]:
    """Zvec has no native scan/list API — returns None to signal unsupported."""
    log.debug(...)
    return None
```

The abstract base method declares `-> Optional[List[str]]` and the docstring says `None = unsupported`. This is a design choice, not a bug — but `SqliteVecBackend.list_all_entity_ids` returns `List[str]` (never `None`), creating an asymmetry that callers must handle. The `VectorStore.list_all_entity_ids` wrapper catches `NotImplementedError` and returns `None`, but `ZvecBackend` doesn't raise `NotImplementedError` — it returns `None`. The `NotImplementedError` catch in the wrapper is therefore dead code for Zvec.

**Fix — standardize the "not supported" signal:**
Either raise `NotImplementedError` from `ZvecBackend.list_all_entity_ids()` (so the wrapper's `except NotImplementedError` actually fires), or remove the `except NotImplementedError` branch from the wrapper and document that `None` is the sentinel.

---

### L-2: No unit tests for critical paths

**File:** `tests/` (only `test_graph_quality.py` exists)  
**Severity:** LOW  
**Category:** Test Coverage

The entire test suite is one golden-set integration test (`test_graph_quality.py`) that requires a running GLiNER2 model and is skipped in most CI environments. The following critical paths have zero test coverage:

| Path | Risk |
|------|------|
| `extraction/pipeline.py` — `_build_relationships`, `_merge_glirel_relations`, embedding tier chain | Relationship deduplication, fallback logic, tier failover |
| `memory/vector_store.py` — `SqliteVecBackend`, `ZvecBackend` | Data persistence, thread safety |
| `memory/graph_suggestions.py` — `run_auto_adoption`, frequency/adoption gate | Adoption criteria, history state machine |
| `extraction/glirel_enrichment.py` — `_build_entity_spans`, `_normalize_relations` | Token alignment, type constraint filtering |
| `extraction/glirel_synonyms.py` — `generate_synonyms_for_label`, suffix/prefix rules | Synonym expansion correctness |
| `audit/llm_audit.py` — `parse_verdicts`, `_extract_json_array` | LLM JSON repair robustness |

**Recommended additions:**
- `tests/test_pipeline_unit.py` — mock graph/vector, test entity dedup, fallback rel, merge logic
- `tests/test_vector_store.py` — thread-safety smoke test for `SqliteVecBackend`
- `tests/test_graph_suggestions.py` — adoption gate with mocked suggestions JSONL
- `tests/test_glirel_synonyms.py` — `generate_synonyms_for_label` for suffix/prefix rules
- `tests/test_llm_audit.py` — `parse_verdicts` with truncated/broken JSON inputs

---

### L-3: `SYSTEM_PROMPT` is only used in LLM-mode, not surfaced in `build_audit_prompt`

**File:** `audit/llm_audit.py:73`  
**Severity:** LOW  
**Category:** Dead Code / Confusion

```python
SYSTEM_PROMPT = (
    "You are auditing a personal knowledge graph extracted from conversations. "
    "Return valid JSON only."
)
```

`SYSTEM_PROMPT` is used in `_invoke_openai_compatible` as the system message. `build_audit_prompt` does NOT use it — it inlines a similar but different persona string ("You are auditing a personal knowledge graph...") directly in the user prompt body. This creates two divergent persona definitions that can drift over time.

**Fix** — ensure `build_audit_prompt` delegates to `SYSTEM_PROMPT` for the user-prompt persona text, or clearly separate the system-level and user-prompt personas with a comment explaining both are intentional.

---

### L-4: Mutable class-level set `_embedding_failed_tiers: set = set()`

**File:** `extraction/pipeline.py:905`  
**Severity:** LOW  
**Category:** Python Anti-pattern

```python
class ExtractionPipeline:
    _embedding_model = None
    _embedding_active_tier: "str | None" = None
    _embedding_failed_tiers: set = set()    # ← mutable default on class body
```

While `invalidate_embedding_cache` resets this with `cls._embedding_failed_tiers = set()`, the initial value is a shared mutable set on the class definition. If any code accesses this before `invalidate_embedding_cache` is called (e.g., in tests), mutations to the set are shared across all instances. The type annotation `set` should also be `set[str]`.

**Fix:**
```python
_embedding_failed_tiers: set[str] = set()
```
This is syntactically identical but the correct annotation. More importantly, ensure no test mutates the class-level set without calling `invalidate_embedding_cache()` as teardown.

---

### L-5: Broad bare `except Exception` blocks swallow context in several critical paths

**File:** Multiple files  
**Severity:** LOW  
**Category:** Error Handling

Several error-handling sites are too broad and silently drop exceptions that should at minimum include context:

**`audit/llm_audit.py` (audit state save):**
```python
def _save_audit_state(state: dict[str, Any]) -> None:
    ...
    except Exception:
        log.debug("Failed to write audit_state.json", exc_info=True)  # ← should be WARNING
```
Silent `log.debug` for a failed state write means audit metrics are silently lost. Should be `log.warning`.

**`evolution/gliner_training.py` (`_load_state`):**
```python
except Exception:
    log.warning("Failed to parse state file %s", self._state_file, exc_info=True)
    return {}
```
Returning `{}` on a corrupted state file silently resets all training state (cursor, history, cooldown timers). Should log at `ERROR` level and potentially raise or surface a health check failure.

**`memory/graph_suggestions.py` (`_append_jsonl`):**
```python
except OSError:
    log.debug("Failed to append graph suggestion", exc_info=True)
```
Only catches `OSError` — other exceptions (e.g., a JSON serialization error from a non-serializable value in `entry`) propagate uncaught to the caller.

**Fix** — review all `log.debug` on critical-path failures, upgrade to `log.warning` or `log.error` as appropriate. For `_append_jsonl`, broaden the exception type or pre-validate `entry` is JSON-serializable.

---

## Test Coverage Gaps Summary

| Critical Path | Has Tests | Risk Without Tests |
|---------------|-----------|-------------------|
| `pipeline.py:process_job` happy path | ❌ | Breakage in entity/rel flow goes undetected |
| `pipeline.py:_merge_glirel_relations` | ❌ | Override/addition logic silently wrong |
| `pipeline.py:_text_embedding` tier failover | ❌ | Tier chain regression undetected |
| `vector_store.py:SqliteVecBackend` threading | ❌ | Thread-safety crash in production |
| `graph_suggestions.py:run_auto_adoption` | ❌ | Adoption gate logic errors |
| `glirel_enrichment.py:_build_entity_spans` | ❌ | Token alignment bugs in NER |
| `glirel_synonyms.py:generate_synonyms_for_label` | ❌ | Wrong synonyms degrade GLiREL quality |
| `llm_audit.py:parse_verdicts` | ❌ | JSON repair failure causes silent data loss |
| `llm_audit.py:apply_verdicts` | ❌ | Graph mutations untested |
| `main.py` delete/query endpoints | ❌ | Regression in API contract |

Only `test_graph_quality.py` (golden set evaluation) exists, and it is skipped by default unless GLiNER2 + torch are installed.

---

## Appendix: Quick Reference

| ID | File | Line | Issue |
|----|------|------|-------|
| C-1 | main.py | 871 | Cypher injection via `rel_type` f-string interpolation |
| H-1 | vector_store.py | 76-84 | `SqliteVecBackend` cross-thread SQLite connection |
| H-2 | pipeline.py | 302-320 | `embedding` unbound if `_text_embedding` raises — job fails |
| H-3 | pipeline.py | 1077,1101,1114 | `_text_embedding` unbounded recursion, no depth limit |
| M-1 | gliner_training.py | 193-203 | `_finetune_running` bool not safe for concurrent async callers |
| M-2 | graph_suggestions.py | EOF | Module-level side effect at import time |
| M-3 | graph_suggestions.py | ~280 | `_save_adoption_history` non-atomic write |
| M-4 | gliner_training.py | `to_gliner_training_record` | `_ALLOWED_REL_TYPES` hot retry loop on schema failure |
| M-5 | glirel_enrichment.py | `_normalize_entity_type` | Returns `""` for Technology/Project/Concept/Event — drops type constraints |
| M-6 | glirel_synonyms.py | ~230 | ThreadPoolExecutor per LLM synonym call, no caching |
| L-1 | vector_store.py | `ZvecBackend` | `list_all_entity_ids` returns None vs NotImplementedError mismatch |
| L-2 | tests/ | — | No unit tests for critical paths |
| L-3 | llm_audit.py | 73 | `SYSTEM_PROMPT` diverges from `build_audit_prompt` persona |
| L-4 | pipeline.py | 905 | Mutable class-level `set` without type annotation |
| L-5 | multiple | — | Bare `except Exception` / `log.debug` swallows critical errors |
