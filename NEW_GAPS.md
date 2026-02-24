# MollyGraph Audit: New Gaps (Not In `BLIND_SPOTS.md`)

## 1) CRITICAL — Training pipeline hard-crashes on undefined class reference
- **Category:** Pipeline failure mode / Missing error handling
- **Evidence:** `evolution/gliner_training.py:179-186`
- **What happens:** `run_gliner_finetune_pipeline()` references `GLiNERTrainingMixin._finetune_running`, but `GLiNERTrainingMixin` is not defined in this module.
- **Concrete failure scenario:** Any call to `/train/gliner` (or nightly step 4) raises `NameError` before training starts, so fine-tuning never runs.

## 2) CRITICAL — Silent relationship loss when either endpoint entity is missing
- **Category:** Data integrity gap
- **Evidence:**
  - `memory/graph/relationships.py:235-271` (`MATCH ... CREATE ...` with no write-count check, always returns a UUID)
  - `extraction/pipeline.py:356-357` (relationships may reference raw head/tail not guaranteed to exist as entities)
  - `extraction/pipeline.py:229-232` (caller ignores returned status)
- **What happens:** If source/target entity nodes are absent, Cypher `MATCH` finds nothing, relationship is not created, but code still returns a synthetic ID and pipeline continues.
- **Concrete failure scenario:** Extractor emits relation `A -> B`; `B` was filtered from entities. Job is marked completed, but the relation is silently dropped.

## 3) HIGH — Partial graph writes on mid-job failure (no atomic unit-of-work)
- **Category:** Data integrity gap / Pipeline failure mode
- **Evidence:**
  - `extraction/pipeline.py:162-165` (entity writes)
  - `extraction/pipeline.py:204` (episode write)
  - `extraction/pipeline.py:229-232` (relationship writes)
  - `extraction/pipeline.py:294-299` (single outer catch marks job failed)
- **What happens:** Entities/episode can be persisted before later steps fail; there is no rollback across the full job.
- **Concrete failure scenario:** First few relationships write, one later upsert throws; job status becomes `failed`, but partial entities/episode/edges remain. Retry can duplicate or skew training signals.

## 4) HIGH — Non-atomic relationship upsert allows duplicate edges under concurrency
- **Category:** Concurrency issue (beyond maintenance lock)
- **Evidence:** `memory/graph/relationships.py:93-107` (read existing), then `:162-169` (create) in separate operations.
- **What happens:** Two workers can both miss `existing` and both create same logical relationship.
- **Concrete failure scenario:** Concurrent ingestion of similar text produces duplicate active edges for the same `(head, rel_type, tail)`.

## 5) HIGH — Queue worker can die silently while API keeps accepting ingest
- **Category:** Pipeline failure mode
- **Evidence:**
  - `main.py:413` (fire-and-forget `asyncio.create_task(queue_worker.start())`)
  - `main.py:463-475` (`/health` always reports `status: healthy`, no worker-task liveness check)
  - `main.py:661-668` (`/ingest` keeps enqueueing)
- **What happens:** If worker task crashes, ingestion appears healthy but queue drains never happen.
- **Concrete failure scenario:** Unhandled exception inside worker stops processing; clients keep getting `status: queued` while backlog grows indefinitely.

## 6) HIGH — Training cursor can skip episodes forever at timestamp boundaries
- **Category:** Data integrity gap / Training data quality
- **Evidence:**
  - `evolution/gliner_training.py:610-613` (next batch filter uses `created_at > cursor`)
  - `evolution/gliner_training.py:636-637` (order only by created_at)
  - `evolution/gliner_training.py:687-688` (cursor set to last seen created_at)
- **What happens:** If many episodes share the same `created_at` and a run hits `LIMIT`, remaining episodes with that exact timestamp are skipped on next run.
- **Concrete failure scenario:** Batch ends at `2026-02-24T01:00:00Z`; next query uses `>` that timestamp, so unprocessed rows at `01:00:00Z` are never accumulated.

## 7) HIGH — Training data can contain contradictory labels (`deleted` relation as positive + negative)
- **Category:** Training data quality
- **Evidence:**
  - Positive relation query excludes only `quarantined`: `evolution/gliner_training.py:1038`
  - Negative relation query includes `deleted`: `evolution/gliner_training.py:1069`
- **What happens:** A relation with `audit_status='deleted'` can still enter positives while also being added as a hard negative.
- **Concrete failure scenario:** Same `(head,label,tail)` appears in both `extracted_relations` and `negative_relations`, poisoning supervision.

## 8) HIGH — Episode examples are labeled from global graph edges, not episode-scoped evidence
- **Category:** Training data quality
- **Evidence:**
  - `evolution/gliner_training.py:1036-1045` and `:1067-1076` filter only by `h.name IN $names AND t.name IN $names`
  - No filter by episode id / edge provenance in these queries.
- **What happens:** Relations from older/unrelated episodes are attached to current text if entity names overlap.
- **Concrete failure scenario:** Episode text mentions "Alice" and "Bob" casually; historical `WORKS_AT` edge between them gets labeled as positive in this episode even if absent from text.

## 9) HIGH — Pre-training audit is fail-open on any exception
- **Category:** Training data quality / Pipeline failure mode
- **Evidence:** `evolution/gliner_training.py:1378-1380`
- **What happens:** If audit model call/parsing fails, method returns `{"passed": True, ...}` and training proceeds.
- **Concrete failure scenario:** Provider outage or parse error bypasses the only pre-train quality gate and still runs LoRA on unvetted data.

## 10) MEDIUM — Deployment fallback can delete active model before copy succeeds
- **Category:** Pipeline failure mode / Data availability
- **Evidence:** `evolution/gliner_training.py:1792-1797`
- **What happens:** On atomic deploy failure, fallback path removes `gliner_active` then does `copytree` without guarding copy failure.
- **Concrete failure scenario:** Disk-full/permission error during fallback copy leaves no active model directory, causing model load failures until manual repair.

## 11) MEDIUM — Concurrent state/training writes can clobber each other
- **Category:** Concurrency issue / Data integrity gap
- **Evidence:**
  - `evolution/gliner_training.py:84-88` (`save_state` uses fixed `.tmp` and no lock)
  - `evolution/gliner_training.py:677-681` (batch filename only second-resolution; opened with `"w"`)
- **What happens:** Concurrent runs can overwrite state updates and/or overwrite same `examples-<timestamp>.jsonl` batch.
- **Concrete failure scenario:** Manual accumulation and nightly accumulation in same second both write `examples-YYYYMMDD-HHMMSS.jsonl`; one batch is lost.

## 12) MEDIUM — Insecure defaults allow accidental prod exposure
- **Category:** Security / Configuration foot-gun
- **Evidence:**
  - `config.py:67` default API key is static (`dev-key-change-in-production`)
  - `config.py:35` default Neo4j password is static (`mollygraph`)
  - `main.py:255-261` accepts whatever `config.API_KEY` is at runtime (no startup hard-fail on default)
- **What happens:** Production deployments that forget env overrides are trivially guessable.
- **Concrete failure scenario:** Service exposed beyond localhost with default key/password; attacker can call protected endpoints and modify graph/training state.

## 13) MEDIUM — Async API endpoints call blocking graph operations directly
- **Category:** Performance bottleneck
- **Evidence:**
  - `main.py:675`, `main.py:679` (`graph.get_current_facts`, `graph.get_entity_context` in async handler)
  - `main.py:734`, `main.py:745-756`, `main.py:787` (sync graph/Neo4j calls inside async `/query`)
- **What happens:** Event loop is blocked by synchronous DB calls; throughput collapses under concurrent requests.
- **Concrete failure scenario:** At ~100+ concurrent queries, long Neo4j round-trips block unrelated requests, increasing latency and timeout rates.

## 14) MEDIUM — Embedding config endpoints do not control actual embedding model used in pipeline
- **Category:** Configuration foot-gun / Performance
- **Evidence:**
  - Hardcoded model load: `extraction/pipeline.py:547-549` (`SentenceTransformer("google/embeddinggemma-300m")`)
  - Hash fallback: `extraction/pipeline.py:551-552`
  - Runtime config endpoints imply dynamic provider/model switching: `main.py:894-916`
- **What happens:** Operators can change embedding provider/model via API, but extraction/query embedding path still uses hardcoded model (or hash fallback).
- **Concrete failure scenario:** Team switches to `ollama` via `/embeddings/config`; retrieval quality/cost behavior does not match configured backend, causing silent misconfiguration.

## 15) LOW — `/health` is unauthenticated and leaks internal operational metadata
- **Category:** Security
- **Evidence:** `main.py:463-475` (no `Depends(verify_api_key)`, returns queue depth and vector stats)
- **What happens:** Anyone reaching the port can probe system load and vector-store size.
- **Concrete failure scenario:** External scanner repeatedly hits `/health` to infer ingestion activity and dataset growth.
