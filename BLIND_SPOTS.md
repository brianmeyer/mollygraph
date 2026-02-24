# MollyGraph Blind Spots Audit — 2026-02-23

## 1. Vector Store ↔ Graph Drift (CRITICAL)

**Failure mode:** When the audit merges/deduplicates entities or deletes orphans in Neo4j, the corresponding vectors in Zvec are NEVER updated or removed. Over time, the vector store accumulates stale embeddings for entities that no longer exist in the graph.

**Evidence:** `memory/graph/entities.py` handles merges and `memory/graph/maintenance.py` handles orphan deletion — neither calls `vector_store.remove()` or `vector_store.update()`. The vector store has no delete API at all.

**Impact:** Vector similarity search returns ghost entities. Query results reference nodes that no longer exist. Graph Lift metric becomes unreliable because vector hits point to dead entities.

**Fix:** Add `VectorStore.remove_entity(entity_id)` and `VectorStore.update_entity()`. Wire into `merge_entities()` and `delete_orphan_entities()`. Add a periodic reconciliation endpoint.

---

## 2. Schema Suggestions Mutate Runtime State (HIGH)

**Failure mode:** `graph_suggestions.py` directly mutates `extractor.RELATION_SCHEMA` and `extractor.ENTITY_SCHEMA` in memory at lines 401-422. These mutations:
- Are lost on restart (not persisted)
- Take effect immediately without audit
- Can add types the audit would reject
- Create divergence between what the extractor accepts and what the schema registry reports

**Evidence:** `apply_schema_suggestions()` does `extractor.RELATION_SCHEMA[relation_name] = {...}` directly.

**Impact:** The extractor starts using unvetted schema types. If a suggestion is bad, every subsequent extraction uses it until restart. No rollback mechanism.

**Fix:** Suggestions should go through the schema registry with an `audit_status='pending'` flag. Only apply after nightly audit approval. Persist to disk.

---

## 3. Concurrent Ingestion + Maintenance Race (HIGH)

**Failure mode:** The extraction queue processes jobs continuously while nightly maintenance runs. Maintenance does bulk Neo4j operations (merge, reclassify, delete) while the extraction pipeline is simultaneously writing new entities and relationships.

**Evidence:** `extraction/queue.py` uses SQLite with row-level locking for the queue but has NO coordination with `maintenance/auditor.py`. The `_finetune_running` flag in `gliner_training.py` (line 179) protects training but nothing protects graph mutations during maintenance.

**Impact:** Entity merged by maintenance → extraction writes relationship to old entity name → orphan created. Maintenance deletes entity → extraction pipeline gets Neo4j error or silently creates a new node.

**Fix:** Add a maintenance lock. Extraction queue should pause during maintenance windows. Or use Neo4j transactions with retry logic for constraint violations.

---

## 4. Episode References Stale After Entity Merge (HIGH)

**Failure mode:** Episodes store `entities_extracted` as a list of entity names. When two entities are merged (e.g., "Brian Meyer" absorbs "Brian M."), the episode still references "Brian M." — an entity that no longer exists.

**Evidence:** `memory/graph/episodes.py` creates Episode nodes with `entities_extracted` array. `memory/graph/entities.py` merge logic doesn't update episode references.

**Impact:** Training accumulator queries episodes → finds "Brian M." → looks up entity → doesn't exist → skips the example. Loss of valid training data. Also breaks episode-to-entity graph traversal.

**Fix:** After entity merge, run: `MATCH (ep:Episode) WHERE $old_name IN ep.entities_extracted SET ep.entities_extracted = [x IN ep.entities_extracted | CASE WHEN x = $old_name THEN $new_name ELSE x END]`

---

## 5. Model Degradation Detection Gap (MEDIUM)

**Failure mode:** `model_health.py` tracks extraction metrics but the fallback benchmark in `gliner_training.py` only runs DURING training. Between training runs, there's no continuous monitoring of whether the active LoRA model is degrading.

**Evidence:** The benchmark at line ~2400 compares candidate vs base model fallback rates, but only at training time. `model_health_monitor.record_extraction()` logs stats but doesn't trigger rollback based on drift from baseline.

**Impact:** If the production model starts producing worse extractions (due to data distribution shift), nobody notices until the next training cycle — which could be weeks if cooldown is active.

**Fix:** Add a rolling window comparison in `model_health.py`. If RELATED_TO fallback rate exceeds baseline + threshold for >100 extractions, auto-rollback to base model and alert.

---

## 6. Embedding Cache Never Invalidated on Context Change (MEDIUM)

**Failure mode:** Entity embeddings are computed at ingestion time using `f"{entity.name} {entity.entity_type} {job.content[:200]}"`. If an entity's type is reclassified by the audit, its vector embedding still reflects the OLD type.

**Evidence:** `extraction/pipeline.py` line 167 computes embedding once. No re-embedding on audit reclassification.

**Impact:** Vector similarity search uses stale semantic content. An entity reclassified from "Person" to "Organization" will still cluster with people in vector space.

**Fix:** Track `embedding_version` on entities. When audit reclassifies, mark embedding as stale. Periodic job re-embeds stale entities.

---

## 7. LOCK File Crash Recovery (MEDIUM)

**Failure mode:** Zvec uses a LOCK file for collection access. If the process crashes (OOM, kill -9), the LOCK file persists. Next startup sees the LOCK, assumes another process owns it, and creates a NEW empty collection — wiping all vectors.

**Evidence:** We experienced this exact bug today during service restarts. Multiple restarts caused repeated vector wipes.

**Impact:** Total loss of vector index. Requires full rebuild (5+ minutes for 894 entities, scales linearly).

**Fix:** Check if LOCK file's PID is still alive before creating new collection. Add startup health check that validates vector count matches entity count. Add `--force-unlock` flag.

---

## 8. Audit Signals One-Way Flow (MEDIUM)

**Failure mode:** When the LLM audit (`audit/llm_audit.py`) generates signals (reclassify, quarantine, merge), these are written to `audit_signals.jsonl` and applied to Neo4j. But:
- The MCP server (`mcp_server.py`) isn't notified
- The metrics system doesn't track audit-induced changes
- The extraction pipeline doesn't learn from audit corrections in real-time

**Evidence:** `audit/llm_audit.py` writes to jsonl file and calls graph methods directly. No event bus or notification system.

**Impact:** MCP clients see stale data until they re-query. Metrics don't distinguish "organic" changes from "audit corrections." Extraction keeps making the same mistakes the audit corrects.

**Fix:** Add an event/notification system. Audit signals should publish to subscribers (metrics, MCP cache invalidation, extraction pipeline hint cache).

---

## 9. Hardcoded Tolerance Thresholds (LOW)

**Failure mode:** Critical thresholds are hardcoded:
- LoRA benchmark tolerance: `1.20` (line 2421)
- Training example minimum: various hardcoded counts
- Cooldown periods: hardcoded in days
- Schema drift alarm: `5%` rel types, `10%` entity types

**Evidence:** Scattered constants in `gliner_training.py`, `main.py`, `config.py`.

**Impact:** Can't tune without code changes. Different deployments may need different thresholds.

**Fix:** Move to config.py or a YAML config with sensible defaults.

---

## 10. No Graph Quality Regression Test (LOW)

**Failure mode:** There's no automated way to know if the graph is getting WORSE over time. We measure entity count, relationship count, and RELATED_TO fallback rate — but not precision, recall, or semantic coherence.

**Evidence:** `/metrics/evolution` tracks growth but not quality. Schema drift alarm tracks type proliferation but not whether the types are correct.

**Impact:** Graph could slowly fill with low-quality entities and relationships. By the time it's noticeable, training data is contaminated.

**Fix:** Create a "golden set" of known-good entities/relations. Run periodic evaluation against this set. Alert if recall drops.
