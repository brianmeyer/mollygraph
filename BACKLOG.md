# MollyGraph Backlog

## Enrichment Pipeline
- [ ] **Specialized GLiNER enrichment** — Run domain-specific GLiNER models as additive passes alongside GLiNER2. Config-driven list of models, entity merge logic, confidence scoring. Models are read-only (can't LoRA train them).
  - `Ihor/gliner-biomed-base-v1.0` — biomedical (drugs, symptoms, conditions)
  - `nvidia/gliner-PII` — PII detection (SSN, bank accounts, medical IDs)
  - `urchade/gliner_multi-v2.1` — multilingual (50+ languages)
  - `knowledgator/gliner-pii-edge-v1.0` — privacy-focused edge model
  - `Mit1208/gliner-fine-tuned-pii-finance-multilingual` — financial PII
- [ ] **GLiREL integration** — `jackboyla/glirel-large-v0` as a second-pass relation extractor. Catches relations GLiNER2 misses (the 8.4% RELATED_TO fallback). Separate lib (`pip install glirel`), separate model, can't LoRA train.
- [ ] **Entity merge & dedup** — When multiple models extract the same entity, merge by confidence score. Higher-confidence specialized model wins over general model.

## Model Infrastructure
- [ ] **Model hot-swap with unload** — when config changes, unload old model from memory, load new one. No restart needed.
- [ ] **GGUF support** — quantized models via llama-cpp-python for lower memory usage
- [ ] **Model download management** — pre-download models, track disk usage, cleanup old versions

## MLX (deprioritized)
- [ ] **MLX embeddings** — `mlx-embeddings` as an embedding tier. Quick win but low impact (MPS already uses GPU).
- [ ] **MLX NER (long-term)** — GLiNER2 uses DeBERTa-v2, no MLX port exists. Multi-week project, marginal perf gain.

## Codex Gaps (from NEW_GAPS.md)
- [ ] Partial graph writes — no rollback on entity+relationship step failures
- [ ] Duplicate edges under concurrency — relationship upsert is read-then-write
- [ ] Queue worker silent death — health says "healthy" even if worker crashed
- [ ] Training cursor skips at timestamp boundaries (`>` vs `>=`)
- [ ] Episode label bleed — relations from old episodes attached to new text
- [ ] Deployment fallback can delete active model before copy succeeds
- [ ] Concurrent state writes can clobber (no lock on `.tmp`)
- [ ] Sync Neo4j calls blocking async event loop

## Security
- [ ] **Cypher injection via rel_type** — DELETE /relationship interpolates user input directly into Cypher f-strings. Validate against VALID_REL_TYPES allowlist before interpolation. Same issue in relationships.py MERGE queries.
- [ ] **API key rotation** — single hardcoded dev key, no key rotation, no per-client keys

## Thread Safety & Data Integrity
- [ ] **SqliteVec single connection across threads** — SqliteVecBackend opens one db connection with check_same_thread=False. Concurrent query + ingestion = silent corruption. Add threading.Lock around all public methods.
- [ ] **Entity-vector count mismatch** — 1001 entities in Neo4j, 944 vectors in Zvec. Partial writes already happening. Need reconciliation + transactional writes.
- [ ] **Queue jobs stuck forever after crash** — Jobs stuck as 'processing' after ungraceful shutdown. Detection exists but no recovery. Add timeout-based reset + retry_count + dead-letter queue.
- [ ] **Maintenance lock doesn't cover delete endpoints** — /entity/{name} and /prune bypass the lock. Concurrent deletes during audit = KeyError/NodeNotFound.

## Nightly Audit
- [ ] **LLM audit not actually firing** — AUDIT_LLM_ENABLED defaults to "0" in config.py. Even with .env override, audit feedback counters show 0/0/0. Moonshot/Kimi is NOT scoring relationships. Trace and fix.
- [ ] **Batch processing for audit** — Large relationship batches truncate Kimi's JSON response. Process in batches of 20-30 rels per LLM call.
- [ ] **Reviewed count metric** — Track relationships_reviewed/approved/flagged/reclassified per nightly run. Surface in /metrics/nightly and /metrics/evolution.
- [ ] **Persist nightly results to disk** — runtime_state.py stores results in memory only. Lost on restart. Write to JSONL file.

## Unwired Modules (Overnight Codex Batch 2/25)
- [ ] **Graph-aware reranker** — query/graph_reranker.py (309 lines) exists but never imported into query path. Needs config flag + integration into api/query.py after merge step.
- [ ] **Source yield API endpoint** — metrics/source_yield.py has get_source_stats(). record_yield() IS called from pipeline.py. Just needs a GET /metrics/sources route in admin.py.
- [ ] **Retrieval quality metrics** — metrics/retrieval_quality.py has compute_retrieval_quality(). Not called anywhere. Wire into query response + add /metrics/retrieval/trend endpoint.

## API & DX
- [ ] **No API versioning** — No /v1/ prefix. Breaking changes will hurt at public launch.
- [ ] **No pagination** — /entities, /training/runs, list endpoints return unbounded results.
- [ ] **40+ undocumented endpoints** — README documents ~20. Full service has 40+. Missing: /maintenance/*, /extractors/*, /entities.
- [ ] **.env.example 25 vars behind** — 92 env vars in config.py, only 73 in .env.example.
- [ ] **MCP search_nodes is a dummy** — Just aliases search_facts. SDK README missing 6 of 9 tools.
- [ ] **No docker-compose for full stack** — Only Neo4j. Need service + Neo4j + optional Ollama.
- [ ] **Duplicate operation_ids** — Legacy route aliases create OpenAPI spec collisions.

## Metrics & Observability  
- [ ] **Metrics reset on restart** — Retrieval counters, model health window, nightly run history all in-memory. Only file-backed logs survive.
- [ ] **Neo4j Cartesian product warnings** — MERGE queries without explicit joins. Perf issue at scale.
- [ ] **No alerting on schema-drift alarm** — Fires but nobody knows.
- [ ] **/maintenance/last-run endpoint** — Persist actual nightly results so they're queryable after restart.

## GLiREL Training Integration
- [ ] **GLiREL silver-label → LoRA corpus** — GLiREL generates training examples during ingestion (persist_training_examples). Verify format compatibility with GLiNER training pipeline. Critical for self-improvement loop.

## Updated: 2026-02-25
