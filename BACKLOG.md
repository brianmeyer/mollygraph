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

## Updated: 2026-02-24
