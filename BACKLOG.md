# MollyGraph Backlog

## Model Infrastructure
- [ ] **Model hot-swap with unload** — when config changes, unload old model from memory, load new one. No restart needed.
- [ ] **MLX embeddings** — `mlx-embeddings` for Apple Silicon optimized embedding (BERT, MiniLM, nomic supported). Quick win, add as embedding tier.
- [ ] **MLX NER (long-term)** — GLiNER2 uses DeBERTa-v2 encoder, no MLX port exists. Would need: DeBERTa-v2 MLX impl, GLiNER head ports, weight conversion. Multi-week project.
- [ ] **MLX BERT NER** — Standard BERT NER (dslim/bert-base-NER) could run via `mlx-transformers` for the hf_token_classification backend. Medium effort.
- [ ] **GGUF support** — quantized models via llama-cpp-python for lower memory usage
- [ ] **Model download management** — pre-download models, track disk usage, cleanup old versions

## Added: 2026-02-24
