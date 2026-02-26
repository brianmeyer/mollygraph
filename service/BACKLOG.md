# MollyGraph Backlog (Unified)

Last updated: 2026-02-25

Status legend:
- ✅ Done
- 🔄 In progress
- 🧪 Implemented, needs hardening
- ⏸️ Parked
- 📝 Planned

---

## Completed Recently (Do not re-open unless broken)

- ✅ Graphiti-style speaker-anchored extraction (per-message processing, speaker flow API→queue→pipeline)
- ✅ Per-source confidence thresholds (chat/email/default)
- ✅ Graph purge + old noisy training data purge
- ✅ Reset to base GLiNER2 after noisy LoRA data reset
- ✅ Contact ingestion overhaul: NLP reformatter for `contacts_json`
- ✅ Direct contact NER training generator (`contact-ner-*.jsonl`)
- ✅ Graph-aware reranker enabled and tuned (direct name-match fix)
- ✅ Nightly audit + LoRA pipeline operational
- ✅ Contact re-ingestion completed; training examples rebuilt from real contacts

---

## P0 — Correctness / Data Quality

### 1) Pre-write semantic relation gate (soft, schema-aware) — ✅
- **Implemented** in `extraction/relation_gate.py` + integrated in `extraction/pipeline.py`.
- Scores every candidate by `(head_type, rel_type, tail_type, source, confidence)`.
- Action policy: **allow** → write normally; **quarantine** → write with `audit_status='quarantined'` + emit audit-bus + suggestion signal; **skip** → emit suggestion signal only (never silently drops).
- Feature flag: `MOLLYGRAPH_RELATION_SOFT_GATE_ENABLED=true` (default on).
- Threshold env vars: `RELATION_GATE_ALLOW_THRESHOLD`, `RELATION_GATE_QUARANTINE_THRESHOLD`, `RELATION_GATE_HIGH_CONF_OVERRIDE`.
- 25 unit + regression tests in `tests/test_relation_gate.py` — all pass.

### 2) Contact relation constraints by source priors — ✅ (bundled with P0-1)
- contacts_json source priors implemented inside the soft gate.
- Boosted rels: `CONTACT_OF / WORKS_AT / LOCATED_IN / MEMBER_OF / CLASSMATE_OF` (+30% plausibility multiplier).
- Down-weighted rels: `CHILD_OF / REPORTS_TO / MENTORED_BY / MENTORS / MANAGES` (×0.55 multiplier, unless extractor confidence ≥ 0.85 → quarantine instead of skip).
- Soft (multiplicative) — not a hard stoplist; evolution/suggestion path preserved.

### 3) Audit selection risk weighting — 📝
- Prioritize risky/high-impact relation labels vs random slices.
- Source: `AUDIT_QA.md`, `AUDIT_PM.md`

### 4) Ontology consistency checks (school/cohort/campus/person roles) — 📝
- Periodic graph consistency job + auto-correct suggestions.
- Source: `mollygraph-architecture-v2.md`, `AUDIT_ARCHITECT.md`

---

## P1 — Specialized Model Pipeline

### 5) Source-routed extraction pipeline — 🔄
- Route extractor strategy by source type (contacts/chat/email/docs).
- Keep base GLiNER2+GLiREL, add per-source model/config routing.
- Source: `mollygraph-extraction-quality.md` (P7), `mollygraph-master-plan.md`

### 6) Specialized relation plausibility layer (post-extractor) — 📝
- Lightweight classifier/reranker for semantic plausibility of relation candidates.
- Source: `AUDIT_DEVELOPER.md`, `AUDIT_ARCHITECT.md`

### 7) Local-first audit model track (Ollama) — 🔄
- Move from Moonshot dependency toward reliable local structured verdicts.
- Current state: Moonshot primary, Gemini fallback; local JSON reliability unresolved.

---

## P1 — Training & Audit Reliability

### 8) Pre-training quality gate robustness — 🧪
- Quarantine/deleted filtering exists; add stronger contradiction negatives and stricter pre-train checks.
- Source: `evolution/gliner_training.py`, `AUDIT_QA.md`

### 9) Audit state concurrency + durability hardening — 📝
- Locking + atomic write + crash-safe recovery.
- Source: `AUDIT_ARCHITECT.md`, `AUDIT_QA.md`

### 10) Auto-delete blast-radius controls — 🧪
- Cap exists conceptually; verify and enforce strict per-cycle safety policy.
- Source: `AUDIT_QA.md`

---

## P2 — Retrieval / Product / Ops

### 11) Retrieval instrumentation v2 — 📝
- Better attribution for graph_exact/vector/merged/reranked lift.

### 12) Reranker stack progression — 🔄
- Graph-aware reranker is live.
- Next: robust blended policy with Jina cross-encoder at larger scale.

### 13) Decision Traces — 🔄
- Planned major differentiator. Implement per `DECISION_TRACES_PLAN.md` phases 1–4.

### 14) API ergonomics — 📝
- Versioned API (`/v1`), pagination, browse/list endpoints.

### 15) Reliability test coverage — 📝
- Expand unit/integration tests for extraction/audit/training/vector sync.

### 16) Operational observability — 📝
- External alerting for worker death, rollback, audit failures, queue anomalies.

### 17) Infra health decision framework (deterministic-first) — ✅
- Implemented `maintenance/infra_health.py` with deterministic policy outputs:
  `healthy | optimize | refresh_embeddings | reindex_embeddings | rebuild_vectors`.
- Added env-driven thresholds (`MOLLYGRAPH_INFRA_*`) with safe defaults.
- Added endpoint: `POST /maintenance/infra-health/evaluate` (dry-run + optional advisory).
- Nightly pipeline now emits infra health snapshot in nightly run payload.
- `POST /maintenance/reconcile-vectors` now supports guarded `mode=rebuild` fallback path.

---

## Parked (Intentional)

- ⏸️ Hard deterministic stoplists/content classifiers as primary guardrail.
- Rationale: brittle across domains; prefer soft semantic gating + audit loop.

---

## Next 3 Execution Items

1. ~~Implement pre-write semantic relation gate in `extraction/pipeline.py` (P0)~~ ✅ Done
2. Add risk-weighted audit selection for high-risk relation labels (P0 #3)
3. Implement first source router slice for `contacts_json` relation policy (P1 #5)
