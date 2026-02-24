# MollyGraph PM Audit — 2026-02-24

> **Persona:** Technical Product Manager reviewing for feature completeness, open-source readiness, and competitive positioning.  
> **Scope:** `service/main.py`, `config.py`, `README.md`, `BACKLOG.md`, `.env.example`, `mcp_server.py`, `BLIND_SPOTS.md`, `NEW_GAPS.md`, `ARCHITECTURE.md`, `sdk/README.md`

---

## Executive Summary

MollyGraph has a genuinely differentiated story: **self-improving extraction via LoRA**, parallel graph+vector retrieval, and a benchmark-gated deployment pipeline that no comparable open-source project has. The architecture is thoughtful and the README sells it well.

However, the codebase has a serious gap between its marketing narrative and its engineering reality:
- Critical correctness bugs (silent relationship loss, training data poisoning, partial graph writes) are **documented in internal gap files but not fixed**.
- The "single endpoint" developer story breaks down quickly due to env-var sprawl (40+ variables).
- The `.env.example` has two versions at different paths with conflicting contents and stale variable names.
- API design is inconsistent with legacy aliases accumulating across 50+ endpoint registrations.
- No versioned API (`/v1/...`), no pagination on list operations, no webhook/event system.

**The project is compelling for early adopters and internal use. To be a credible open-source project, it needs a stability sprint, then a polish sprint, before a public launch.**

---

## Priority Index

| ID | Priority | Category | One-liner |
|----|----------|----------|-----------|
| A1 | P0 | Correctness | Silent relationship loss when entity is missing |
| A2 | P0 | Correctness | Training data poisoned with deleted relations as positives |
| A3 | P0 | Correctness | `GLiNERTrainingMixin` NameError crashes all training |
| A4 | P0 | Correctness | Embedding config endpoint doesn't control actual embeddings |
| B1 | P1 | API | No API versioning (`/v1/`) — breaking changes will hurt |
| B2 | P1 | API | `GET /query` should be `POST` (query strings in URLs leak) |
| B3 | P1 | API | `/entities` list endpoint missing (no way to browse graph) |
| B4 | P1 | API | No pagination on any list operation |
| B5 | P1 | API | Async event loop blocked by sync Neo4j calls in hot paths |
| C1 | P1 | DX | Two `.env.example` files with conflicting content |
| C2 | P1 | DX | 40+ env vars with inconsistent prefix/naming conventions |
| C3 | P1 | DX | No `docker-compose.yml` for the full stack (only Neo4j) |
| C4 | P1 | DX | No OpenAPI/Swagger interactive documentation served |
| D1 | P1 | Features | No multi-tenant / namespace isolation |
| D2 | P1 | Features | MCP tool `search_nodes` is a dummy (just aliases `search_facts`) |
| D3 | P2 | Features | `prune_entities` MCP tool can't prune orphans (no `orphans` param) |
| E1 | P2 | Observability | Nightly pipeline success/failure not surfaced in metrics |
| E2 | P2 | Observability | `/training/runs` has no pagination, returns raw 20 item limit |
| E3 | P2 | Observability | No alerting/webhook when schema-drift alarm fires |
| F1 | P2 | Docs | Architecture doc is minimal stub; doesn't match current code |
| F2 | P2 | Docs | No deployment guide (Docker, systemd, cloud) |
| F3 | P2 | Docs | SDK README missing 6 of 9 MCP tools |
| G1 | P2 | Competitive | GLiREL integration exists in config but isn't surfaced in README |
| G2 | P3 | Competitive | No web UI for graph exploration (competitors have this) |
| G3 | P3 | Config | `STRICT_AI` mode is undocumented in README and `.env.example` |
| G4 | P3 | Config | `AUDIT_MODEL_SECONDARY` / `TERTIARY` in `.env.example` don't exist in `config.py` |

---

## 1. API Completeness

### A1 · P0 · Correctness: Silent relationship loss

**Finding:** When the extractor emits `A -> B` and entity `B` was filtered out, the Cypher `MATCH ... CREATE` finds nothing, returns a synthetic UUID, and the pipeline continues as if the write succeeded. The relationship is silently dropped.  
**Impact:** Retrieval miss. Training data missing real examples. No signal to the operator.  
**Recommendation:** Return a write-count from the relationship upsert. Fail (or at minimum log and metric) any job where `relationships_written < relationships_extracted`.

---

### A2 · P0 · Correctness: Training data poisoned

**Finding:** Relations with `audit_status='deleted'` can appear in both the positive examples query (which only excludes `quarantined`) and the negative examples query (which includes `deleted`). The same `(head, label, tail)` tuple can supervise in opposite directions in the same training batch.  
**Impact:** LoRA training receives contradictory gradients. F1 improvements may be artefacts, not real gains. The "self-improving" story depends on clean labels.  
**Recommendation:** Unify exclusion: positives must exclude BOTH `quarantined` AND `deleted`. Negatives must source from a different pool (e.g., type-confused extractions), not deleted relations.

---

### A3 · P0 · Correctness: `NameError` crashes all training

**Finding:** `evolution/gliner_training.py` references `GLiNERTrainingMixin._finetune_running` but `GLiNERTrainingMixin` is not defined in that module. Every call to `POST /train/gliner` (and nightly step 4) raises `NameError`.  
**Impact:** The headline differentiator — self-improving extraction — is broken for anyone who triggers training.  
**Recommendation:** Fix the reference. Add a smoke test that calls `POST /train/gliner?force=true` in a dry-run mode as part of CI.

---

### A4 · P0 · Config disconnect: Embedding API vs pipeline

**Finding:** `POST /embeddings/config` updates the embedding registry JSON. But `ExtractionPipeline._text_embedding()` reads from a hardcoded model path (or hash fallback) — it does not consult the registry at runtime.  
**Impact:** Operators are told they can "swap embedding providers live." They cannot. The registry is cosmetic. Vector searches use different embeddings than documents are indexed with, causing silent quality regression.  
**Recommendation:** Wire `_text_embedding()` to read the active provider from the registry at query time. Add an integration test that switches providers and verifies embeddings change.

---

### B1 · P1 · API: No versioning

**Finding:** All endpoints live at `/` with no `/v1/` prefix. Legacy aliases (`/training/gliner`, `/maintenance/audit`, `/suggestions_digest`) are piling up.  
**Impact:** Any breaking API change requires versioned clients to manage. As the endpoint list grows (50+ decorators), backwards compatibility will become unmanageable.  
**Recommendation:** Add `/v1/` prefix now, before public release. Keep current paths as deprecated aliases for 1 release cycle. Update README and SDK accordingly.

---

### B2 · P1 · API: `GET /query` is wrong for this use case

**Finding:** `/query?q=<text>` passes potentially long, sensitive query text as a URL query parameter. These appear in server access logs, browser history, CDN caches, and reverse proxy logs.  
**Impact:** Privacy leak. Also limits query length to URL limits.  
**Recommendation:** Change to `POST /query` with `{"q": "...", "limit": 5, "filters": {...}}`. Keep `GET /query` as a deprecated alias for backward compat for one version.

---

### B3 · P1 · API: No entity listing endpoint

**Finding:** There is `GET /entity/{name}` (point lookup) but no `GET /entities` (list/browse). To explore the graph via API, you must know the entity name already. The `/stats` endpoint gives aggregate counts but not the actual entity list.  
**Impact:** Cannot build a UI, cannot audit what's in the graph without direct Neo4j access. Blocks Web UI, SDK graph browsing, and basic debugging.  
**Recommendation:** Add `GET /entities?page=1&limit=50&type=Person&sort=name` returning paginated entity summaries. This is the most-requested endpoint for any graph memory system.

---

### B4 · P1 · API: No pagination anywhere

**Finding:** `/training/runs` has a hardcoded `limit=20`. `/entities/prune` returns the full pruned list. `/audit` scans up to `limit=5000` in one shot. No endpoint returns `next_cursor` or `total_count`.  
**Impact:** Any graph with 1000+ entities/runs cannot be browsed safely. Large audit requests block the event loop.  
**Recommendation:** Add cursor-based pagination (`cursor`, `limit`, `has_more`) to: `/training/runs`, future `/entities`, `/audit` result listing. Use keyset pagination for Neo4j queries.

---

### B5 · P1 · API: Sync Neo4j calls blocking async event loop

**Finding:** `graph.get_current_facts()`, `graph.get_entity_context()`, and Neo4j session operations run synchronously inside `async def` handlers without `asyncio.to_thread()`. Under concurrent load, this blocks all other requests.  
**Impact:** At ~100 concurrent queries, latency spikes. The `/query` parallel branches are undermined by blocking execution.  
**Recommendation:** Wrap all graph calls in `asyncio.to_thread()` consistently. Consider `neo4j-driver`'s async API (`neo4j.AsyncDriver`) for the hot paths.

---

## 2. Developer Experience

### C1 · P1 · DX: Two `.env.example` files with conflicting content

**Finding:**
- `/service/.env.example` — newer, organized with tier-chain comments, references `MOLLYGRAPH_AUDIT_TIER_PRIMARY`
- `root .env.example` — older, references `AUDIT_MODEL_SECONDARY`, `AUDIT_MODEL_TERTIARY`, `MOLLYGRAPH_EXTRACTOR_RELATION_MODEL=Babelscape/rebel-large` (a model that may not be in use), and `MOLLYGRAPH_EMBEDDING_BACKEND=hash` (which would disable the primary embedding)

**Impact:** New contributors follow the wrong file. Root `.env.example` sets `hash` embeddings — a first-run experience with only hash embeddings produces garbage retrieval.  
**Recommendation:** Delete the root `.env.example`. Symlink or reference `service/.env.example` from root. Remove stale variables (`AUDIT_MODEL_SECONDARY`, `AUDIT_MODEL_TERTIARY`, `MOLLYGRAPH_EXTRACTOR_RELATION_MODEL`).

---

### C2 · P1 · DX: 40+ env vars with inconsistent naming

**Finding:** Variables use 4+ different prefix conventions:
- `MOLLYGRAPH_*` — most new variables
- `GRAPH_MEMORY_*` — legacy host/port (`GRAPH_MEMORY_HOST`, `GRAPH_MEMORY_PORT`)
- `GLINER_*` — training variables (no MOLLYGRAPH prefix)
- `AUDIT_*` — some without prefix, some with `MOLLYGRAPH_AUDIT_*`
- `NEO4J_*` — database (separate namespace, reasonable)
- `OLLAMA_*` — some with `MOLLYGRAPH_` prefix, some without

**Specific confusion points:**
- `GRAPH_MEMORY_HOST` vs `MOLLYGRAPH_HOST` (both exist in README, only former in config.py)
- `MOLLYGRAPH_EMBEDDING_OLLAMA_MODEL` vs `MOLLYGRAPH_OLLAMA_EMBED_MODEL` (both accepted as aliases)
- `AUDIT_MODEL_LOCAL` vs `MOLLYGRAPH_AUDIT_MODEL_LOCAL` (both parsed, different variable in config.py)

**Recommendation:** Standardize to `MOLLYGRAPH_*` for all service variables. Keep `NEO4J_*` as an external namespace. Document migration of `GRAPH_MEMORY_HOST` → `MOLLYGRAPH_HOST` with a deprecation warning at startup.

---

### C3 · P1 · DX: No full-stack Docker Compose

**Finding:** `docker-compose.neo4j.yml` only starts Neo4j. The MollyGraph service itself requires a manual `./scripts/install.sh && ./scripts/start.sh` flow.  
**Impact:** Quick-start experience requires multiple steps. No easy path for CI, Kubernetes, or cloud deployment. Competitors (Mem0, Graphiti) have Docker-first setups.  
**Recommendation:** Add `docker-compose.yml` (full stack: Neo4j + MollyGraph service). Include a `Dockerfile` for the service. The Quick Start in README should be 3 commands: `git clone`, `cp .env.example .env`, `docker compose up`.

---

### C4 · P1 · DX: No interactive API documentation

**Finding:** FastAPI auto-generates `/docs` (Swagger UI) and `/redoc` at runtime, but these aren't mentioned in the README. `operation_id` is set inconsistently — some endpoints have it, many don't. No OpenAPI spec is exported or committed.  
**Impact:** New developers can't discover the API without reading 2000 lines of `main.py`. The Python SDK README lists 7 MCP tools but the README lists 9 — gap isn't obvious.  
**Recommendation:**
1. Add "API Explorer: `http://localhost:7422/docs`" to README Quick Start
2. Commit a generated `openapi.json` to the repo (can be automated in CI)
3. Add `operation_id` to all endpoints (currently ~half have it)

---

## 3. Feature Gaps

### D1 · P1 · Features: No multi-tenant / namespace isolation

**Finding:** All entities share a single Neo4j namespace. There's no concept of "workspace," "user," or "agent" isolation. `source` field on Episode nodes exists but isn't used for retrieval filtering.  
**Impact:** Can't use MollyGraph for multiple AI assistants or users from the same service. Every assistant sees every other assistant's memory. This is the #1 blocker for production multi-agent use cases.  
**Recommendation:** Add a `namespace` field to Entity, Episode, and Relationship nodes. Make it a required or optional query parameter on `/ingest` and `/query`. Default to `"default"` for backward compatibility.

---

### D2 · P1 · Features: MCP `search_nodes` is a dummy tool

**Finding:** `mcp_server.py` `search_nodes` calls `GET /query` and returns `item["entity"]` names from the response. It's functionally identical to `search_facts` but returns less data. It does NOT search by entity type, fuzzy name match, or any property filter. It has no relationship to the `B3` missing `/entities` list endpoint.  
**Impact:** MCP clients advertising "9 distinct tools" are misrepresenting capability. Claude or any LLM using MCP will attempt `search_nodes` for node browsing and get undifferentiated results.  
**Recommendation:** Once `GET /entities` endpoint exists, wire `search_nodes` to it with `type` filter support. Make the semantic distinction clear: `search_facts` = semantic query, `search_nodes` = entity type/name browse.

---

### D3 · P2 · Features: MCP `prune_entities` can't prune orphans

**Finding:** `POST /entities/prune` supports `{"orphans": true}` to bulk-delete zero-relationship entities. But the MCP `prune_entities` tool signature is `prune_entities(names: list[str])` — only accepts explicit names.  
**Impact:** LLM agents can't trigger orphan pruning via MCP without first fetching the entity list themselves (which they can't, per D2/B3).  
**Recommendation:** Add `orphans: bool = False` parameter to MCP `prune_entities` tool. If `orphans=True` and `names=[]`, call the API with `{"orphans": true}`.

---

## 4. Configuration UX

### G3 · P3 · Config: `STRICT_AI` mode undocumented

**Finding:** `MOLLYGRAPH_STRICT_AI` and `MOLLYGRAPH_RUNTIME_PROFILE=strict_ai` exist in `config.py` and are checked at startup, but are not documented in README or `.env.example`. A user discovering this in logs or code has no idea what it does.  
**Recommendation:** Add a "Runtime Profiles" section to README: `hybrid` (default), `local` (no cloud providers), `strict_ai` (startup validation of all AI models). Document what each mode blocks.

---

### G4 · P3 · Config: Stale variables in `.env.example`

**Finding (root `.env.example`):**
- `AUDIT_MODEL_SECONDARY`, `AUDIT_MODEL_TERTIARY` — not parsed in `config.py`
- `MOLLYGRAPH_EXTRACTOR_RELATION_MODEL=Babelscape/rebel-large` — not referenced in active pipeline
- `MOLLYGRAPH_EMBEDDING_CONFIG_FILE` — parsed in embedding registry but not in `config.py`

**Impact:** Contributors add these to their `.env`, nothing happens, they wonder why.  
**Recommendation:** Audit every variable in both `.env.example` files against `config.py`. Remove any that aren't parsed. Add a CI test that loads `.env.example` and validates all variables against the config module.

---

## 5. Observability

### E1 · P2 · Observability: Nightly pipeline health not surfaced

**Finding:** `POST /maintenance/nightly` runs 4 sequential steps in a background task. If step 2 (model health check) triggers a rollback, or step 3 (training cleanup) removes 500 examples, this is logged to stdout but not persisted to metrics.  
**Impact:** Operator can't know from `/metrics/dashboard` whether the last nightly run succeeded. No way to detect "nightly ran but LLM audit failed silently for 7 days."  
**Recommendation:** Write a `nightly_run_log.jsonl` with per-step status/duration/outcome. Add a `nightly` section to `/metrics/dashboard`:
```json
"nightly": {
  "last_run_at": "2026-02-24T03:00:00Z",
  "steps": {"audit": "ok", "model_health": "ok", "cleanup": "ok", "lora": "skipped"},
  "next_run_scheduled": null
}
```

---

### E2 · P2 · Observability: Training runs endpoint needs pagination

**Finding:** `GET /training/runs?limit=20` is hardcoded. For a deployment with 100+ training runs, there's no way to page through history.  
**Recommendation:** Add `offset` parameter. Return `total_runs` in response. Expose per-run detail at `GET /training/runs/{run_id}`.

---

### E3 · P2 · Observability: Schema drift alarm has no delivery mechanism

**Finding:** `GET /metrics/schema-drift` sets `"alarm": true` in the JSON response but nothing actively notifies anyone. The alarm only fires if someone polls the endpoint.  
**Impact:** Schema proliferation can run unchecked unless someone manually checks `/metrics/schema-drift` each day. The README says "alarm at >5%/day growth" but the alarm is passive.  
**Recommendation:** Add optional webhook configuration: `MOLLYGRAPH_ALERT_WEBHOOK_URL`. When alarm fires during nightly maintenance, POST a JSON payload to the webhook. Works with Slack, Discord, PagerDuty, and generic HTTP.

---

## 6. Documentation Gaps

### F1 · P2 · Docs: Architecture doc is a stub

**Finding:** `docs/ARCHITECTURE.md` is ~60 lines and describes a simpler version of the system (it still says "disabled" as the default audit chain, references `AUDIT_PROVIDER_ORDER=none`). It doesn't document:
- The bi-temporal graph model
- The embedding tier chain
- The parallel query architecture
- The LoRA pipeline and benchmark gates
- The maintenance lock mechanism

**Recommendation:** Expand to a full architecture document with a flow diagram (Mermaid). This is essential for contributors and the open-source README already has a partial ASCII diagram that could be the basis.

---

### F2 · P2 · Docs: No deployment guide

**Finding:** CONTRIBUTING.md covers local development. README covers Quick Start. Nothing covers:
- Production deployment (systemd service, Docker)
- Nginx/reverse proxy setup
- How to change the default port
- Backup/restore of Neo4j + vector store
- Upgrading between versions

**Recommendation:** Add `docs/DEPLOYMENT.md` covering: Docker, systemd, nginx, and cloud (fly.io, Railway). The FAQ-style "How do I expose this beyond localhost?" is the most common new-user question for local-first services.

---

### F3 · P2 · Docs: SDK README vs README tool count mismatch

**Finding:**
- `README.md` advertises **9 MCP tools**: `search_facts`, `search_nodes`, `get_entity_context`, `add_episode`, `delete_entity`, `prune_entities`, `run_audit`, `get_training_status`, `get_queue_status`
- `sdk/README.md` lists only **7 tools** (missing `delete_entity`, `prune_entities`)
- `mcp_server.py` implements all 9

**Recommendation:** Update `sdk/README.md` tool table to match. These management tools are differentiators — they should be marketed, not hidden.

---

## 7. Competitive Positioning

### Summary vs. Competitors

| Feature | Mem0 | LightRAG | Graphiti | MollyGraph |
|---------|------|----------|----------|-----------|
| Self-improving extraction | ❌ | ❌ | ❌ | ✅ (LoRA) |
| Parallel graph+vector retrieval | ❌ | ⚠️ (waterfall) | ❌ | ✅ |
| Benchmark-gated deployment | ❌ | ❌ | ❌ | ✅ |
| Local-first (no cloud required) | ❌ | ⚠️ | ❌ | ✅ |
| Entity/rel delete | ✅ | ❌ | ⚠️ | ✅ |
| Multi-tenant namespaces | ✅ | ❌ | ✅ | ❌ **GAP** |
| Web UI | ✅ | ❌ | ❌ | ❌ **GAP** |
| Docker-first setup | ✅ | ✅ | ✅ | ⚠️ (Neo4j only) |
| MCP integration | ✅ | ❌ | ✅ | ✅ |
| OpenAPI docs | ✅ | ❌ | ✅ | ⚠️ (hidden) |
| Swappable embeddings | ❌ | ❌ | ❌ | ✅ (tier chain) |
| Bi-temporal graph | ❌ | ❌ | ✅ | ✅ |

---

### G1 · P2 · Competitive: GLiREL is half-built and undersold

**Finding:** GLiREL (`jackboyla/glirel-large-v0`) is the answer to the "8.4% RELATED_TO fallback" problem and directly addresses the most-cited gap in any GLiNER-based system. It's configured in `config.py`, there's an `extraction/glirel_enrichment.py`, but:
- It's disabled by default with no mention of this in README
- The Roadmap section lists it as `[ ]` to-do
- The competitive table in README doesn't mention it

**Impact:** If GLiREL works (even partially), it's a headline feature. A system that auto-names relationships is substantially more useful than one that labels everything `RELATED_TO`.  
**Recommendation:** If GLiREL is functional, enable it behind a flag and benchmark the RELATED_TO fallback rate improvement. Document it in README as "Beta: GLiREL relation enrichment — reduce RELATED_TO fallback from 2.3% further with second-pass extraction."

---

### G2 · P3 · Competitive: No web UI for graph exploration

**Finding:** All competitors with graph memory offer some form of visual graph exploration. The knowledge graph is the core value proposition — users should be able to see it without connecting directly to Neo4j.  
**Recommendation:**
- **Short-term (weeks):** Add a `GET /graph/export` endpoint that returns nodes/edges in a format suitable for D3 or Vis.js (JSON, Cytoscape.js format). Let users embed the visualization themselves.
- **Medium-term (months):** Ship a single-page `index.html` at `GET /` using a lightweight graph lib (Cytoscape.js). Authentication via same Bearer token. No external dependencies.
- **Long-term:** A proper React/Vue app in `web/` directory.

The short-term option is a single endpoint + a 200-line HTML file. High impact, low effort.

---

## Appendix A: Endpoint Audit Summary

**Total endpoint registrations:** 51 (including legacy aliases)  
**Unique functional endpoints:** ~36  
**Legacy alias debt:** 15 duplicate registrations

**Missing CRUD:**
| Resource | GET list | GET by ID | POST create | PUT/PATCH update | DELETE |
|----------|----------|-----------|-------------|------------------|--------|
| Entity | ❌ **missing** | ✅ | via `/ingest` | ❌ **missing** | ✅ |
| Relationship | ❌ **missing** | ❌ **missing** | via `/ingest` | ❌ **missing** | ✅ |
| Episode | ❌ **missing** | ❌ **missing** | ✅ `/ingest` | ❌ | ❌ **missing** |
| Training run | ✅ `/training/runs` | ❌ **missing** | ✅ `/train/gliner` | ❌ | ❌ |

**Naming inconsistencies:**
- `/train/gliner` and `/train/status` use `train/` prefix
- `/training/runs` uses `training/` prefix
- `/maintenance/nightly` and `/maintenance/run` overlap in purpose (two ways to trigger maintenance)
- `/metrics/model-health` and `/model-health/status` are the same resource at different paths

---

## Appendix B: `.env.example` Variable Audit

Variables in root `.env.example` NOT in `config.py`:
- `AUDIT_MODEL_SECONDARY` — stale
- `AUDIT_MODEL_TERTIARY` — stale
- `MOLLYGRAPH_EXTRACTOR_RELATION_MODEL` — stale (REBEL-large not active)
- `MOLLYGRAPH_EMBEDDING_CONFIG_FILE` — parsed in registry but not in `config.py`

Variables in `config.py` NOT in either `.env.example`:
- `MOLLYGRAPH_AUDIT_PROVIDER_TIERS`
- `MOLLYGRAPH_AUDIT_TIER_LOCAL`
- `MOLLYGRAPH_AUDIT_TIER_FALLBACK`
- `MOLLYGRAPH_AUDIT_MODEL_FALLBACK`
- `MOLLYGRAPH_STRICT_AI`
- `MOLLYGRAPH_RUNTIME_PROFILE`
- `MOLLYGRAPH_GLIREL_LLM_SYNONYMS`
- `MOLLYGRAPH_DRIFT_ALARM_REL`
- `MOLLYGRAPH_DRIFT_ALARM_ENT`
- `MOLLYGRAPH_LOCK_TIMEOUT`
- `MOLLYGRAPH_MAX_NEW_RELATIONS`
- `MOLLYGRAPH_MAX_NEW_ENTITIES`
- `MOLLYGRAPH_DEGRADATION_WINDOW`
- `MOLLYGRAPH_DEGRADATION_THRESHOLD`
- `MOLLYGRAPH_LORA_BENCHMARK_TOLERANCE`
- `GLINER_LORA_COOLDOWN_DAYS`

---

## Recommended Sprint Plan

### Sprint 0: Stability (Pre-release blocker)
Fix P0 bugs. Nothing ships without these.
1. Fix `GLiNERTrainingMixin` NameError → training unblocked
2. Fix training data positive/negative contamination
3. Fix silent relationship loss (write-count validation)
4. Wire embedding config to actual pipeline

### Sprint 1: Developer Experience
The open-source story depends on a clean first-run.
1. Consolidate `.env.example` (one file, no stale vars, no `hash` as default embedding)
2. Add `docker-compose.yml` for full stack
3. Standardize env var prefixes + add startup deprecation warnings
4. Add "API Explorer" link to README Quick Start

### Sprint 2: API Polish
Make the API something contributors can build on.
1. Add `GET /entities` with pagination
2. Add `POST /v1/query` (deprecate `GET /query`)
3. Add `/v1/` prefix across all endpoints
4. Fix `search_nodes` MCP tool to use entity listing
5. Add `orphans` param to MCP `prune_entities`

### Sprint 3: Observability + Positioning
Make operators confident and competitors nervous.
1. Add nightly pipeline result logging to dashboard
2. Add webhook alerting for schema-drift alarm
3. Add `GET /graph/export` + embedded Cytoscape viewer at `GET /`
4. Enable and benchmark GLiREL, document in README
5. Add namespace/multi-tenant support

---

*Audit completed 2026-02-24 by PM review of commit state at that date.*
