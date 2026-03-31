# MollyGraph Service Backlog

Last updated: 2026-03-27

Status legend:

- ✅ Done
- 🔄 In progress
- 🧪 Implemented, needs hardening
- 📝 Planned
- ⏸️ Parked / later-phase

Use [docs/DOCS_MAP.md](/Users/brianmeyer/mollygraph/docs/DOCS_MAP.md) before treating any markdown file as current.

## Default local core

This is the product MollyGraph is building:

- local-first graph memory for agents
- Ladybug graph backend
- Ladybug vector backend
- GLiNER2 extraction
- local embeddings by default
- MCP, HTTP, and SDK as the agent-facing surface

Everything else is secondary to this core. Audit chains, training loops, decision traces, GLiREL, and spaCy remain available as later-phase capabilities, but they are not required for the default runtime.

## Completed

- ✅ Snowflake local embedder is the default local path
- ✅ Ladybug vector backend landed and tested
- ✅ Ladybug graph backend landed and tested
- ✅ runtime graph selection via `MOLLYGRAPH_GRAPH_BACKEND`
- ✅ capability-aware `501` responses for unsupported legacy surfaces
- ✅ default OpenAPI cleanup: legacy aliases removed, experimental surface hidden by default
- ✅ Ladybug core-flow API harness covering ingest -> process -> graph/vector write -> query/stats
- ✅ root README rewritten around the local-first product story

## P0 — Keep the default local core dependable

### 1) Local runtime install and test harness — 🔄
- Keep the default setup boring and dependable for non-developers.
- Reduce environment surprises between scripts, docs, and tests.
- Keep `service/.venv`, `service/.env`, and the root pytest path aligned.

### 2) Shared Ladybug runtime owner — 📝
- Decide whether graph and vector should keep separate `.lbug` files or move to one shared Ladybug-backed runtime owner.
- Keep the safe split until the migration story is explicit.

### 3) Core API / MCP / SDK surface hardening — 🔄
- Keep the default surface focused on ingest, query, entity context, cleanup, health, and stats.
- Continue trimming compatibility-only surface and hiding unsupported experimental routes from the default docs.
- Keep MCP tools aligned with the runtime capability set.

### 4) Source-routed extraction for the highest-value sources — 📝
- Start with `contacts_json` and other sources that clearly improve graph quality without re-expanding the whole product surface.

### 5) Graph quality regression coverage — 🔄
- Keep the golden-set and core-flow harnesses growing so cleanup does not recreate bad graphs.

## P1 — Make the local product easier to operate

### 6) Backup / export / restore story — 📝
- Make local data ownership clearer.
- Define how users can back up or rebuild their local memory safely.

### 7) Query ergonomics — 📝
- Improve browse/list/search ergonomics without exploding the API surface.
- Keep the default user story simple for agents and non-developers.

### 8) Docs and repo cleanup — 🔄
- Keep the current markdown set aligned with the Ladybug-first product story.
- Remove or rewrite docs that drift back toward the old Neo4j/audit-heavy framing.

## P2 — Experimental or later-phase work

### 9) Local audit model track — ⏸️
- Later-phase only.

### 10) GLiNER training loops — ⏸️
- Preserve the work, but keep it out of the base product.

### 11) Decision traces — ⏸️
- Later-phase differentiator.
- See `service/DECISION_TRACES_PLAN.md`.

### 12) GLiREL, spaCy, and extra enrichment layers — ⏸️
- Optional quality layers, not required for the default product.
