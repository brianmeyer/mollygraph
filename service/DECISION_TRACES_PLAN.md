# Decision Traces — Later-Phase Context Graph Extension

Status:
- experimental, later-phase plan
- not part of the default Ladybug local memory core
- current service behavior keeps decision traces gated or disabled on the default backend

## Inspiration
[Foundation Capital: Context Graphs — AI's Trillion-Dollar Opportunity](https://foundationcapital.com/context-graphs-ais-trillion-dollar-opportunity/)

**Core thesis:** MollyGraph can eventually store decisions as first-class graph data, but the default product stays focused on local memory, retrieval, and extraction.

## What MollyGraph Has Today

| Capability | Status |
|---|---|
| Entities connected over time | ✅ graph backend with temporal metadata |
| Episodes (what was ingested) | ✅ Episode nodes with temporal metadata |
| Local structured extraction | ✅ GLiNER2 default path |
| Parallel retrieval | ✅ Graph + vector search |
| Cross-system ingestion | ✅ Email, conversations, calendar |
| Audit trail for schema changes | ✅ Optional adoption pipeline with gates |

## What's Missing: Decision Traces

Episodes capture *what* was ingested but not *why decisions were made*. A decision trace is a future node type that could capture:

```
DecisionTrace {
  id: uuid
  timestamp: datetime
  decision: "Switched embedding model to Snowflake Arctic Embed S"
  reasoning: "ungated, local-first, simpler runtime defaults"
  alternatives_considered: ["nomic-embed-text", "BAAI/bge-small-en-v1.5", "keep current model"]
  inputs: ["MTEB benchmarks", "model size constraints", "Mac Mini M4 16GB"]
  outcome: "Reindexed 903 entities, 0 failures"
  decided_by: "Brian"  
  approved_by: "Brian"
  precedent_cited: null
  source_episode: episode_id
  confidence: 0.95
}
```

## Later-Phase Plan

### Phase 1: Decision Node Type (1 day)
- Add a `Decision` node type to the graph backend only when the later-phase work starts
- Keep the schema intentionally small at first: `decision`, `reasoning`, `alternatives`, `inputs`, `outcome`, `decided_by`, `timestamp`
- Add API and MCP surfaces only if the feature earns a dedicated delivery slice

### Phase 2: Auto-Detection from Conversations (3 days)
- When ingesting conversation transcripts, detect decision moments:
  - "Let's go with X"
  - "I decided to..."
  - "We should use X instead of Y"
  - "Approved" / "Let's do it" / "Ship it"
- Use a classifier only if the decision feature is worth the added complexity
- Create Decision nodes alongside regular entity/relationship extraction

### Phase 3: Precedent Search (2 days)  
- When a new decision is being made, search for similar past decisions
- Use vector similarity on decision text plus graph traversal on related entities
- Surface a related precedent only when it adds clear value

### Phase 4: Decision Audit Trail (1 day)
- Extend the audit pipeline only if decision traces become a maintained feature
- Weekly summaries and contradiction flags are optional follow-ons, not part of the default path

## What This Enables

1. **Searchable precedent** — "How did we handle X last time?"
2. **Decision consistency** — "This contradicts what we decided on Feb 20"
3. **Onboarding context** — New team members can see WHY things are the way they are
4. **Agent autonomy** — AI agent can reference past decisions to make similar ones without asking

## Competitive Edge

No open-source graph memory system has decision traces:
- **Mem0**: User preferences only, no decision history
- **Graphiti**: Temporal knowledge, no decision modeling
- **LightRAG**: Static extraction, no temporal or decision layer
- **Tiny-GraphRAG**: GLiNER+GLiREL extraction, no decision concept

Decision traces are one possible differentiator for MollyGraph, but they do not define the default product.

## Open Questions
- Should decisions be extracted from ALL sources or only conversations?
- How to handle decisions that get reversed later?
- Should the audit LLM or a dedicated classifier detect decisions?
- What's the minimum viable decision trace? (just decision + timestamp + who?)
