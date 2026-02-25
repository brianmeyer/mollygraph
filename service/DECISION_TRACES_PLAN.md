# Decision Traces — MollyGraph Context Graph Extension

## Inspiration
[Foundation Capital: Context Graphs — AI's Trillion-Dollar Opportunity](https://foundationcapital.com/context-graphs-ais-trillion-dollar-opportunity/)

**Core thesis:** The next systems of record won't store objects (like Salesforce/Workday) — they'll store *decisions*. Who decided what, why, what alternatives existed, what precedent was cited, and what happened as a result.

## What MollyGraph Has Today

| Capability | Status |
|---|---|
| Entities connected over time | ✅ Bi-temporal graph (Neo4j) |
| Episodes (what was ingested) | ✅ Episode nodes with temporal metadata |
| Self-improving extraction | ✅ LoRA training loop + GLiREL |
| Parallel retrieval | ✅ Graph + vector search |
| Cross-system ingestion | ✅ Email, conversations, calendar |
| Audit trail for schema changes | ✅ Adoption pipeline with gates |

## What's Missing: Decision Traces

Episodes capture *what* was ingested but not *why decisions were made*. A decision trace is a first-class node type that captures:

```
DecisionTrace {
  id: uuid
  timestamp: datetime
  decision: "Switched embedding model from embeddingGemma to Jina v5-nano"
  reasoning: "71.0 vs 62 MTEB, smaller params, 8192 context, newer"
  alternatives_considered: ["nomic-embed-text", "BAAI/bge-m3", "keep embeddingGemma"]
  inputs: ["MTEB benchmarks", "model size constraints", "Mac Mini M4 16GB"]
  outcome: "Reindexed 903 entities, 0 failures"
  decided_by: "Brian"  
  approved_by: "Brian"
  precedent_cited: null
  source_episode: episode_id
  confidence: 0.95
}
```

## Implementation Plan

### Phase 1: Decision Node Type (1 day)
- Add `Decision` node type to Neo4j schema
- Properties: `decision`, `reasoning`, `alternatives`, `inputs`, `outcome`, `decided_by`, `timestamp`
- Relationships: `DECIDED_BY` (→ Person), `RELATES_TO` (→ Entity), `PRECEDED_BY` (→ Decision), `SOURCED_FROM` (→ Episode)
- API: `POST /decisions` to create, `GET /decisions` to query
- MCP tool: `record_decision`, `query_decisions`

### Phase 2: Auto-Detection from Conversations (3 days)
- When ingesting conversation transcripts, detect decision moments:
  - "Let's go with X"
  - "I decided to..."
  - "We should use X instead of Y"
  - "Approved" / "Let's do it" / "Ship it"
- Use the audit LLM to classify: "Is this text recording a decision? If so, extract: decision, reasoning, alternatives"
- Create Decision nodes alongside regular entity/relationship extraction

### Phase 3: Precedent Search (2 days)  
- When a new decision is being made, search for similar past decisions
- Vector similarity on decision text + graph traversal on related entities
- "Last time we changed embedding models, we did X and it took Y"
- Surface in retrieval results: "Related precedent: [decision]"

### Phase 4: Decision Audit Trail (1 day)
- The audit pipeline already tracks schema changes — extend to track all Decision nodes
- Weekly summary: "5 decisions recorded this week, 2 with precedent matches"
- Flag decisions that contradict previous decisions on the same topic

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

MollyGraph with decision traces = **the self-improving context graph for AI agents**.

## Open Questions
- Should decisions be extracted from ALL sources or only conversations?
- How to handle decisions that get reversed later?
- Should the audit LLM or a dedicated classifier detect decisions?
- What's the minimum viable decision trace? (just decision + timestamp + who?)
