# Decision Traces — Later-Phase Context Graph Extension

Status:
- later-phase plan
- not part of the default Ladybug local memory core
- service behavior keeps decision traces gated or disabled on the default backend

## Inspiration
[Foundation Capital: Context Graphs — AI's Trillion-Dollar Opportunity](https://foundationcapital.com/context-graphs-ais-trillion-dollar-opportunity/)

**Core thesis:** MollyGraph can eventually store decisions as first-class graph data, but the default product stays focused on local memory, retrieval, and extraction.

## Current Capabilities

| Capability | Status |
|---|---|
| Entities connected over time | ✅ graph backend with temporal metadata |
| Episodes (what was ingested) | ✅ Episode nodes with temporal metadata |
| Local structured extraction | ✅ GLiNER2 default path |
| Parallel retrieval | ✅ Graph + vector search |
| Cross-system ingestion | ✅ Email, conversations, calendar |
| Audit trail for schema changes | ✅ Optional adoption pipeline with gates |

## What Decision Traces Would Add

Episodes capture *what* was ingested but not *why decisions were made*. A decision trace could capture:

```
DecisionTrace {
  id: uuid
  timestamp: datetime
  decision: "Adopt a smaller local embedding model"
  reasoning: "lower setup friction and more predictable local runtime behavior"
  alternatives_considered: ["keep the current model", "use an Ollama-only path", "use a larger multilingual model"]
  inputs: ["local benchmark notes", "hardware limits", "install complexity"]
  outcome: "embedding backend updated and local index rebuilt"
  decided_by: "operator"
  approved_by: "maintainer"
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
- Use a classifier only if the feature is worth the added complexity
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

## Positioning Hypothesis

Decision traces could become a meaningful differentiator if MollyGraph eventually treats decisions as first-class graph data instead of only storing facts and episodes.

Comparable systems to re-check when this work becomes active:
- **Mem0**
- **Graphiti**
- **LightRAG**
- **Tiny-GraphRAG**

Re-evaluate the landscape before treating this as a durable differentiator.

## Open Questions
- Should decisions be extracted from ALL sources or only conversations?
- How to handle decisions that get reversed later?
- Should a lightweight classifier or a dedicated extractor detect decisions?
- What's the minimum viable decision trace? (just decision + timestamp + who?)
