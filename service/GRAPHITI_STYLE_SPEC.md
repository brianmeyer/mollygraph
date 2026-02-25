# Graphiti-Style Extraction Upgrade for MollyGraph

## Overview
Upgrade the extraction pipeline to process messages individually with speaker-anchored entity extraction, instead of bulk-chunking 50 messages together.

## Files to Modify

### 1. `extraction/pipeline.py` (~50 lines)

**A. `_normalize_source()` (line ~920)**
Add these sources to the allowed set:
- "session", "conversation", "correction", "contacts_json"
Current: `{"manual", "whatsapp", "voice", "email", "imessage"}`

**B. `process_job()` (line 296)**
- Add optional `speaker` field to ExtractionJob
- If speaker is provided, prepend "Statement by {speaker}: " to content before GLiNER extraction
- After extraction, if speaker is provided, ensure speaker is always an entity node and all extracted relationships radiate FROM speaker (speaker is always head node)

**C. Per-source confidence thresholds (line 317)**
Replace hardcoded `0.4` with config-driven threshold:
```python
threshold = getattr(config, 'EXTRACTION_CONFIDENCE', {}).get(
    self._normalize_source(job.source), 
    config.EXTRACTION_CONFIDENCE_DEFAULT
)
extracted = await asyncio.to_thread(gliner_extractor.extract, job.content, threshold)
```

**D. spaCy fallback (around line 327)**
Disable spaCy enrichment for chat/session sources:
```python
chat_sources = {"session", "conversation", "whatsapp", "imessage"}
if service_config.SPACY_ENRICHMENT and ingest_source not in chat_sources:
    spacy_entities = self._spacy_enrich_entities(...)
```

### 2. `api/ingest.py` (~10 lines)

**Add `speaker` field to IngestRequest body:**
```python
class IngestRequest(BaseModel):
    content: str
    source: str = "manual"
    speaker: str | None = None  # NEW: who said this
```

Pass `speaker` through to ExtractionJob.

### 3. `config.py` (~15 lines)

Add per-source confidence thresholds:
```python
EXTRACTION_CONFIDENCE_DEFAULT = float(os.environ.get("MOLLYGRAPH_EXTRACTION_CONFIDENCE_DEFAULT", "0.4"))
EXTRACTION_CONFIDENCE = {
    "session": float(os.environ.get("MOLLYGRAPH_EXTRACTION_CONFIDENCE_SESSION", "0.55")),
    "conversation": float(os.environ.get("MOLLYGRAPH_EXTRACTION_CONFIDENCE_CONVERSATION", "0.55")),
    "whatsapp": float(os.environ.get("MOLLYGRAPH_EXTRACTION_CONFIDENCE_WHATSAPP", "0.55")),
    "imessage": float(os.environ.get("MOLLYGRAPH_EXTRACTION_CONFIDENCE_IMESSAGE", "0.55")),
    "email": float(os.environ.get("MOLLYGRAPH_EXTRACTION_CONFIDENCE_EMAIL", "0.45")),
    "voice": float(os.environ.get("MOLLYGRAPH_EXTRACTION_CONFIDENCE_VOICE", "0.50")),
}
```

### 4. `extraction/queue.py` (~5 lines)

Add `speaker` field to ExtractionJob dataclass.

## Testing
After changes:
1. Restart service
2. Ingest a test message with speaker:
```bash
curl -s -X POST http://127.0.0.1:7422/ingest \
  -H "Authorization: Bearer dev-key-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{"content": "Marcela mentioned that Greyson did great at the parent-teacher conference", "source": "session", "speaker": "Brian"}'
```
3. Verify: Brian is an entity, Marcela is an entity, Greyson is an entity, relationships radiate FROM Brian
4. Verify: source is "session" (not "manual")
5. Run existing tests: `cd ~/mollygraph/service && python -m pytest tests/ -x`

## Constraints
- Do NOT break existing /ingest calls without speaker field
- All changes must be backwards compatible
- Do NOT modify the GLiNER model itself
- Keep the extraction/pipeline.py changes minimal and surgical
- Commit with descriptive message when done
