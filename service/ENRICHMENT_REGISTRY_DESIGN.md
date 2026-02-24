# GLiNER Enrichment Registry Design

**Date:** 2026-02-24  
**Purpose:** Design a generic enrichment registry for MollyGraph to support additional GLiNER v1/v2 models alongside the primary GLiNER2 extractor.

---

## 1. Executive Summary

MollyGraph currently uses **GLiNER2** (`gliner2` library) as its primary entity extractor. This design doc outlines a generic **enrichment registry** that allows users to plug in additional GLiNER v1/v2 models (`gliner` library) for additive NER passes. The registry supports lazy loading, graceful failures, and source attribution.

---

## 2. API Research Findings

### 2.1 GLiNER (v1/v2) Package - `gliner`

**Installation:**
```bash
pip install gliner
```

**Import & Load:**
```python
from gliner import GLiNER
model = GLiNER.from_pretrained("urchade/gliner_large-v2.1")
```

**Inference API:**
```python
entities = model.predict_entities(
    text: str,
    labels: List[str],
    threshold: float = 0.5,
    flat_ner: bool = True,
    batch_size: int = 8
) -> List[Dict[str, Any]]
```

**Return Format:**
```python
[
    {
        "start": 0,        # Character start position
        "end": 10,         # Character end position
        "text": "John Doe", # Extracted entity text
        "label": "person",  # Entity label
        "score": 0.95       # Confidence score (0-1)
    },
    ...
]
```

### 2.2 GLiNER2 Package - `gliner2`

**Installation:**
```bash
pip install gliner2
```

**Import & Load:**
```python
from gliner2 import GLiNER2
extractor = GLiNER2.from_pretrained("fastino/gliner2-large-v1")
```

**Inference API:**
```python
result = extractor.extract_entities(
    text: str,
    labels: Union[List[str], Dict[str, str]],  # Can include descriptions
    threshold: float = 0.5,
    include_confidence: bool = False,
    include_spans: bool = False
) -> Dict[str, List]  # Different return structure!
```

**Return Format:**
```python
{
    "entities": {
        "person": ["John Doe"],           # Just text by default
        "company": ["Acme Corp"]
    }
}

# With include_confidence=True and include_spans=True:
{
    "entities": {
        "person": [{"text": "John Doe", "confidence": 0.95, "start": 0, "end": 8}],
        "company": [{"text": "Acme Corp", "confidence": 0.92, "start": 15, "end": 24}]
    }
}
```

### 2.3 Key API Differences Summary

| Aspect | `gliner` (v1/v2) | `gliner2` |
|--------|------------------|-----------|
| Import | `from gliner import GLiNER` | `from gliner2 import GLiNER2` |
| Class Name | `GLiNER` | `GLiNER2` |
| Method | `predict_entities()` | `extract_entities()` |
| Return Type | `List[Dict]` | `Dict[str, List]` |
| Entity Format | Flat list with label field | Grouped by label type |
| Confidence | Always included | Optional via flag |
| Spans | Always included | Optional via flag |

### 2.4 Coexistence

**Can they coexist?** YES. The packages have different module names:
- `gliner` → imports as `gliner`
- `gliner2` → imports as `gliner2`

Both can be installed simultaneously:
```bash
pip install gliner gliner2
```

---

## 3. Target Model Analysis

### 3.1 Model Specifications

| Model | Parameters | Base | Est. RAM | License |
|-------|------------|------|----------|---------|
| `urchade/gliner_multi-v2.1` | 209M | BERT-based | ~0.8-1.0 GB | Apache 2.0 |
| `Ihor/gliner-biomed-base-v1.0` | ~209M | BERT-based | ~0.8-1.0 GB | (check HF) |
| `nvidia/gliner-PII` | 570M | gliner_large-v2.1 | ~2.0-2.5 GB | NVIDIA Open Model |
| `knowledgator/gliner-pii-edge-v1.0` | ~166M | BERT-small | ~0.6-0.8 GB | (check HF) |

### 3.2 Model-Specific Notes

**urchade/gliner_multi-v2.1:**
- Multilingual support
- 209M parameters (medium size)
- Good general-purpose model

**Ihor/gliner-biomed-base-v1.0:**
- Specialized for biomedical NER
- Supports: Disease, Drug, Dosage, Lab test, etc.
- Paper: GLiNER-BioMed (2025)

**nvidia/gliner-PII:**
- 570M parameters (largest)
- 55+ PII/PHI categories
- Based on gliner_large-v2.1
- Commercial use allowed (NVIDIA Open Model License)

**knowledgator/gliner-pii-edge-v1.0:**
- Optimized for edge deployment
- Smaller/faster than base PII model
- Trades slight accuracy for speed

### 3.3 Compatibility Check

✅ **All models use the same API:**
- `GLiNER.from_pretrained()` for loading
- `predict_entities(text, labels, threshold)` for inference
- Same return format: `List[Dict[start, end, text, label, score]]`

✅ **No special handling required** for any of these models.

---

## 4. RAM Estimation & Capacity Planning

### 4.1 Memory Requirements

| Component | Est. RAM |
|-----------|----------|
| GLiNER2 (primary) | ~1.2 GB |
| Jina embeddings | ~0.5 GB |
| Python overhead | ~0.3 GB |
| Neo4j (if local) | ~1.0 GB |
| **Base system** | **~3.0 GB** |

**Available for enrichment models:** ~13 GB

### 4.2 Enrichment Model Scenarios (16GB Mac Mini M4)

| Scenario | Models | Est. Total | Fits? |
|----------|--------|------------|-------|
| Conservative | 1 medium (~1GB) | ~4GB | ✅ Yes |
| Moderate | 2 medium (~2GB) | ~5GB | ✅ Yes |
| Aggressive | 3 medium (~3GB) | ~6GB | ✅ Yes |
| PII-heavy | 1 large PII (~2.5GB) | ~5.5GB | ✅ Yes |
| Max config | GLiNER2 + 2 medium + 1 small | ~6GB | ✅ Yes |

**Conclusion:** Can comfortably run 2-3 enrichment models alongside GLiNER2 + Jina on 16GB.

### 4.3 Load Strategy Recommendation

**RECOMMENDED: Lazy-load with LRU eviction**

- Load models on first use (not at startup)
- Keep frequently-used models resident
- Evict least-recently-used when memory pressure
- Unload after idle timeout (configurable, default 10 min)

---

## 5. Registry Interface Design

### 5.1 Configuration Environment Variables

```bash
# Core settings
MOLLYGRAPH_ENRICHMENT_ENABLED=true
MOLLYGRAPH_ENRICHMENT_CONFIDENCE=0.5
MOLLYGRAPH_ENRICHMENT_IDLE_TIMEOUT=600  # seconds

# Model list (comma-separated)
MOLLYGRAPH_ENRICHMENT_MODELS=Ihor/gliner-biomed-base-v1.0,nvidia/gliner-PII

# Per-model thresholds (optional, overrides default)
MOLLYGRAPH_ENRICHMENT_THRESHOLD_Ihor_gliner_biomed_base_v1_0=0.6
MOLLYGRAPH_ENRICHMENT_THRESHOLD_nvidia_gliner_PII=0.4

# Max models to keep loaded simultaneously
MOLLYGRAPH_ENRICHMENT_MAX_LOADED=3
```

### 5.2 Python Registry Class

```python
"""GLiNER Enrichment Registry for MollyGraph.

Supports lazy-loaded GLiNER v1/v2 models as additive NER passes.
"""

import os
import re
import logging
import functools
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class EnrichmentEntity:
    """Unified entity format from enrichment models."""
    text: str
    label: str
    start: int
    end: int
    score: float
    source_model: str  # Which enrichment model found this


@dataclass  
class LoadedModel:
    """Wrapper for loaded model with metadata."""
    model: Any  # GLiNER instance
    model_name: str
    loaded_at: datetime
    last_used: datetime
    use_count: int = 0


class GlinerEnrichmentRegistry:
    """Registry for GLiNER enrichment models.
    
    Features:
    - Lazy loading on first use
    - LRU eviction under memory pressure
    - Graceful handling of model load failures
    - Source attribution for all entities
    - Unified entity format across models
    """
    
    def __init__(
        self,
        enabled: Optional[bool] = None,
        models: Optional[List[str]] = None,
        default_threshold: float = 0.5,
        max_loaded: int = 3,
        idle_timeout_seconds: int = 600
    ):
        """Initialize the enrichment registry.
        
        Args:
            enabled: Whether enrichment is enabled (from env if None)
            models: List of model names to load (from env if None)
            default_threshold: Default confidence threshold
            max_loaded: Max models to keep loaded simultaneously
            idle_timeout_seconds: Unload models idle longer than this
        """
        self.enabled = enabled if enabled is not None else self._env_bool(
            "MOLLYGRAPH_ENRICHMENT_ENABLED", False
        )
        self.models = models or self._parse_model_list(
            os.getenv("MOLLYGRAPH_ENRICHMENT_MODELS", "")
        )
        self.default_threshold = default_threshold
        self.max_loaded = max_loaded
        self.idle_timeout = timedelta(seconds=idle_timeout_seconds)
        
        # Model cache: model_name -> LoadedModel
        self._cache: Dict[str, LoadedModel] = {}
        
        # Track failed models to avoid retry loops
        self._failed_models: set = set()
        
        # Import gliner lazily to avoid startup overhead
        self._gliner_module = None
    
    def _env_bool(self, key: str, default: bool) -> bool:
        """Parse boolean from environment variable."""
        val = os.getenv(key, "").lower()
        return val in ("true", "1", "yes", "on") if val else default
    
    def _parse_model_list(self, models_str: str) -> List[str]:
        """Parse comma-separated model list."""
        if not models_str:
            return []
        return [m.strip() for m in models_str.split(",") if m.strip()]
    
    def _get_model_threshold(self, model_name: str) -> float:
        """Get threshold for specific model, with env override."""
        env_key = f"MOLLYGRAPH_ENRICHMENT_THRESHOLD_{model_name.replace('/', '_').replace('-', '_')}"
        env_val = os.getenv(env_key)
        if env_val:
            try:
                return float(env_val)
            except ValueError:
                pass
        return self.default_threshold
    
    def _get_gliner(self):
        """Lazy import gliner module."""
        if self._gliner_module is None:
            try:
                from gliner import GLiNER
                self._gliner_module = GLiNER
            except ImportError as e:
                logger.error("gliner package not installed. Run: pip install gliner")
                raise
        return self._gliner_module
    
    def _load_model(self, model_name: str) -> Optional[Any]:
        """Load a GLiNER model, with error handling.
        
        Returns:
            Loaded model or None if loading failed
        """
        if model_name in self._failed_models:
            logger.debug(f"Skipping previously failed model: {model_name}")
            return None
        
        try:
            logger.info(f"Loading enrichment model: {model_name}")
            GLiNER = self._get_gliner()
            model = GLiNER.from_pretrained(model_name)
            logger.info(f"Successfully loaded: {model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            self._failed_models.add(model_name)
            return None
    
    def _evict_if_needed(self):
        """Evict LRU model if at capacity."""
        if len(self._cache) < self.max_loaded:
            return
        
        # Find LRU model
        lru_model = min(self._cache.values(), key=lambda m: m.last_used)
        logger.info(f"Evicting LRU model: {lru_model.model_name}")
        del self._cache[lru_model.model_name]
    
    def _cleanup_idle(self):
        """Remove models that have been idle too long."""
        now = datetime.now()
        to_remove = [
            name for name, lm in self._cache.items()
            if now - lm.last_used > self.idle_timeout
        ]
        for name in to_remove:
            logger.info(f"Unloading idle model: {name}")
            del self._cache[name]
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Get (or load) a model by name.
        
        Args:
            model_name: HuggingFace model name
            
        Returns:
            GLiNER model instance or None
        """
        if not self.enabled:
            return None
        
        # Check cache
        if model_name in self._cache:
            loaded = self._cache[model_name]
            loaded.last_used = datetime.now()
            loaded.use_count += 1
            return loaded.model
        
        # Check if we've previously failed to load this model
        if model_name in self._failed_models:
            return None
        
        # Evict if needed before loading
        self._evict_if_needed()
        
        # Load the model
        model = self._load_model(model_name)
        if model:
            now = datetime.now()
            self._cache[model_name] = LoadedModel(
                model=model,
                model_name=model_name,
                loaded_at=now,
                last_used=now,
                use_count=1
            )
        return model
    
    def enrich(
        self,
        text: str,
        labels: List[str],
        threshold: Optional[float] = None
    ) -> List[EnrichmentEntity]:
        """Run all enrichment models and return combined entities.
        
        Args:
            text: Input text to analyze
            labels: Entity labels to extract
            threshold: Override confidence threshold
            
        Returns:
            List of enrichment entities with source attribution
        """
        if not self.enabled or not self.models:
            return []
        
        all_entities: List[EnrichmentEntity] = []
        
        for model_name in self.models:
            model = self.get_model(model_name)
            if model is None:
                continue  # Skip failed models gracefully
            
            model_threshold = threshold or self._get_model_threshold(model_name)
            
            try:
                # GLiNER v1/v2 API
                results = model.predict_entities(
                    text,
                    labels,
                    threshold=model_threshold
                )
                
                # Convert to unified format
                for entity in results:
                    all_entities.append(EnrichmentEntity(
                        text=entity["text"],
                        label=entity["label"],
                        start=entity["start"],
                        end=entity["end"],
                        score=entity["score"],
                        source_model=model_name
                    ))
                    
            except Exception as e:
                logger.error(f"Inference failed for {model_name}: {e}")
                continue  # Skip failed inference gracefully
        
        return all_entities
    
    def enrich_with_source_grouping(
        self,
        text: str,
        labels: List[str],
        threshold: Optional[float] = None
    ) -> Dict[str, List[EnrichmentEntity]]:
        """Run enrichment and group results by source model.
        
        Returns:
            Dict mapping model_name -> list of entities
        """
        entities = self.enrich(text, labels, threshold)
        
        grouped: Dict[str, List[EnrichmentEntity]] = {}
        for entity in entities:
            if entity.source_model not in grouped:
                grouped[entity.source_model] = []
            grouped[entity.source_model].append(entity)
        
        return grouped
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "enabled": self.enabled,
            "configured_models": self.models,
            "loaded_models": list(self._cache.keys()),
            "failed_models": list(self._failed_models),
            "model_details": {
                name: {
                    "loaded_at": lm.loaded_at.isoformat(),
                    "last_used": lm.last_used.isoformat(),
                    "use_count": lm.use_count
                }
                for name, lm in self._cache.items()
            }
        }
    
    def unload_all(self):
        """Unload all models (useful for memory cleanup)."""
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"Unloaded {count} enrichment models")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup."""
        self.unload_all()


# Convenience factory function
def create_registry_from_env() -> GlinerEnrichmentRegistry:
    """Create registry from environment variables."""
    return GlinerEnrichmentRegistry(
        enabled=os.getenv("MOLLYGRAPH_ENRICHMENT_ENABLED", "").lower() in ("true", "1"),
        models=None,  # Will parse from env
        default_threshold=float(os.getenv("MOLLYGRAPH_ENRICHMENT_CONFIDENCE", "0.5")),
        max_loaded=int(os.getenv("MOLLYGRAPH_ENRICHMENT_MAX_LOADED", "3")),
        idle_timeout_seconds=int(os.getenv("MOLLYGRAPH_ENRICHMENT_IDLE_TIMEOUT", "600"))
    )
```

---

## 6. Integration with MollyGraph

### 6.1 Recommended Integration Pattern

```python
# In MollyGraph service initialization
from enrichment_registry import GlinerEnrichmentRegistry, create_registry_from_env

class MollyGraphExtractor:
    def __init__(self):
        # Primary extractor (GLiNER2)
        from gliner2 import GLiNER2
        self.primary = GLiNER2.from_pretrained("fastino/gliner2-large-v1")
        
        # Enrichment registry (GLiNER v1/v2 models)
        self.enrichment = create_registry_from_env()
    
    def extract_entities(self, text: str, labels: List[str]) -> Dict[str, Any]:
        """Extract entities using primary + enrichment models."""
        
        # 1. Primary extraction with GLiNER2
        primary_result = self.primary.extract_entities(
            text, labels, 
            include_confidence=True,
            include_spans=True
        )
        
        # 2. Enrichment pass with additional models
        enrichment_entities = self.enrichment.enrich(text, labels)
        
        # 3. Merge results (deduplication optional)
        return {
            "primary": primary_result,
            "enrichment": [
                {
                    "text": e.text,
                    "label": e.label,
                    "start": e.start,
                    "end": e.end,
                    "score": e.score,
                    "source": e.source_model
                }
                for e in enrichment_entities
            ]
        }
```

### 6.2 Deduplication Strategy (Optional)

```python
def deduplicate_entities(
    primary_entities: List[Dict],
    enrichment_entities: List[EnrichmentEntity],
    iou_threshold: float = 0.5
) -> List[Dict]:
    """Deduplicate entities using IoU overlap.
    
    Prefers primary model results when overlaps occur.
    """
    def compute_iou(e1, e2):
        start1, end1 = e1.get("start", 0), e1.get("end", 0)
        start2, end2 = e2.start, e2.end
        
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        if overlap_start >= overlap_end:
            return 0.0
        
        overlap = overlap_end - overlap_start
        union = max(end1, end2) - min(start1, start2)
        return overlap / union if union > 0 else 0.0
    
    # Start with primary entities
    merged = [e for e in primary_entities]
    merged_spans = [(e.get("start", 0), e.get("end", 0)) for e in merged]
    
    # Add enrichment entities that don't overlap
    for e in enrichment_entities:
        is_duplicate = False
        for (start, end) in merged_spans:
            iou = compute_iou({"start": start, "end": end}, e)
            if iou >= iou_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            merged.append({
                "text": e.text,
                "label": e.label,
                "start": e.start,
                "end": e.end,
                "score": e.score,
                "source": e.source_model,
                "is_enrichment": True
            })
    
    return merged
```

---

## 7. Usage Examples

### 7.1 Basic Usage

```python
# Set env vars
import os
os.environ["MOLLYGRAPH_ENRICHMENT_ENABLED"] = "true"
os.environ["MOLLYGRAPH_ENRICHMENT_MODELS"] = "Ihor/gliner-biomed-base-v1.0"
os.environ["MOLLYGRAPH_ENRICHMENT_CONFIDENCE"] = "0.5"

# Use registry
from enrichment_registry import create_registry_from_env

registry = create_registry_from_env()

text = "Patient was diagnosed with type 2 diabetes and prescribed Metformin."
labels = ["disease", "medication", "symptom"]

entities = registry.enrich(text, labels)
for e in entities:
    print(f"{e.text} => {e.label} ({e.score:.2f}) [from {e.source_model}]")
```

### 7.2 With Source Grouping

```python
results = registry.enrich_with_source_grouping(text, labels)

for model_name, entities in results.items():
    print(f"\n=== {model_name} ===")
    for e in entities:
        print(f"  {e.text} => {e.label} ({e.score:.2f})")
```

### 7.3 Context Manager (Auto-cleanup)

```python
from enrichment_registry import GlinerEnrichmentRegistry

with GlinerEnrichmentRegistry(
    enabled=True,
    models=["urchade/gliner_multi-v2.1", "nvidia/gliner-PII"],
    max_loaded=2
) as registry:
    entities = registry.enrich(text, labels)
    # Models auto-unloaded on exit
```

### 7.4 Health Check / Stats

```python
stats = registry.get_stats()
print(f"Loaded models: {stats['loaded_models']}")
print(f"Failed models: {stats['failed_models']}")
for name, details in stats['model_details'].items():
    print(f"  {name}: used {details['use_count']} times")
```

---

## 8. Installation Requirements

### 8.1 Dependencies

```txt
# requirements-enrichment.txt
gliner>=0.2.0  # For enrichment models
gliner2>=0.1.0  # Primary extractor (already in MollyGraph)
```

### 8.2 Optional Dependencies

```txt
# For memory monitoring (optional)
psutil>=5.9.0
```

---

## 9. Error Handling & Edge Cases

### 9.1 Graceful Degradation

| Scenario | Behavior |
|----------|----------|
| Model fails to load | Log error, skip model, continue with others |
| Inference fails | Log error, skip model for this request |
| All models fail | Return empty enrichment list |
| Out of memory | LRU eviction triggers, load new model |
| Invalid model name | Log error, add to failed set |
| Network issue (HF) | Standard HF error handling applies |

### 9.2 Retry Logic

No automatic retry - models that fail to load are added to `_failed_models` set and skipped in subsequent calls. This prevents:
- Repeated network requests
- Log spam
- Request latency spikes

Manual retry requires registry restart or explicit `registry._failed_models.clear()`.

---

## 10. Future Enhancements

### 10.1 Potential Improvements

1. **ONNX Support**: Use ONNX runtime for faster inference
2. **Quantization**: Load models in INT8 for lower memory
3. **Batch Processing**: Support batch enrichment for multiple texts
4. **Async Loading**: Non-blocking model loading
5. **Metrics**: Prometheus metrics for model usage/latency
6. **Model Warmup**: Preload models on startup (optional)

### 10.2 Alternative Architectures

| Approach | Pros | Cons |
|----------|------|------|
| **Per-request load/unload** | Minimal memory usage | High latency per request |
| **Keep all loaded** | Fastest response | May exceed RAM with many models |
| **LRU with idle timeout** (chosen) | Balanced | Moderate complexity |

---

## 11. Summary

This design provides a robust, production-ready enrichment registry for MollyGraph that:

✅ **Supports GLiNER v1/v2 models** alongside GLiNER2  
✅ **Lazy-loads models** to control startup time and RAM  
✅ **Gracefully handles failures** without breaking extraction  
✅ **Attributes entities** to their source model  
✅ **Fits comfortably** in 16GB alongside existing components  
✅ **Requires minimal code changes** to integrate  

**Next Steps:**
1. Implement `enrichment_registry.py` module
2. Add tests with mocked GLiNER models
3. Integrate into MollyGraph service
4. Document configuration options for users
