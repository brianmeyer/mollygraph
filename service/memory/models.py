"""
MollyGraph V2 Data Models
Pydantic models for bi-temporal graph memory.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
import uuid

class Episode(BaseModel):
    """
    A source episode (message, voice transcript, etc.) that generated facts.
    Provenance tracking - every fact traces back to an Episode.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str = "manual"
    source_id: Optional[str] = None  # External reference (message_id, etc.)
    
    # Content
    content_preview: str = Field(..., max_length=500)  # First 500 chars
    content_hash: str  # SHA256 for deduplication
    
    # Bi-temporal
    occurred_at: datetime  # When the event happened (valid time)
    ingested_at: datetime = Field(default_factory=datetime.utcnow)  # When we learned it
    
    # Processing metadata
    processed_at: Optional[datetime] = None
    processor_version: str = "v2.0"
    entities_extracted: List[str] = []  # List of entity names extracted
    
    # Status
    status: Literal["active", "archived", "quarantined"] = "active"
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class Entity(BaseModel):
    """
    A node in the knowledge graph representing a person, organization, etc.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str  # Canonical name
    entity_type: Literal[
        "Person", "Organization", "Technology", "Place", 
        "Project", "Concept", "Event"
    ]
    
    # Identity
    aliases: List[str] = []  # Alternative names
    description: Optional[str] = None
    
    # Temporal tracking
    first_mentioned: datetime
    last_mentioned: datetime
    
    # Quality metrics
    confidence: float = Field(ge=0.0, le=1.0)  # Highest extraction confidence
    mention_count: int = 1  # Number of episodes mentioning this entity
    strength: float = 1.0  # Decayed importance score
    
    # Audit
    created_from_episode: str  # Episode ID where first seen
    verified: bool = False
    zvec_id: Optional[str] = None  # Reference to vector DB
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class Relationship(BaseModel):
    """
    A bi-temporal edge between entities with provenance tracking.
    """
    # Identifiers
    source_entity: str  # Source entity name/ID
    target_entity: str  # Target entity name/ID
    relation_type: str
    
    # Bi-temporal tracking
    valid_at: Optional[datetime] = None  # When fact is true (valid time)
    valid_until: Optional[datetime] = None  # When fact expires (for job changes)
    observed_at: datetime = Field(default_factory=datetime.utcnow)  # When we learned it
    
    # Evidence
    confidence: float = Field(ge=0.0, le=1.0)
    strength: float = 1.0  # Decayed based on mentions
    mention_count: int = 1
    context_snippets: List[str] = Field(default=[], max_length=3)  # Evidence quotes
    
    # Source tracking
    episode_ids: List[str] = []  # All episodes supporting this relationship
    
    # Audit status
    audit_status: Literal["unverified", "verified", "quarantined", "contradicted"] = "unverified"
    verified_by: Optional[str] = None  # Model or user ID
    verified_at: Optional[datetime] = None
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ExtractionJob(BaseModel):
    """
    A job in the async extraction queue.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    source: str
    priority: int = 1  # 0=realtime, 1=normal, 2=batch
    
    # Context
    reference_time: datetime  # For temporal resolution
    episode_id: Optional[str] = None
    
    # Status
    status: Literal["pending", "processing", "completed", "failed"] = "pending"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    
    # Results
    extracted_entities: List[Entity] = []
    extracted_relationships: List[Relationship] = []


class RetrievalResult(BaseModel):
    """
    Result from a graph memory query.
    """
    entities: List[Entity]
    relationships: List[Relationship]
    episodes: List[Episode]
    
    # Ranking info
    scores: Dict[str, float]  # Entity/rel ID -> final score
    query_time_ms: float
    
    # Context for LLM
    context_string: str  # Formatted for injection


class AuditReport(BaseModel):
    """
    Result of a nightly audit run.
    """
    timestamp: datetime
    episodes_audited: int
    entities_checked: int
    relationships_checked: int
    
    # Issues found
    contradictions: List[Dict[str, Any]]
    low_confidence_extractions: List[Dict[str, Any]]
    orphans: List[str]  # Entity IDs with no relationships
    zombies: List[str]  # Entity IDs stale >30 days
    
    # Training data generated
    training_examples_added: int
    suggested_entity_types: List[str]
    suggested_rel_types: List[str]
    
    # Overall health
    health_score: float  # 0-1
