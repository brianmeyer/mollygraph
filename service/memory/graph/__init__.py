"""Unified graph module exposing both Neo4j and Ladybug backends."""
from __future__ import annotations

import hashlib
import logging
import math
import re
import uuid
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from neo4j import GraphDatabase

from memory.models import Entity, Episode, Relationship

from .constants import (
    DECAY_HALFLIVES,
    ENTITY_BLOCKLIST,
    RELATION_TIERS,
    VALID_REL_TYPES,
    _REL_TYPE_RE,
    _SAFE_PROPERTY_KEY,
    _tier_for_rel_type,
    _to_iso_datetime,
    build_relationship_half_life_case,
    calculate_strength,
    log,
    recency_score,
)
from .core import BiTemporalGraph
from .ladybug import LadybugGraph

GraphBackend = BiTemporalGraph | LadybugGraph

# Preserve the legacy module path for reflection/pickling compatibility.
BiTemporalGraph.__module__ = __name__
LadybugGraph.__module__ = __name__
