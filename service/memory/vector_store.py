"""
Vector Store - Pluggable implementation
Supports: sqlite-vec, Zvec
"""
import os
import math
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from pathlib import Path
import logging

import config as service_config

log = logging.getLogger(__name__)

# Try to import Zvec
try:
    import zvec
    HAVE_ZVEC = True
    log.info("Zvec available")
except ImportError:
    HAVE_ZVEC = False
    zvec = None
    log.info("Zvec not available")


class VectorStoreBackend(ABC):
    """Abstract base for vector storage backends."""
    
    @abstractmethod
    def add_entity(self, entity_id: str, name: str, entity_type: str,
                   dense_embedding: List[float], content: str, confidence: float = 1.0):
        pass
    
    @abstractmethod
    def similarity_search(self, query_embedding: List[float], top_k: int = 10,
                          entity_type: Optional[str] = None) -> List[Dict]:
        pass
    
    @abstractmethod
    def keyword_search(self, query: str, top_k: int = 10) -> List[Dict]:
        pass
    
    @abstractmethod
    def hybrid_search(self, query_embedding: List[float], query_text: str,
                      top_k: int = 10, dense_weight: float = 0.7) -> List[Dict]:
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict:
        pass


class SqliteVecBackend(VectorStoreBackend):
    """sqlite-vec implementation."""
    
    def __init__(self, db_path: str | Path | None = None):
        import sqlite3
        import sqlite_vec
        import numpy as np
        
        self.db_path = (
            Path(db_path).expanduser()
            if db_path is not None
            else service_config.SQLITE_VEC_DB_PATH
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.np = np
        
        self.db = sqlite3.connect(str(self.db_path))
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        
        self._init_tables()
    
    def _init_tables(self):
        self.db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS dense_vectors USING vec0(
                entity_id TEXT PRIMARY KEY,
                embedding FLOAT[768]
            )
        """)
        self.db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS sparse_vectors USING fts5(
                entity_id, content, tokenize='porter'
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS entity_meta (
                entity_id TEXT PRIMARY KEY,
                name TEXT, entity_type TEXT, confidence REAL, last_updated TEXT
            )
        """)
        self.db.commit()
    
    def add_entity(self, entity_id: str, name: str, entity_type: str,
                   dense_embedding: List[float], content: str, confidence: float = 1.0):
        from datetime import datetime
        
        self.db.execute(
            "INSERT OR REPLACE INTO dense_vectors (entity_id, embedding) VALUES (?, ?)",
            (entity_id, self.np.array(dense_embedding, dtype=self.np.float32))
        )
        self.db.execute(
            "INSERT OR REPLACE INTO sparse_vectors (entity_id, content) VALUES (?, ?)",
            (entity_id, content)
        )
        self.db.execute(
            """INSERT OR REPLACE INTO entity_meta 
               (entity_id, name, entity_type, confidence, last_updated)
               VALUES (?, ?, ?, ?, ?)""",
            (entity_id, name, entity_type, confidence, datetime.utcnow().isoformat())
        )
        self.db.commit()
    
    def similarity_search(self, query_embedding: List[float], top_k: int = 10,
                          entity_type: Optional[str] = None) -> List[Dict]:
        query_vec = self.np.array(query_embedding, dtype=self.np.float32)
        
        # sqlite-vec requires k = ? constraint for KNN
        if entity_type:
            sql = """
                SELECT v.entity_id, m.name, m.entity_type, v.distance
                FROM dense_vectors v
                JOIN entity_meta m ON v.entity_id = m.entity_id
                WHERE v.embedding MATCH ? AND k = ? AND m.entity_type = ?
            """
            cursor = self.db.execute(sql, (query_vec, top_k, entity_type))
        else:
            sql = """
                SELECT v.entity_id, m.name, m.entity_type, v.distance
                FROM dense_vectors v
                JOIN entity_meta m ON v.entity_id = m.entity_id
                WHERE v.embedding MATCH ? AND k = ?
            """
            cursor = self.db.execute(sql, (query_vec, top_k))
        
        # Convert distance to similarity score (0-1)
        results = []
        for r in cursor.fetchall():
            distance = r[3]
            score = math.exp(-max(0, distance))
            results.append({"entity_id": r[0], "name": r[1], "entity_type": r[2], "score": score})
        return results
    
    def keyword_search(self, query: str, top_k: int = 10) -> List[Dict]:
        sql = """
            SELECT s.entity_id, m.name, m.entity_type, rank
            FROM sparse_vectors s
            JOIN entity_meta m ON s.entity_id = m.entity_id
            WHERE sparse_vectors MATCH ? ORDER BY rank LIMIT ?
        """
        cursor = self.db.execute(sql, (query, top_k))
        return [{"entity_id": r[0], "name": r[1], "entity_type": r[2],
                 "score": 1.0 / (1.0 + abs(r[3]))} for r in cursor.fetchall()]
    
    def hybrid_search(self, query_embedding: List[float], query_text: str,
                      top_k: int = 10, dense_weight: float = 0.7) -> List[Dict]:
        dense_results = self.similarity_search(query_embedding, top_k=top_k*2)
        sparse_results = self.keyword_search(query_text, top_k=top_k*2)
        
        combined = {}
        for r in dense_results:
            combined[r["entity_id"]] = {**r, "dense_score": r["score"], "sparse_score": 0.0}
        for r in sparse_results:
            if r["entity_id"] in combined:
                combined[r["entity_id"]]["sparse_score"] = r["score"]
            else:
                combined[r["entity_id"]] = {**r, "dense_score": 0.0, "sparse_score": r["score"]}
        
        for data in combined.values():
            data["score"] = dense_weight * data["dense_score"] + (1 - dense_weight) * data["sparse_score"]
        
        return sorted(combined.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    
    def get_stats(self) -> Dict:
        cursor = self.db.execute("SELECT COUNT(*) FROM dense_vectors")
        dense_count = cursor.fetchone()[0]
        cursor = self.db.execute("SELECT COUNT(*) FROM sparse_vectors")
        sparse_count = cursor.fetchone()[0]
        return {"dense_vectors": dense_count, "sparse_vectors": sparse_count,
                "db_size_mb": self.db_path.stat().st_size / (1024 * 1024)}


class ZvecBackend(VectorStoreBackend):
    """Zvec implementation using proper API."""
    
    def __init__(self, db_path: str | Path | None = None):
        import numpy as np
        
        self.db_path = (
            Path(db_path).expanduser()
            if db_path is not None
            else service_config.ZVEC_COLLECTION_DIR
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.np = np
        
        # Create or open collection
        self._init_collection()
    
    def _init_collection(self):
        """Initialize Zvec collection with proper schema."""
        collection_exists = self.db_path.exists()
        
        if collection_exists:
            # Clear stale lock file to prevent "Can't open lock file" on restart
            lock_file = self.db_path / "LOCK"
            if lock_file.exists():
                try:
                    lock_file.unlink()
                    log.info("Cleared stale Zvec LOCK file")
                except OSError:
                    pass
            # Open existing
            self.collection = zvec.open(str(self.db_path))
            log.info(f"Opened existing Zvec collection: {self.db_path}")
        else:
            # Create new collection with schema
            # Vector schema with HNSW index for cosine similarity
            from zvec import HnswIndexParam, MetricType
            hnsw_param = HnswIndexParam(
                metric_type=MetricType.COSINE,
                m=16,
                ef_construction=200
            )
            
            schema = zvec.CollectionSchema(
                name="graph_entities",
                fields=[
                    zvec.FieldSchema("entity_id", zvec.DataType.STRING, nullable=False),
                    zvec.FieldSchema("name", zvec.DataType.STRING, nullable=False),
                    zvec.FieldSchema("entity_type", zvec.DataType.STRING, nullable=False),
                    zvec.FieldSchema("content", zvec.DataType.STRING, nullable=True),
                    zvec.FieldSchema("confidence", zvec.DataType.FLOAT, nullable=True),
                ],
                vectors=zvec.VectorSchema(
                    "embedding", 
                    zvec.DataType.VECTOR_FP32, 
                    dimension=768, 
                    index_param=hnsw_param
                )
            )
            
            option = zvec.CollectionOption(
                enable_mmap=True,
                read_only=False
            )
            
            self.collection = zvec.create_and_open(
                str(self.db_path),
                schema,
                option
            )
            log.info(f"Created new Zvec collection: {self.db_path}")
    
    def add_entity(self, entity_id: str, name: str, entity_type: str,
                   dense_embedding: List[float], content: str, confidence: float = 1.0):
        """Insert entity into Zvec collection."""
        doc = zvec.Doc(
            id=entity_id,
            vectors={"embedding": dense_embedding},
            fields={
                "entity_id": entity_id,
                "name": name,
                "entity_type": entity_type,
                "content": content,
                "confidence": confidence
            }
        )
        self.collection.upsert(doc)
    
    def similarity_search(self, query_embedding: List[float], top_k: int = 10,
                          entity_type: Optional[str] = None) -> List[Dict]:
        """Search similar vectors using cosine similarity."""
        # Build filter if entity_type specified
        filter_expr = None
        if entity_type:
            filter_expr = f"entity_type == '{entity_type}'"
        
        # Query
        results = self.collection.query(
            vectors=zvec.VectorQuery(
                field_name="embedding",
                vector=query_embedding
            ),
            topk=top_k,
            filter=filter_expr,
            output_fields=["entity_id", "name", "entity_type"]
        )
        
        # Parse results - results is a list of Doc objects
        output = []
        for doc in results:
            output.append({
                "entity_id": doc.id,
                "name": doc.fields.get("name", ""),
                "entity_type": doc.fields.get("entity_type", ""),
                "score": doc.score if doc.score is not None else 0.0
            })
        return output
    
    def keyword_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Zvec doesn't support BM25 directly, use sparse embedding fallback."""
        # For now, return empty - would need sparse embedding index
        log.warning("Keyword search not implemented in Zvec backend")
        return []
    
    def hybrid_search(self, query_embedding: List[float], query_text: str,
                      top_k: int = 10, dense_weight: float = 0.7) -> List[Dict]:
        """Hybrid search - currently just dense search."""
        # Zvec doesn't have built-in hybrid search, use dense only
        return self.similarity_search(query_embedding, top_k=top_k)
    
    def get_stats(self) -> Dict:
        """Get collection stats."""
        stats = self.collection.stats
        entities = getattr(stats, "num_entities", None)
        if entities is None:
            entities = getattr(stats, "doc_count", 0)
        return {
            "entities": int(entities or 0),
            "backend": "zvec"
        }


class VectorStore:
    """Pluggable vector store - auto-selects best backend."""
    
    def __init__(self, backend: Optional[str] = None, **kwargs):
        """
        Initialize vector store.
        
        Args:
            backend: 'auto', 'zvec', or 'sqlite-vec'. Auto picks best available.
        """
        if backend is None or backend == "auto":
            backend = "zvec" if HAVE_ZVEC else "sqlite-vec"
        
        if backend == "zvec":
            if not HAVE_ZVEC:
                raise ImportError("Zvec not available. Install: pip install zvec")
            self.backend = ZvecBackend(**kwargs)
        elif backend == "sqlite-vec":
            self.backend = SqliteVecBackend(**kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        log.info(f"VectorStore using {self.backend.__class__.__name__}")
    
    def add_entity(self, entity_id: str, name: str, entity_type: str,
                   dense_embedding: List[float], content: str, confidence: float = 1.0):
        return self.backend.add_entity(entity_id, name, entity_type, 
                                       dense_embedding, content, confidence)
    
    def similarity_search(self, query_embedding: List[float], top_k: int = 10,
                          entity_type: Optional[str] = None) -> List[Dict]:
        return self.backend.similarity_search(query_embedding, top_k, entity_type)
    
    def keyword_search(self, query: str, top_k: int = 10) -> List[Dict]:
        return self.backend.keyword_search(query, top_k)
    
    def hybrid_search(self, query_embedding: List[float], query_text: str,
                      top_k: int = 10, dense_weight: float = 0.7) -> List[Dict]:
        return self.backend.hybrid_search(query_embedding, query_text, top_k, dense_weight)
    
    def get_stats(self) -> Dict:
        return self.backend.get_stats()
