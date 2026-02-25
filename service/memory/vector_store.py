"""
Vector Store - Pluggable implementation
Supports: sqlite-vec, Zvec
"""
import os
import math
import threading
import time
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
    # Initialize Zvec's C++ runtime once at import time.
    # log_level=WARN means the C++ logger emits WARN, ERROR, and FATAL messages.
    # Notably, the benign "[ERROR] segment.cc:711 vector indexer not found for
    # field embedding" will still appear at ERROR level — this is Zvec's own
    # logger writing directly to stdout/stderr and cannot be suppressed from
    # Python.  The root cause (un-indexed WAL segments) is fixed at startup by
    # calling collection.optimize() in _init_collection().  See ZvecBackend
    # class docstring for the full explanation.
    try:
        from zvec.typing.enum import LogLevel as _ZvecLogLevel
        zvec.init(log_level=_ZvecLogLevel.WARN)
        log.info("Zvec runtime initialized (log_level=WARN)")
    except RuntimeError:
        # init() raises RuntimeError if called more than once (e.g. test reload)
        log.debug("Zvec already initialized — skipping zvec.init()")
    except Exception as _e:
        log.warning("zvec.init() failed (non-fatal): %s", _e)
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
    def remove_entity(self, entity_id: str) -> bool:
        """Delete a vector by entity_id. Returns True if deleted, False if not found."""
        pass

    @abstractmethod
    def update_entity(self, entity_id: str, name: str, entity_type: str,
                      dense_embedding: List[float], content: str,
                      confidence: float = 1.0) -> None:
        """Upsert/replace a vector for an existing entity."""
        pass

    @abstractmethod
    def list_all_entity_ids(self) -> Optional[List[str]]:
        """Return all stored entity_ids, or None if not supported by this backend."""
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
        self._lock = threading.Lock()
        
        self.db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        
        self._init_tables()
    
    def _init_tables(self):
        with self._lock:
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
        from datetime import datetime, UTC
        
        with self._lock:
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
                (entity_id, name, entity_type, confidence, datetime.now(UTC).isoformat())
            )
            self.db.commit()
    
    def similarity_search(self, query_embedding: List[float], top_k: int = 10,
                          entity_type: Optional[str] = None) -> List[Dict]:
        query_vec = self.np.array(query_embedding, dtype=self.np.float32)
        
        # sqlite-vec requires k = ? constraint for KNN
        with self._lock:
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
        with self._lock:
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
    
    def remove_entity(self, entity_id: str) -> bool:
        """Delete vectors and metadata for entity_id. Returns True if a row was deleted."""
        with self._lock:
            cursor = self.db.execute(
                "DELETE FROM entity_meta WHERE entity_id = ?", (entity_id,)
            )
            self.db.execute("DELETE FROM dense_vectors WHERE entity_id = ?", (entity_id,))
            self.db.execute("DELETE FROM sparse_vectors WHERE entity_id = ?", (entity_id,))
            self.db.commit()
            return cursor.rowcount > 0

    def update_entity(self, entity_id: str, name: str, entity_type: str,
                      dense_embedding: List[float], content: str,
                      confidence: float = 1.0) -> None:
        """Upsert/replace via add_entity (which uses INSERT OR REPLACE)."""
        self.add_entity(entity_id, name, entity_type, dense_embedding, content, confidence)

    def list_all_entity_ids(self) -> List[str]:
        """Return every entity_id currently stored in the metadata table."""
        with self._lock:
            cursor = self.db.execute("SELECT entity_id FROM entity_meta")
            return [row[0] for row in cursor.fetchall()]

    def get_stats(self) -> Dict:
        with self._lock:
            cursor = self.db.execute("SELECT COUNT(*) FROM dense_vectors")
            dense_count = cursor.fetchone()[0]
            cursor = self.db.execute("SELECT COUNT(*) FROM sparse_vectors")
            sparse_count = cursor.fetchone()[0]
        return {"dense_vectors": dense_count, "sparse_vectors": sparse_count,
                "db_size_mb": self.db_path.stat().st_size / (1024 * 1024)}


class ZvecBackend(VectorStoreBackend):
    """Zvec-backed vector store for MollyGraph graph entities.

    ## Segment health and the ``segment.cc:711`` error
    -------------------------------------------------------
    Zvec uses a log-structured segment model backed by RocksDB.  Each ``upsert``
    call appends to a WAL (write-ahead log) segment.  A background thread
    (governed by ``optimize_threads``) later merges small segments and builds
    HNSW indexes on each merged segment.

    Until a segment is merged + indexed it exists as a *raw* (un-indexed)
    segment.  When a query touches such a segment Zvec's C++ core logs:

        [ERROR] segment.cc:711 vector indexer not found for field embedding

    at ``ERROR`` level to stdout/stderr via its **own** logger (not Python's
    logging system), so it cannot be silenced from Python.

    **This is expected and benign.**  Zvec automatically falls back to brute-
    force cosine scan for un-indexed segments, so search results are still
    correct — just slightly slower.  The message disappears after the background
    optimizer has caught up (typically within seconds of a flush).

    To accelerate indexing call ``collection.optimize(OptimizeOption())``
    which synchronously merges all segments and builds HNSW indexes.  This is
    exposed via the ``/admin/zvec/optimize`` HTTP endpoint and is called
    automatically at startup when ``index_completeness["embedding"] < 1.0``.

    ``CollectionStats.index_completeness`` (``dict[str, float]``) maps each
    vector field to the fraction of segments that are fully indexed (0.0–1.0).
    A value of ``1.0`` means all segments are indexed and the error will not
    appear.  This metric is surfaced in ``/health`` as
    ``vector_index_completeness``.

    ## Degraded mode
    -----------------
    If the collection cannot be opened (e.g. another process holds the RocksDB
    LOCK), ``self.collection`` is set to ``None`` and all operations become
    no-ops / return empty results.  ``VectorStore.is_degraded()`` reflects this.
    """

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
            # RocksDB LOCK files are advisory locks. After unclean shutdown they
            # may be stale. The fix is NOT to delete them — Zvec/RocksDB needs
            # them to exist (empty). We truncate them to release any stale flock,
            # then ensure they exist as empty files so zvec.open() succeeds.
            for lock_file in self.db_path.rglob("LOCK"):
                try:
                    # Truncate to release stale advisory lock, keep file present
                    lock_file.write_bytes(b"")
                    log.info("Reset stale Zvec LOCK file: %s", lock_file)
                except OSError:
                    pass

            # Also ensure LOCK files exist at known paths (in case they were
            # deleted by a prior version of this code)
            for lock_path in [self.db_path / "LOCK", self.db_path / "idmap.0" / "LOCK"]:
                if not lock_path.exists():
                    try:
                        lock_path.touch()
                        log.info("Recreated missing Zvec LOCK file: %s", lock_path)
                    except OSError:
                        pass

            try:
                self.collection = zvec.open(str(self.db_path))
                log.info("Opened existing Zvec collection: %s", self.db_path)
                # Trigger optimize() if any vector field is not fully indexed.
                # This resolves stale WAL segments that cause:
                #   [ERROR] segment.cc:711 vector indexer not found for field embedding
                # The error is benign (searches fall back to brute-force) but
                # calling optimize() here ensures the HNSW index is built
                # synchronously so the error stops appearing in normal operation.
                try:
                    stats = self.collection.stats
                    completeness: dict = getattr(stats, "index_completeness", {}) or {}
                    embedding_completeness = float(completeness.get("embedding", 1.0))
                    if embedding_completeness < 1.0:
                        log.warning(
                            "Zvec embedding index completeness=%.2f — stale segments detected "
                            "(this causes benign [ERROR] segment.cc:711 messages). "
                            "Running optimize() to rebuild indexes synchronously...",
                            embedding_completeness,
                        )
                        self.collection.optimize(zvec.OptimizeOption())
                        log.info("Zvec optimize() complete — stale segment indexes rebuilt")
                    else:
                        log.info(
                            "Zvec index completeness OK (embedding=%.2f) — "
                            "no stale segments to rebuild",
                            embedding_completeness,
                        )
                except Exception as opt_exc:
                    # Non-fatal: optimize failure does not block serving
                    log.warning(
                        "Zvec startup optimize() failed (non-fatal): %s. "
                        "The [ERROR] segment.cc:711 messages may continue to appear "
                        "but search results will still be correct.",
                        opt_exc,
                    )
            except RuntimeError as exc:
                # Don't destroy the data — run degraded instead
                log.error(
                    "Failed to open Zvec collection (%s). "
                    "NOT wiping data. Vector search will be degraded until "
                    "next successful restart.",
                    exc,
                )
                self.collection = None
                return
        if not collection_exists:
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
        if self.collection is None:
            log.error("Zvec degraded — add_entity(%s) is a no-op; vector store is unavailable", entity_id)
            return
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
        self._upsert_count = getattr(self, '_upsert_count', 0) + 1
        # Flush WAL periodically to make data visible to queries and stats
        if self._upsert_count % 100 == 0:
            try:
                self.collection.flush()
            except Exception:
                pass

    def flush(self):
        """Explicitly flush WAL to disk. Call after bulk operations."""
        if self.collection is not None:
            try:
                self.collection.flush()
            except Exception:
                pass

    def similarity_search(self, query_embedding: List[float], top_k: int = 10,
                          entity_type: Optional[str] = None) -> List[Dict]:
        """Search similar vectors using cosine similarity."""
        if self.collection is None:
            log.error("Zvec degraded — similarity_search returning empty; vector store is unavailable")
            return []
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
    
    def remove_entity(self, entity_id: str) -> bool:
        """Delete a vector by entity_id using zvec collection.delete()."""
        if self.collection is None:
            return False
        try:
            result = self.collection.delete(ids=entity_id)
            # result is a Status-like object; code=0 means success
            if isinstance(result, dict):
                return result.get("code", -1) == 0
            code = getattr(result, "code", None)
            return code == 0 if code is not None else True
        except Exception:
            log.debug("ZvecBackend remove_entity failed for %s", entity_id, exc_info=True)
            return False

    def update_entity(self, entity_id: str, name: str, entity_type: str,
                      dense_embedding: List[float], content: str,
                      confidence: float = 1.0) -> None:
        """Upsert/replace via add_entity (which already calls collection.upsert)."""
        self.add_entity(entity_id, name, entity_type, dense_embedding, content, confidence)

    def list_all_entity_ids(self) -> Optional[List[str]]:
        """Zvec has no native scan/list API — returns None to signal unsupported."""
        log.debug(
            "ZvecBackend.list_all_entity_ids(): native listing not supported; "
            "returning None — reconciliation will be partial."
        )
        return None

    def optimize(self, concurrency: int = 0) -> dict:
        """Trigger Zvec segment compaction and HNSW index rebuild.

        Merges all WAL/small segments and ensures every segment has a fully
        built vector index.  After this call ``index_completeness["embedding"]``
        should be 1.0 and the ``[ERROR] segment.cc:711`` messages will stop.

        Args:
            concurrency: number of threads to use (0 = auto-detect).

        Returns:
            dict with ``status`` and ``index_completeness_after``.
        """
        if self.collection is None:
            return {"status": "degraded", "message": "collection not open"}
        try:
            from zvec import OptimizeOption
            opt = OptimizeOption(concurrency=concurrency)
            self.collection.optimize(opt)
            # Re-read stats to report new completeness
            stats_after = self.collection.stats
            completeness = getattr(stats_after, "index_completeness", {}) or {}
            return {
                "status": "ok",
                "index_completeness_after": {str(k): float(v) for k, v in completeness.items()},
            }
        except Exception as exc:
            log.error("ZvecBackend.optimize() failed: %s", exc, exc_info=True)
            return {"status": "error", "message": str(exc)}

    def get_segment_health(self) -> dict:
        """Return Zvec index completeness metrics for the /health endpoint.

        ``index_completeness`` maps each vector field name to a float in
        [0.0, 1.0].  A value < 1.0 means some segments are un-indexed and
        Zvec will log ``[ERROR] segment.cc:711`` for queries that touch them.
        Searches still return correct results via brute-force fallback.

        Returns:
            dict with ``index_completeness`` and ``embedding_indexed`` bool.
        """
        if self.collection is None:
            return {
                "index_completeness": {},
                "embedding_indexed": False,
                "degraded": True,
            }
        try:
            stats = self.collection.stats
            completeness: dict = getattr(stats, "index_completeness", {}) or {}
            completeness_safe = {str(k): float(v) for k, v in completeness.items()}
            embedding_complete = float(completeness_safe.get("embedding", 1.0))
            return {
                "index_completeness": completeness_safe,
                "embedding_indexed": embedding_complete >= 1.0,
                "embedding_completeness": embedding_complete,
            }
        except Exception as exc:
            log.debug("ZvecBackend.get_segment_health() failed: %s", exc)
            return {"index_completeness": {}, "embedding_indexed": False, "error": str(exc)}

    def get_stats(self) -> Dict:
        """Get collection stats including segment health metrics.

        The returned ``index_completeness`` dict maps each vector field to the
        fraction of segments that are fully indexed (0.0–1.0).  When
        ``index_completeness["embedding"] < 1.0``, Zvec logs the benign
        ``[ERROR] segment.cc:711 vector indexer not found for field embedding``
        for queries that touch un-indexed segments.  Call ``optimize()`` to
        resolve this immediately.
        """
        if self.collection is None:
            return {"entities": 0, "backend": "zvec", "degraded": True}
        stats = self.collection.stats
        entities = getattr(stats, "num_entities", None)
        if entities is None:
            entities = getattr(stats, "doc_count", 0)
        completeness: dict = getattr(stats, "index_completeness", {}) or {}
        completeness_safe = {str(k): float(v) for k, v in completeness.items()}
        return {
            "entities": int(entities or 0),
            "backend": "zvec",
            "index_completeness": completeness_safe,
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

        self._search_lock = threading.Lock()
        self._similarity_search_durations_ms: list[float] = []
        self._similarity_search_count = 0
        self._similarity_search_error_count = 0

        log.info(f"VectorStore using {self.backend.__class__.__name__}")
    
    def add_entity(self, entity_id: str, name: str, entity_type: str,
                   dense_embedding: List[float], content: str, confidence: float = 1.0):
        return self.backend.add_entity(entity_id, name, entity_type,
                                       dense_embedding, content, confidence)

    def remove_entity(self, entity_id: str) -> bool:
        """Delete a vector by entity_id. Returns True if deleted."""
        try:
            return self.backend.remove_entity(entity_id)
        except Exception:
            log.debug("VectorStore remove_entity failed for %s", entity_id, exc_info=True)
            return False

    def update_entity(self, entity_id: str, name: str, entity_type: str,
                      dense_embedding: List[float], content: str,
                      confidence: float = 1.0) -> None:
        """Upsert/replace a vector for an existing entity."""
        try:
            self.backend.update_entity(entity_id, name, entity_type,
                                       dense_embedding, content, confidence)
        except Exception:
            log.debug("VectorStore update_entity failed for %s", entity_id, exc_info=True)

    def list_all_entity_ids(self) -> Optional[List[str]]:
        """Return all stored entity_ids, or None if not supported by the backend."""
        try:
            return self.backend.list_all_entity_ids()
        except NotImplementedError:
            return None
        except Exception:
            log.debug("VectorStore list_all_entity_ids failed", exc_info=True)
            return None

    def similarity_search(self, query_embedding: List[float], top_k: int = 10,
                          entity_type: Optional[str] = None) -> List[Dict]:
        start = time.perf_counter()
        result_count = 0
        failed = False
        try:
            results = self.backend.similarity_search(query_embedding, top_k, entity_type)
            result_count = len(results)
            return results
        except Exception:
            failed = True
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            with self._search_lock:
                self._similarity_search_count += 1
                if failed:
                    self._similarity_search_error_count += 1
                self._similarity_search_durations_ms.append(duration_ms)
                if len(self._similarity_search_durations_ms) > 10_000:
                    del self._similarity_search_durations_ms[:5_000]

            log.info(
                "vector_similarity_search backend=%s top_k=%s entity_type=%s duration_ms=%.2f results=%s failed=%s",
                self.backend.__class__.__name__,
                top_k,
                entity_type or "",
                duration_ms,
                result_count,
                failed,
            )
    
    def keyword_search(self, query: str, top_k: int = 10) -> List[Dict]:
        return self.backend.keyword_search(query, top_k)
    
    def hybrid_search(self, query_embedding: List[float], query_text: str,
                      top_k: int = 10, dense_weight: float = 0.7) -> List[Dict]:
        return self.backend.hybrid_search(query_embedding, query_text, top_k, dense_weight)

    def is_degraded(self) -> bool:
        """Return True if the backend is in degraded mode (e.g. Zvec collection failed to open)."""
        if isinstance(self.backend, ZvecBackend):
            return self.backend.collection is None
        return False

    def zvec_optimize(self, concurrency: int = 0) -> dict:
        """Trigger Zvec segment compaction and HNSW index rebuild (no-op for non-Zvec backends).

        After this call all WAL segments are merged and fully indexed, so the
        ``[ERROR] segment.cc:711 vector indexer not found for field embedding``
        messages will stop appearing.

        Args:
            concurrency: threads to use for optimization (0 = auto).

        Returns:
            dict with ``status`` and ``index_completeness_after``.
        """
        if isinstance(self.backend, ZvecBackend):
            return self.backend.optimize(concurrency=concurrency)
        return {"status": "not_applicable", "backend": self.backend.__class__.__name__}

    def get_segment_health(self) -> dict:
        """Return Zvec segment index completeness metrics (empty dict for non-Zvec backends).

        ``index_completeness["embedding"]`` < 1.0 indicates un-indexed segments
        that produce the benign ``[ERROR] segment.cc:711`` log line.
        """
        if isinstance(self.backend, ZvecBackend):
            return self.backend.get_segment_health()
        return {}

    def flush(self):
        """Flush backend WAL to disk. Call after bulk operations (reindex, etc.)."""
        if hasattr(self.backend, 'flush'):
            self.backend.flush()

    def get_stats(self) -> Dict:
        backend_stats = self.backend.get_stats()
        with self._search_lock:
            timings = list(self._similarity_search_durations_ms)
            search_count = self._similarity_search_count
            error_count = self._similarity_search_error_count

        avg_latency = sum(timings) / len(timings) if timings else 0.0
        return {
            **backend_stats,
            "similarity_search_count": search_count,
            "similarity_search_error_count": error_count,
            "similarity_search_avg_ms": round(avg_latency, 2),
            "similarity_search_p95_ms": round(self._percentile(timings, 95), 2),
            "similarity_search_last_ms": round(timings[-1], 2) if timings else 0.0,
        }

    @staticmethod
    def _percentile(values: List[float], pct: float) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = int(len(sorted_vals) * pct / 100)
        idx = min(idx, len(sorted_vals) - 1)
        return sorted_vals[idx]
