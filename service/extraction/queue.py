"""
Async Extraction Queue - SQLite WAL Mode
Local-first, persistent queue for background processing.
"""
import asyncio
import json
import logging
import sqlite3
from datetime import datetime, UTC
from pathlib import Path
from typing import List, Optional, Callable
from contextlib import contextmanager

import config as service_config
from maintenance.lock import is_maintenance_locked
from memory.models import ExtractionJob

log = logging.getLogger(__name__)


class ExtractionQueue:
    """
    Persistent queue for extraction jobs using SQLite WAL mode.
    
    Benefits over Redis for local use:
    - Zero config (no daemon)
    - ACID transactions
    - Survives restarts
    - Minimal resource usage on Mac Mini
    """
    
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = (
            Path(db_path).expanduser()
            if db_path is not None
            else service_config.QUEUE_DB_PATH
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database with WAL mode for concurrent access."""
        with self._get_conn() as conn:
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")  # Balance safety/speed
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    priority INTEGER DEFAULT 1,
                    reference_time TEXT NOT NULL,
                    episode_id TEXT,
                    status TEXT DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    error TEXT,
                    result_json TEXT,
                    retry_count INTEGER DEFAULT 0
                )
            """)

            # Migration: add retry_count column to existing databases.
            try:
                conn.execute("ALTER TABLE jobs ADD COLUMN retry_count INTEGER DEFAULT 0")
                conn.commit()
            except Exception:
                pass  # Column already exists — no-op
            
            # Index for efficient queue polling
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status_priority 
                ON jobs(status, priority, created_at)
            """)
            
            conn.commit()
    
    @contextmanager
    def _get_conn(self):
        """Get database connection with proper isolation."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def submit(self, job: ExtractionJob) -> str:
        """Add a job to the queue."""
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO jobs (
                    id, content, source, priority, reference_time,
                    episode_id, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job.id,
                job.content,
                job.source,
                job.priority,
                job.reference_time.isoformat(),
                job.episode_id,
                job.status,
                job.created_at.isoformat()
            ))
            conn.commit()
        
        log.debug(f"Job {job.id} submitted (priority={job.priority})")
        return job.id
    
    _STUCK_JOB_TIMEOUT_SECONDS: int = 300   # jobs processing > 5 min are considered stuck
    _MAX_RETRIES: int = 3                    # jobs retried ≥ this many times become 'dead'

    def claim_next(self, timeout: float = 5.0) -> Optional[ExtractionJob]:
        """
        Atomically claim the next pending job.
        Uses SELECT FOR UPDATE pattern for concurrency safety.

        Before claiming, resets any stuck 'processing' jobs whose worker died:
        - Jobs in 'processing' with started_at older than STUCK_JOB_TIMEOUT_SECONDS
          are reset to 'pending' (retry_count incremented).
        - Jobs that have exceeded MAX_RETRIES are moved to 'dead' instead.
        """
        with self._get_conn() as conn:
            # ── Step 1: Reset stuck processing jobs ──────────────────────────
            conn.execute("BEGIN IMMEDIATE")
            try:
                result = conn.execute("""
                    UPDATE jobs
                    SET status = CASE
                            WHEN COALESCE(retry_count, 0) + 1 >= ? THEN 'dead'
                            ELSE 'pending'
                        END,
                        retry_count = COALESCE(retry_count, 0) + 1
                    WHERE status = 'processing'
                      AND started_at IS NOT NULL
                      AND started_at < datetime('now', '-' || ? || ' seconds')
                """, (self._MAX_RETRIES, self._STUCK_JOB_TIMEOUT_SECONDS))
                if result.rowcount:
                    log.warning(
                        "Reset %d stuck processing job(s) (timeout=%ds, max_retries=%d)",
                        result.rowcount, self._STUCK_JOB_TIMEOUT_SECONDS, self._MAX_RETRIES,
                    )
                conn.commit()
            except Exception as e:
                conn.rollback()
                log.warning("Failed to reset stuck jobs: %s", e)

            # ── Step 2: Claim the next pending job ────────────────────────────
            conn.execute("BEGIN IMMEDIATE")

            try:
                # Find highest priority, oldest pending job
                cursor = conn.execute("""
                    SELECT * FROM jobs 
                    WHERE status = 'pending'
                    ORDER BY priority ASC, created_at ASC
                    LIMIT 1
                """)
                
                row = cursor.fetchone()
                if not row:
                    conn.rollback()
                    return None
                
                # Mark as processing
                conn.execute("""
                    UPDATE jobs 
                    SET status = 'processing', started_at = ?
                    WHERE id = ? AND status = 'pending'
                """, (datetime.now(UTC).isoformat(), row['id']))
                
                conn.commit()
                
                return self._row_to_job(row)
                
            except Exception as e:
                conn.rollback()
                raise e
    
    def complete(self, job_id: str, success: bool = True, error: str = None,
                 result: dict = None):
        """Mark a job as completed or failed."""
        status = 'completed' if success else 'failed'
        
        def _serialize(obj):
            """Recursively serialize datetime objects."""
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, list):
                return [_serialize(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}
            return obj
        
        with self._get_conn() as conn:
            conn.execute("""
                UPDATE jobs 
                SET status = ?, completed_at = ?, error = ?, result_json = ?
                WHERE id = ?
            """, (
                status,
                datetime.now(UTC).isoformat(),
                error,
                json.dumps(_serialize(result)) if result else None,
                job_id
            ))
            conn.commit()
    
    def get_pending_count(self) -> int:
        """Get count of pending jobs."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status = 'pending'"
            )
            return cursor.fetchone()[0]
    
    def get_processing_count(self) -> int:
        """Get count of jobs currently processing."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status = 'processing'"
            )
            return cursor.fetchone()[0]

    def get_stuck_count(self) -> int:
        """Get count of jobs stuck in 'processing' beyond the timeout threshold."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM jobs
                WHERE status = 'processing'
                  AND started_at IS NOT NULL
                  AND started_at < datetime('now', '-' || ? || ' seconds')
                """,
                (self._STUCK_JOB_TIMEOUT_SECONDS,),
            )
            return cursor.fetchone()[0]

    def get_dead_count(self) -> int:
        """Get count of jobs that exhausted all retries and are permanently dead."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status = 'dead'"
            )
            return cursor.fetchone()[0]

    def get_recent_jobs(self, limit: int = 100) -> List[ExtractionJob]:
        """Get recent jobs for monitoring."""
        with self._get_conn() as conn:
            cursor = conn.execute("""
                SELECT * FROM jobs 
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
            
            return [self._row_to_job(row) for row in cursor.fetchall()]
    
    def cleanup_old(self, days: int = 7):
        """Remove completed jobs older than N days."""
        with self._get_conn() as conn:
            conn.execute("""
                DELETE FROM jobs 
                WHERE status IN ('completed', 'failed')
                AND completed_at < datetime('now', '-{} days')
            """.format(days))
            conn.commit()
            
            # Vacuum to reclaim space
            conn.execute("VACUUM")
    
    def _row_to_job(self, row: sqlite3.Row) -> ExtractionJob:
        """Convert database row to ExtractionJob model."""
        return ExtractionJob(
            id=row['id'],
            content=row['content'],
            source=row['source'],
            priority=row['priority'],
            reference_time=datetime.fromisoformat(row['reference_time']),
            episode_id=row['episode_id'],
            status=row['status'],
            created_at=datetime.fromisoformat(row['created_at']),
            started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else None,
            completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
            error=row['error']
        )


class QueueWorker:
    """
    Background worker that polls the queue and processes jobs.
    """
    
    def __init__(self, queue: ExtractionQueue, processor: Callable, 
                 poll_interval: float = 1.0, max_concurrent: int = 3):
        self.queue = queue
        self.processor = processor
        self.poll_interval = poll_interval
        self.max_concurrent = max_concurrent
        self.running = False
        self._tasks = set()
    
    async def start(self):
        """Start the worker loop."""
        self.running = True
        log.info(f"Queue worker started (poll_interval={self.poll_interval}s)")
        
        while self.running:
            # Pause extraction during maintenance to avoid race conditions
            if is_maintenance_locked():
                log.debug("Maintenance lock active — extraction paused")
                await asyncio.sleep(5.0)
                continue

            # Limit concurrent processing
            active = len([t for t in self._tasks if not t.done()])
            
            if active < self.max_concurrent:
                job = await asyncio.to_thread(self.queue.claim_next)
                
                if job:
                    # Process in background
                    task = asyncio.create_task(self._process_job(job))
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)
                else:
                    # No jobs, sleep
                    await asyncio.sleep(self.poll_interval)
            else:
                # At capacity, wait for slot
                await asyncio.sleep(0.1)
    
    async def _process_job(self, job: ExtractionJob):
        """Process a single job."""
        try:
            log.debug(f"Processing job {job.id}")
            result = await self.processor(job)
            
            await asyncio.to_thread(
                self.queue.complete,
                job.id,
                result.status == 'completed',
                result.error,
                result.model_dump() if result.status == 'completed' else None,
            )
            
        except Exception as e:
            log.error(f"Job {job.id} failed: {e}", exc_info=True)
            await asyncio.to_thread(self.queue.complete, job.id, False, str(e), None)
    
    def stop(self):
        """Signal the worker to stop."""
        self.running = False
        log.info("Queue worker stopping...")
