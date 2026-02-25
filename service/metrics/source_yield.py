"""Per-source entity yield tracking for MollyGraph.

Stores ingestion quality records in a SQLite table (rolling 10K rows).
Exposes per-source aggregate stats via helper functions used by the
GET /metrics/sources endpoint.

Schema
------
source_yield_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT    NOT NULL,   -- ISO-8601 UTC
    date         TEXT    NOT NULL,   -- YYYY-MM-DD
    source       TEXT    NOT NULL,   -- ingest source label
    entity_count INTEGER NOT NULL,
    relationship_count INTEGER NOT NULL,
    unique_entity_count INTEGER NOT NULL,
    total_entity_count  INTEGER NOT NULL,
    unique_entity_rate  REAL NOT NULL,  -- unique_entity_count / total_entity_count
    quality_score       REAL NOT NULL   -- (unique_entities / total_entities) * (1 + rels * 0.1)
)
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Derive DB path from config (fallback to ~/.graph-memory/)
try:
    import config as _cfg
    _DB_PATH: Path = _cfg.GRAPH_MEMORY_DIR / "source_yield.db"
except Exception:
    _DB_PATH = Path.home() / ".graph-memory" / "source_yield.db"

_MAX_ROWS = 10_000
_TRIM_TO = 8_000   # trim to this size when limit is hit

_db_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is not None:
        return _conn
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    _conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    _conn.row_factory = sqlite3.Row
    _init_schema(_conn)
    return _conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS source_yield_log (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            ts                 TEXT    NOT NULL,
            date               TEXT    NOT NULL,
            source             TEXT    NOT NULL,
            entity_count       INTEGER NOT NULL DEFAULT 0,
            relationship_count INTEGER NOT NULL DEFAULT 0,
            unique_entity_count INTEGER NOT NULL DEFAULT 0,
            total_entity_count  INTEGER NOT NULL DEFAULT 0,
            unique_entity_rate  REAL NOT NULL DEFAULT 0.0,
            quality_score       REAL NOT NULL DEFAULT 0.0
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_syl_source ON source_yield_log(source)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_syl_date ON source_yield_log(date)"
    )
    conn.commit()


def _compute_quality(unique_entities: int, total_entities: int, rels: int) -> float:
    """quality = (unique_entities / total_entities) * (1 + relationship_count * 0.1)"""
    if total_entities <= 0:
        return 0.0
    unique_rate = unique_entities / total_entities
    return unique_rate * (1.0 + rels * 0.1)


def record_yield(
    source: str,
    entity_count: int,
    relationship_count: int,
    new_entity_count: int,   # entities NOT previously seen (unique to this ingest)
    total_entity_count: int, # total entities extracted (new + existing)
) -> None:
    """Record one ingestion event's yield metrics."""
    now = datetime.now(tz=timezone.utc)
    unique_rate = round(new_entity_count / total_entity_count, 4) if total_entity_count > 0 else 0.0
    quality = round(_compute_quality(new_entity_count, total_entity_count, relationship_count), 6)

    try:
        with _db_lock:
            conn = _get_conn()
            conn.execute(
                """
                INSERT INTO source_yield_log
                    (ts, date, source, entity_count, relationship_count,
                     unique_entity_count, total_entity_count, unique_entity_rate, quality_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now.isoformat(),
                    now.strftime("%Y-%m-%d"),
                    str(source)[:128],
                    int(entity_count),
                    int(relationship_count),
                    int(new_entity_count),
                    int(total_entity_count),
                    unique_rate,
                    quality,
                ),
            )
            conn.commit()

            # Rolling trim — keep at most _MAX_ROWS
            row_count: int = conn.execute(
                "SELECT COUNT(*) FROM source_yield_log"
            ).fetchone()[0]
            if row_count > _MAX_ROWS:
                # Delete oldest rows to bring count to _TRIM_TO
                to_delete = row_count - _TRIM_TO
                conn.execute(
                    f"DELETE FROM source_yield_log WHERE id IN "
                    f"(SELECT id FROM source_yield_log ORDER BY id ASC LIMIT {to_delete})"
                )
                conn.commit()
    except Exception:
        log.warning("source_yield: failed to record", exc_info=True)


def get_source_stats() -> dict[str, Any]:
    """Return per-source aggregate stats and top/bottom 10 by quality."""
    try:
        with _db_lock:
            conn = _get_conn()
            rows = conn.execute(
                """
                SELECT
                    source,
                    COUNT(*)               AS ingest_count,
                    AVG(entity_count)      AS avg_entity_yield,
                    AVG(relationship_count) AS avg_rel_yield,
                    AVG(unique_entity_rate) AS avg_unique_entity_rate,
                    AVG(quality_score)     AS avg_quality_score,
                    SUM(entity_count)      AS total_entities,
                    SUM(relationship_count) AS total_relationships,
                    MIN(ts)                AS first_seen,
                    MAX(ts)                AS last_seen
                FROM source_yield_log
                GROUP BY source
                ORDER BY avg_quality_score DESC
                """
            ).fetchall()
    except Exception:
        log.warning("source_yield: get_source_stats failed", exc_info=True)
        return {"sources": [], "top_10": [], "bottom_10": [], "total_records": 0}

    sources = [
        {
            "source": row["source"],
            "ingest_count": int(row["ingest_count"]),
            "avg_entity_yield": round(float(row["avg_entity_yield"] or 0), 2),
            "avg_relationship_yield": round(float(row["avg_rel_yield"] or 0), 2),
            "avg_unique_entity_rate": round(float(row["avg_unique_entity_rate"] or 0), 4),
            "avg_quality_score": round(float(row["avg_quality_score"] or 0), 6),
            "total_entities": int(row["total_entities"] or 0),
            "total_relationships": int(row["total_relationships"] or 0),
            "first_seen": row["first_seen"],
            "last_seen": row["last_seen"],
        }
        for row in rows
    ]

    top_10 = sources[:10]
    bottom_10 = list(reversed(sources[-10:])) if len(sources) >= 2 else sources

    try:
        with _db_lock:
            conn = _get_conn()
            total_records = conn.execute(
                "SELECT COUNT(*) FROM source_yield_log"
            ).fetchone()[0]
    except Exception:
        total_records = 0

    return {
        "sources": sources,
        "top_10_by_quality": top_10,
        "bottom_10_by_quality": bottom_10,
        "total_records": total_records,
    }


def get_daily_source_breakdown(date_str: str | None = None) -> list[dict[str, Any]]:
    """Return per-source breakdown for a specific date."""
    if date_str is None:
        date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    try:
        with _db_lock:
            conn = _get_conn()
            rows = conn.execute(
                """
                SELECT source,
                       COUNT(*)                AS ingest_count,
                       SUM(entity_count)       AS total_entities,
                       SUM(relationship_count) AS total_relationships,
                       AVG(unique_entity_rate) AS avg_unique_rate,
                       AVG(quality_score)      AS avg_quality
                FROM source_yield_log
                WHERE date = ?
                GROUP BY source
                ORDER BY total_entities DESC
                """,
                (date_str,),
            ).fetchall()
    except Exception:
        return []

    return [
        {
            "source": row["source"],
            "ingest_count": int(row["ingest_count"]),
            "total_entities": int(row["total_entities"] or 0),
            "total_relationships": int(row["total_relationships"] or 0),
            "avg_unique_entity_rate": round(float(row["avg_unique_rate"] or 0), 4),
            "avg_quality_score": round(float(row["avg_quality"] or 0), 6),
        }
        for row in rows
    ]
